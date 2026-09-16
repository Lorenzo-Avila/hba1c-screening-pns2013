"""Testes dos invariantes de ortogonalidade e do preprocessador (§4 da spec).

Dois blocos. O primeiro valida os conjuntos de atributos como estrutura — não
depende dos microdados e roda sempre. O segundo executa o pipeline de ponta a
ponta sobre a base real e depende do CSV bruto, não versionado.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src import config as cfg
from src import features as ft
from src.data import carregar_bruto, limpar, separar

# As 16 variáveis conceituais da §6.4.1, agrupadas nos cinco blocos temáticos.
VARIAVEIS_TCC = {
    "antropometrico": ["idade", "sexo", "imc", "circ_cintura", "rcest"],
    "pressao": ["pas", "pad", "has_autorrelato", "has_combinada"],
    "diagnostico": ["colesterol_alto", "multimorbidade"],
    "sociodemografico": ["raca_cor", "escolaridade", "plano_saude", "regiao"],
    "autopercepcao": ["saude_autoaval"],
}

# Componentes do bloco de pressão e de diagnóstico autorrelatado — as variáveis
# cuja duplicação a §4.1 identifica como violação do Princípio 4.
COMPONENTES_CRITICOS = {"pas", "pad", "has_autorrelato", "colesterol_alto"}


# ---------------------------------------------------------------------------
# §4.2 — invariantes dos conjuntos de atributos
# ---------------------------------------------------------------------------


def test_conjuntos_declarados():
    assert set(ft.FEATURE_SETS) == {"A_atomico", "B_composto", "C_completo"}


def test_toda_coluna_tem_proveniencia_declarada():
    """Sem proveniência não há como auditar canal duplo."""
    for nome in ft.FEATURE_SETS:
        for coluna in ft.colunas_do_conjunto(nome):
            assert coluna in ft.FONTES, f"{nome}: {coluna} sem entrada em FONTES"


def test_A_tem_ortogonalidade_plena():
    """Em A nenhuma variável fonte alimenta o modelo por dois canais (§4.2)."""
    assert ft.fontes_em_canal_duplo("A_atomico") == set()


def test_A_e_B_cumprem_o_invariante_estrito():
    """Nenhuma fonte entra como coluna própria e parcela idêntica de agregada (§4.2 v2)."""
    assert ft.fontes_isoladas_e_agregadas("A_atomico") == set()
    assert ft.fontes_isoladas_e_agregadas("B_composto") == set()


def test_B_tem_sobreposicao_apenas_por_coarsening():
    """A redundância admitida em B é antropométrica e binarizada por corte clínico."""
    coarsening_b = ft.fontes_em_coarsening("B_composto")
    assert coarsening_b == {"peso_kg", "altura_cm", "circ_cintura", "sexo"}
    # Nada de diagnóstico autorrelatado nem de pressão aferida.
    assert coarsening_b.isdisjoint(COMPONENTES_CRITICOS)


def test_B_nao_carrega_has_combinada_ao_lado_do_escore():
    """D-012: `has_combinada` é parcela literal de `multimorbidade`.

    Mantê-la ao lado do escore reproduziria em B a violação que a §4.1 acusa no
    baseline, e o modelo recuperaria a parcela linearmente — esvaziando a
    hipótese de sinergia que B existe para testar.
    """
    colunas = ft.colunas_do_conjunto("B_composto")
    assert "has_combinada" not in colunas
    assert "multimorbidade" in colunas


@pytest.mark.parametrize("nome", ["A_atomico", "B_composto"])
def test_diagnostico_autorrelatado_entra_uma_unica_vez(nome):
    """`has_autorrelato` e `colesterol_alto` não entram isolados e agregados em A e B.

    É exatamente a duplicação do baseline corrigida por D-001: três canais para
    `has_autorrelato`, dois para `colesterol_alto`.
    """
    colunas = ft.colunas_do_conjunto(nome)
    for fonte in ft.DIAGNOSTICO_AUTORRELATADO:
        isolada = any(
            ft.FONTES[c] == {fonte} for c in colunas if c not in ft.AGREGADAS
        )
        agregada = any(fonte in ft.FONTES[c] for c in colunas if c in ft.AGREGADAS)
        assert not (isolada and agregada), f"{nome}: {fonte} entra por dois canais"


def test_C_reproduz_a_redundancia_do_baseline():
    """Falha se a redundância desaparecer: em C ela é proposital (§4.2 v2)."""
    duplicadas = ft.fontes_isoladas_e_agregadas("C_completo")
    assert {"has_autorrelato", "colesterol_alto"} <= duplicadas
    assert "has_combinada" in ft.colunas_do_conjunto("C_completo")


def test_rcest_apenas_em_C():
    """Remoção de `rcest` em A e B é a experimentação prevista na §6.5."""
    assert "rcest" in ft.colunas_do_conjunto("C_completo")
    assert "rcest" not in ft.colunas_do_conjunto("A_atomico")
    assert "rcest" not in ft.colunas_do_conjunto("B_composto")


def test_C_contem_as_16_variaveis_da_secao_641():
    """Nenhuma variável justificada no TCC I pode deixar de ser avaliada."""
    esperadas = [v for bloco in VARIAVEIS_TCC.values() for v in bloco]
    assert len(esperadas) == 16

    colunas = ft.colunas_do_conjunto("C_completo")
    fontes_de_c = {f for c in colunas for f in ft.FONTES[c]} | set(colunas)
    # `sexo` e `plano_saude` entram recodificadas (§4.3); `imc`, pelas fontes.
    for variavel in esperadas:
        assert variavel in fontes_de_c, f"§6.4.1: {variavel} ausente de C"


def test_nenhum_conjunto_usa_hba1c_peso_lab_ou_comportamentais():
    """R6 (HbA1c só como desfecho), R10 (`peso_lab` fora do treino) e R5."""
    proibidas = {"hba1c", "target", "peso_lab",
                 "dieta", "atividade_fisica", "tabagismo"}
    for nome in ft.FEATURE_SETS:
        colunas = set(ft.colunas_do_conjunto(nome))
        fontes = {f for c in colunas for f in ft.FONTES[c]}
        assert proibidas.isdisjoint(colunas | fontes), nome


def test_raca_cor_nao_agrupada():
    """D-007 está com status 'não implementada' — as cinco categorias permanecem."""
    for nome in ft.FEATURE_SETS:
        colunas = ft.colunas_do_conjunto(nome)
        assert "raca_cor" in colunas
        assert not any(c.startswith("raca_cor_") for c in colunas)


def test_conjunto_desconhecido_falha_alto():
    with pytest.raises(ValueError):
        ft.construir_preprocessador("D_inexistente")


# ---------------------------------------------------------------------------
# §4.4 e §4.5 — compostas e imputação dentro do pipeline
# ---------------------------------------------------------------------------


def test_compostas_sao_transformador_nao_funcao_de_dataframe():
    """R2: as compostas dependem de imputação, logo só existem dentro do pipeline."""
    pre = ft.construir_preprocessador("B_composto")
    assert [nome for nome, _ in pre.steps] == ["imputacao", "compostas", "preparo"]
    assert isinstance(pre.named_steps["compostas"], ft.ConstrutorCompostas)
    assert isinstance(pre.named_steps["imputacao"], ft.ImputadorPNS)


def test_imputador_categorico_arredonda_para_codigo_valido():
    """§4.5: o KNN devolve médias; o one-hot exige código inteiro da PNS."""
    treino = pd.DataFrame({
        "raca_cor": [1.0, 1.0, 2.0, 4.0, 5.0, 4.0],
        "regiao": [1.0, 2.0, 3.0, 4.0, 5.0, 3.0],
    })
    teste = pd.DataFrame({"raca_cor": [np.nan, 2.0], "regiao": [3.0, np.nan]})

    imp = ft.KNNImputerCategorico(n_neighbors=3).fit(treino)
    saida = imp.transform(teste)

    assert saida.isna().sum().sum() == 0
    assert (saida == saida.round(0)).all().all()
    assert saida["raca_cor"].between(1, 5).all()
    assert saida["regiao"].between(1, 5).all()


def test_multimorbidade_e_has_combinada_seguem_a_secao_44():
    """Escore ordinal 0–3; código 2 de Q002 ('apenas na gravidez') não é HAS crônica."""
    entrada = pd.DataFrame({
        "idade": [50.0, 60.0, 40.0],
        "sexo": [2.0, 1.0, 2.0],            # feminino, masculino, feminino
        "imc": [22.0, 35.0, 24.0],
        "circ_cintura": [70.0, 100.0, 85.0],
        "rcest": [0.45, 0.58, 0.52],
        "pas": [120.0, 145.0, 130.0],
        "pad": [80.0, 85.0, 85.0],
        "has_autorrelato": [3.0, 3.0, 2.0],  # não, não, apenas na gravidez
        "colesterol_alto": [2.0, 1.0, 2.0],
        "raca_cor": [1.0, 2.0, 4.0],
        "escolaridade": [3.0, 2.0, 5.0],
        "plano_saude": [1.0, 2.0, 2.0],
        "regiao": [3.0, 1.0, 4.0],
        "saude_autoaval": [2.0, 4.0, 3.0],
    })
    saida = ft.ConstrutorCompostas().fit_transform(entrada)

    # Linha 0: sem HAS, sem colesterol, sem obesidade -> 0
    # Linha 1: PAS 145 -> HAS; colesterol 1; IMC 35 e cintura 100 (M) -> 3
    # Linha 2: gravidez não conta como HAS; cintura 85 >= 80 (F) -> 1
    assert list(saida["has_combinada"]) == [0, 1, 0]
    assert list(saida["multimorbidade"]) == [0, 3, 1]
    assert saida["multimorbidade"].between(0, 3).all()
    assert list(saida["sexo_fem"]) == [1, 0, 1]
    assert list(saida["plano_sim"]) == [1, 0, 0]
    assert list(saida["colesterol_sim"]) == [0, 1, 0]


# ---------------------------------------------------------------------------
# Pipeline sobre a base real
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def split():
    """Divisão 80/20 estratificada pelo alvo (R1), com a semente do projeto (R10)."""
    if not cfg.PATH_RAW.exists():
        pytest.skip(f"CSV bruto ausente em {cfg.PATH_RAW} (não versionado).")
    X, y, _ = separar(limpar(carregar_bruto()))
    return train_test_split(
        X, y, test_size=cfg.TEST_SIZE, stratify=y, random_state=cfg.RANDOM_STATE
    )


@pytest.mark.parametrize("nome", ["A_atomico", "B_composto", "C_completo"])
def test_pipeline_roda_e_nao_deixa_ausentes(split, nome):
    X_train, X_test, y_train, _ = split
    pre = ft.construir_preprocessador(nome)
    matriz_treino = pre.fit_transform(X_train, y_train)
    matriz_teste = pre.transform(X_test)

    assert matriz_treino.shape[0] == len(X_train)
    assert matriz_teste.shape[0] == len(X_test)
    assert matriz_treino.shape[1] == matriz_teste.shape[1]
    assert np.isnan(matriz_treino).sum() == 0
    assert np.isnan(matriz_teste).sum() == 0


def test_imputacao_ajustada_apenas_no_treino(split):
    """R2: o `fit` não pode ter visto o conjunto de teste."""
    X_train, X_test, y_train, _ = split
    pre = ft.construir_preprocessador("A_atomico")
    pre.fit(X_train, y_train)

    escalonador = pre.named_steps["preparo"].named_transformers_["num"]
    assert escalonador.n_samples_seen_ == len(X_train)
    # Transformar só uma fatia do teste dá o mesmo resultado que transformar tudo:
    # prova de que nenhuma estatística é recalculada em `transform`.
    completo = pre.transform(X_test)
    fatia = pre.transform(X_test.iloc[:50])
    assert np.allclose(completo[:50], fatia)


def test_raca_cor_gera_quatro_dummies(split):
    """Cinco categorias com `drop='first'` -> quatro colunas. D-007 não implementada."""
    X_train, _, y_train, _ = split
    pre = ft.construir_preprocessador("A_atomico")
    pre.fit(X_train, y_train)
    nomes = list(pre.named_steps["preparo"].get_feature_names_out())
    assert len([n for n in nomes if n.startswith("raca_cor_")]) == 4


def test_colesterol_alto_imputado_por_knn(split):
    """R3: o KNN é o método primário; D-008 (categoria explícita) não entra agora."""
    X_train, _, y_train, _ = split
    assert X_train["colesterol_alto"].isna().sum() > 0

    pre = ft.construir_preprocessador("A_atomico")
    pre.fit(X_train, y_train)
    imputado = pre.named_steps["imputacao"].transform(X_train)

    assert imputado["colesterol_alto"].isna().sum() == 0
    # Nenhuma categoria nova: a imputação devolve apenas códigos oficiais.
    assert set(imputado["colesterol_alto"].unique()) <= set(cfg.COD_COLEST)


def test_preprocessador_nao_recebe_peso_lab(split):
    """R10/§9: nenhuma função de `features.py` aceita o peso amostral."""
    X_train, _, y_train, _ = split
    assert cfg.PESO not in X_train.columns
    pre = ft.construir_preprocessador("C_completo")
    pre.fit(X_train, y_train)
    assert cfg.PESO not in pre.named_steps["imputacao"].columns_


# ---------------------------------------------------------------------------
# D-014 — espaço métrico da imputação
# ---------------------------------------------------------------------------


def _participacao_na_distancia(matriz: pd.DataFrame) -> pd.Series:
    """Fração da distância euclidiana quadrática média atribuível a cada coluna.

    Para pares de pontos, a contribuição esperada de cada dimensão à distância
    quadrática é proporcional à variância daquela coluna. A participação relativa
    mede, portanto, quanto cada variável de fato pesa na escolha dos vizinhos.
    """
    variancias = matriz.astype("float64").var(ddof=0)
    return variancias / variancias.sum()


def test_espaco_metrico_equaliza_a_contribuicao_das_continuas(split):
    """D-014: padronizadas, todas as contínuas pesam igual na distância."""
    X_train, _, _, _ = split
    imp = ft.ImputadorPNS().fit(X_train)
    metrico = imp.espaco_metrico(X_train)

    desvios = metrico[imp.continuas_].std(ddof=0)
    assert np.allclose(desvios, 1.0, atol=1e-6)
    # `sexo_fem` entra em 0/1, sem padronizar: peso deliberadamente menor.
    assert 0.3 < metrico[ft.ImputadorPNS.METRICA_BINARIA].std(ddof=0) < 0.6


def test_imc_e_rcest_deixam_de_ser_irrelevantes_na_distancia(split):
    """O defeito corrigido: na escala bruta, quem tem mais amplitude decide o vizinho."""
    X_train, _, _, _ = split
    imp = ft.ImputadorPNS().fit(X_train)

    bruta = _participacao_na_distancia(X_train[imp.continuas_])
    metrica = _participacao_na_distancia(imp.espaco_metrico(X_train)[imp.continuas_])

    # Antes: `rcest` some, e as três de maior amplitude decidem quase tudo.
    assert bruta["rcest"] < 0.001
    assert bruta[["pas", "idade", "circ_cintura"]].sum() > 0.80
    # Depois: nenhuma contínua abaixo de 10% nem acima de 25% da distância.
    assert metrica.min() > 0.10
    assert metrica.max() < 0.25
    assert metrica["imc"] > 5 * bruta["imc"]
    assert metrica["rcest"] > 100 * bruta["rcest"]


def test_vizinhos_mudam_em_relacao_a_escala_bruta(split):
    """A correção precisa alterar resultados — caso contrário não corrigiu nada.

    Comparador: a versão anterior imputava as categóricas por distância entre os
    próprios códigos nominais, sem padronização e sem as contínuas.
    """
    X_train, _, _, _ = split
    imp = ft.ImputadorPNS().fit(X_train)
    novo = imp.transform(X_train)

    anterior = ft.KNNImputerCategorico().fit(X_train[imp.categoricas_])
    antigo = anterior.transform(X_train[imp.categoricas_])

    divergencias = {}
    for coluna in ["colesterol_alto", "has_autorrelato"]:
        faltantes = X_train[coluna].isna()
        assert faltantes.sum() > 0, coluna
        divergentes = (
            antigo.loc[faltantes, coluna].to_numpy()
            != novo.loc[faltantes, coluna].to_numpy()
        )
        divergencias[coluna] = divergentes.mean()

    # Divergência substantiva, não ruído de arredondamento.
    assert divergencias["colesterol_alto"] > 0.02
    assert divergencias["has_autorrelato"] > 0.10


def test_nominais_ficam_fora_do_espaco_metrico(split):
    """Códigos de `raca_cor` e `regiao` não são distância — mas seguem sendo alvo."""
    X_train, _, _, _ = split
    imp = ft.ImputadorPNS().fit(X_train)
    metrico = imp.espaco_metrico(X_train)

    fora = ["raca_cor", "regiao", "has_autorrelato", "escolaridade", "saude_autoaval"]
    assert set(metrico.columns) == set(imp.continuas_) | {ft.ImputadorPNS.METRICA_BINARIA}
    for coluna in fora:
        assert coluna not in metrico.columns
        # Fora da métrica, dentro dos alvos.
        assert coluna in imp.imputadores_categoricos_


def test_continuas_voltam_a_escala_original(split):
    """As compostas da §4.4 comparam com cortes clínicos em unidades reais."""
    X_train, _, _, _ = split
    imp = ft.ImputadorPNS().fit(X_train)
    saida = imp.transform(X_train)

    # Valores já observados não podem ser alterados pela ida e volta da escala.
    for coluna in imp.continuas_:
        observados = X_train[coluna].notna()
        assert np.allclose(
            saida.loc[observados, coluna], X_train.loc[observados, coluna].astype(float)
        )
    # Faixas clínicas plausíveis, não desvios-padrão.
    assert 100 < saida["pas"].mean() < 140
    assert 20 < saida["imc"].mean() < 35


def test_sexo_ausente_nao_vira_masculino():
    """`sexo_fem` preserva `NaN`: a dimensão deixa de votar em vez de votar errado."""
    imp = ft.ImputadorPNS()
    entrada = pd.DataFrame({"sexo": [1.0, 2.0, np.nan]})
    sexo_fem = imp._sexo_fem(entrada)
    assert list(sexo_fem[:2]) == [0.0, 1.0]
    assert pd.isna(sexo_fem.iloc[2])


def test_escalonador_interno_ajustado_apenas_no_treino(split):
    """R2: o espaço métrico é definido pelo treino e aplicado ao teste."""
    X_train, X_test, _, _ = split
    imp = ft.ImputadorPNS().fit(X_train)

    assert imp.escalonador_.n_samples_seen_ == len(X_train)
    # No teste, as médias não são zero — sinal de que a escala veio do treino.
    metrico_teste = imp.espaco_metrico(X_test)
    assert not np.allclose(metrico_teste[imp.continuas_].mean(), 0.0, atol=1e-9)
    # E o escalonador do ColumnTransformer segue sendo outro, intacto.
    pre = ft.construir_preprocessador("C_completo").fit(X_train)
    assert pre.named_steps["preparo"].named_transformers_["num"] is not imp.escalonador_
