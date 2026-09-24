"""Testes do diagnóstico de multicolinearidade.

O ponto central é que o VIF precisa enxergar a matriz de design **inteira**, dummies
incluídas. Foi por calcular VIF só sobre o bloco numérico cru que o baseline não
detectou a redundância entre as variáveis compostas e seus componentes.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src import colinearidade as col
from src import config as cfg
from src import features as ft
from src.data import carregar_bruto, limpar, separar


@pytest.fixture(scope="module")
def split():
    """Divisão 80/20 estratificada, com a semente do projeto."""
    if not cfg.PATH_RAW.exists():
        pytest.skip(f"CSV bruto ausente em {cfg.PATH_RAW} (não versionado).")
    X, y, _ = separar(limpar(carregar_bruto()))
    return train_test_split(
        X, y, test_size=cfg.TEST_SIZE, stratify=y, random_state=cfg.RANDOM_STATE
    )


@pytest.fixture(scope="module")
def preprocessadores(split):
    """Um preprocessador por conjunto de atributos, ajustado no treino."""
    X_train, _, y_train, _ = split
    return {
        nome: ft.construir_preprocessador(nome).fit(X_train, y_train)
        for nome in ft.FEATURE_SETS
    }


# ---------------------------------------------------------------------------
# VIF sobre a matriz de design
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nome", ["A_atomico", "B_composto", "C_completo"])
def test_linhas_batem_com_get_feature_names_out(split, preprocessadores, nome):
    """Uma linha por coluna da matriz de design: nem a mais, nem a menos."""
    X_train, _, _, _ = split
    pre = preprocessadores[nome]

    esperado = list(pre.get_feature_names_out())
    tabela = col.vif_matriz_design(pre, X_train, nome)

    assert len(tabela) == len(esperado)
    assert set(tabela["variavel"]) == set(esperado)
    assert (tabela["feature_set"] == nome).all()


@pytest.mark.parametrize("nome", ["A_atomico", "B_composto", "C_completo"])
def test_vif_inclui_colunas_dummy(split, preprocessadores, nome):
    """Sem as dummies no cálculo, o diagnóstico não vê a redundância nominal."""
    X_train, _, _, _ = split
    tabela = col.vif_matriz_design(preprocessadores[nome], X_train, nome)

    dummies = [v for v in tabela["variavel"] if v.startswith(("raca_cor_", "regiao_"))]
    assert len(dummies) >= 8  # 4 de raça/cor + 4 de região, com drop="first"
    assert tabela.loc[tabela["variavel"].isin(dummies), "vif"].notna().all()


def test_vif_de_C_inclui_has_combinada_e_has_autorrelato(split, preprocessadores):
    """As duas colunas que o baseline deixava fora do diagnóstico."""
    X_train, _, _, _ = split
    tabela = col.vif_matriz_design(preprocessadores["C_completo"], X_train, "C_completo")
    variaveis = set(tabela["variavel"])

    assert "has_combinada" in variaveis
    assert any(v.startswith("has_autorrelato_") for v in variaveis)
    assert "colesterol_sim" in variaveis


def test_matriz_de_design_tem_mais_colunas_que_o_bloco_numerico(split, preprocessadores):
    """A matriz pós-transformação é maior que as contínuas, e o ponto é esse."""
    X_train, _, _, _ = split
    design = col.matriz_de_design(preprocessadores["C_completo"], X_train)
    numericas = ft.FEATURE_SETS["C_completo"]["num"]

    assert design.shape[1] > len(numericas)
    assert len(design) == len(X_train)


def test_pipeline_completo_descarta_o_estimador(split, preprocessadores):
    """`pipeline[:-1]` só vale quando o último estágio é o estimador."""
    X_train, _, y_train, _ = split
    pre = preprocessadores["A_atomico"]
    modelo = Pipeline([
        ("pre", ft.construir_preprocessador("A_atomico")),
        ("clf", LogisticRegression(max_iter=200, class_weight="balanced",
                                   random_state=cfg.RANDOM_STATE)),
    ]).fit(X_train, y_train)

    do_preprocessador = col.matriz_de_design(pre, X_train)
    do_modelo = col.matriz_de_design(modelo, X_train)

    assert list(do_modelo.columns) == list(do_preprocessador.columns)
    assert np.allclose(do_modelo.to_numpy(), do_preprocessador.to_numpy())


def test_pipeline_nao_ajustado_falha_alto():
    """Não há diagnóstico válido sobre pipeline que nunca viu o treino."""
    with pytest.raises(NotFittedError):
        col.matriz_de_design(
            ft.construir_preprocessador("A_atomico"),
            pd.DataFrame({"idade": [40.0, 50.0]}),
        )


# ---------------------------------------------------------------------------
# Aritmética do VIF
# ---------------------------------------------------------------------------


def test_colinearidade_perfeita_devolve_infinito():
    """`R² >= 1` não pode estourar a divisão: VIF infinito é o diagnóstico correto."""
    base = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "b": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
    })
    base["c"] = base["a"] + base["b"]  # combinação linear exata

    assert np.isinf(col._vif_de_coluna(base, "c"))
    assert np.isinf(col._vif_de_coluna(base, "a"))


def test_coluna_constante_devolve_infinito():
    """Dummy de categoria ausente no treino não tem regressão auxiliar possível."""
    base = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0],
        "constante": [1.0, 1.0, 1.0, 1.0],
    })
    assert np.isinf(col._vif_de_coluna(base, "constante"))


def test_colunas_independentes_tem_vif_proximo_de_um():
    """Sem correlação, VIF fica perto de 1, que é o piso da escala."""
    rng = np.random.default_rng(cfg.RANDOM_STATE)
    base = pd.DataFrame(rng.normal(size=(500, 3)), columns=["a", "b", "c"])
    for coluna in base.columns:
        assert 0.9 < col._vif_de_coluna(base, coluna) < 1.2


def test_vif_cresce_com_a_correlacao():
    """Monotonicidade: mais correlação, VIF maior."""
    rng = np.random.default_rng(cfg.RANDOM_STATE)
    a = rng.normal(size=1000)
    fraca = pd.DataFrame({"a": a, "b": a * 0.3 + rng.normal(size=1000)})
    forte = pd.DataFrame({"a": a, "b": a * 0.97 + rng.normal(scale=0.1, size=1000)})

    assert col._vif_de_coluna(forte, "a") > col._vif_de_coluna(fraca, "a")


def test_sinalizacao_usa_o_limiar_do_config():
    """O limiar é 10 e vem de `config.VIF_LIMIAR`."""
    assert cfg.VIF_LIMIAR == 10
    tabela = pd.DataFrame({
        "feature_set": ["X", "X"],
        "variavel": ["baixo", "alto"],
        "vif": [cfg.VIF_LIMIAR - 0.5, cfg.VIF_LIMIAR + 0.5],
    })
    tabela["acima_do_limiar"] = tabela["vif"] > cfg.VIF_LIMIAR
    assert list(tabela["acima_do_limiar"]) == [False, True]


# ---------------------------------------------------------------------------
# Pearson entre contínuas
# ---------------------------------------------------------------------------


def test_matriz_pearson_forma_e_simetria(split):
    """A metodologia pede Pearson **e** VIF; o baseline entregava só o segundo."""
    X_train, _, _, _ = split
    continuas = ["idade", "imc", "circ_cintura", "pas", "pad"]
    matriz = col.matriz_pearson(X_train, continuas)

    assert list(matriz.columns) == continuas
    assert list(matriz.index) == continuas
    assert np.allclose(np.diag(matriz), 1.0)
    assert np.allclose(matriz.to_numpy(), matriz.to_numpy().T)
    assert matriz.to_numpy().min() >= -1 and matriz.to_numpy().max() <= 1


def test_matriz_pearson_so_no_treino(split):
    """Correlação é estatística entre linhas, estimada só no treino."""
    X_train, X_test, _, _ = split
    continuas = ["idade", "imc"]
    do_treino = col.matriz_pearson(X_train, continuas)
    do_teste = col.matriz_pearson(X_test, continuas)
    # Valores distintos confirmam que nada foi calculado sobre a base inteira.
    assert not np.allclose(do_treino.to_numpy(), do_teste.to_numpy())


def test_matriz_pearson_rejeita_coluna_ausente(split):
    X_train, _, _, _ = split
    with pytest.raises(KeyError):
        col.matriz_pearson(X_train, ["idade", "variavel_inexistente"])


def test_heatmap_salva_figura_e_devolve_path(split, tmp_path):
    """Convenção do projeto: função que produz figura salva e retorna o caminho."""
    X_train, _, _, _ = split
    matriz = col.matriz_pearson(X_train, ["idade", "imc", "pas"])
    destino = col.heatmap_pearson(matriz, "A_atomico", path_figuras=tmp_path)

    assert destino.name == "pearson_A_atomico.png"
    assert destino.exists() and destino.stat().st_size > 0


# ---------------------------------------------------------------------------
# Tabela comparativa: resultado do TCC
# ---------------------------------------------------------------------------


def test_tabela_comparativa_cobre_os_tres_conjuntos(split):
    X_train, _, y_train, _ = split
    tabela = col.tabela_comparativa(X_train, y_train)

    assert set(tabela["feature_set"]) == set(ft.FEATURE_SETS)
    for nome in ft.FEATURE_SETS:
        pre = ft.construir_preprocessador(nome).fit(X_train, y_train)
        esperado = len(pre.get_feature_names_out())
        assert (tabela["feature_set"] == nome).sum() == esperado


def test_escore_e_reconstruido_exatamente_pelas_parcelas(split, preprocessadores):
    """Identidade algébrica: o escore É a soma das três parcelas."""
    X_train, _, _, _ = split
    preditoras = col.preditoras_pos_compostas(preprocessadores["C_completo"], X_train)
    soma = (
        preditoras["has_combinada"]
        + preditoras["colesterol_sim"]
        + col.obesidade_reconstruida(preditoras)
    )
    assert (soma == preditoras["multimorbidade"]).all()


def test_r2_da_identidade_e_um_e_o_do_vif_e_menor(split, preprocessadores):
    """A distância entre os dois R² é a redundância que o VIF não enxerga."""
    X_train, _, _, _ = split
    d = col.decomposicao_do_escore(preprocessadores["C_completo"], X_train)

    # Identidade, não estimativa: R² é exatamente 1.
    assert d["r2_identidade"] == pytest.approx(1.0, abs=1e-9)
    # A regressão auxiliar do VIF vê `imc` e `circ_cintura` contínuas, não o limiar.
    assert d["r2_vif"] < 0.95
    assert d["vif"] == pytest.approx(1.0 / (1.0 - d["r2_vif"]), rel=1e-9)


def test_obesidade_reconstruida_segue_os_cortes_da_secao_44():
    """Disjuntiva e sensível ao sexo: IMC ≥ 30, cintura ≥ 80 (F) ou ≥ 94 (M)."""
    preditoras = pd.DataFrame({
        "sexo_fem": [1, 0, 1, 0, 1],
        "imc": [22.0, 31.0, 24.0, 25.0, 29.9],
        "circ_cintura": [70.0, 80.0, 85.0, 95.0, 79.9],
    })
    assert list(col.obesidade_reconstruida(preditoras)) == [0, 1, 1, 1, 0]


def test_tabela_traz_r2_auxiliar_coerente_com_o_vif(split, preprocessadores):
    """`R² = 1 - 1/VIF`: o VIF sozinho subestima a leitura da redundância."""
    X_train, _, _, _ = split
    tabela = col.vif_matriz_design(preprocessadores["C_completo"], X_train, "C_completo")

    assert "r2_auxiliar" in tabela.columns
    assert np.allclose(tabela["r2_auxiliar"], 1.0 - 1.0 / tabela["vif"])
    assert (tabela["r2_auxiliar"] >= 0).all() and (tabela["r2_auxiliar"] <= 1).all()

    escore = tabela.loc[tabela["variavel"] == "multimorbidade"].iloc[0]
    assert escore["vif"] == pytest.approx(8.60, abs=0.05)
    assert escore["r2_auxiliar"] == pytest.approx(0.884, abs=0.005)
    # Abaixo do limiar, mas com 88% da variância já explicada pelas demais.
    assert not escore["acima_do_limiar"]


def test_r2_de_vif_trata_colinearidade_perfeita():
    assert col.r2_de_vif(pd.Series([np.inf])).iloc[0] == 1.0
    assert col.r2_de_vif(pd.Series([1.0])).iloc[0] == pytest.approx(0.0)
    assert col.r2_de_vif(pd.Series([10.0])).iloc[0] == pytest.approx(0.9)


def test_pares_pearson_traz_os_coeficientes_exigidos(split, preprocessadores):
    """Pearson é resultado, não só figura: os coeficientes vão para a tabela."""
    X_train, _, _, _ = split
    preditoras = col.preditoras_pos_compostas(preprocessadores["C_completo"], X_train)
    matriz = col.matriz_pearson(preditoras, ft.FEATURE_SETS["C_completo"]["num"])
    pares = col.pares_pearson(matriz, "C_completo")

    exigidos = [("rcest", "circ_cintura"), ("imc", "circ_cintura"),
                ("pas", "pad"), ("idade", "imc")]
    for a, b in exigidos:
        linha = pares[(pares[["var_1", "var_2"]].apply(frozenset, axis=1)
                       == frozenset({a, b}))]
        assert len(linha) == 1, f"par {a}×{b} ausente"
        assert -1 <= linha["pearson"].iloc[0] <= 1

    # Sem diagonal e sem pares repetidos em ordem trocada.
    n = len(matriz)
    assert len(pares) == n * (n - 1) // 2
    assert pares["abs_pearson"].is_monotonic_decreasing


def test_ablacao_de_rcest_reduz_o_vif_da_cintura(split):
    """Protocolo de ablação: remover a variável de maior VIF do bloco e recalcular."""
    X_train, _, y_train, _ = split
    comparativo = col.ablacao_vif(X_train, y_train, "C_completo", "rcest")

    assert "rcest" not in set(comparativo["variavel"])
    cintura = comparativo.loc[comparativo["variavel"] == "circ_cintura"].iloc[0]
    assert cintura["vif_antes"] > cfg.VIF_LIMIAR
    assert cintura["vif_depois"] < cfg.VIF_LIMIAR
    assert cintura["delta_vif"] < 0
    # Sem `rcest`, nenhuma coluna de C cruza o limiar.
    assert not comparativo["acima_depois"].any()


def test_ablacao_rejeita_variavel_ausente_do_conjunto():
    with pytest.raises(ValueError):
        ft.conjunto_sem("A_atomico", "rcest")


def test_conjunto_sem_preserva_o_restante():
    original = ft.FEATURE_SETS["C_completo"]
    derivado = ft.conjunto_sem("C_completo", "rcest")

    assert "rcest" not in derivado["num"]
    assert len(derivado["num"]) == len(original["num"]) - 1
    assert derivado["bin"] == original["bin"]
    assert derivado["nom"] == original["nom"]
    # Não registra um quarto conjunto de modelagem.
    assert set(ft.FEATURE_SETS) == {"A_atomico", "B_composto", "C_completo"}


def test_resumo_e_exportacao_latex(split, tmp_path):
    X_train, _, y_train, _ = split
    tabela = col.tabela_comparativa(X_train, y_train)
    resumo = col.resumo_por_conjunto(tabela)

    assert list(resumo["feature_set"]) == sorted(ft.FEATURE_SETS)
    assert {"colunas", "vif_maximo", "vif_mediano", "acima_do_limiar",
            "limiar_vif"} <= set(resumo)
    assert (resumo["limiar_vif"] == cfg.VIF_LIMIAR).all()

    destino = col.exportar_latex(resumo, "vif_resumo.tex", path_reports=tmp_path)
    conteudo = destino.read_text(encoding="utf-8")
    assert destino.exists()
    assert "tabular" in conteudo
    assert "A_atomico" in conteudo


def test_limiar_versionado_decide_acima_do_limiar():
    """`acima_do_limiar` do comparativo é `vif > limiar_vif` do resumo, linha a linha.

    O painel mostra o limiar ao lado da contagem. Se os dois CSV discordarem, a tela
    exibe um corte que não foi o usado.
    """
    caminho_vif = cfg.PATH_TABELAS / "vif_comparativo.csv"
    caminho_resumo = cfg.PATH_TABELAS / "vif_resumo.csv"
    for caminho in (caminho_vif, caminho_resumo):
        if not caminho.exists():
            pytest.skip(f"Artefato ausente em {caminho}. Rode make notebook antes.")
    resumo = pd.read_csv(caminho_resumo)
    vif = pd.read_csv(caminho_vif).merge(
        resumo[["feature_set", "limiar_vif"]], on="feature_set", how="left",
        validate="many_to_one")

    assert (resumo["limiar_vif"] == cfg.VIF_LIMIAR).all()
    assert vif["limiar_vif"].notna().all()
    assert (vif["acima_do_limiar"] == (vif["vif"] > vif["limiar_vif"])).all()
    contagem = vif.groupby("feature_set")["acima_do_limiar"].sum()
    assert (resumo.set_index("feature_set")["acima_do_limiar"] == contagem).all()


def test_decomposicao_versionada_reproduz_os_dois_r2():
    """R² 1,000 da identidade contra 0,884 do VIF, o argumento do descarte de C.

    Os dois números eram impressos pelo notebook e não tinham arquivo. Fixados aqui
    porque são resultado medido: se mudarem, investiga-se (D-016).
    """
    caminho = cfg.PATH_TABELAS / "decomposicao_escore.csv"
    if not caminho.exists():
        pytest.skip(f"Artefato ausente em {caminho}. Rode make notebook antes.")
    linha = pd.read_csv(caminho).iloc[0]
    assert linha["feature_set"] == "C_completo"
    assert round(linha["r2_identidade"], 3) == 1.000
    assert round(linha["r2_vif"], 3) == 0.884
    assert linha["parcelas"] == "has_combinada, colesterol_sim, obesidade_reconstruida"
