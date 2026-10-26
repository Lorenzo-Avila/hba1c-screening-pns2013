"""Contratos da explicabilidade via SHAP sobre a Random Forest publicada.

As guardas se dividem em três naturezas. As de **identidade numérica**
conferem que a atribuição descreve o modelo: aditividade nos dois regimes e agregação
por variável conceitual. As de **vazamento e determinismo** conferem o protocolo: o
background sai do treino, e a seleção dos casos locais reproduz sozinha. As de
**artefato** conferem que o que está em `reports/tabelas/` é o que o código produz hoje:
composição completa no padrão de D-044, consistência entre duas tabelas que se olham,
reprodução do regime principal na terceira casa, o topo coincidente entre regimes, e as
colunas que o painel lê prontas.

**Custo, que decidiu o desenho.** Identificar as árvores que o explicador atribui com
erro custa minutos, porque explica as 500 uma a uma sobre o teste inteiro. Guarda cara é
guarda que ninguém roda, então o portão do regime de contraste é conferido **lendo**
`shap_arvores_excluidas.csv` como está versionado, e não recalculando. A reprodução
recalcula só o regime principal, que é barato, e vai marcada `lento` mesmo assim.

**O ajuste das fixtures passa por `models.montar` direto**, sem a validação cruzada de
`evaluate.avaliar`: o que as guardas precisam é da floresta e da matriz de design, e a
calibração custaria cinco ajustes a mais por célula sem mudar nenhum dos dois. Onde o
limiar faz falta, ele sai do artefato publicado, que é a mesma fonte que o módulo
confere. A guarda de vazamento é a exceção e chama `explicabilidade.preparar`, porque ali
o que está sob teste é o sorteio do background, não um valor derivado dele.

Nenhuma guarda escreve em `reports/tabelas/`, então a fixture `tabelas_preservadas` de
D-032 não aparece aqui. Todas leem.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import colinearidade as col
from src import config as cfg
from src import explicabilidade as ex
from src import experimento as xp
from src import models as md

# Nominais conferidas na guarda de agregação. As duas existem nas duas células, o que
# deixa a guarda valer para o conjunto inteiro em vez de para um caso.
NOMINAIS_CONFERIDAS = ["raca_cor", "regiao"]

# Folga da agregação. Os CSV saem arredondados em `artefatos.CASAS_CSV`, então a soma de
# quatro dummies pode afastar-se do agregado por alguns múltiplos da última casa.
TOLERANCIA_AGREGACAO = 1e-8

CASAS = ex.CASAS_CONFERENCIA


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def particao():
    return xp.preparar_particao()


@pytest.fixture(scope="module")
def ajustadas(particao):
    """Floresta ajustada, matriz de design e mapa de proveniência, por célula.

    Escopo de módulo porque o ajuste é a parte cara e nenhuma guarda o modifica.
    """
    saida = {}
    for celula in ex.CELULAS:
        pipeline = md.montar(md.estimadores()[ex.MODELO], celula)
        pipeline.fit(particao.X_tr, particao.y_tr)

        saida[celula] = {
            "pipeline": pipeline,
            "preprocessador": pipeline.named_steps["pre"],
            "floresta": pipeline.named_steps["clf"],
            "design_tr": col.matriz_de_design(pipeline, particao.X_tr),
            "design_te": col.matriz_de_design(pipeline, particao.X_te),
            "mapa": ex.proveniencia(pipeline.named_steps["pre"]),
            "proba": pipeline.predict_proba(particao.X_te)[:, 1],
            "y_te": particao.y_te,
        }
    return saida


@pytest.fixture(scope="module")
def artefatos():
    """As cinco tabelas versionadas, lidas uma vez.

    A categoria de referência vem por conversor de texto. Sem ele o pandas vê uma
    coluna que mistura código e célula vazia, infere float, e o código `1` volta da
    leitura como `1.0`, que não é o que está no arquivo.
    """
    tabelas = {}
    for nome in [ex.NOME_GLOBAL_ATRIBUTO, ex.NOME_GLOBAL_COLUNA, ex.NOME_PARES,
                 ex.NOME_CASOS, ex.NOME_ARVORES]:
        caminho = cfg.PATH_TABELAS / f"{nome}.csv"
        if not caminho.exists():
            pytest.skip(f"Artefato ausente em {caminho}. Rode make shap antes.")
        tabelas[nome] = pd.read_csv(
            caminho, converters={"categoria_referencia_omitida": str}
        )
    return tabelas


@pytest.fixture(scope="module")
def preparada(particao):
    """Uma célula pelo caminho completo de `explicabilidade.preparar`.

    Paga a calibração de propósito: a guarda de vazamento testa o sorteio do background
    como o módulo o faz, e replicá-lo no teste transformaria a guarda numa cópia da
    linha que ela deveria vigiar.
    """
    return ex.preparar(ex.CELULAS[0], particao)


def _divergencias(referencia: pd.DataFrame, obtido: pd.DataFrame,
                  chaves: list[str], colunas: list[str],
                  casas: int = CASAS) -> pd.DataFrame:
    """Compara duas tabelas linha a linha e devolve o que não bate.

    Nomeia linha, coluna, valor de referência e valor obtido, que é o formato que
    `experimento.conferir_reproducao` já usa: quem lê a falha precisa saber onde olhar
    sem reabrir o CSV.
    """
    esperado = referencia.set_index(chaves)
    atual = obtido.set_index(chaves)

    linhas = []
    for chave in esperado.index:
        if chave not in atual.index:
            linhas.append({"linha": " · ".join(map(str, chave)),
                           "coluna": "(linha inteira)",
                           "referencia": "presente no artefato",
                           "obtido": "ausente do recalculado"})
            continue
        for coluna in colunas:
            a = round(float(esperado.loc[chave, coluna]), casas)
            b = round(float(atual.loc[chave, coluna]), casas)
            if a != b:
                linhas.append({"linha": " · ".join(map(str, chave)),
                               "coluna": coluna, "referencia": a, "obtido": b})
    return pd.DataFrame(linhas, columns=["linha", "coluna", "referencia", "obtido"])


# ---------------------------------------------------------------------------
# Identidade numérica
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("celula", ex.CELULAS)
def test_aditividade_do_regime_principal_fecha(ajustadas, celula):
    """A soma das atribuições mais o valor de base reproduz `predict_proba`.

    É a conferência que separa atribuição que descreve o modelo de atribuição que
    descreve outra coisa. O regime principal é o que sustenta número publicado, então o
    portão dele é o apertado (D-053).
    """
    material = ajustadas[celula]
    design = np.asarray(material["design_te"])

    valores, base = ex.explicar(material["floresta"], design, ex.REGIME_PRINCIPAL)
    proba = material["floresta"].predict_proba(design)[:, 1]
    erro = ex.erro_de_aditividade(valores, base, proba)

    assert erro < ex.TOL_PRINCIPAL, (
        f"{celula}: aditividade do regime {ex.REGIME_PRINCIPAL} com erro máximo "
        f"{erro:.3e}, acima do portão {ex.TOL_PRINCIPAL:g}."
    )


def test_aditividade_do_regime_de_contraste_fica_abaixo_do_piso(artefatos):
    """O erro da floresta reduzida não passa do erro de uma árvore isolada sem defeito.

    O portão do regime de contraste é relativo ao piso do algoritmo sobre este modelo, e
    não a um valor fixo por analogia com o regime principal: nenhuma tolerância se fixa
    abaixo do ruído do método que ela mede (D-053).

    Lê os dois valores do artefato em vez de recalcular. Identificar as árvores com
    defeito custa minutos, e guarda cara é guarda que ninguém roda.

    A comparação é entre ordens de grandeza, porque é assim que o artefato grava as duas
    colunas de aditividade da floresta: a mantissa delas se move entre execuções e nunca
    carregou informação.
    """
    tabela = artefatos[ex.NOME_ARVORES]
    assert sorted(tabela["celula"]) == sorted(ex.CELULAS)

    for _, linha in tabela.iterrows():
        erro = ex.ordem_de_grandeza(linha["erro_aditividade_contraste"])
        piso = ex.ordem_de_grandeza(linha["piso_por_arvore"])
        assert erro <= piso, (
            f"{linha['celula']}: aditividade do regime {ex.REGIME_CONTRASTE} na ordem "
            f"de {erro:.0e}, acima do piso por árvore, que está em {piso:.0e}. Sinal de "
            f"que a identificação das árvores ficou incompleta."
        )
        principal = ex.ordem_de_grandeza(linha["erro_aditividade_principal"])
        assert principal < ex.TOL_PRINCIPAL, (
            f"{linha['celula']}: aditividade do regime {ex.REGIME_PRINCIPAL} na ordem "
            f"de {principal:.0e}, acima do portão {ex.TOL_PRINCIPAL:g}."
        )


@pytest.mark.parametrize("nominal", NOMINAIS_CONFERIDAS)
def test_agregacao_por_nominal_fecha(artefatos, nominal):
    """A soma das dummies de uma nominal bate com a linha agregada daquela variável.

    As duas tabelas globais saem do mesmo vetor de atribuições, uma por coluna da matriz
    e outra por variável conceitual. Divergência entre elas significa que o mapa de
    proveniência perdeu ou duplicou coluna, que é o defeito silencioso que o parsing de
    prefixo produziria.
    """
    por_coluna = artefatos[ex.NOME_GLOBAL_COLUNA]
    por_atributo = artefatos[ex.NOME_GLOBAL_ATRIBUTO]

    conferidas = 0
    for (celula, regime), grupo in por_coluna.groupby(["celula", "regime"]):
        membros = grupo[grupo["atributo"] == nominal]
        if membros.empty:
            continue
        agregada = por_atributo[(por_atributo["celula"] == celula)
                                & (por_atributo["regime"] == regime)
                                & (por_atributo["atributo"] == nominal)]
        assert len(agregada) == 1, (
            f"{celula} · {regime}: {nominal} aparece {len(agregada)} vezes na tabela "
            f"agregada, e deveria aparecer uma."
        )

        soma = float(membros["shap_medio_abs"].sum())
        valor = float(agregada.iloc[0]["shap_medio_abs"])
        assert abs(soma - valor) <= TOLERANCIA_AGREGACAO, (
            f"{celula} · {regime}: as {len(membros)} colunas de {nominal} somam "
            f"{soma:.9f} e a linha agregada traz {valor:.9f}."
        )
        assert int(agregada.iloc[0]["n_colunas"]) == len(membros)
        conferidas += 1

    assert conferidas > 0, f"{nominal} não apareceu em nenhuma célula do artefato."


def test_categoria_de_referencia_e_declarada(ajustadas, artefatos):
    """Nominal de k categorias produz k menos uma colunas, e a omitida vai na tabela.

    A codificação usa `drop="first"`, então a categoria de referência tem atribuição
    zero por construção e não por irrelevância. Sem essa declaração ao lado da tabela, a
    leitura correta depende de alguém lembrar do detalhe da codificação (D-053).
    """
    por_atributo = artefatos[ex.NOME_GLOBAL_ATRIBUTO]

    for celula, material in ajustadas.items():
        encoder = material["preprocessador"].named_steps["preparo"].named_transformers_["nom"]
        nominais = material["preprocessador"].named_steps["preparo"].transformers_
        colunas_nominais = next(cols for nome, _, cols in nominais if nome == "nom")

        for atributo, categorias in zip(colunas_nominais, encoder.categories_):
            esperado_colunas = len(categorias) - 1
            omitida = ex._formatar_categoria(categorias[0])

            no_mapa = material["mapa"]
            geradas = no_mapa[no_mapa["atributo"] == atributo]
            assert len(geradas) == esperado_colunas, (
                f"{celula} · {atributo}: {len(categorias)} categorias observadas "
                f"deveriam gerar {esperado_colunas} colunas, e o mapa traz "
                f"{len(geradas)}."
            )

            linhas = por_atributo[(por_atributo["celula"] == celula)
                                  & (por_atributo["atributo"] == atributo)]
            assert not linhas.empty, f"{celula} · {atributo} ausente do artefato."
            for _, linha in linhas.iterrows():
                assert int(linha["n_colunas"]) == esperado_colunas
                declarada = str(linha["categoria_referencia_omitida"])
                assert declarada == omitida, (
                    f"{celula} · {atributo}: o encoder deixou de fora a categoria "
                    f"{omitida!r} e a tabela declara {declarada!r}."
                )


# ---------------------------------------------------------------------------
# Vazamento e determinismo
# ---------------------------------------------------------------------------

def test_background_sai_do_treino(preparada, particao):
    """Os índices do background são do treino e nenhum deles está no teste.

    O regime de contraste estima a expectativa marginal sobre esse conjunto, e estimá-la
    sobre o teste faria o explicador olhar o que R1 e R2 mantêm fora de qualquer ajuste.
    A guarda espelha a que já existe para o recall da regra medido no treino.
    """
    indices = set(preparada["indices_background"])

    assert len(indices) == ex.N_BACKGROUND
    assert indices <= set(particao.X_tr.index), (
        f"{len(indices - set(particao.X_tr.index))} linhas do background fora do treino."
    )
    assert not (indices & set(particao.X_te.index)), (
        f"{len(indices & set(particao.X_te.index))} linhas do background também estão "
        f"no teste."
    )


def test_selecao_local_e_deterministica(ajustadas, artefatos):
    """Duas chamadas devolvem os mesmos índices, e eles são os do artefato.

    A regra é um `argmin` por célula da confusão, com desempate pelo menor índice: ela
    não admite julgamento, e é isso que separa escolha de caso de escolha por
    conveniência. Fixar os índices publicados junto disso amarra a regra ao que está no
    `.csv`.
    """
    publicados = artefatos[ex.NOME_CASOS]

    for celula, material in ajustadas.items():
        limiar = ex.limiar_publicado(celula)
        primeira = ex.selecionar_casos(material["y_te"], material["proba"], limiar)
        segunda = ex.selecionar_casos(material["y_te"], material["proba"], limiar)
        assert list(primeira["indice"]) == list(segunda["indice"])
        assert list(primeira["caso"]) == list(segunda["caso"])

        no_artefato = (publicados[publicados["celula"] == celula]
                       .groupby("caso")["indice"].first().to_dict())
        obtidos = dict(zip(primeira["caso"], primeira["indice"]))
        assert obtidos == no_artefato, (
            f"{celula}: a regra escolhe {obtidos} e o artefato publica {no_artefato}."
        )


# ---------------------------------------------------------------------------
# Artefato
# ---------------------------------------------------------------------------

def test_composicao_das_tabelas(artefatos):
    """Duas células, dois regimes, os três blocos nomeados e os oito pares locais.

    Guarda que compara só a interseção dos índices deixa passar linha que sumiu, e uma
    tabela truncada tem todos os números certos nas linhas que sobraram. Aqui a
    composição é declarada e a falha nomeia o que faltou (D-044).

    **Oito pares, sete indivíduos.** Os casos locais são um por célula da confusão em
    cada célula de atributos, e nada na regra impede que o mesmo indivíduo vença nas
    duas: o falso negativo de fronteira é o mesmo. Uma asserção sobre oito índices
    distintos reprovaria um resultado correto.
    """
    regimes = {ex.REGIME_PRINCIPAL, ex.REGIME_CONTRASTE}

    for nome in [ex.NOME_GLOBAL_ATRIBUTO, ex.NOME_GLOBAL_COLUNA, ex.NOME_PARES]:
        tabela = artefatos[nome]
        faltando = [c for c in ex.CELULAS if c not in set(tabela["celula"])]
        assert not faltando, f"{nome}: célula ausente do emitido: {faltando}"
        ausentes = sorted(regimes - set(tabela["regime"]))
        assert not ausentes, f"{nome}: regime ausente do emitido: {ausentes}"

    pares = artefatos[ex.NOME_PARES]
    esperados = {(celula, par["bloco"], regime)
                 for par in ex.PARES_COLINEARES
                 for celula in par["celulas"]
                 for regime in regimes}
    obtidos = set(zip(pares["celula"], pares["bloco"], pares["regime"]))
    faltando = sorted(esperados - obtidos)
    sobrando = sorted(obtidos - esperados)
    assert not faltando, f"bloco ausente do emitido: {faltando}"
    assert not sobrando, f"bloco inesperado no emitido: {sobrando}"

    casos = artefatos[ex.NOME_CASOS]
    esperados_locais = {(celula, caso) for celula in ex.CELULAS
                        for caso in ex.CELULAS_CONFUSAO}
    obtidos_locais = set(zip(casos["celula"], casos["caso"]))
    faltando = sorted(esperados_locais - obtidos_locais)
    sobrando = sorted(obtidos_locais - esperados_locais)
    assert not faltando, f"caso local ausente do emitido: {faltando}"
    assert not sobrando, f"caso local inesperado no emitido: {sobrando}"
    assert len(obtidos_locais) == 8

    # Um indivíduo por par, e a contagem de indivíduos distintos fica livre: o mesmo
    # falso negativo vence nas duas células, e isso é resultado, não defeito.
    por_par = casos.groupby(["celula", "caso"])["indice"].nunique()
    assert set(por_par) == {1}, (
        f"par com mais de um indivíduo: {por_par[por_par > 1].to_dict()}"
    )


def test_contagem_de_arvores_bate_com_a_floresta_ajustada(ajustadas, artefatos):
    """O artefato e o modelo contam as mesmas árvores, e a aritmética dele fecha.

    Duas tabelas que se conferem uma à outra valem mais que duas que repetem o mesmo
    cálculo, e é a ideia de `test_equiparado_bate_com_a_carga_operacional`. Aqui um dos
    lados é a floresta ajustada agora e o outro é o que ficou gravado.

    A guarda confere o total e as contagens, não quais árvores foram excluídas:
    identificá-las custa minutos, e o portão do regime de contraste já responde por elas.
    """
    tabela = artefatos[ex.NOME_ARVORES].set_index("celula")

    for celula, material in ajustadas.items():
        linha = tabela.loc[celula]
        ajustadas_agora = len(material["floresta"].estimators_)

        assert int(linha["arvores_ajustadas"]) == ajustadas_agora, (
            f"{celula}: o artefato registra {int(linha['arvores_ajustadas'])} árvores "
            f"ajustadas e a floresta de hoje tem {ajustadas_agora}."
        )
        mantidas = int(linha["arvores_mantidas"])
        excluidas = int(linha["arvores_excluidas"])
        assert mantidas + excluidas == ajustadas_agora, (
            f"{celula}: {mantidas} mantidas mais {excluidas} excluídas não somam "
            f"{ajustadas_agora}."
        )
        assert 0 < excluidas < ajustadas_agora

        # A floresta sobre a qual o contraste rodou tem que ser construtível com essa
        # contagem, sem reajuste e predizendo sobre o teste.
        reduzida = ex.reduzir_floresta(material["floresta"], np.arange(mantidas))
        assert len(reduzida.estimators_) == mantidas
        assert reduzida.n_estimators == mantidas
        assert reduzida.predict_proba(np.asarray(material["design_te"])).shape == (
            len(material["design_te"]), 2
        )


@pytest.mark.lento
def test_reproducao_do_regime_principal(ajustadas, artefatos):
    """Recalcula o regime principal e compara com os `.csv` versionados.

    Só o regime principal, que é o barato e o que sustenta número publicado. O de
    contraste fica de fora por custo e está coberto pela guarda de composição e pelo
    portão lido do artefato.

    Quando esta guarda falha, investiga-se: ela fixa resultado medido, e ajustar a
    asserção para o número novo passar é o que D-016 proíbe.
    """
    esperado_atributo = artefatos[ex.NOME_GLOBAL_ATRIBUTO]
    esperado_coluna = artefatos[ex.NOME_GLOBAL_COLUNA]

    obtido_atributo, obtido_coluna = [], []
    for celula, material in ajustadas.items():
        valores, _ = ex.explicar(material["floresta"],
                                 np.asarray(material["design_te"]),
                                 ex.REGIME_PRINCIPAL)
        por_atributo = ex.ranking_por_atributo(valores, material["mapa"])
        por_coluna = ex.ranking_por_coluna(valores, material["mapa"])
        por_atributo["celula"] = celula
        por_coluna["celula"] = celula
        obtido_atributo.append(por_atributo)
        obtido_coluna.append(por_coluna)

    principal = esperado_atributo["regime"] == ex.REGIME_PRINCIPAL
    divergencias = _divergencias(
        esperado_atributo[principal], pd.concat(obtido_atributo, ignore_index=True),
        chaves=["celula", "atributo"], colunas=["shap_medio_abs", "posicao"],
    )
    principal_coluna = esperado_coluna["regime"] == ex.REGIME_PRINCIPAL
    divergencias = pd.concat([
        divergencias,
        _divergencias(esperado_coluna[principal_coluna],
                      pd.concat(obtido_coluna, ignore_index=True),
                      chaves=["celula", "coluna"],
                      colunas=["shap_medio_abs", "posicao"]),
    ], ignore_index=True)

    assert divergencias.empty, (
        "O regime principal recalculado não reproduz o artefato na "
        f"{CASAS}ª casa:\n{divergencias.to_string(index=False)}"
    )


# ---------------------------------------------------------------------------
# Topo coincidente
# ---------------------------------------------------------------------------

def test_concordancia_topo_conta_o_prefixo_comum():
    """`k` é o maior prefixo idêntico, e uma troca no meio corta a contagem ali."""
    linhas = []
    ordens = {
        (ex.REGIME_PRINCIPAL, ex.OBJETO_COMPLETA): ["a", "b", "c", "d"],
        ex.PRINCIPAL_REDUZIDA: ["a", "c", "b", "d"],
        (ex.REGIME_CONTRASTE, ex.OBJETO_REDUZIDA): ["a", "b", "d", "c"],
    }
    for (regime, objeto), ordem in ordens.items():
        for posicao, atributo in enumerate(ordem, start=1):
            linhas.append({"celula": "X", "regime": regime, "objeto_explicado": objeto,
                           "atributo": atributo, "posicao": posicao})

    tabela = ex.concordancia_topo(pd.DataFrame(linhas)).set_index("comparacao")
    completa, reduzida = (nome for nome, _, _ in ex.COMPARACOES_TOPO)
    assert tabela.loc[completa, "topo_coincidente"] == 2
    assert tabela.loc[completa, "atributos_no_topo"] == "a, b"
    assert tabela.loc[reduzida, "topo_coincidente"] == 1
    assert (tabela["n_atributos"] == 4).all()

    # O CSV publicado não traz o principal sobre a reduzida. Sem este lado a comparação
    # daria zero calado, e a função tem de recusar.
    so_publicado = pd.DataFrame(linhas)
    so_publicado = so_publicado[so_publicado["objeto_explicado"].ne(ex.OBJETO_REDUZIDA)
                                | so_publicado["regime"].eq(ex.REGIME_CONTRASTE)]
    with pytest.raises(KeyError, match="ausente"):
        ex.concordancia_topo(so_publicado)


def test_concordancia_topo_versionada_confere_com_o_ranking_publicado(artefatos):
    """Sete posições em `A_atomico` e cinco em `B_composto`, e o CSV confere com o ranking.

    A primeira comparação se recalcula sobre `shap_global_atributo.csv`, porque os dois
    lados dela estão publicados. A segunda depende do principal sobre a reduzida, que
    só existe dentro de `make shap`, e aqui ela entra só pela composição.
    """
    caminho = cfg.PATH_TABELAS / f"{ex.NOME_CONCORDANCIA}.csv"
    if not caminho.exists():
        pytest.skip(f"Artefato ausente em {caminho}. Rode make shap antes.")
    versionada = pd.read_csv(caminho)

    esperadas = {(celula, nome) for celula in ex.CELULAS
                 for nome, _, _ in ex.COMPARACOES_TOPO}
    obtidas = set(zip(versionada["celula"], versionada["comparacao"]))
    assert obtidas == esperadas, sorted(obtidas ^ esperadas)

    publicada = ex.COMPARACOES_TOPO[0]
    recalculada = ex.concordancia_topo(artefatos[ex.NOME_GLOBAL_ATRIBUTO],
                                       comparacoes=[publicada])
    na_versionada = versionada[versionada["comparacao"] == publicada[0]]
    assert (recalculada.set_index("celula")["topo_coincidente"].to_dict()
            == na_versionada.set_index("celula")["topo_coincidente"].to_dict())
    assert recalculada.set_index("celula")["topo_coincidente"].to_dict() == {
        "A_atomico": 7, "B_composto": 5,
    }


# ---------------------------------------------------------------------------
# Colunas que o painel lê prontas
# ---------------------------------------------------------------------------

def test_tabelas_shap_declaram_o_modelo(artefatos):
    """As cinco tabelas de atribuição dizem de qual modelo são, em todas as linhas."""
    caminho = cfg.PATH_TABELAS / f"{ex.NOME_CONCORDANCIA}.csv"
    if not caminho.exists():
        pytest.skip(f"Artefato ausente em {caminho}. Rode make shap antes.")
    tabelas = {nome: artefatos[nome] for nome in
               [ex.NOME_GLOBAL_ATRIBUTO, ex.NOME_GLOBAL_COLUNA, ex.NOME_PARES,
                ex.NOME_CASOS]}
    tabelas[ex.NOME_CONCORDANCIA] = pd.read_csv(caminho)

    for nome, tabela in tabelas.items():
        assert (tabela["modelo"] == ex.MODELO).all(), nome


def test_indice_nas_duas_celulas_marca_o_mesmo_individuo(artefatos):
    """A marca vale exatamente para os índices que aparecem nas duas células."""
    casos = artefatos[ex.NOME_CASOS]
    celulas_por_indice = casos.groupby("indice")["celula"].nunique()
    repetidos = set(celulas_por_indice[celulas_por_indice > 1].index)

    assert repetidos, "nenhum índice nas duas células"
    assert (casos["indice_nas_duas_celulas"] == casos["indice"].isin(repetidos)).all()


def test_waterfall_mostra_categoria_e_corta_contribuicao_minuscula():
    """O rótulo da figura usa a categoria da PNS. Contribuição abaixo de 0,0005 some."""
    assert ex._rotulo_observado("regiao", 3) == "Região = Sudeste"
    assert ex._rotulo_observado("raca_cor", 4) == "Raça/cor = Parda"
    assert ex._rotulo_observado("sexo_fem", 0) == "Sexo = masculino"
    assert ex._rotulo_observado("sexo_fem", 1) == "Sexo = feminino"
    assert ex._rotulo_observado("plano_sim", 0) == "Plano de saúde = não"
    assert ex._rotulo_observado("escolaridade", 5) == (
        "Escolaridade = Médio completo ou equivalente"
    )
    assert ex._rotulo_observado("saude_autoaval", 2) == "Saúde autoavaliada = Boa"
    assert ex._rotulo_observado("multimorbidade", 2) == "Multimorbidade Metabólica = 2"
    assert ex._texto_contribuicao(0.000173) == "< 0,001"
    assert ex._texto_contribuicao(-0.0004) == "< 0,001"
    assert ex._texto_contribuicao(0.04) == "+0,040"
    assert ex._texto_contribuicao(-0.1) == "\u22120,100"
