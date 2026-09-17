"""Testes dos comparadores.

A regra ADA precisa ser exatamente a regra: determinística, em escala clínica, sem
nada estimado. E precisa atravessar o mesmo harness dos modelos, sobre as mesmas
partições, senão a comparação não é comparação.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src import benchmark as bm
from src import config as cfg
from src import evaluate as ev
from src import features as ft
from src.data import carregar_bruto, limpar, separar


@pytest.fixture(scope="module")
def split():
    if not cfg.PATH_RAW.exists():
        pytest.skip(f"CSV bruto ausente em {cfg.PATH_RAW} (não versionado).")
    X, y, _ = separar(limpar(carregar_bruto()))
    return train_test_split(
        X, y, test_size=cfg.TEST_SIZE, stratify=y, random_state=cfg.RANDOM_STATE
    )


@pytest.fixture(scope="module")
def comparadores(split):
    X_train, X_test, y_train, y_test = split
    return bm.avaliar_comparadores(X_train, y_train, X_test, y_test,
                                   salvar_figuras=False)


# ---------------------------------------------------------------------------
# A regra
# ---------------------------------------------------------------------------


def test_regra_aplica_os_cortes_da_ada():
    """`idade >= 45` OU `imc >= 30`, incluindo as bordas."""
    X = pd.DataFrame({
        "idade": [45, 44, 44, 46, 30, 44],
        "imc": [22.0, 30.0, 29.9, 35.0, 18.0, 30.1],
    })
    esperado = [1, 1, 0, 1, 0, 1]
    assert list(bm.RegraClinicaADA().fit(X).predict(X)) == esperado


def test_regra_nao_aprende_nada(split):
    """Ajustar em 6.760 linhas ou em 10 produz o mesmo classificador."""
    X_train, X_test, y_train, _ = split
    completa = bm.RegraClinicaADA().fit(X_train, y_train)
    minima = bm.RegraClinicaADA().fit(X_train.head(10), y_train.head(10))

    assert np.array_equal(completa.predict(X_test), minima.predict(X_test))
    # Nada estimado além do que a API do scikit-learn exige.
    aprendidos = {a for a in vars(completa) if a.endswith("_")}
    assert aprendidos <= {"n_features_in_", "feature_names_in_", "classes_"}


def test_regra_e_clonavel_e_tem_parametros_explicitos():
    regra = bm.RegraClinicaADA()
    assert regra.get_params() == {"corte_idade": 45, "corte_imc": 30}
    assert clone(regra).get_params() == regra.get_params()


def test_predict_proba_devolve_a_indicadora_como_zero_ou_um(split):
    """Compatibilidade com o harness, e não são probabilidades."""
    X_train, _, y_train, _ = split
    regra = bm.RegraClinicaADA().fit(X_train, y_train)
    proba = regra.predict_proba(X_train)

    assert proba.shape == (len(X_train), 2)
    assert set(np.unique(proba)) <= {0.0, 1.0}
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.array_equal(proba[:, 1], regra.predict(X_train))


def test_regra_opera_em_escala_clinica_original(split):
    """Sobre a matriz padronizada a regra deixaria de ser a regra."""
    X_train, _, y_train, _ = split
    pre = ft.construir_preprocessador("C_completo").fit(X_train, y_train)
    design = pd.DataFrame(pre.transform(X_train),
                          columns=pre.get_feature_names_out())

    # A matriz de design tem `idade` e `imc` padronizados: comparar z-score com
    # 45 e 30 daria zero positivos. O estimador recusa a entrada errada por exigir
    # as colunas cruas; aqui elas existem, mas em escala trocada.
    padronizada = bm.RegraClinicaADA().fit(design).predict(design)
    clinica = bm.RegraClinicaADA().fit(X_train).predict(X_train)
    assert padronizada.sum() == 0
    assert clinica.sum() > 0

    # Preditoras em escala clínica: idade em anos, IMC em kg/m².
    assert X_train["idade"].between(18, 101).all()
    assert X_train["imc"].median() > 20


def test_regra_exige_idade_e_imc():
    with pytest.raises(KeyError, match="idade"):
        bm.RegraClinicaADA().fit(pd.DataFrame({"pas": [120.0, 130.0]}))


# ---------------------------------------------------------------------------
# Avaliação pelo mesmo harness
# ---------------------------------------------------------------------------


def test_comparadores_usam_limiar_fixo(comparadores):
    """Calibrar limiar sobre uma indicadora binária não produz ponto de operação."""
    for r in comparadores:
        assert r["limiar"] == bm.LIMIAR_INDICADORA
        assert r["limiar_calibrado"] is False


def test_dummy_seria_destruido_por_calibracao(split):
    """Sem limiar fixo, o piso viraria "prevê sempre positivo" e deixaria de ser piso."""
    X_train, X_test, y_train, y_test = split
    dummy = bm.construir_comparadores()["Dummy (maioria)"]

    fixo = ev.avaliar(dummy, X_train, y_train, X_test, y_test, "Dummy",
                      limiar=bm.LIMIAR_INDICADORA, salvar_figuras=False)
    calibrado = ev.avaliar(dummy, X_train, y_train, X_test, y_test, "Dummy",
                           salvar_figuras=False)

    assert fixo["recall"] == 0.0 and fixo["vp"] == 0
    assert calibrado["recall"] == 1.0  # classifica todo mundo como positivo
    assert calibrado["fp"] == fixo["vn"]


def test_dummy_expoe_a_armadilha_da_acuracia(comparadores):
    """Acurácia perto de 0,93 com recall zero."""
    dummy = next(r for r in comparadores if r["modelo"] == "Dummy (maioria)")
    assert dummy["acuracia"] == pytest.approx(0.93, abs=0.01)
    assert dummy["recall"] == 0.0
    assert dummy["especificidade"] == 1.0
    assert np.isinf(dummy["nns"])


def test_regra_avaliada_nas_mesmas_particoes(split, comparadores):
    _, X_test, _, _ = split
    for r in comparadores:
        assert r["vp"] + r["vn"] + r["fp"] + r["fn"] == len(X_test)


@pytest.mark.parametrize("metrica", ev.METRICAS_ASSINADAS)
def test_comparadores_trazem_as_metricas_assinadas(comparadores, metrica):
    """Mesmo harness, mesma tabela: as métricas assinadas."""
    for r in comparadores:
        assert metrica in r and np.isfinite(r[metrica])


# ---------------------------------------------------------------------------
# NNS e critério de sucesso
# ---------------------------------------------------------------------------


def test_tabela_nns_traduz_carga_operacional(split, comparadores):
    _, X_test, _, _ = split
    tabela = bm.tabela_nns(comparadores)

    assert set(tabela["estrategia"]) == {"Regra ADA", "Dummy (maioria)"}
    positivos_no_teste = int(sum(comparadores[0][k] for k in ("vp", "fn")))

    for _, linha in tabela.iterrows():
        assert linha["encaminhados"] <= len(X_test)
        # Todo caso do teste ou foi detectado, ou foi perdido.
        assert linha["casos_detectados"] + linha["casos_perdidos"] == positivos_no_teste
        assert linha["taxa_encaminhamento"] == pytest.approx(
            linha["encaminhados"] / len(X_test)
        )

    regra = tabela[tabela["estrategia"] == "Regra ADA"].iloc[0]
    assert regra["nns"] == pytest.approx(
        regra["encaminhados"] / regra["casos_detectados"], rel=1e-6
    )
    # Ordenada por NNS: o Dummy, que não encaminha ninguém, fica por último.
    assert tabela.iloc[-1]["estrategia"] == "Dummy (maioria)"


def test_criterio_de_sucesso_exige_as_duas_condicoes():
    """Recall maior ao custo de encaminhar mais gente não é ganho."""
    base = {"conjunto": "", "vp": 0, "vn": 0, "fp": 0, "fn": 0}
    resultados = [
        {**base, "modelo": "Regra ADA", "recall": 0.80, "nns": 9.0},
        {**base, "modelo": "so_recall", "recall": 0.90, "nns": 10.0},
        {**base, "modelo": "so_nns", "recall": 0.70, "nns": 7.0},
        {**base, "modelo": "ambos", "recall": 0.85, "nns": 8.0},
    ]
    tabela = bm.supera_a_regra_clinica(resultados).set_index("estrategia")

    assert tabela.loc["so_recall", "supera"] is np.False_ or not tabela.loc["so_recall", "supera"]
    assert not tabela.loc["so_nns", "supera"]
    assert tabela.loc["ambos", "supera"]


def test_criterio_falha_sem_a_regra_nos_resultados():
    with pytest.raises(ValueError, match="Regra ADA"):
        bm.supera_a_regra_clinica([{"modelo": "LogReg", "conjunto": "A",
                                    "recall": 0.8, "nns": 7.0}])


def test_recall_da_regra_no_treino_e_medido_so_no_treino(split):
    """Usar o recall da regra no teste vazaria o teste para dentro da calibração."""
    X_train, X_test, y_train, y_test = split
    no_treino = bm.recall_da_regra_no_treino(X_train, y_train)
    no_teste = bm.recall_da_regra_no_treino(X_test, y_test)

    assert 0.0 < no_treino < 1.0
    assert no_treino != no_teste  # partições distintas, valores distintos


def test_tabela_separa_recall_alvo_de_recall_obtido(split):
    """A equiparação é do alvo, no treino; o obtido no teste desvia, e isso aparece."""
    X_train, X_test, y_train, y_test = split
    alvo = bm.recall_da_regra_no_treino(X_train, y_train)
    modelo = Pipeline([
        ("pre", ft.construir_preprocessador("B_composto")),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000,
                                   random_state=cfg.RANDOM_STATE)),
    ])
    resultados = [
        ev.avaliar(modelo, X_train, y_train, X_test, y_test, "LogReg",
                   conjunto="B_composto", recall_alvo=alvo, salvar_figuras=False),
        *bm.avaliar_comparadores(X_train, y_train, X_test, y_test,
                                 salvar_figuras=False),
    ]
    tabela = bm.supera_a_regra_clinica(resultados)

    assert {"recall_alvo", "recall_obtido", "recall_regra"} <= set(tabela.columns)
    linha = tabela[tabela["estrategia"].str.contains("LogReg")].iloc[0]
    assert linha["recall_alvo"] == pytest.approx(alvo)
    # O obtido não é o alvo, e a tabela não finge que é.
    assert linha["recall_obtido"] != linha["recall_alvo"]
    assert linha["recall_obtido"] != linha["recall_regra"]
    # Comparadores de limiar fixo não têm alvo: a célula fica vazia, não 0,80.
    dummy = tabela[tabela["estrategia"] == "Dummy (maioria)"].iloc[0]
    assert np.isnan(dummy["recall_alvo"])


def test_nota_da_equiparacao_acompanha_a_tabela(split, comparadores):
    """A tabela carrega a própria advertência, não depende de quem a lê lembrar."""
    resultados = [
        {"modelo": "LogReg", "conjunto": "A", "recall": 0.9, "nns": 8.0,
         "recall_alvo": 0.89, "limiar_calibrado": True},
        *comparadores,
    ]
    tabela = bm.supera_a_regra_clinica(resultados)
    nota = tabela.attrs["nota"]

    assert "TREINO" in nota
    assert "recall_obtido" in nota and "recall_alvo" in nota
    assert "NNS" in nota


def test_leitura_operacional_enuncia_a_taxa_de_encaminhamento(split, comparadores):
    """57,6% é o número que sustenta o argumento de aplicabilidade."""
    X_train, X_test, y_train, y_test = split
    modelo = Pipeline([
        ("pre", ft.construir_preprocessador("B_composto")),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000,
                                   random_state=cfg.RANDOM_STATE)),
    ])
    resultados = [
        ev.avaliar(modelo, X_train, y_train, X_test, y_test, "LogReg",
                   conjunto="B_composto", salvar_figuras=False),
        *comparadores,
    ]
    frase = bm.leitura_operacional(resultados)

    regra = next(r for r in comparadores if r["modelo"] == "Regra ADA")
    taxa = 100 * (regra["vp"] + regra["fp"]) / len(X_test)
    assert f"{taxa:.1f}%" in frase
    assert str(regra["vp"] + regra["fp"]) in frase
    assert "metade da população" in frase


def test_leitura_explica_o_volume_maior_no_regime_equiparado(split, comparadores):
    """1.000 > 974 parece desvantagem até se ver que os recalls diferem."""
    X_train, X_test, y_train, y_test = split
    modelo = Pipeline([
        ("pre", ft.construir_preprocessador("B_composto")),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000,
                                   random_state=cfg.RANDOM_STATE)),
    ])
    alvo = bm.recall_da_regra_no_treino(X_train, y_train)
    padrao = [ev.avaliar(modelo, X_train, y_train, X_test, y_test, "LogReg",
                         conjunto="B_composto", salvar_figuras=False), *comparadores]
    equiparados = [ev.avaliar(modelo, X_train, y_train, X_test, y_test, "LogReg",
                              conjunto="B_composto", recall_alvo=alvo,
                              salvar_figuras=False), *comparadores]

    frase = bm.leitura_operacional(padrao, equiparados=equiparados)
    regra = next(r for r in comparadores if r["modelo"] == "Regra ADA")
    modelo_eq = equiparados[0]

    assert "regime de recall equiparado" in frase
    assert "pessoas a mais" in frase          # o volume maior é dito, não escondido
    assert "casos a mais" in frase
    assert "NNS inferior" in frase
    assert f"{modelo_eq['recall']:.3f} contra {regra['recall']:.3f}" in frase
    # A mesma estratégia nas duas metades da frase.
    assert frase.count("B_composto") == 2


def test_tabela_nns_carrega_a_advertencia_de_volume(comparadores):
    """A tabela não deve poder ser lida isolada da comparação de recall."""
    nota = bm.tabela_nns(comparadores).attrs["nota"]
    assert "mesmo recall" in nota
    assert "NNS" in nota


def test_comparacao_em_pe_de_igualdade_de_recall(split):
    """No mesmo regime de recall, a comparação de NNS passa a significar algo."""
    X_train, X_test, y_train, y_test = split
    alvo = bm.recall_da_regra_no_treino(X_train, y_train)

    modelo = Pipeline([
        ("pre", ft.construir_preprocessador("B_composto")),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000,
                                   random_state=cfg.RANDOM_STATE)),
    ])
    resultados = [
        ev.avaliar(modelo, X_train, y_train, X_test, y_test, "LogReg",
                   conjunto="B_composto", recall_alvo=alvo, salvar_figuras=False),
        *bm.avaliar_comparadores(X_train, y_train, X_test, y_test,
                                 salvar_figuras=False),
    ]
    logreg = resultados[0]
    regra = next(r for r in resultados if r["modelo"] == "Regra ADA")

    # Calibrado para o recall da regra, o modelo opera na mesma faixa.
    assert abs(logreg["recall"] - regra["recall"]) < 0.10
    # E aí a comparação de NNS é entre pontos comparáveis.
    assert logreg["nns"] < regra["nns"]
