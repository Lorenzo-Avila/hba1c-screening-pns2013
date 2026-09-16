"""Testes do harness de avaliação (§6 da spec · §6.3 fase 5).

O ponto crítico: o limiar é decidido no treino, por validação cruzada. Nenhuma
estatística do conjunto de teste pode influenciá-lo — há teste específico para isso.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

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


def _logreg(conjunto: str) -> Pipeline:
    """Estimador já embrulhado no preprocessador — como `avaliar` o espera."""
    return Pipeline([
        ("pre", ft.construir_preprocessador(conjunto)),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000,
                                   random_state=cfg.RANDOM_STATE)),
    ])


@pytest.fixture(scope="module")
def resultados(split, tmp_path_factory):
    """LogReg sobre os três conjuntos, avaliada uma única vez por módulo."""
    X_train, X_test, y_train, y_test = split
    destino = tmp_path_factory.mktemp("figuras")
    return [
        ev.avaliar(_logreg(nome), X_train, y_train, X_test, y_test,
                   nome="LogReg", conjunto=nome, path_figuras=destino)
        for nome in ft.FEATURE_SETS
    ]


# ---------------------------------------------------------------------------
# Seleção de limiar — no treino, nunca no teste
# ---------------------------------------------------------------------------


def test_limiar_nao_depende_do_conjunto_de_teste(split):
    """A garantia central do harness: o teste mede, não decide."""
    X_train, X_test, y_train, y_test = split
    est = _logreg("A_atomico")

    completo = ev.avaliar(est, X_train, y_train, X_test, y_test, "LogReg",
                          salvar_figuras=False)
    fatia = ev.avaliar(est, X_train, y_train, X_test.iloc[:200], y_test.iloc[:200],
                       "LogReg", salvar_figuras=False)
    embaralhado = ev.avaliar(
        est, X_train, y_train, X_test,
        pd.Series(np.random.default_rng(0).permutation(y_test.to_numpy()),
                  index=y_test.index),
        "LogReg", salvar_figuras=False,
    )

    assert completo["limiar"] == fatia["limiar"] == embaralhado["limiar"]
    # E as métricas de teste, essas sim, mudam com o teste.
    assert completo["recall"] != embaralhado["recall"]


def test_limiar_e_o_maior_que_atinge_o_alvo():
    """Entre os limiares que cumprem o recall-alvo, o maior maximiza a precisão."""
    rng = np.random.default_rng(cfg.RANDOM_STATE)
    y = np.repeat([0, 1], [900, 100])
    proba = np.clip(rng.normal(0.3, 0.15, 1000) + y * 0.25, 0.001, 0.999)

    escolha = ev.selecionar_limiar(y, proba, recall_alvo=0.80)
    assert escolha["recall_alvo_atingido"] is True
    assert escolha["recall_cv"] >= 0.80
    # Qualquer limiar maior perderia o alvo.
    acima = (proba >= escolha["limiar"] + 0.01).astype(int)
    recall_acima = acima[y == 1].sum() / (y == 1).sum()
    assert recall_acima < 0.80 or np.isclose(recall_acima, escolha["recall_cv"])


def test_alvo_inatingivel_sinaliza_em_vez_de_falhar_em_silencio(monkeypatch):
    """Sem limiar que alcance o alvo: devolve o de maior recall e avisa.

    Ramo defensivo. Em `precision_recall_curve` o menor limiar classifica tudo
    como positivo, então o recall máximo é sempre 1,0 e nenhum alvo válido cai
    aqui — por isso a curva é substituída para exercitar o caminho.
    """
    curva_degenerada = (
        np.array([0.5, 0.5, 1.0]),   # precisão
        np.array([0.60, 0.40, 0.0]),  # recall: nunca alcança 0,8
        np.array([0.30, 0.70]),       # limiares
    )
    monkeypatch.setattr(ev, "precision_recall_curve",
                        lambda *a, **k: curva_degenerada)

    y = np.array([1, 0, 1, 0, 1, 0])
    proba = np.array([0.9, 0.1, 0.2, 0.3, 0.8, 0.4])
    with pytest.warns(UserWarning, match="Nenhum limiar atinge recall"):
        escolha = ev.selecionar_limiar(y, proba, recall_alvo=0.80)

    assert escolha["recall_alvo_atingido"] is False
    assert escolha["limiar"] == pytest.approx(0.30)  # o de maior recall


def test_recall_alvo_fora_do_intervalo_falha_alto():
    """Alvo inválido é erro de chamada, não aviso a ignorar."""
    y = np.array([0, 1, 0, 1])
    proba = np.array([0.1, 0.9, 0.2, 0.8])
    for invalido in (0.0, 1.5, -0.2):
        with pytest.raises(ValueError, match="recall_alvo"):
            ev.selecionar_limiar(y, proba, recall_alvo=invalido)


def test_cv_e_estratificada_com_a_semente_do_projeto(split):
    """R11: mesma semente, mesmas probabilidades fora da amostra."""
    X_train, _, y_train, _ = split
    est = _logreg("B_composto")
    primeira = ev.probabilidades_por_cv(est, X_train, y_train)
    segunda = ev.probabilidades_por_cv(est, X_train, y_train)

    assert np.allclose(primeira, segunda)
    assert len(primeira) == len(X_train)
    assert ((primeira >= 0) & (primeira <= 1)).all()


def test_estimador_recebido_nao_e_modificado(split):
    """`avaliar` opera sobre clones: o objeto do chamador segue não ajustado."""
    X_train, X_test, y_train, y_test = split
    est = _logreg("A_atomico")
    ev.avaliar(est, X_train, y_train, X_test, y_test, "LogReg", salvar_figuras=False)

    with pytest.raises(NotFittedError):
        check_is_fitted(est.named_steps["clf"])


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------


def test_metricas_no_limiar_com_valores_conhecidos():
    """Aritmética conferida à mão, sem depender do estimador."""
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.6, 0.7, 0.3, 0.8, 0.9, 0.95])
    m = ev.metricas_no_limiar(y, proba, limiar=0.5)

    # Preditos positivos: índices 2,3 (negativos) e 5,6,7 (positivos).
    assert (m["vp"], m["fp"], m["fn"], m["vn"]) == (3, 2, 1, 2)
    assert m["recall"] == pytest.approx(3 / 4)
    assert m["especificidade"] == pytest.approx(2 / 4)
    assert m["precisao"] == pytest.approx(3 / 5)
    assert m["nns"] == pytest.approx(5 / 3)
    assert m["f1"] == pytest.approx(2 * (0.6 * 0.75) / (0.6 + 0.75))


def test_nns_e_o_inverso_da_precisao(resultados):
    """NNS: quantas pessoas encaminhar a exame para identificar um caso."""
    for r in resultados:
        assert r["nns"] == pytest.approx(1.0 / r["precisao"])


def test_nns_infinito_quando_nenhum_positivo_acerta():
    """Precisão zero não pode estourar a divisão."""
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.9, 0.8, 0.1, 0.2])
    m = ev.metricas_no_limiar(y, proba, limiar=0.5)
    assert m["precisao"] == 0.0
    assert np.isinf(m["nns"])


def test_especificidade_complementa_a_taxa_de_falsos_positivos(resultados):
    for r in resultados:
        assert r["especificidade"] == pytest.approx(r["vn"] / (r["vn"] + r["fp"]))


@pytest.mark.parametrize("metrica", ev.METRICAS_ASSINADAS)
def test_metricas_assinadas_presentes_em_todo_resultado(resultados, metrica):
    """§6.3 fase 5: obrigatórias em toda tabela, sem exceção."""
    for r in resultados:
        assert metrica in r
        assert np.isfinite(r[metrica])
        assert 0.0 <= r[metrica] <= 1.0


def test_matriz_de_confusao_soma_o_teste(split, resultados):
    _, X_test, _, _ = split
    for r in resultados:
        assert r["vp"] + r["vn"] + r["fp"] + r["fn"] == len(X_test)


# ---------------------------------------------------------------------------
# Execução end-to-end
# ---------------------------------------------------------------------------


def test_end_to_end_nos_tres_conjuntos(resultados):
    """LogReg × {A, B, C} com as cinco métricas assinadas presentes."""
    assert len(resultados) == 3
    assert {r["conjunto"] for r in resultados} == set(ft.FEATURE_SETS)

    for r in resultados:
        assert r["recall_alvo"] == cfg.RECALL_ALVO
        assert 0.0 < r["limiar"] < 1.0
        assert r["auc_roc"] > 0.5  # melhor que o acaso
        assert r["nns"] > 1.0


def test_recall_alvo_atingido_na_calibracao(resultados):
    """O ponto de operação precisa cumprir a restrição que o define."""
    for r in resultados:
        assert r["recall_alvo_atingido"] is True
        assert r["recall_cv"] >= cfg.RECALL_ALVO


def test_figuras_geradas_e_caminhos_devolvidos(resultados):
    """Convenção do projeto: figura salva em disco, caminho no retorno."""
    esperadas = {"confusao", "roc", "precision_recall", "calibracao"}
    for r in resultados:
        assert set(r["figuras"]) == esperadas
        for caminho in r["figuras"].values():
            assert caminho.exists() and caminho.stat().st_size > 0


def test_avaliar_nao_constroi_pipeline(split):
    """O estimador chega pronto: sem preprocessador embutido, `avaliar` falha."""
    X_train, X_test, y_train, y_test = split
    cru = LogisticRegression(max_iter=1000)
    with pytest.raises(Exception):
        ev.avaliar(cru, X_train, y_train, X_test, y_test, "LogReg",
                   salvar_figuras=False)


# ---------------------------------------------------------------------------
# Tabela comparativa
# ---------------------------------------------------------------------------


def test_comparar_uma_linha_por_modelo_e_conjunto(resultados):
    tabela = ev.comparar(resultados)
    assert len(tabela) == 3
    assert list(tabela["conjunto"]) == list(ft.FEATURE_SETS)
    assert tabela.duplicated(["modelo", "conjunto"]).sum() == 0


def test_comparar_traz_assinadas_antes_das_complementares(resultados):
    tabela = ev.comparar(resultados)
    for metrica in ev.METRICAS_ASSINADAS + ev.METRICAS_COMPLEMENTARES:
        assert metrica in tabela.columns

    posicoes = list(tabela.columns)
    assert max(posicoes.index(m) for m in ev.METRICAS_ASSINADAS) < min(
        posicoes.index(m) for m in ev.METRICAS_COMPLEMENTARES
    )
    # Matriz de confusão também entra na tabela (§6.3 fase 5).
    assert {"vp", "fp", "fn", "vn"} <= set(tabela.columns)


def test_exportacao_latex(resultados, tmp_path):
    tabela = ev.comparar(resultados)
    destino = ev.exportar_latex(tabela, "comparativo.tex", path_reports=tmp_path)
    conteudo = destino.read_text(encoding="utf-8")

    assert destino.exists()
    assert "tabular" in conteudo
    assert "LogReg" in conteudo
