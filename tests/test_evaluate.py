"""Testes do harness de avaliação.

O ponto crítico é que o limiar se decide no treino, por validação cruzada. Nenhuma
estatística do conjunto de teste pode influenciá-lo, e há teste específico para isso.
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
from src import experimento as xp
from src import features as ft
from src import models as md
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
    """Estimador já embrulhado no preprocessador, como `avaliar` espera receber."""
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


@pytest.fixture(scope="module")
def resultados_floresta(split):
    """A floresta publicada, nos conjuntos em que ela publica.

    Sai de `models.estimadores`, e não de uma floresta montada aqui, senão o teste
    passa a medir uma configuração que não é a do resultado (D-029). Sem figuras: o
    que interessa é o diagnóstico de calibração.
    """
    X_train, X_test, y_train, y_test = split
    floresta = md.estimadores()["Random Forest"]
    return [
        ev.avaliar(md.montar(floresta, nome), X_train, y_train, X_test, y_test,
                   nome="Random Forest", conjunto=nome, salvar_figuras=False)
        for nome in xp.CONJUNTOS_FLORESTA
    ]


# ---------------------------------------------------------------------------
# Seleção de limiar: no treino, nunca no teste
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
    """Sem limiar que alcance o alvo, devolve o de maior recall e avisa.

    Ramo defensivo. Em `precision_recall_curve` o menor limiar classifica tudo como
    positivo, então o recall máximo é sempre 1,0 e nenhum alvo válido cai aqui; daí
    a curva ser substituída para exercitar o caminho.
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
    """Mesma semente, mesmas probabilidades fora da amostra."""
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
    """Métricas assinadas: obrigatórias em toda tabela, sem exceção."""
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
    # A matriz de confusão também entra na tabela.
    assert {"vp", "fp", "fn", "vn"} <= set(tabela.columns)


# ---------------------------------------------------------------------------
# Dispersão entre folds
# ---------------------------------------------------------------------------


def test_metricas_por_fold_cobrem_o_treino_inteiro(split, resultados):
    """Cinco folds disjuntos que somam o conjunto de treino."""
    X_train, _, _, _ = split
    for r in resultados:
        por_fold = r["metricas_por_fold"]
        assert len(por_fold) == cfg.CV_FOLDS
        assert por_fold["n"].sum() == len(X_train)
        assert set(ev.METRICAS_ASSINADAS) <= set(por_fold.columns)


def test_dispersao_presente_para_cada_metrica_assinada(resultados):
    """Sem dispersão não há como julgar se a diferença entre conjuntos é sinal."""
    for r in resultados:
        for metrica in ev.METRICAS_ASSINADAS:
            media, dp = r[f"{metrica}_cv_media"], r[f"{metrica}_cv_dp"]
            assert np.isfinite(media) and np.isfinite(dp)
            assert 0.0 <= media <= 1.0
            assert dp >= 0.0


def test_resumir_folds_confere_com_o_dataframe(resultados):
    for r in resultados:
        esperado = ev.resumir_folds(r["metricas_por_fold"])
        for chave, valor in esperado.items():
            assert r[chave] == pytest.approx(valor)


def test_folds_reconstruidos_sao_os_da_validacao_cruzada(split):
    """Mesma semente, mesmas partições: a média dos folds bate com o agregado."""
    X_train, _, y_train, _ = split
    est = _logreg("A_atomico")
    proba_cv = ev.probabilidades_por_cv(est, X_train, y_train)
    escolha = ev.selecionar_limiar(y_train, proba_cv, cfg.RECALL_ALVO)
    por_fold = ev.metricas_por_fold(y_train, proba_cv, escolha["limiar"],
                                    X_tr=X_train)

    # Folds de tamanho igual: a média simples aproxima o recall agregado.
    assert por_fold["recall"].mean() == pytest.approx(escolha["recall_cv"], abs=0.01)


def test_comparar_traz_media_e_dp_ao_lado_do_teste(resultados):
    tabela = ev.comparar(resultados)
    for metrica in ev.METRICAS_ASSINADAS:
        assert {metrica, f"{metrica}_cv_media", f"{metrica}_cv_dp"} <= set(tabela.columns)
        # O valor de teste vem imediatamente antes da sua dispersão.
        colunas = list(tabela.columns)
        assert colunas.index(metrica) + 1 == colunas.index(f"{metrica}_cv_media")


def test_comparar_formatado_exibe_media_e_dp(resultados):
    tabela = ev.comparar_formatado(resultados)
    assert len(tabela) == 3
    for metrica in ev.METRICAS_ASSINADAS:
        assert tabela[metrica].str.contains("±").all()


# ---------------------------------------------------------------------------
# Calibração: o número, não só a figura
# ---------------------------------------------------------------------------


def test_brier_trivial_e_a_variancia_da_prevalencia():
    y = np.repeat([0, 1], [930, 70])
    p = y.mean()
    assert ev.brier_trivial(y) == pytest.approx(p * (1 - p))


def test_tabela_calibracao_particiona_o_teste(split, resultados):
    _, X_test, _, _ = split
    for r in resultados:
        tabela = r["calibracao"]["tabela"]
        assert tabela["n"].sum() == len(X_test)
        assert (tabela["proba_media"].between(0, 1)).all()
        assert (tabela["freq_observada"].between(0, 1)).all()


def _logito_da_prevalencia(y_train) -> float:
    """Onde `class_weight='balanced'` põe o intercepto da logística, por álgebra."""
    p = float(np.mean(y_train))
    return float(np.log(p / (1 - p)))


def test_probabilidades_da_logistica_infladas_por_class_weight(split, resultados):
    """`class_weight='balanced'` desloca o nível da logística, não a forma.

    O modelo treina como se as classes fossem equilibradas, então a probabilidade
    média sobe muito acima da prevalência. A discriminação fica preservada, com slope
    perto de 1, e o nível sai deslocado para o logito da prevalência.

    Os limites são estreitos de propósito. Eles descrevem a identidade algébrica da
    logística e **não valem para a floresta**, que chega a um deslocamento parecido por
    outro mecanismo. O par deste teste é
    `test_probabilidades_da_floresta_deslocadas_sem_a_identidade`, e D-040 registra por
    que os dois não podem compartilhar limite.
    """
    _, _, y_train, _ = split
    logito = _logito_da_prevalencia(y_train)

    for r in resultados:
        cal = r["calibracao"]
        assert cal["razao_inflacao"] > 5.0
        assert cal["proba_media"] > 5 * cal["prevalencia"]
        # Pior que prever sempre a prevalência.
        assert r["brier_skill"] < 0
        assert r["brier"] > r["brier_trivial"]
        # Forma preservada, nível deslocado para onde a álgebra manda.
        assert 0.7 < cal["slope"] < 1.2
        assert cal["intercepto"] == pytest.approx(logito, abs=0.1)


def test_intercepto_da_logistica_reproduz_o_logito_da_prevalencia(split, resultados):
    """O deslocamento é exatamente o reponderamento de classes, não ruído.

    Vale para a logística e só para ela. Com o modelo linear, reponderar as classes
    equivale a somar uma constante ao logito, e a constante é o logito da prevalência
    de treino. Não há o que transpor para a floresta, que não tem intercepto.
    """
    _, _, y_train, _ = split
    logito = _logito_da_prevalencia(y_train)

    for r in resultados:
        assert r["calibracao"]["intercepto"] == pytest.approx(logito, abs=0.1)


def test_probabilidades_da_floresta_deslocadas_sem_a_identidade(split,
                                                               resultados_floresta):
    """A floresta também desloca o nível, e por um mecanismo que a álgebra não descreve.

    A saída da floresta é **fração de votos**, não a inversa de um logito somado a um
    intercepto: não existe termo aditivo a que atribuir o deslocamento. O intercepto que
    `diagnostico_calibracao` mede é o de uma regressão ajustada sobre o logito das
    probabilidades preditas, e ele erra o logito da prevalência por larga margem, ao
    contrário do que acontece na logística.

    Por isso os limites são outros, e a distância mínima em relação ao logito é asserção
    e não observação: transpor os limites da logística faria este teste passar por
    pouco, dando a entender que os dois modelos têm a mesma patologia. D-040 mostra que
    não têm.
    """
    _, _, y_train, _ = split
    logito = _logito_da_prevalencia(y_train)

    for r in resultados_floresta:
        cal = r["calibracao"]
        # O deslocamento existe, e é grande.
        assert cal["razao_inflacao"] > 3.0
        assert cal["proba_media"] > 3 * cal["prevalencia"]
        assert r["brier_skill"] < 0
        # A ordenação sobrevive, que é a condição para o limiar calibrado valer.
        assert 0.7 < cal["slope"] < 1.3
        # E a identidade da logística não transfere: o intercepto fica longe do logito.
        assert abs(cal["intercepto"] - logito) > 0.3


def test_modelo_bem_calibrado_tem_slope_um_e_intercepto_zero():
    """Controle: com probabilidades honestas, o diagnóstico acusa calibração boa."""
    rng = np.random.default_rng(cfg.RANDOM_STATE)
    proba = rng.uniform(0.01, 0.99, 40_000)
    y = rng.binomial(1, proba)

    cal = ev.diagnostico_calibracao(y, proba)
    assert cal["slope"] == pytest.approx(1.0, abs=0.12)
    assert cal["intercepto"] == pytest.approx(0.0, abs=0.12)
    assert cal["razao_inflacao"] == pytest.approx(1.0, abs=0.05)
    assert cal["ece"] < 0.02
    assert cal["brier_skill"] > 0  # melhor que o previsor trivial


def test_tabela_calibracao_comparada_uma_linha_por_conjunto(resultados):
    tabela = ev.tabela_calibracao_comparada(resultados)
    assert len(tabela) == 3
    assert {"brier", "brier_trivial", "brier_skill", "razao_inflacao",
            "ece", "slope", "intercepto"} <= set(tabela.columns)


def test_corte_unico_detectado_pela_saida(resultados):
    """Modelo com escore contínuo não é classificador de corte único."""
    for r in resultados:
        assert r["corte_unico"] is False
        assert len(np.unique(r["proba_teste"])) > 2


def test_nota_de_rodape_ausente_sem_corte_unico(resultados):
    """Sem linhas de corte único, não há advertência a fazer."""
    tabela = ev.comparar(resultados)
    assert ev.nota_corte_unico(tabela) == ""


def test_nota_de_rodape_gerada_automaticamente(resultados, tmp_path):
    """A advertência depende do conteúdo, não de alguém lembrar de pedi-la."""
    com_corte = ev.comparar(resultados)
    com_corte.loc[0, "corte_unico"] = True
    com_corte.loc[0, "modelo"] = "Regra ADA"

    nota = ev.nota_corte_unico(com_corte)
    assert "Regra ADA" in nota
    assert "AUC-ROC" in nota and "NNS" in nota

    destino = ev.exportar_latex(com_corte, "com_nota.tex", path_reports=tmp_path)
    conteudo = destino.read_text(encoding="utf-8")
    assert "footnotesize" in conteudo and "corte único" in conteudo

    # Suprimível de propósito, nunca por esquecimento.
    sem = ev.exportar_latex(com_corte, "sem_nota.tex", path_reports=tmp_path, nota="")
    assert "footnotesize" not in sem.read_text(encoding="utf-8")


def test_exportacao_latex(resultados, tmp_path):
    tabela = ev.comparar(resultados)
    destino = ev.exportar_latex(tabela, "comparativo.tex", path_reports=tmp_path)
    conteudo = destino.read_text(encoding="utf-8")

    assert destino.exists()
    assert "tabular" in conteudo
    assert "LogReg" in conteudo


# ---------------------------------------------------------------------------
# McNemar pareado
# ---------------------------------------------------------------------------

def test_mcnemar_conta_discordancias_e_nao_concordancias():
    """A 2x2 separa quem acerta o quê, e o p-valor olha só para as discordâncias.

    Monta as predições à mão para o resultado ser verificável na unha: dez casos, dois
    em que só A acerta e um em que só B acerta.
    """
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    a = {"modelo": "A", "conjunto": "", "limiar": 0.5,
         "proba_teste": np.array([.9, .9, .9, .1, .1, .1, .1, .1, .9, .1])}
    b = {"modelo": "B", "conjunto": "", "limiar": 0.5,
         "proba_teste": np.array([.9, .1, .1, .1, .1, .1, .1, .1, .1, .1])}

    teste = ev.mcnemar_pareado(y, a, b)

    assert teste["so_a_acerta"] == 2      # casos 2 e 3: A pega, B perde
    assert teste["so_b_acerta"] == 1      # caso 9: B acerta o negativo, A erra
    assert teste["discordantes"] == 3
    assert teste["concordam"] == 7
    assert teste["positivos_so_a_detecta"] == 2
    assert teste["positivos_so_b_detecta"] == 0
    assert teste["tabela"].to_numpy().sum() == len(y)


def test_mcnemar_sem_discordancia_devolve_p_igual_a_um():
    """Modelos idênticos não têm o que testar, e o teste não pode dividir por zero."""
    y = np.array([1, 0, 1, 0])
    igual = {"modelo": "A", "conjunto": "", "limiar": 0.5,
             "proba_teste": np.array([.9, .1, .9, .1])}
    outro = dict(igual, modelo="B")

    teste = ev.mcnemar_pareado(y, igual, outro)
    assert teste["discordantes"] == 0
    assert teste["p_valor"] == 1.0


def test_mcnemar_usa_o_limiar_de_cada_modelo():
    """Limiar por modelo, porque a escala de probabilidade de um não é a do outro."""
    y = np.array([1, 1, 0, 0])
    alto = {"modelo": "A", "conjunto": "", "limiar": 0.8,
            "proba_teste": np.array([.9, .5, .1, .1])}
    baixo = {"modelo": "B", "conjunto": "", "limiar": 0.3,
             "proba_teste": np.array([.9, .5, .1, .1])}

    teste = ev.mcnemar_pareado(y, alto, baixo)
    # Com limiar 0,8 o segundo positivo passa despercebido; com 0,3 ele é detectado.
    assert teste["positivos_so_b_detecta"] == 1
    assert teste["positivos_so_a_detecta"] == 0
