"""Harness de avaliação (§6 da spec · §6.3 fase 5).

Todo modelo passa por aqui — Regressão Logística, Random Forest e comparadores
não-ML —, garantindo que a comparação seja entre modelos e não entre protocolos de
avaliação.

Duas regras estruturam o módulo:

1. **O limiar é calibrado no treino, por validação cruzada, e nunca no teste.** Com
   7% de positivos, o 0,5 implícito do `predict()` é uma decisão silenciosa e
   inadequada (D-004). Aqui ele é parâmetro explícito, escolhido sobre
   probabilidades fora da amostra obtidas por `cross_val_predict`.
2. **O estimador chega pronto.** Já vem embrulhado no preprocessador da §4.5 —
   `avaliar` não constrói pipeline, não escolhe conjunto de atributos e não toca em
   imputação. Assim R2 e R3 são responsabilidade de um lugar só.

`peso_lab` não é parâmetro de nenhuma função deste módulo (R10, §9).
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import config as cfg

# Métricas assinadas na §6.3 fase 5 — obrigatórias em toda tabela de resultado.
METRICAS_ASSINADAS = ["recall", "especificidade", "f1", "auc_roc"]
# Complementares (§10 item 4) — somam-se às assinadas, jamais as substituem.
METRICAS_COMPLEMENTARES = ["pr_auc", "brier", "nns", "precisao"]


def _slug(*partes: str) -> str:
    """Nome de arquivo seguro a partir do modelo e do conjunto de atributos."""
    bruto = "_".join(p for p in partes if p)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", bruto).strip("_")


# ---------------------------------------------------------------------------
# Etapas 1 e 2 — validação cruzada no treino e seleção de limiar
# ---------------------------------------------------------------------------


def probabilidades_por_cv(estimador, X_tr, y_tr, cv=None) -> np.ndarray:
    """Probabilidades fora da amostra, por `StratifiedKFold(5)` no treino.

    `cross_val_predict` devolve, para cada observação de treino, a probabilidade
    estimada pelo fold em que ela **não** participou do ajuste. É o que permite
    calibrar o limiar sem olhar para o teste e sem otimismo de reajuste.
    """
    cv = cv if cv is not None else StratifiedKFold(
        n_splits=cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE
    )
    proba = cross_val_predict(
        clone(estimador), X_tr, y_tr, cv=cv, method="predict_proba"
    )
    return np.asarray(proba)[:, 1]


def selecionar_limiar(y_tr, proba_cv, recall_alvo: float) -> dict:
    """Maior limiar que ainda garante o recall-alvo, medido por CV no treino.

    Entre todos os limiares que atingem o alvo, o maior é o que produz a melhor
    precisão — menos encaminhamentos desnecessários para o mesmo recall. Em
    triagem, recall é restrição; precisão é o que se otimiza sob ela.

    Se **nenhum** limiar atinge o alvo, devolve o de maior recall e sinaliza:
    `recall_alvo_atingido=False`, mais um `UserWarning`. Silenciar isso faria uma
    tabela de resultados reportar um ponto de operação que não cumpre o que
    promete.

    Nota sobre esse ramo: em `precision_recall_curve` o menor limiar classifica
    todas as observações como positivas, logo o recall máximo da curva é 1,0 e
    qualquer alvo válido encontra candidato. O fallback é defesa contra curvas
    degeneradas, não caminho esperado — alvo fora de (0, 1] é erro de chamada e
    falha alto, em vez de virar aviso.

    Returns:
        Dicionário com `limiar`, `recall_alvo`, `recall_alvo_atingido`,
        `recall_cv` e `precisao_cv` no limiar escolhido.
    """
    if not 0.0 < recall_alvo <= 1.0:
        raise ValueError(
            f"recall_alvo deve estar em (0, 1]; recebido {recall_alvo!r}."
        )
    prec, rec, thr = precision_recall_curve(y_tr, proba_cv)
    # `precision_recall_curve` devolve um ponto a mais que os limiares; o zip
    # descarta o par terminal (recall 0, precisão 1), que não tem limiar associado.
    candidatos = [t for p, r, t in zip(prec, rec, thr) if r >= recall_alvo]

    if candidatos:
        limiar = float(max(candidatos))
        atingido = True
    else:
        # Sem nenhum limiar que alcance o alvo: o de maior recall é o melhor
        # disponível, e o fato precisa aparecer no resultado.
        melhor = int(np.argmax(rec[:-1]))
        limiar = float(thr[melhor])
        atingido = False
        warnings.warn(
            f"Nenhum limiar atinge recall >= {recall_alvo:.2f} na CV do treino. "
            f"Usando o de maior recall ({rec[melhor]:.3f}, limiar {limiar:.4f}).",
            UserWarning, stacklevel=2,
        )

    predito_cv = (proba_cv >= limiar).astype(int)
    return {
        "limiar": limiar,
        "recall_alvo": recall_alvo,
        "recall_alvo_atingido": atingido,
        "recall_cv": float(recall_score(y_tr, predito_cv, zero_division=0)),
        "precisao_cv": float(precision_score(y_tr, predito_cv, zero_division=0)),
    }


def metricas_por_fold(y_tr, proba_cv, limiar: float, cv=None,
                      X_tr=None) -> pd.DataFrame:
    """Métricas assinadas dentro de cada fold da CV, com o limiar já fixado.

    Reaproveita as probabilidades fora da amostra de `probabilidades_por_cv`: os
    folds são reconstruídos com a mesma semente (R11), então cada partição de
    validação é exatamente a que gerou aquelas probabilidades. Nenhum reajuste.

    A dispersão entre folds é o que permite dizer se a diferença entre A, B e C é
    sinal ou variação de partição. Sem ela, comparar 0,748 com 0,765 é comparar
    dois números sem escala de referência.

    Returns:
        DataFrame com uma linha por fold e as quatro métricas assinadas.
    """
    cv = cv if cv is not None else StratifiedKFold(
        n_splits=cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE
    )
    y_tr = np.asarray(y_tr)
    proba_cv = np.asarray(proba_cv)
    base = np.zeros((len(y_tr), 1)) if X_tr is None else X_tr

    linhas = []
    for i, (_, idx) in enumerate(cv.split(base, y_tr), start=1):
        m = metricas_no_limiar(y_tr[idx], proba_cv[idx], limiar)
        linhas.append({"fold": i, "n": len(idx),
                       **{k: m[k] for k in METRICAS_ASSINADAS}})
    return pd.DataFrame(linhas)


def resumir_folds(por_fold: pd.DataFrame) -> dict:
    """Média e desvio padrão amostral de cada métrica assinada entre os folds."""
    resumo = {}
    for metrica in METRICAS_ASSINADAS:
        resumo[f"{metrica}_cv_media"] = float(por_fold[metrica].mean())
        resumo[f"{metrica}_cv_dp"] = float(por_fold[metrica].std(ddof=1))
    return resumo


# ---------------------------------------------------------------------------
# Etapas 3 e 4 — métricas no teste
# ---------------------------------------------------------------------------


def metricas_no_limiar(y_te, proba_te, limiar: float) -> dict:
    """Métricas assinadas e complementares no teste, com o limiar já fixado.

    NNS (Number Needed to Screen) é `1 / precisão`: quantas pessoas a UBS encaminha
    a exame laboratorial para identificar um caso. Precisão zero devolve NNS
    infinito — nenhum encaminhamento acerta.
    """
    y_pred = (np.asarray(proba_te) >= limiar).astype(int)
    tn, fp, fn, vp = confusion_matrix(y_te, y_pred, labels=[0, 1]).ravel()

    precisao = float(precision_score(y_te, y_pred, zero_division=0))
    especificidade = float(tn / (tn + fp)) if (tn + fp) else float("nan")

    return {
        # Assinadas (§6.3 fase 5).
        "recall": float(recall_score(y_te, y_pred, zero_division=0)),
        "especificidade": especificidade,
        "f1": float(f1_score(y_te, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_te, proba_te)),
        # Complementares (§10 item 4).
        "pr_auc": float(average_precision_score(y_te, proba_te)),
        "brier": float(brier_score_loss(y_te, proba_te)),
        "precisao": precisao,
        "nns": float(1.0 / precisao) if precisao > 0 else float("inf"),
        # Matriz de confusão.
        "vp": int(vp), "vn": int(tn), "fp": int(fp), "fn": int(fn),
    }


# ---------------------------------------------------------------------------
# Calibração — o número, não só a figura
# ---------------------------------------------------------------------------


def brier_trivial(y) -> float:
    """Brier de um previsor que sempre responde a prevalência observada.

    Para uma constante `p`, o Brier é `p(1-p)`. É o piso de referência: um modelo
    com Brier acima disso tem probabilidades piores que a resposta mais ingênua
    possível, ainda que discrimine bem.
    """
    p = float(np.mean(y))
    return p * (1.0 - p)


def tabela_calibracao(y_te, proba_te, n_bins: int = 10,
                      strategy: str = "quantile") -> pd.DataFrame:
    """Calibração em tabela: por faixa de probabilidade, o predito e o observado.

    Returns:
        DataFrame com `bin`, `n`, `proba_media`, `freq_observada`, `diferenca` e
        `razao` (predito ÷ observado).
    """
    proba_te = np.asarray(proba_te, dtype="float64")
    y_te = np.asarray(y_te)

    if strategy == "quantile":
        cortes = np.quantile(proba_te, np.linspace(0, 1, n_bins + 1))
        cortes = np.unique(cortes)
    else:
        cortes = np.linspace(proba_te.min(), proba_te.max(), n_bins + 1)
    indices = np.clip(np.digitize(proba_te, cortes[1:-1], right=False),
                      0, len(cortes) - 2)

    linhas = []
    for b in range(len(cortes) - 1):
        sel = indices == b
        if not sel.any():
            continue
        predito = float(proba_te[sel].mean())
        observado = float(y_te[sel].mean())
        linhas.append({
            "bin": b + 1,
            "n": int(sel.sum()),
            "proba_media": predito,
            "freq_observada": observado,
            "diferenca": predito - observado,
            "razao": predito / observado if observado > 0 else float("inf"),
        })
    return pd.DataFrame(linhas)


def diagnostico_calibracao(y_te, proba_te, n_bins: int = 10) -> dict:
    """Quantifica o desvio entre probabilidade estimada e frequência observada.

    Métricas devolvidas:

    - `brier` e `brier_trivial` — o modelo contra o previsor constante na
      prevalência. `brier_skill` negativo significa pior que o trivial.
    - `proba_media` e `prevalencia` — se a média das probabilidades não bate com a
      prevalência, o modelo está deslocado em nível, não só em forma.
    - `razao_inflacao` — quantas vezes a probabilidade média excede a prevalência.
    - `ece` — erro de calibração esperado, média ponderada de |predito − observado|
      entre as faixas.
    - `slope` e `intercepto` — calibração de Cox: regressão de `y` sobre o logito
      das probabilidades. Modelo calibrado tem slope 1 e intercepto 0; intercepto
      fortemente negativo indica probabilidades sistematicamente infladas.
    """
    y_te = np.asarray(y_te)
    proba_te = np.asarray(proba_te, dtype="float64")

    tabela = tabela_calibracao(y_te, proba_te, n_bins=n_bins)
    ece = float(np.average(tabela["diferenca"].abs(), weights=tabela["n"]))

    brier = float(brier_score_loss(y_te, proba_te))
    trivial = brier_trivial(y_te)
    prevalencia = float(np.mean(y_te))
    proba_media = float(proba_te.mean())

    # Calibração de Cox: y ~ logito(p), sem `class_weight`, para medir o
    # deslocamento que o `class_weight` do modelo original introduziu.
    p = np.clip(proba_te, 1e-6, 1 - 1e-6)
    logito = np.log(p / (1 - p)).reshape(-1, 1)
    cox = LogisticRegression(max_iter=1000).fit(logito, y_te)

    return {
        "brier": brier,
        "brier_trivial": trivial,
        "brier_skill": float(1.0 - brier / trivial) if trivial else float("nan"),
        "prevalencia": prevalencia,
        "proba_media": proba_media,
        "razao_inflacao": proba_media / prevalencia if prevalencia else float("inf"),
        "ece": ece,
        "slope": float(cox.coef_[0][0]),
        "intercepto": float(cox.intercept_[0]),
        "tabela": tabela,
    }


# ---------------------------------------------------------------------------
# Etapa 5 — figuras
# ---------------------------------------------------------------------------


def _preparar_eixo(ax, titulo: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(titulo, fontweight="bold", color=cfg.INK)
    ax.set_xlabel(xlabel, color=cfg.INK)
    ax.set_ylabel(ylabel, color=cfg.INK)
    ax.grid(color=cfg.GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figura_matriz_confusao(y_te, proba_te, limiar: float, rotulo: str,
                           path_figuras: Path | None = None) -> Path:
    """Matriz de confusão no limiar escolhido. Salva e devolve o caminho."""
    path_figuras = cfg.PATH_FIGURAS if path_figuras is None else Path(path_figuras)
    path_figuras.mkdir(parents=True, exist_ok=True)
    destino = path_figuras / f"confusao_{_slug(rotulo)}.png"

    y_pred = (np.asarray(proba_te) >= limiar).astype(int)
    matriz = confusion_matrix(y_te, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(matriz, cmap="Greys", vmin=0, vmax=matriz.max())
    rotulos = ["HbA1c < 6,5%", "HbA1c ≥ 6,5%"]
    ax.set_xticks([0, 1], rotulos)
    ax.set_yticks([0, 1], rotulos, rotation=90, va="center")
    _preparar_eixo(ax, f"Matriz de confusão — {rotulo}\nlimiar {limiar:.3f}",
                   "Classificado como", "Observado")
    ax.grid(visible=False)

    nomes = [["VN", "FP"], ["FN", "VP"]]
    for i in range(2):
        for j in range(2):
            destaque = cfg.CORAL if nomes[i][j] in ("FN", "VP") else cfg.TEAL
            ax.text(j, i, f"{nomes[i][j]}\n{matriz[i, j]:,}".replace(",", "."),
                    ha="center", va="center", fontweight="bold", color=destaque)

    fig.tight_layout()
    fig.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return destino


def figura_roc(y_te, proba_te, rotulo: str,
               path_figuras: Path | None = None) -> Path:
    """Curva ROC no teste. Salva e devolve o caminho."""
    path_figuras = cfg.PATH_FIGURAS if path_figuras is None else Path(path_figuras)
    path_figuras.mkdir(parents=True, exist_ok=True)
    destino = path_figuras / f"roc_{_slug(rotulo)}.png"

    fpr, tpr, _ = roc_curve(y_te, proba_te)
    auc = roc_auc_score(y_te, proba_te)

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.plot(fpr, tpr, color=cfg.TEAL, lw=2.2, label=f"AUC-ROC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color=cfg.INK, lw=1, ls=":", label="aleatório")
    _preparar_eixo(ax, f"Curva ROC — {rotulo}",
                   "1 − especificidade (falsos positivos)",
                   "Recall (sensibilidade)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return destino


def figura_precision_recall(y_te, proba_te, limiar: float, rotulo: str,
                            path_figuras: Path | None = None) -> Path:
    """Curva Precision-Recall com o ponto de operação marcado."""
    path_figuras = cfg.PATH_FIGURAS if path_figuras is None else Path(path_figuras)
    path_figuras.mkdir(parents=True, exist_ok=True)
    destino = path_figuras / f"pr_{_slug(rotulo)}.png"

    prec, rec, _ = precision_recall_curve(y_te, proba_te)
    pr_auc = average_precision_score(y_te, proba_te)
    y_pred = (np.asarray(proba_te) >= limiar).astype(int)
    rec_op = recall_score(y_te, y_pred, zero_division=0)
    prec_op = precision_score(y_te, y_pred, zero_division=0)
    prevalencia = float(np.mean(y_te))

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.plot(rec, prec, color=cfg.TEAL, lw=2.2, label=f"PR-AUC = {pr_auc:.3f}")
    ax.axhline(prevalencia, color=cfg.INK, lw=1, ls=":",
               label=f"prevalência = {prevalencia:.3f}")
    ax.scatter([rec_op], [prec_op], color=cfg.CORAL, s=90, zorder=5,
               label=f"operação: limiar {limiar:.3f}")
    ax.annotate(f"recall {rec_op:.3f}\nprecisão {prec_op:.3f}\nNNS "
                f"{(1 / prec_op if prec_op else float('inf')):.1f}",
                (rec_op, prec_op), textcoords="offset points", xytext=(10, 10),
                fontsize=9, color=cfg.CORAL, fontweight="bold")
    _preparar_eixo(ax, f"Precision-Recall — {rotulo}",
                   "Recall (sensibilidade)", "Precisão")
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    fig.tight_layout()
    fig.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return destino


def figura_calibracao(y_te, proba_te, rotulo: str, n_bins: int = 10,
                      path_figuras: Path | None = None) -> Path:
    """Curva de calibração — subsidia o backlog de calibração probabilística."""
    path_figuras = cfg.PATH_FIGURAS if path_figuras is None else Path(path_figuras)
    path_figuras.mkdir(parents=True, exist_ok=True)
    destino = path_figuras / f"calibracao_{_slug(rotulo)}.png"

    observado, predito = calibration_curve(y_te, proba_te, n_bins=n_bins,
                                           strategy="quantile")
    brier = brier_score_loss(y_te, proba_te)

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.plot([0, 1], [0, 1], color=cfg.INK, lw=1, ls=":", label="calibração perfeita")
    ax.plot(predito, observado, color=cfg.TEAL, lw=2.2, marker="o",
            markersize=5, label=f"Brier = {brier:.4f}")
    _preparar_eixo(ax, f"Curva de calibração — {rotulo}",
                   "Probabilidade estimada", "Frequência observada")
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return destino


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def avaliar(estimador, X_tr, y_tr, X_te, y_te, nome: str,
            recall_alvo: float = cfg.RECALL_ALVO, conjunto: str = "",
            salvar_figuras: bool = True,
            path_figuras: Path | None = None) -> dict:
    """Avalia um estimador pelo protocolo da §6.3 fase 5.

    Args:
        estimador: estimador **já embrulhado** no preprocessador da §4.5. Nada
            aqui constrói pipeline: o conjunto de atributos e a imputação são
            decididos fora, em `features.construir_preprocessador`.
        X_tr, y_tr: treino. Único conjunto que participa da escolha do limiar.
        X_te, y_te: teste. Usado exclusivamente para medir, nunca para decidir.
        nome: identificação do modelo na tabela comparativa.
        recall_alvo: restrição de recall do ponto de operação (D-004, provisório
            em 0,80 — confirmar com o orientador).
        conjunto: conjunto de atributos, registrado para a comparação.

    Returns:
        Dicionário com o limiar, as métricas assinadas e complementares, a matriz
        de confusão, os caminhos das figuras e o estimador ajustado.

    O estimador recebido não é modificado: a CV e o ajuste final operam sobre
    clones.
    """
    rotulo = f"{nome}_{conjunto}" if conjunto else nome

    # Etapas 1 e 2 — tudo no treino.
    proba_cv = probabilidades_por_cv(estimador, X_tr, y_tr)
    escolha = selecionar_limiar(y_tr, proba_cv, recall_alvo)
    limiar = escolha["limiar"]

    # Dispersão entre folds, no limiar já fixado: escala de referência para
    # julgar se a diferença entre conjuntos é sinal ou variação de partição.
    por_fold = metricas_por_fold(y_tr, proba_cv, limiar, X_tr=X_tr)
    dispersao = resumir_folds(por_fold)

    # Etapa 3 — ajuste final no treino completo e medição no teste.
    ajustado = clone(estimador).fit(X_tr, y_tr)
    proba_te = ajustado.predict_proba(X_te)[:, 1]
    metricas = metricas_no_limiar(y_te, proba_te, limiar)
    calibracao = diagnostico_calibracao(y_te, proba_te)

    figuras = {}
    if salvar_figuras:
        figuras = {
            "confusao": figura_matriz_confusao(y_te, proba_te, limiar, rotulo,
                                               path_figuras),
            "roc": figura_roc(y_te, proba_te, rotulo, path_figuras),
            "precision_recall": figura_precision_recall(y_te, proba_te, limiar,
                                                        rotulo, path_figuras),
            "calibracao": figura_calibracao(y_te, proba_te, rotulo, path_figuras=
                                            path_figuras),
        }

    return {
        "modelo": nome,
        "conjunto": conjunto,
        "auc_roc_cv": float(roc_auc_score(y_tr, proba_cv)),
        **escolha,
        **metricas,
        **dispersao,
        "brier_trivial": calibracao["brier_trivial"],
        "brier_skill": calibracao["brier_skill"],
        "razao_inflacao": calibracao["razao_inflacao"],
        "ece": calibracao["ece"],
        "calibracao": calibracao,
        "metricas_por_fold": por_fold,
        "figuras": figuras,
        "estimador": ajustado,
        "proba_teste": proba_te,
        "proba_cv_treino": proba_cv,
    }


def comparar(resultados: list[dict]) -> pd.DataFrame:
    """Tabela comparativa: uma linha por (modelo × conjunto de atributos).

    As métricas assinadas na §6.3 fase 5 vêm primeiro e estão sempre presentes,
    cada uma seguida da média e do desvio padrão entre os folds da CV; as
    complementares vêm depois. Exportável via `.to_latex()`.
    """
    assinadas_com_dispersao = [
        coluna
        for metrica in METRICAS_ASSINADAS
        for coluna in (metrica, f"{metrica}_cv_media", f"{metrica}_cv_dp")
    ]
    colunas = (
        ["modelo", "conjunto", "limiar", "recall_alvo_atingido"]
        + assinadas_com_dispersao
        + METRICAS_COMPLEMENTARES
        + ["brier_trivial", "brier_skill", "razao_inflacao", "ece"]
        + ["vp", "fp", "fn", "vn", "auc_roc_cv", "recall_cv"]
    )
    tabela = pd.DataFrame([
        {k: r[k] for k in colunas if k in r} for r in resultados
    ])
    return tabela[[c for c in colunas if c in tabela.columns]]


def comparar_formatado(resultados: list[dict], casas: int = 3) -> pd.DataFrame:
    """Tabela das métricas assinadas no formato `teste (CV: média ± dp)`.

    Versão para o texto do TCC: o valor de teste é o resultado; a dispersão entre
    folds é a escala que diz se a diferença entre conjuntos significa algo.
    """
    linhas = []
    for r in resultados:
        linha = {"modelo": r["modelo"], "conjunto": r["conjunto"],
                 "limiar": round(r["limiar"], casas)}
        for metrica in METRICAS_ASSINADAS:
            linha[metrica] = (
                f"{r[metrica]:.{casas}f} "
                f"({r[f'{metrica}_cv_media']:.{casas}f} ± "
                f"{r[f'{metrica}_cv_dp']:.{casas}f})"
            )
        linha["nns"] = round(r["nns"], 2)
        linhas.append(linha)
    return pd.DataFrame(linhas)


def tabela_calibracao_comparada(resultados: list[dict]) -> pd.DataFrame:
    """Uma linha por conjunto com o diagnóstico numérico de calibração."""
    return pd.DataFrame([
        {
            "modelo": r["modelo"],
            "conjunto": r["conjunto"],
            "brier": r["brier"],
            "brier_trivial": r["brier_trivial"],
            "brier_skill": r["brier_skill"],
            "prevalencia": r["calibracao"]["prevalencia"],
            "proba_media": r["calibracao"]["proba_media"],
            "razao_inflacao": r["razao_inflacao"],
            "ece": r["ece"],
            "slope": r["calibracao"]["slope"],
            "intercepto": r["calibracao"]["intercepto"],
        }
        for r in resultados
    ])


def exportar_latex(tabela: pd.DataFrame, nome_arquivo: str,
                   path_reports: Path | None = None, **kwargs) -> Path:
    """Exporta a tabela comparativa para `.tex` em `reports/`."""
    path_reports = cfg.PATH_REPORTS if path_reports is None else Path(path_reports)
    path_reports.mkdir(parents=True, exist_ok=True)
    destino = path_reports / nome_arquivo

    opcoes = {"index": False, "float_format": "%.3f"}
    opcoes.update(kwargs)
    destino.write_text(tabela.to_latex(**opcoes), encoding="utf-8")
    return destino
