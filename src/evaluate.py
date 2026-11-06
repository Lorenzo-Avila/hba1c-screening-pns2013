"""Harness de avaliação.

Todo modelo passa por aqui: Regressão Logística, Random Forest e os comparadores
não-ML. Assim a comparação é entre modelos, não entre protocolos de avaliação.

Duas regras estruturam o módulo:

1. **O limiar é calibrado no treino, por validação cruzada, nunca no teste.** Com
   7% de positivos, o 0,5 implícito do `predict()` é uma decisão silenciosa e
   ruim. Aqui ele é parâmetro explícito, escolhido sobre probabilidades fora da
   amostra obtidas por `cross_val_predict`.
2. **O estimador chega pronto**, já embrulhado no preprocessador. `avaliar` não
   constrói pipeline, não escolhe conjunto de atributos e não toca em imputação,
   de modo que o cuidado com vazamento fica concentrado num lugar só.

`peso_lab` não é parâmetro de nenhuma função deste módulo.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.base import clone
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
from . import figuras as figu

# Métricas assinadas no TCC I: obrigatórias em toda tabela de resultado.
METRICAS_ASSINADAS = ["recall", "especificidade", "f1", "auc_roc"]
# Complementares: somam-se às assinadas, nunca as substituem.
METRICAS_COMPLEMENTARES = ["pr_auc", "brier", "nns", "precisao", "acuracia"]


def _slug(*partes: str) -> str:
    """Monta um nome de arquivo seguro a partir do modelo e do conjunto."""
    bruto = "_".join(p for p in partes if p)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", bruto).strip("_")


# ---------------------------------------------------------------------------
# Validação cruzada no treino e seleção de limiar
# ---------------------------------------------------------------------------


def probabilidades_por_cv(estimador, X_tr, y_tr, cv=None) -> np.ndarray:
    """Probabilidades fora da amostra, por `StratifiedKFold(5)` no treino.

    Para cada observação de treino, `cross_val_predict` devolve a probabilidade
    estimada pelo fold em que ela **não** entrou no ajuste. É o que permite calibrar
    o limiar sem olhar para o teste e sem o otimismo de quem reavalia no que treinou.
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

    Entre os limiares que atingem o alvo, o maior é o de melhor precisão, ou seja,
    menos encaminhamento desnecessário para o mesmo recall. Em triagem, recall é
    restrição e precisão é o que se otimiza sob ela.

    Se **nenhum** limiar atinge o alvo, devolve o de maior recall e sinaliza com
    `recall_alvo_atingido=False` mais um `UserWarning`. Silenciar isso faria a
    tabela de resultados reportar um ponto de operação que não cumpre o que promete.

    Sobre esse ramo: em `precision_recall_curve` o menor limiar classifica todo
    mundo como positivo, então o recall máximo da curva é 1,0 e qualquer alvo válido
    encontra candidato. O fallback é defesa contra curva degenerada, não caminho
    esperado. Alvo fora de (0, 1] é erro de chamada e falha alto, em vez de virar
    aviso.

    Returns:
        Dicionário com `limiar`, `recall_alvo`, `recall_alvo_atingido`, `recall_cv`
        e `precisao_cv` no limiar escolhido.
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
        # Nenhum limiar alcança o alvo: o de maior recall é o melhor disponível, e
        # isso precisa aparecer no resultado.
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
    folds são reconstruídos com a mesma semente, então cada partição de validação é
    exatamente a que gerou aquelas probabilidades. Nada é reajustado.

    A dispersão entre folds é o que permite dizer se a diferença entre A, B e C é
    sinal ou variação de partição. Sem ela, comparar 0,748 com 0,765 é comparar dois
    números sem escala de referência.

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
# Métricas no teste
# ---------------------------------------------------------------------------


def metricas_no_limiar(y_te, proba_te, limiar: float) -> dict:
    """Métricas assinadas e complementares no teste, com o limiar já fixado.

    NNS (Number Needed to Screen) é `1 / precisão`: quantas pessoas a UBS encaminha
    a exame laboratorial para achar um caso. Precisão zero devolve NNS infinito,
    porque nenhum encaminhamento acerta.
    """
    y_pred = (np.asarray(proba_te) >= limiar).astype(int)
    tn, fp, fn, vp = confusion_matrix(y_te, y_pred, labels=[0, 1]).ravel()

    precisao = float(precision_score(y_te, y_pred, zero_division=0))
    especificidade = float(tn / (tn + fp)) if (tn + fp) else float("nan")

    return {
        # Assinadas.
        "recall": float(recall_score(y_te, y_pred, zero_division=0)),
        "especificidade": especificidade,
        "f1": float(f1_score(y_te, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_te, proba_te)),
        # Complementares.
        "pr_auc": float(average_precision_score(y_te, proba_te)),
        "brier": float(brier_score_loss(y_te, proba_te)),
        "precisao": precisao,
        "nns": float(1.0 / precisao) if precisao > 0 else float("inf"),
        # Acurácia entra só para expor a própria armadilha: com 7% de positivos,
        # chutar negativo para todo mundo já dá uns 0,93.
        "acuracia": float((vp + tn) / (vp + tn + fp + fn)),
        # Matriz de confusão.
        "vp": int(vp), "vn": int(tn), "fp": int(fp), "fn": int(fn),
    }


# ---------------------------------------------------------------------------
# Calibração: o número, não só a figura
# ---------------------------------------------------------------------------


def brier_trivial(y) -> float:
    """Brier de quem sempre responde a prevalência observada.

    Para uma constante `p`, o Brier é `p(1-p)`. Serve de piso de referência: modelo
    com Brier acima disso tem probabilidades piores que a resposta mais ingênua
    possível, mesmo que discrimine bem.
    """
    p = float(np.mean(y))
    return p * (1.0 - p)


def tabela_calibracao(y_te, proba_te, n_bins: int = 10,
                      strategy: str = "quantile") -> pd.DataFrame:
    """Calibração em tabela: por faixa de probabilidade, o predito e o observado.

    Returns:
        DataFrame com `bin`, `n`, `proba_media`, `freq_observada`, `diferenca` e
        `razao` (predito dividido por observado).
    """
    proba_te = np.asarray(proba_te, dtype="float64")
    y_te = np.asarray(y_te)

    distintos = np.unique(proba_te)
    if len(distintos) <= n_bins:
        # Saída discreta, caso dos comparadores, que devolvem indicadoras. Cada
        # valor vira seu próprio grupo: com dois valores, a tabela mostra a
        # frequência observada entre encaminhados e entre não encaminhados. Forçar
        # quantis aqui daria faixas vazias ou um único bin degenerado.
        grupos = [(v, proba_te == v) for v in distintos]
    else:
        if strategy == "quantile":
            cortes = np.unique(np.quantile(proba_te, np.linspace(0, 1, n_bins + 1)))
        else:
            cortes = np.linspace(proba_te.min(), proba_te.max(), n_bins + 1)
        indices = np.clip(np.digitize(proba_te, cortes[1:-1], right=False),
                          0, max(len(cortes) - 2, 0))
        grupos = [(b, indices == b) for b in range(max(len(cortes) - 1, 1))]

    linhas = []
    for _, sel in grupos:
        if not sel.any():
            continue
        predito = float(proba_te[sel].mean())
        observado = float(y_te[sel].mean())
        linhas.append({
            "bin": len(linhas) + 1,
            "n": int(sel.sum()),
            "proba_media": predito,
            "freq_observada": observado,
            "diferenca": predito - observado,
            "razao": predito / observado if observado > 0 else float("inf"),
        })
    return pd.DataFrame(
        linhas,
        columns=["bin", "n", "proba_media", "freq_observada", "diferenca", "razao"],
    )


def calibracao_decis(y_te, proba_te, n_bins: int = 10) -> pd.DataFrame:
    """Curva de calibração por decis, a mesma que `diagnostico_calibracao` resume.

    Sai de `tabela_calibracao`, e não de `sklearn.calibration.calibration_curve`. As
    duas cortam por quantis e diferem na terceira casa, porque o scikit-learn atribui
    as faixas de outro jeito. Esta é a que dá o ECE publicado e o décimo decil de
    D-040, e é a que a figura desenha.

    Returns:
        DataFrame com `decil`, `n`, `prob_media` e `freq_observada`.
    """
    tabela = tabela_calibracao(y_te, proba_te, n_bins=n_bins)
    return tabela.rename(columns={"bin": "decil", "proba_media": "prob_media"})[
        ["decil", "n", "prob_media", "freq_observada"]
    ]


def diagnostico_calibracao(y_te, proba_te, n_bins: int = 10) -> dict:
    """Quantifica o desvio entre probabilidade estimada e frequência observada.

    Métricas devolvidas:

    - `brier` e `brier_trivial`: o modelo contra o previsor constante na
      prevalência. `brier_skill` negativo é pior que o trivial.
    - `proba_media` e `prevalencia`: se a média das probabilidades não bate com a
      prevalência, o modelo está deslocado em nível, não só em forma.
    - `razao_inflacao`: quantas vezes a probabilidade média excede a prevalência.
    - `ece`: erro de calibração esperado, média ponderada de |predito − observado|
      entre as faixas.
    - `slope` e `intercepto`: calibração de Cox, regredindo `y` sobre o logito das
      probabilidades. Modelo calibrado tem slope 1 e intercepto 0; intercepto bem
      negativo indica probabilidades infladas de forma sistemática.
    """
    y_te = np.asarray(y_te)
    proba_te = np.asarray(proba_te, dtype="float64")

    tabela = tabela_calibracao(y_te, proba_te, n_bins=n_bins)
    ece = float(np.average(tabela["diferenca"].abs(), weights=tabela["n"]))

    brier = float(brier_score_loss(y_te, proba_te))
    trivial = brier_trivial(y_te)
    prevalencia = float(np.mean(y_te))
    proba_media = float(proba_te.mean())

    # Calibração de Cox: y ~ logito(p), estimada sem penalização e sem
    # `class_weight`, para medir o deslocamento que o `class_weight` do modelo
    # original introduziu.
    p = np.clip(proba_te, 1e-6, 1 - 1e-6)
    logito = np.log(p / (1 - p)).reshape(-1, 1)
    cox = LogisticRegression(C=np.inf, max_iter=1000).fit(logito, y_te)

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
# Curvas: os pontos que as figuras desenham e que os artefatos gravam
# ---------------------------------------------------------------------------


def curva_roc(y_te, proba_te) -> pd.DataFrame:
    """Pontos da curva ROC, na ordem de `roc_curve`.

    O primeiro limiar é infinito: é o ponto (0, 0), em que ninguém é encaminhado.
    A área por trapézio sobre estes pontos é a `auc_roc` da tabela consolidada.

    Returns:
        DataFrame com `fpr`, `tpr` e `limiar`.
    """
    fpr, tpr, limiares = roc_curve(y_te, proba_te)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "limiar": limiares})


def curva_pr(y_te, proba_te) -> pd.DataFrame:
    """Pontos da curva precision-recall, na ordem de `precision_recall_curve`.

    O último ponto, recall 0 e precisão 1, não tem limiar associado e sai com limiar
    vazio. A PR-AUC publicada é `average_precision`, a soma em degrau
    `Σ (recall_n − recall_{n−1}) · precisao_n` sobre estes pontos. Trapézio aqui dá
    outro número.

    Returns:
        DataFrame com `precisao`, `recall` e `limiar`.
    """
    precisao, recall, limiares = precision_recall_curve(y_te, proba_te)
    return pd.DataFrame({
        "precisao": precisao,
        "recall": recall,
        "limiar": np.append(limiares, np.nan),
    })


# Decis fixos de sensibilidade, fechados à direita: (início; fim]. Definidos antes da
# execução. A divisão em duas faixas com corte em 0,1 saiu de olhar os dados e não entra.
FAIXAS_RECALL = [(i / 10, (i + 1) / 10) for i in range(10)]


def conferir_faixas(faixas) -> None:
    """Falha se as faixas não forem contíguas, sem sobreposição, cobrindo (0; 1].

    Raises:
        ValueError: com a primeira lacuna, sobreposição ou faixa vazia encontrada.
    """
    if not faixas or faixas[0][0] != 0 or faixas[-1][1] != 1:
        raise ValueError(f"As faixas não cobrem (0; 1]: {faixas}")
    for inicio, fim in faixas:
        if inicio >= fim:
            raise ValueError(f"Faixa vazia ou invertida: ({inicio}; {fim}]")
    for (_, fim), (inicio, _) in zip(faixas, faixas[1:]):
        if fim != inicio:
            tipo = "lacuna" if fim < inicio else "sobreposição"
            raise ValueError(f"{tipo.capitalize()} entre {fim} e {inicio}")


def pr_auc_por_faixa(curva: pd.DataFrame, faixas=FAIXAS_RECALL) -> pd.DataFrame:
    """Decompõe a soma em degrau da PR-AUC por faixa de sensibilidade.

    Recebe a saída de `curva_pr`, para que curva e decomposição venham da mesma chamada
    de `precision_recall_curve`. Com os pontos em recall crescente e `recall_0 = 0`, a
    contribuição de cada ponto é `(recall_n − recall_{n−1}) · precisao_n` e vai para a
    faixa que contém `recall_n`, extremidade direita incluída. Ponto que repete o recall
    contribui com zero.

    Returns:
        DataFrame com `faixa_inicio`, `faixa_fim` e `contribuicao`, uma linha por faixa.
    """
    conferir_faixas(faixas)
    recall = curva["recall"].to_numpy()[::-1]
    precisao = curva["precisao"].to_numpy()[::-1]
    contribuicao = np.diff(recall, prepend=0.0) * precisao
    return pd.DataFrame([
        {"faixa_inicio": inicio, "faixa_fim": fim,
         "contribuicao": float(contribuicao[(recall > inicio) & (recall <= fim)].sum())}
        for inicio, fim in faixas
    ])


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------


def _preparar_eixo(ax, xlabel: str, ylabel: str) -> None:
    """Rótulos e grade. O título da figura fica na legenda do LaTeX."""
    figu.eixo(ax, xlabel, ylabel)


def figura_matriz_confusao(y_te, proba_te, limiar: float, rotulo: str,
                           path_figuras: Path | None = None) -> Path:
    """Matriz de confusão no limiar escolhido. Salva a figura e devolve o caminho."""
    path_figuras = cfg.PATH_FIGURAS if path_figuras is None else Path(path_figuras)
    path_figuras.mkdir(parents=True, exist_ok=True)
    destino = path_figuras / f"confusao_{_slug(rotulo)}.png"

    y_pred = (np.asarray(proba_te) >= limiar).astype(int)
    matriz = confusion_matrix(y_te, y_pred, labels=[0, 1])

    fig, ax = figu.subplots(figu.LARGURA_INTEIRA, figu.LARGURA_INTEIRA * 0.78)
    # O máximo da escala fica no quarto do cinza. No vmax da própria matriz a célula
    # VN saía preta e o texto verde-azulado sumia.
    ax.imshow(matriz, cmap="Greys", vmin=0, vmax=float(matriz.max()) * 4)
    rotulos = ["HbA1c < 6,5%", "HbA1c ≥ 6,5%"]
    ax.set_xticks([0, 1], rotulos)
    ax.set_yticks([0, 1], rotulos, rotation=90, va="center")
    _preparar_eixo(ax, "Classificado como", "Observado")
    ax.grid(visible=False)

    nomes = [["VN", "FP"], ["FN", "VP"]]
    for i in range(2):
        for j in range(2):
            destaque = cfg.CORAL if nomes[i][j] in ("FN", "VP") else cfg.TEAL
            ax.text(j, i, f"{nomes[i][j]}\n{figu.numero(matriz[i, j], 0)}",
                    ha="center", va="center", fontweight="bold", color=destaque,
                    fontsize=9)

    return figu.salvar(fig, destino)


def figura_roc(y_te, proba_te, rotulo: str,
               path_figuras: Path | None = None) -> Path:
    """Curva ROC no teste. Salva a figura e devolve o caminho."""
    path_figuras = cfg.PATH_FIGURAS if path_figuras is None else Path(path_figuras)
    path_figuras.mkdir(parents=True, exist_ok=True)
    destino = path_figuras / f"roc_{_slug(rotulo)}.png"

    curva = curva_roc(y_te, proba_te)
    auc = roc_auc_score(y_te, proba_te)

    fig, ax = figu.subplots(figu.LARGURA_INTEIRA, figu.LARGURA_INTEIRA * 0.78)
    ax.plot(curva["fpr"], curva["tpr"], color=cfg.TEAL, lw=2.2,
            label=f"AUC-ROC = {figu.numero(auc, 3)}")
    ax.plot([0, 1], [0, 1], color=cfg.INK, lw=1, ls=":", label="aleatório")
    _preparar_eixo(ax, "1 − especificidade (falsos positivos)",
                   "Recall (sensibilidade)")
    ax.xaxis.set_major_formatter(figu.formatador(2))
    ax.yaxis.set_major_formatter(figu.formatador(2))
    ax.legend(frameon=False, loc="lower right")

    return figu.salvar(fig, destino)


def figura_precision_recall(y_te, proba_te, limiar: float, rotulo: str,
                            path_figuras: Path | None = None) -> Path:
    """Curva Precision-Recall com o ponto de operação marcado."""
    path_figuras = cfg.PATH_FIGURAS if path_figuras is None else Path(path_figuras)
    path_figuras.mkdir(parents=True, exist_ok=True)
    destino = path_figuras / f"pr_{_slug(rotulo)}.png"

    curva = curva_pr(y_te, proba_te)
    prec, rec = curva["precisao"], curva["recall"]
    pr_auc = average_precision_score(y_te, proba_te)
    y_pred = (np.asarray(proba_te) >= limiar).astype(int)
    rec_op = recall_score(y_te, y_pred, zero_division=0)
    prec_op = precision_score(y_te, y_pred, zero_division=0)
    prevalencia = float(np.mean(y_te))

    nns = 1 / prec_op if prec_op else float("inf")
    fig, ax = figu.subplots(figu.LARGURA_INTEIRA, figu.LARGURA_INTEIRA * 0.78)
    ax.plot(rec, prec, color=cfg.TEAL, lw=2.2,
            label=f"PR-AUC = {figu.numero(pr_auc, 3)}")
    ax.axhline(prevalencia, color=cfg.INK, lw=1, ls=":",
               label=f"prevalência = {figu.numero(prevalencia, 3)}")
    ax.scatter([rec_op], [prec_op], color=cfg.CORAL, s=90, zorder=5, marker="o",
               label=f"operação, limiar {figu.numero(limiar, 3)}")
    ax.annotate(f"recall {figu.numero(rec_op, 3)}\nprecisão {figu.numero(prec_op, 3)}\n"
                f"NNS {figu.numero(nns, 2)}",
                (rec_op, prec_op), textcoords="offset points", xytext=(10, 10),
                fontsize=9, color=cfg.CORAL, fontweight="bold")
    _preparar_eixo(ax, "Recall (sensibilidade)", "Precisão")
    ax.xaxis.set_major_formatter(figu.formatador(2))
    ax.yaxis.set_major_formatter(figu.formatador(2))
    ax.legend(frameon=False, loc="upper right")

    return figu.salvar(fig, destino)


def figura_calibracao(y_te, proba_te, rotulo: str, n_bins: int = 10,
                      path_figuras: Path | None = None) -> Path:
    """Curva de calibração por decis, a mesma de `calibracao_decis`."""
    path_figuras = cfg.PATH_FIGURAS if path_figuras is None else Path(path_figuras)
    path_figuras.mkdir(parents=True, exist_ok=True)
    destino = path_figuras / f"calibracao_{_slug(rotulo)}.png"

    decis = calibracao_decis(y_te, proba_te, n_bins=n_bins)
    brier = brier_score_loss(y_te, proba_te)

    fig, ax = figu.subplots(figu.LARGURA_INTEIRA, figu.LARGURA_INTEIRA * 0.78)
    ax.plot([0, 1], [0, 1], color=cfg.INK, lw=1, ls=":", label="calibração perfeita")
    ax.plot(decis["prob_media"], decis["freq_observada"], color=cfg.TEAL, lw=2.2,
            marker="o", markersize=5, label=f"Brier = {figu.numero(brier, 4)}")
    _preparar_eixo(ax, "Probabilidade estimada", "Frequência observada")
    ax.xaxis.set_major_formatter(figu.formatador(2))
    ax.yaxis.set_major_formatter(figu.formatador(2))
    ax.legend(frameon=False, loc="upper left")

    return figu.salvar(fig, destino)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def avaliar(estimador, X_tr, y_tr, X_te, y_te, nome: str,
            recall_alvo: float = cfg.RECALL_ALVO, conjunto: str = "",
            salvar_figuras: bool = True, limiar: float | None = None,
            path_figuras: Path | None = None) -> dict:
    """Avalia um estimador pelo protocolo de validação do projeto.

    Args:
        estimador: estimador **já embrulhado** no preprocessador. Nada aqui constrói
            pipeline: conjunto de atributos e imputação são decididos fora, em
            `features.construir_preprocessador`.
        X_tr, y_tr: treino. Único conjunto que participa da escolha do limiar.
        X_te, y_te: teste. Serve só para medir, nunca para decidir.
        nome: identificação do modelo na tabela comparativa.
        recall_alvo: restrição de recall do ponto de operação. 0,80 é o ponto
            principal, na faixa de sensibilidade do FINDRISC.
        conjunto: conjunto de atributos, registrado para a comparação.
        limiar: quando informado, **dispensa a calibração** e usa este valor. É o
            caso dos comparadores não-ML: a regra clínica ADA devolve uma indicadora
            0/1, não um escore, e calibrar limiar sobre ela daria um ponto de
            operação sem significado. `limiar_calibrado` no retorno distingue os
            dois regimes.

    Returns:
        Dicionário com o limiar, as métricas assinadas e complementares, a matriz de
        confusão, os caminhos das figuras e o estimador ajustado.

    O estimador recebido não é modificado: a CV e o ajuste final operam sobre clones.
    """
    rotulo = f"{nome}_{conjunto}" if conjunto else nome

    # Calibração do limiar: tudo no treino.
    proba_cv = probabilidades_por_cv(estimador, X_tr, y_tr)
    if limiar is None:
        escolha = selecionar_limiar(y_tr, proba_cv, recall_alvo)
        escolha["limiar_calibrado"] = True
    else:
        # Limiar imposto: nenhuma busca, nenhuma curva. `recall_alvo_atingido`
        # deixa de ser garantia e vira constatação.
        predito_cv = (proba_cv >= limiar).astype(int)
        recall_cv = float(recall_score(y_tr, predito_cv, zero_division=0))
        escolha = {
            "limiar": float(limiar),
            "limiar_calibrado": False,
            "recall_alvo": recall_alvo,
            "recall_alvo_atingido": bool(recall_cv >= recall_alvo),
            "recall_cv": recall_cv,
            "precisao_cv": float(precision_score(y_tr, predito_cv,
                                                 zero_division=0)),
        }
    limiar = escolha["limiar"]

    # Dispersão entre folds, no limiar já fixado: escala de referência para
    # julgar se a diferença entre conjuntos é sinal ou variação de partição.
    por_fold = metricas_por_fold(y_tr, proba_cv, limiar, X_tr=X_tr)
    dispersao = resumir_folds(por_fold)

    # Ajuste final no treino completo e medição no teste.
    ajustado = clone(estimador).fit(X_tr, y_tr)
    proba_te = ajustado.predict_proba(X_te)[:, 1]
    metricas = metricas_no_limiar(y_te, proba_te, limiar)
    calibracao = diagnostico_calibracao(y_te, proba_te)

    # Classificador de corte único: a saída assume no máximo dois valores, então
    # não há curva a percorrer. Detectado pela saída, não pelo tipo do estimador,
    # porque o que vale é o que ele produz. É isso que governa a nota de rodapé de
    # `comparar`.
    corte_unico = bool(len(np.unique(proba_te)) <= 2)

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
        "corte_unico": corte_unico,
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
    """Tabela comparativa: uma linha por modelo e conjunto de atributos.

    As métricas assinadas vêm primeiro e estão sempre presentes, cada uma seguida da
    média e do desvio padrão entre os folds da CV; as complementares vêm depois.
    Exportável via `.to_latex()`.
    """
    assinadas_com_dispersao = [
        coluna
        for metrica in METRICAS_ASSINADAS
        for coluna in (metrica, f"{metrica}_cv_media", f"{metrica}_cv_dp")
    ]
    colunas = (
        ["modelo", "conjunto", "limiar", "limiar_calibrado", "recall_alvo_atingido"]
        + assinadas_com_dispersao
        + METRICAS_COMPLEMENTARES
        + ["brier_trivial", "brier_skill", "razao_inflacao", "ece"]
        + ["vp", "fp", "fn", "vn", "auc_roc_cv", "recall_cv", "corte_unico"]
    )
    tabela = pd.DataFrame([
        {k: r[k] for k in colunas if k in r} for r in resultados
    ])
    return tabela[[c for c in colunas if c in tabela.columns]]


def comparar_formatado(resultados: list[dict], casas: int = 3) -> pd.DataFrame:
    """Tabela das métricas assinadas no formato `teste (CV: média ± dp)`.

    É a versão para o texto do TCC: o valor de teste é o resultado, e a dispersão
    entre folds é a escala que diz se a diferença entre conjuntos significa algo.
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


def mcnemar_pareado(y_te, resultado_a: dict, resultado_b: dict) -> dict:
    """Teste de McNemar entre dois modelos, nas predições do teste.

    Compara os dois no mesmo conjunto de teste e caso a caso, que é o que a comparação
    de médias não faz: duas tabelas de recall podem diferir porque os modelos acertam
    pessoas diferentes ou porque um acerta um subconjunto do outro, e a diferença entre
    esses dois cenários muda o que se pode afirmar.

    Cada modelo entra com o **seu** ponto de operação, o limiar que `selecionar_limiar`
    calibrou para ele. Igualar o valor numérico do limiar entre modelos não significa
    nada, porque a escala de probabilidade de um não é a do outro.

    O p-valor sai do teste binomial exato sobre as discordâncias, sem aproximação por
    qui-quadrado: com poucas dezenas de casos discordantes, a aproximação assintótica é
    justamente onde ela vale menos.

    Args:
        y_te: rótulos do teste.
        resultado_a, resultado_b: saídas de `avaliar`, cada uma com `proba_teste` e
            `limiar`.

    Returns:
        Dicionário com `tabela` (a 2x2 de acertos), `p_valor`, as quatro células da
        2x2 como contagens soltas, o total de positivos do teste e, entre esses
        positivos, quantos cada modelo detecta que o outro perde.
    """
    y_te = np.asarray(y_te)
    pred_a = (np.asarray(resultado_a["proba_teste"]) >= resultado_a["limiar"]).astype(int)
    pred_b = (np.asarray(resultado_b["proba_teste"]) >= resultado_b["limiar"]).astype(int)

    acerta_a, acerta_b = pred_a == y_te, pred_b == y_te
    ambos_acertam = int(np.sum(acerta_a & acerta_b))
    ambos_erram = int(np.sum(~acerta_a & ~acerta_b))
    so_a = int(np.sum(acerta_a & ~acerta_b))
    so_b = int(np.sum(~acerta_a & acerta_b))

    rotulo_a = _rotulo_resultado(resultado_a)
    rotulo_b = _rotulo_resultado(resultado_b)
    tabela = pd.DataFrame(
        [[ambos_acertam, so_a],
         [so_b, ambos_erram]],
        index=[f"{rotulo_a} acerta", f"{rotulo_a} erra"],
        columns=[f"{rotulo_b} acerta", f"{rotulo_b} erra"],
    )

    discordantes = so_a + so_b
    p_valor = (float(binomtest(min(so_a, so_b), n=discordantes, p=0.5).pvalue)
               if discordantes else 1.0)

    positivo = y_te == 1
    return {
        "modelo_a": rotulo_a,
        "modelo_b": rotulo_b,
        "tabela": tabela,
        "n": int(len(y_te)),
        "positivos_teste": int(np.sum(positivo)),
        "concordam": int(np.sum(acerta_a == acerta_b)),
        "ambos_acertam": ambos_acertam,
        "ambos_erram": ambos_erram,
        "discordantes": discordantes,
        "so_a_acerta": so_a,
        "so_b_acerta": so_b,
        # Entre os positivos reais: quem cada um detecta que o outro deixa passar. É a
        # leitura que interessa na triagem, onde o custo do falso negativo é o caso
        # perdido, não o erro em abstrato.
        "positivos_so_a_detecta": int(np.sum(positivo & (pred_a == 1) & (pred_b == 0))),
        "positivos_so_b_detecta": int(np.sum(positivo & (pred_b == 1) & (pred_a == 0))),
        "p_valor": p_valor,
    }


def _rotulo_resultado(resultado: dict) -> str:
    """Nome do modelo com o conjunto, quando ele depende de um."""
    conjunto = resultado.get("conjunto", "")
    return (f"{resultado['modelo']} · {conjunto}" if conjunto and conjunto != "—"
            else resultado["modelo"])


def tabela_mcnemar(teste: dict) -> pd.DataFrame:
    """A saída do McNemar em uma linha, pronta para virar artefato.

    Leva as quatro células da 2x2 e o total de positivos como colunas próprias, para
    que quem lê o CSV não precise somar nada (D-055).
    """
    return pd.DataFrame([{
        "modelo_a": teste["modelo_a"],
        "modelo_b": teste["modelo_b"],
        "n": teste["n"],
        "concordam": teste["concordam"],
        "so_a_acerta": teste["so_a_acerta"],
        "so_b_acerta": teste["so_b_acerta"],
        "positivos_so_a_detecta": teste["positivos_so_a_detecta"],
        "positivos_so_b_detecta": teste["positivos_so_b_detecta"],
        "p_valor": teste["p_valor"],
        "ambos_acertam": teste["ambos_acertam"],
        "ambos_erram": teste["ambos_erram"],
        "discordancias": teste["discordantes"],
        "positivos_teste": teste["positivos_teste"],
    }])


def leitura_do_mcnemar(teste: dict, alfa: float = 0.05) -> str:
    """Escreve o resultado do McNemar, com o veredito derivado do p-valor.

    Duas frases que dizem coisas diferentes e precisam andar juntas. A primeira é o
    acerto global: se a diferença entre os modelos sobrevive ao teste sobre as
    discordâncias. A segunda é o recorte dos positivos, que é o que a triagem compra,
    e ela pode ser assimétrica mesmo quando a primeira dá empate.

    Nenhum adjetivo fica escrito à mão: significância sai da comparação com `alfa`, e
    a assimetria dos positivos sai da contagem. Ver D-043 e o defeito que D-045
    registra.
    """
    significativo = teste["p_valor"] < alfa
    veredito = ("é significativa" if significativo
                else "não é significativa")

    so_a, so_b = teste["positivos_so_a_detecta"], teste["positivos_so_b_detecta"]
    if so_a and not so_b:
        leitura = (f"{teste['modelo_a']} detecta um superconjunto estrito dos "
                   f"positivos que {teste['modelo_b']} pega: {so_a} a mais e nenhum "
                   f"no sentido inverso")
    elif so_b and not so_a:
        leitura = (f"{teste['modelo_b']} detecta um superconjunto estrito dos "
                   f"positivos que {teste['modelo_a']} pega: {so_b} a mais e nenhum "
                   f"no sentido inverso")
    elif so_a or so_b:
        leitura = (f"cada um detecta positivos que o outro perde, {so_a} contra "
                   f"{so_b}, então nenhum domina o outro caso a caso")
    else:
        leitura = "os dois detectam exatamente os mesmos positivos"

    return (
        f"Entre {teste['n']} casos do teste, os dois modelos discordam em "
        f"{teste['discordantes']}: {teste['so_a_acerta']} em que só "
        f"{teste['modelo_a']} acerta e {teste['so_b_acerta']} em que só "
        f"{teste['modelo_b']} acerta. Pelo teste binomial exato, p = "
        f"{teste['p_valor']:.4f}, e a diferença de acerto global {veredito} ao nível "
        f"de {alfa:.2f}. No recorte que interessa à triagem, {leitura}."
    )


def nota_corte_unico(tabela: pd.DataFrame) -> str:
    """Nota de rodapé sobre as linhas cuja saída assume no máximo dois valores.

    Devolve string vazia quando a tabela não tem nenhuma. A nota existe porque a
    tabela comparativa põe lado a lado objetos diferentes: sem ela, quem lê compara
    a AUC-ROC 0,656 da regra ADA com a 0,781 do modelo como se medissem a mesma
    coisa.
    """
    if "corte_unico" not in tabela.columns or not tabela["corte_unico"].any():
        return ""
    nomes = ", ".join(sorted(tabela.loc[tabela["corte_unico"], "modelo"].unique()))
    return (
        f"Nota: {nomes} são classificadores de corte único — a saída assume dois "
        f"valores, não um escore. AUC-ROC, PR-AUC, Brier e a curva de calibração "
        f"dependem de percorrer limiares e, para essas linhas, ficam degradadas por "
        f"construção, não por desempenho. As métricas comparáveis entre todas as "
        f"linhas são Recall, Especificidade e NNS."
    )


def exportar_latex(tabela: pd.DataFrame, nome_arquivo: str,
                   path_reports: Path | None = None, nota: str | None = None,
                   **kwargs) -> Path:
    """Exporta a tabela comparativa para `.tex` em `reports/tabelas/`.

    Args:
        nota: rodapé anexado depois do `tabular`. Omitido, a nota de corte único sai
            sozinha quando a tabela tem linhas que a exijam: advertência que depende
            do conteúdo não deveria depender de alguém lembrar de pedir. Passe
            `nota=""` para suprimir.
    """
    path_reports = cfg.PATH_TABELAS if path_reports is None else Path(path_reports)
    path_reports.mkdir(parents=True, exist_ok=True)
    destino = path_reports / nome_arquivo

    opcoes = {"index": False, "float_format": "%.3f"}
    opcoes.update(kwargs)
    conteudo = tabela.to_latex(**opcoes)

    if nota is None:
        nota = nota_corte_unico(tabela)
    if nota:
        conteudo += f"\n\\begin{{flushleft}}\n\\footnotesize {nota}\n\\end{{flushleft}}\n"

    destino.write_text(conteudo, encoding="utf-8")
    return destino
