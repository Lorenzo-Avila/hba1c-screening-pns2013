"""Estilo, rótulos e gravação das figuras do texto.

Único lugar com `savefig`. As funções de figura de `colinearidade`, `evaluate` e
`explicabilidade` chamam daqui. A largura já é a da página, para o LaTeX incluir
na mesma medida e a escala ficar 1. O heatmap entra no texto com 0,85 da linha,
então nasce com 13,6 cm. As demais nascem com 16 cm, a linha inteira.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from . import config as cfg

# 1 pol = 2,54 cm. A figura guarda a largura em polegadas porque é a unidade do
# figsize, e o PNG a 300 dpi preserva esse tamanho físico.
_CM = 2.54
LARGURA_INTEIRA = 16 / _CM
LARGURA_PARCIAL = 13.6 / _CM
DPI = 300

# Nomes por extenso, os mesmos do texto. Indicadora one-hot não mora aqui: sai
# do prefixo mais o dicionário de código de `config`.
ROTULOS = {
    "idade": "Idade",
    "sexo": "Sexo",
    "sexo_fem": "Sexo feminino",
    "imc": "Índice de massa corporal",
    "circ_cintura": "Circunferência da cintura",
    "rcest": "Relação cintura-estatura",
    "pas": "Pressão arterial sistólica",
    "pad": "Pressão arterial diastólica",
    "has_autorrelato": "Hipertensão autorreferida",
    "has_combinada": "Hipertensão combinada",
    "colesterol_alto": "Colesterol alto autorreferido",
    "colesterol_sim": "Colesterol alto",
    "multimorbidade": "Multimorbidade Metabólica",
    "raca_cor": "Raça/cor",
    "escolaridade": "Escolaridade",
    "plano_saude": "Plano de saúde",
    "plano_sim": "Plano de saúde",
    "regiao": "Região",
    "saude_autoaval": "Saúde autoavaliada",
    "LogReg": "Regressão Logística",
    "A_atomico": "A, atômico",
    "B_composto": "B, composto",
    "C_completo": "C, completo",
}

_CODIGOS = {
    "regiao": cfg.COD_REGIAO,
    "raca_cor": cfg.COD_RACA,
    "sexo": cfg.COD_SEXO,
    "has_autorrelato": cfg.COD_HAS,
    "colesterol_alto": cfg.COD_COLEST,
}


def configurar() -> None:
    """Fonte serifada e tamanhos mínimos, aplicados antes de desenhar."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.labelcolor": cfg.INK,
        "xtick.color": cfg.INK,
        "ytick.color": cfg.INK,
        "text.color": cfg.INK,
        "axes.edgecolor": cfg.INK,
        "axes.unicode_minus": False,
    })


def numero(valor, casas: int | None = None) -> str:
    """Vírgula decimal e ponto de milhar, sem `locale`.

    `casas` fixa as decimais. Sem ele, a casa acompanha a magnitude, para o tick
    de eixo não sair com cauda de float.
    """
    valor = float(valor)
    if math.isnan(valor):
        return ""
    if math.isinf(valor):
        return "∞" if valor > 0 else "-∞"
    sinal = "-" if valor < 0 else ""
    valor = abs(valor)
    if casas is None:
        if valor >= 100:
            casas = 0
        elif valor >= 10:
            casas = 1
        elif valor >= 1:
            casas = 2
        else:
            casas = 3
    texto = f"{valor:.{casas}f}"
    inteiro, _, frac = texto.partition(".")
    grupos = []
    while inteiro:
        grupos.append(inteiro[-3:])
        inteiro = inteiro[:-3]
    inteiro_fmt = ".".join(reversed(grupos))
    if casas == 0:
        return sinal + inteiro_fmt
    return f"{sinal}{inteiro_fmt},{frac}"


def formatador(casas: int | None = None):
    """Formatter de eixo que usa `numero`."""
    return FuncFormatter(lambda valor, _pos: numero(valor, casas))


def rotulo(nome: str) -> str:
    """Nome por extenso. `regiao_2.0` vira `Região: Nordeste`."""
    if nome in ROTULOS:
        return ROTULOS[nome]
    for variavel, codigos in _CODIGOS.items():
        prefixo = f"{variavel}_"
        if not nome.startswith(prefixo):
            continue
        try:
            codigo = int(float(nome[len(prefixo):]))
        except ValueError:
            break
        if codigo in codigos:
            return f"{ROTULOS[variavel]}: {codigos[codigo]}"
    return nome


def subplots(largura: float, altura: float, *args, **kwargs):
    """Figura na largura pedida, já com o estilo aplicado."""
    configurar()
    return plt.subplots(*args, figsize=(largura, altura), **kwargs)


def eixo(ax, xlabel: str, ylabel: str = "", grade: str = "both") -> None:
    """Rótulos e grade. O título fica na legenda do LaTeX, não no eixo."""
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if grade:
        ax.grid(color=cfg.GRID, linewidth=0.8, axis=grade)
    ax.set_axisbelow(True)


def salvar(fig, destino, nota: str | None = None) -> Path:
    """Grava o PNG a 300 dpi e devolve o caminho. Único `savefig` do `src`."""
    if nota:
        fig.tight_layout(rect=(0, 0.18, 1, 1))
        fig.text(0.0, 0.01, nota, fontsize=8, color=cfg.INK, va="bottom", ha="left")
    else:
        fig.tight_layout()
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=DPI)
    plt.close(fig)
    return destino
