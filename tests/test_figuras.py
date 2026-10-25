"""Guardas das figuras do texto.

O estilo, o número e o `savefig` moram em `src/figuras.py`. As funções que desenham
não põem título, não gravam arquivo e não usam travessão nem ponto médio. A leitura
é do código, por AST, no mesmo molde das guardas estáticas do painel.
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src import colinearidade as col
from src import evaluate as ev
from src import figuras as figu

RAIZ = Path(__file__).resolve().parent.parent
SRC = RAIZ / "src"
MODULOS_DE_FIGURA = (
    SRC / "colinearidade.py",
    SRC / "evaluate.py",
    SRC / "explicabilidade.py",
)
FUNCOES_DE_FIGURA = {
    "heatmap_pearson",
    "_preparar_eixo",
    "figura_matriz_confusao",
    "figura_roc",
    "figura_precision_recall",
    "figura_calibracao",
    "_eixo_figura",
    "figura_beeswarm",
    "figura_contraste_de_regimes",
    "_painel_waterfall",
    "figura_waterfall_par",
}
PROIBIDOS = ("—", "·")


def _arvore(arquivo: Path) -> ast.AST:
    return ast.parse(arquivo.read_text(encoding="utf-8"), filename=str(arquivo))


def _funcoes(arquivo: Path):
    for no in _arvore(arquivo).body:
        if isinstance(no, ast.FunctionDef) and no.name in FUNCOES_DE_FIGURA:
            yield no


def _textos(no: ast.AST):
    for filho in ast.walk(no):
        if isinstance(filho, ast.Constant) and isinstance(filho.value, str):
            yield filho


def test_modulos_de_figura_nao_tem_titulo():
    violacoes = []
    for arquivo in MODULOS_DE_FIGURA:
        for no in ast.walk(_arvore(arquivo)):
            if isinstance(no, ast.Attribute) and no.attr in ("set_title", "suptitle"):
                violacoes.append(f"{arquivo.name}:{no.lineno} {no.attr}")
    assert not violacoes, "título dentro da figura:\n" + "\n".join(violacoes)


def test_savefig_so_em_figuras():
    violacoes = []
    for arquivo in SRC.glob("*.py"):
        if arquivo.name == "figuras.py":
            continue
        for no in ast.walk(_arvore(arquivo)):
            if isinstance(no, ast.Attribute) and no.attr == "savefig":
                violacoes.append(f"{arquivo.name}:{no.lineno}")
    assert not violacoes, "savefig fora de src/figuras.py:\n" + "\n".join(violacoes)


def test_texto_de_figura_nao_tem_travessao_nem_ponto_medio():
    violacoes = []
    for no in _textos(_arvore(SRC / "figuras.py")):
        if any(marca in no.value for marca in PROIBIDOS):
            violacoes.append(f"figuras.py:{no.lineno} {no.value!r}")
    for arquivo in MODULOS_DE_FIGURA:
        for funcao in _funcoes(arquivo):
            for no in _textos(funcao):
                if any(marca in no.value for marca in PROIBIDOS):
                    violacoes.append(
                        f"{arquivo.name}:{no.lineno} {funcao.name} {no.value!r}")
    assert not violacoes, "travessão ou ponto médio na figura:\n" + "\n".join(violacoes)


def test_numero_usa_virgula_decimal_e_ponto_de_milhar():
    assert figu.numero(0.78, 3) == "0,780"
    assert figu.numero(1421, 0) == "1.421"
    assert figu.numero(-2.571, 3) == "-2,571"
    assert figu.formatador(2)(0.5, 0) == "0,50"


def test_rotulo_sai_por_extenso_e_a_indicadora_do_codigo():
    assert figu.rotulo("circ_cintura") == "Circunferência da cintura"
    assert figu.rotulo("multimorbidade") == "Multimorbidade Metabólica"
    assert figu.rotulo("LogReg") == "Regressão Logística"
    assert figu.rotulo("B_composto") == "B, composto"
    assert figu.rotulo("regiao_2.0") == "Região: Nordeste"


def test_png_grava_a_300_dpi_na_largura_pedida(tmp_path):
    fig, ax = figu.subplots(figu.LARGURA_PARCIAL, 2)
    ax.plot([0, 1], [0, 1], ls="-")
    ax.plot([0, 1], [1, 0], ls=":")
    destino = figu.salvar(fig, tmp_path / "amostra.png")
    imagem = Image.open(destino)
    dpi = imagem.info["dpi"][0]
    assert dpi == pytest.approx(300, abs=0.01)
    assert imagem.size[0] / dpi == pytest.approx(figu.LARGURA_PARCIAL, abs=0.02)


def test_curvas_e_heatmap_gravam_png_sem_titulo(tmp_path):
    y = np.array([0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.4, 0.7, 0.9])
    roc = ev.figura_roc(y, proba, "LogReg_B_composto", path_figuras=tmp_path)
    assert roc.name == "roc_LogReg_B_composto.png"
    matriz = pd.DataFrame(
        [[1.0, 0.2], [0.2, 1.0]],
        index=["imc", "circ_cintura"], columns=["imc", "circ_cintura"],
    )
    heatmap = col.heatmap_pearson(matriz, "A_atomico", path_figuras=tmp_path)
    assert heatmap.name == "pearson_A_atomico.png"
    imagem = Image.open(heatmap)
    assert imagem.size[0] / imagem.info["dpi"][0] == pytest.approx(
        figu.LARGURA_PARCIAL, abs=0.02)
