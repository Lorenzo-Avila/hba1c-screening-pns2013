"""Formatação e construtores de gráfico do painel.

Funções puras: recebem DataFrame, devolvem DataFrame de exibição ou gráfico Altair.
Nenhuma chama `st`, nenhuma lê arquivo. A única operação numérica é a soma
acumulada do waterfall, que posiciona as barras e não vira número exibido.
"""

import math
import re

import altair as alt
import numpy as np
import pandas as pd

# Formato brasileiro nos eixos do Vega-Lite: vírgula decimal, ponto de milhar.
LOCALE = {
    "number": {
        "decimal": ",",
        "thousands": ".",
        "grouping": [3],
        "currency": ["R$ ", ""],
    }
}

COR_POSITIVA = "#c2410c"
COR_NEGATIVA = "#1d4ed8"
COR_NEUTRA = "#475569"


def _vazio(valor) -> bool:
    return valor is None or (isinstance(valor, float) and math.isnan(valor))


def numero(valor, casas) -> str:
    """Formata no padrão brasileiro com as casas do `.tex`.

    `casas` inteiro fixa as casas decimais, "e3" usa notação científica com três
    casas. Vazio vira string vazia.
    """
    if _vazio(valor):
        return ""
    if casas == "e3":
        return f"{float(valor):.3e}".replace(".", ",")
    texto = f"{float(valor):,.{casas}f}"
    return texto.replace(",", "\0").replace(".", ",").replace("\0", ".")


def observado(valor) -> str:
    """Valor observado com até seis casas e sem zeros à direita."""
    if _vazio(valor):
        return ""
    texto = f"{float(valor):.6f}".rstrip("0").rstrip(".")
    return texto.replace(".", ",")


def tabela(df: pd.DataFrame, casas: dict, nomes: dict | None = None,
           colunas: list | None = None) -> pd.DataFrame:
    """DataFrame de exibição: números como texto formatado, colunas renomeadas.

    Booleanos viram "sim" e "não". Colunas fora de `casas` passam como texto.
    """
    exibida = df[colunas] if colunas else df
    saida = pd.DataFrame(index=exibida.index)
    for coluna in exibida.columns:
        serie = exibida[coluna]
        if coluna in casas:
            saida[coluna] = [numero(v, casas[coluna]) for v in serie]
        elif serie.dtype == bool:
            saida[coluna] = serie.map({True: "sim", False: "não"})
        else:
            saida[coluna] = serie.astype("string").fillna("")
    saida = saida.rename(columns=nomes or {})
    return saida.reset_index(drop=True)


def _frases(texto: str) -> list:
    """Quebra um parágrafo em frases, uma por linha de subtítulo."""
    return re.split(r"(?<=\.) ", texto)


def ranking(df: pd.DataFrame, titulo: str, casas: int) -> alt.Chart:
    """Barras horizontais de `shap_medio_abs` por atributo, na ordem de `posicao`."""
    # Só as colunas desenhadas vão ao navegador.
    dados = df.sort_values("posicao")[["atributo", "shap_medio_abs"]]
    dados = dados.assign(rotulo=[numero(v, casas) for v in dados["shap_medio_abs"]])
    ordem = list(dados["atributo"])
    base = alt.Chart(dados).encode(
        y=alt.Y("atributo:N", sort=ordem, title=None,
                     axis=alt.Axis(labelOverlap=False)),
        x=alt.X("shap_medio_abs:Q", title="|SHAP| médio"),
    )
    barras = base.mark_bar(color=COR_NEUTRA)
    rotulos = base.mark_text(align="left", dx=4, fontSize=11).encode(text="rotulo:N")
    return (
        (barras + rotulos)
        .properties(title=titulo, height=max(160, 24 * len(dados)))
        .configure(locale=LOCALE)
    )


def barras_vif(df: pd.DataFrame, limiar: float, rotulo_limiar: str,
               casas: int) -> alt.Chart:
    """VIF por coluna da matriz de design, com o limiar marcado."""
    dados = df[["variavel", "vif", "acima_do_limiar"]].assign(
        rotulo=[numero(v, casas) for v in df["vif"]],
        situacao=df["acima_do_limiar"].map({True: "acima do limiar",
                                            False: "abaixo do limiar"}),
    )
    ordem = list(dados.sort_values("vif", ascending=False)["variavel"])
    base = alt.Chart(dados).encode(
        y=alt.Y("variavel:N", sort=ordem, title=None,
                     axis=alt.Axis(labelOverlap=False, labelLimit=200)),
        x=alt.X("vif:Q", title="VIF"),
    )
    barras = base.mark_bar().encode(
        color=alt.Color(
            "situacao:N", title=None,
            scale=alt.Scale(domain=["abaixo do limiar", "acima do limiar"],
                            range=[COR_NEUTRA, COR_POSITIVA]),
        )
    )
    rotulos = base.mark_text(align="left", dx=4, fontSize=11).encode(text="rotulo:N")
    linha = pd.DataFrame({"limiar": [limiar], "texto": [rotulo_limiar]})
    regra = alt.Chart(linha).mark_rule(strokeDash=[6, 4], color="black").encode(
        x="limiar:Q"
    )
    legenda = alt.Chart(linha).mark_text(
        align="left", dx=4, dy=-6, baseline="bottom", fontSize=11
    ).encode(x="limiar:Q", y=alt.value(0), text="texto:N")
    return (
        (barras + rotulos + regra + legenda)
        .properties(height=max(160, 22 * len(dados)))
        .configure(locale=LOCALE)
    )


def waterfall(caso: pd.DataFrame, ressalva: str, casas: int) -> alt.Chart:
    """Waterfall de um caso local na escala do escore de ordenação.

    A barra parte de `valor_base`, soma as contribuições na ordem do artefato e
    termina em `escore`. O ponto final vem da coluna `escore`, não da soma. A linha
    de referência fica no `limiar`, com o rótulo "limiar calibrado". A ressalva de
    D-053 vai no subtítulo, dentro da figura.
    """
    valor_base = caso["valor_base"].iloc[0]
    escore = caso["escore"].iloc[0]
    limiar = caso["limiar"].iloc[0]

    acumulado = np.cumsum([valor_base, *caso["contribuicao"]])
    barras = pd.DataFrame({
        "rotulo": [f"{a} (observado {observado(v)})"
                   for a, v in zip(caso["atributo"], caso["valor_observado"])],
        "inicio": acumulado[:-1],
        "fim": acumulado[1:],
        "contribuicao": [numero(c, casas) for c in caso["contribuicao"]],
        "sinal": ["positiva" if c > 0 else "negativa" for c in caso["contribuicao"]],
    })
    extremos = pd.DataFrame({
        "rotulo": [f"valor de base ({numero(valor_base, casas)})",
                   f"escore de ordenação ({numero(escore, casas)})"],
        "valor": [valor_base, escore],
    })
    ordem = [extremos["rotulo"][0], *barras["rotulo"], extremos["rotulo"][1]]
    eixo_y = alt.Y("rotulo:N", sort=ordem, title=None,
                   axis=alt.Axis(labelLimit=320, labelOverlap=False))
    eixo_x = alt.X("inicio:Q", title="escore de ordenação",
                   scale=alt.Scale(zero=False))

    camada_barras = alt.Chart(barras).mark_bar(height=16).encode(
        y=eixo_y, x=eixo_x, x2="fim:Q",
        color=alt.Color("sinal:N", title="contribuição",
                        scale=alt.Scale(domain=["positiva", "negativa"],
                                        range=[COR_POSITIVA, COR_NEGATIVA])),
    )
    texto_pos = alt.Chart(barras[barras["sinal"] == "positiva"]).mark_text(
        align="left", dx=4, fontSize=11
    ).encode(y=eixo_y, x="fim:Q", text="contribuicao:N")
    texto_neg = alt.Chart(barras[barras["sinal"] == "negativa"]).mark_text(
        align="right", dx=-4, fontSize=11
    ).encode(y=eixo_y, x="fim:Q", text="contribuicao:N")
    pontos = alt.Chart(extremos).mark_point(
        filled=True, size=90, color="black", shape="diamond"
    ).encode(y=eixo_y, x="valor:Q")

    linha = pd.DataFrame({"limiar": [limiar], "texto": ["limiar calibrado"]})
    regra = alt.Chart(linha).mark_rule(strokeDash=[6, 4], color="black").encode(
        x="limiar:Q"
    )
    rotulo_regra = alt.Chart(linha).mark_text(
        align="left", dx=4, baseline="top", fontSize=11, fontWeight="bold"
    ).encode(x="limiar:Q", y=alt.value(2), text="texto:N")

    return (
        (camada_barras + texto_pos + texto_neg + pontos + regra + rotulo_regra)
        .properties(
            title=alt.TitleParams(
                "Decomposição do escore de ordenação",
                subtitle=_frases(ressalva),
                anchor="start",
                subtitleFontSize=13,
                subtitleColor="#111827",
            ),
            height=34 * len(ordem),
        )
        .configure(locale=LOCALE)
    )
