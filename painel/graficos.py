"""Formatação, gráficos Altair e o diagrama do pipeline.

Funções puras: recebem DataFrame e devolvem DataFrame de exibição, gráfico ou texto
DOT. Nenhuma chama `st` nem lê arquivo. A única operação numérica é a soma acumulada
do waterfall, que posiciona as barras e não vira número exibido. Cor pelo sinal,
gradiente e comprimento são codificação visual de coluna existente.
"""

import math
import re

import altair as alt
import numpy as np
import pandas as pd

import estilo

# Formato brasileiro nos eixos do Vega-Lite: vírgula decimal, ponto de milhar.
LOCALE = {
    "number": {
        "decimal": ",",
        "thousands": ".",
        "grouping": [3],
        "currency": ["R$ ", ""],
    }
}


def _vazio(valor) -> bool:
    return valor is None or (isinstance(valor, float) and math.isnan(valor))


def numero(valor, casas) -> str:
    """Formata no padrão brasileiro com as casas do `.tex`.

    `casas` inteiro fixa as casas decimais, "e3" usa notação científica com três
    casas e "%3" escreve a proporção de três casas como porcentagem. Vazio vira
    string vazia, infinito vira "∞".
    """
    if _vazio(valor):
        return ""
    if isinstance(valor, float) and math.isinf(valor):
        return "∞"
    if casas == "e3":
        return f"{float(valor):.3e}".replace(".", ",")
    if casas == "%3":
        return f"{float(valor):.1%}".replace(".", ",")
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


def _configurar(grafico):
    """Fonte, locale e eixos comuns a todos os gráficos."""
    return (
        grafico.configure(locale=LOCALE, font=estilo.FONTE, background=estilo.CARTAO)
        .configure_axis(labelColor=estilo.TEXTO_SECUNDARIO,
                        titleColor=estilo.TEXTO_SECUNDARIO,
                        gridColor=estilo.BORDA, domainColor=estilo.BORDA)
        .configure_view(stroke=None)
        .configure_title(color=estilo.TEXTO, anchor="start")
        .configure_legend(labelColor=estilo.TEXTO, titleColor=estilo.TEXTO_SECUNDARIO)
    )


def _ranking(df: pd.DataFrame, titulo: str, casas: int, largura: int):
    """Barras horizontais de `shap_medio_abs` por atributo, na ordem de `posicao`."""
    # Só as colunas desenhadas vão ao navegador.
    dados = df.sort_values("posicao")[["atributo", "shap_medio_abs"]]
    dados = dados.assign(rotulo=[numero(v, casas) for v in dados["shap_medio_abs"]])
    ordem = list(dados["atributo"])
    base = alt.Chart(dados).encode(
        y=alt.Y("atributo:N", sort=ordem, title=None,
                axis=alt.Axis(labelOverlap=False, labelFontSize=12)),
        x=alt.X("shap_medio_abs:Q", title="|SHAP| médio"),
    )
    barras = base.mark_bar(color=estilo.AZUL, cornerRadiusEnd=3)
    rotulos = base.mark_text(align="left", dx=4, fontSize=11,
                             color=estilo.TEXTO).encode(text="rotulo:N")
    return (barras + rotulos).properties(title=titulo, width=largura,
                                         height=26 * len(dados))


def rankings_lado_a_lado(recortes: list, casas: int, largura: int = 300) -> alt.Chart:
    """Rankings de vários regimes num só gráfico, lado a lado.

    Cada recorte é (título, DataFrame). Um gráfico só evita que o Streamlit reduza
    cada metade como imagem dentro de colunas estreitas.
    """
    return _configurar(alt.hconcat(
        *[_ranking(df, titulo, casas, largura) for titulo, df in recortes], spacing=70
    ).resolve_scale(x="independent", y="independent"))


def fatores(df: pd.DataFrame, rotulos: dict, casas: int, titulo_eixo: str) -> alt.Chart:
    """Barras horizontais em gradiente de azul, com o nome clínico de cada atributo."""
    dados = df.sort_values("posicao")[["atributo", "shap_medio_abs"]]
    dados = dados.assign(
        nome=[rotulos.get(a, a) for a in dados["atributo"]],
        rotulo=[numero(v, casas) for v in dados["shap_medio_abs"]],
    )
    ordem = list(dados["nome"])
    base = alt.Chart(dados).encode(
        y=alt.Y("nome:N", sort=ordem, title=None,
                axis=alt.Axis(labelOverlap=False, labelLimit=260, labelFontSize=13)),
        x=alt.X("shap_medio_abs:Q", title=titulo_eixo),
    )
    barras = base.mark_bar(cornerRadiusEnd=4, height=22).encode(
        color=alt.Color("shap_medio_abs:Q", legend=None,
                        scale=alt.Scale(range=[estilo.AZUL_SUAVE, estilo.AZUL_ESCURO]))
    )
    valores = base.mark_text(align="left", dx=6, fontSize=12,
                             color=estilo.TEXTO).encode(text="rotulo:N")
    return _configurar((barras + valores).properties(height=38 * len(dados)))


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
    barras = base.mark_bar(cornerRadiusEnd=3).encode(
        color=alt.Color(
            "situacao:N", title=None,
            scale=alt.Scale(domain=["abaixo do limiar", "acima do limiar"],
                            range=[estilo.AZUL, estilo.VERMELHO]),
        )
    )
    rotulos = base.mark_text(align="left", dx=4, fontSize=11,
                             color=estilo.TEXTO).encode(text="rotulo:N")
    linha = pd.DataFrame({"limiar": [limiar], "texto": [rotulo_limiar]})
    regra = alt.Chart(linha).mark_rule(strokeDash=[6, 4], color=estilo.TEXTO).encode(
        x="limiar:Q"
    )
    legenda = alt.Chart(linha).mark_text(
        align="left", dx=4, dy=-6, baseline="bottom", fontSize=11, color=estilo.TEXTO
    ).encode(x="limiar:Q", y=alt.value(0), text="texto:N")
    return _configurar(
        (barras + rotulos + regra + legenda).properties(height=max(160, 22 * len(dados)))
    )


def desbalanceamento(itens: list) -> alt.Chart:
    """Barra única empilhada. Cada item é (rótulo com n, n)."""
    dados = pd.DataFrame(itens, columns=["grupo", "n"])
    dados["ordem"] = range(len(dados))
    # Categoria única no eixo y: a espessura da barra vem da banda, não da marca.
    dados["linha"] = "amostra"
    barras = alt.Chart(dados).mark_bar(cornerRadius=4).encode(
        x=alt.X("n:Q", stack="zero", axis=None),
        y=alt.Y("linha:N", axis=None, scale=alt.Scale(paddingInner=0, paddingOuter=0)),
        order="ordem:O",
        color=alt.Color("grupo:N", title=None,
                        scale=alt.Scale(domain=list(dados["grupo"]),
                                        range=[estilo.AZUL, estilo.AZUL_SUAVE]),
                        legend=alt.Legend(orient="top", labelFontSize=13)),
    )
    return _configurar(barras.properties(height=110))


def barras_encaminhados(df: pd.DataFrame, casas) -> alt.Chart:
    """Percentual de encaminhados por regime, na ordem do artefato."""
    dados = df[["regime", "taxa_encaminhamento"]].assign(
        rotulo=[numero(v, casas) for v in df["taxa_encaminhamento"]],
    )
    base = alt.Chart(dados).encode(
        y=alt.Y("regime:N", sort=list(dados["regime"]), title=None,
                axis=alt.Axis(labelLimit=440, labelOverlap=False, labelFontSize=12)),
        x=alt.X("taxa_encaminhamento:Q", title="encaminhados a exame",
                axis=alt.Axis(format="%")),
    )
    barras = base.mark_bar(height=26, cornerRadiusEnd=4).encode(
        color=alt.Color("regime:N", legend=None,
                        scale=alt.Scale(domain=list(dados["regime"]),
                                        range=[estilo.CINZA, estilo.AZUL_CLARO,
                                               estilo.AZUL]))
    )
    valores = base.mark_text(align="left", dx=6, fontSize=13, fontWeight="bold",
                             color=estilo.TEXTO).encode(text="rotulo:N")
    return _configurar((barras + valores).properties(height=150))


def _ponto(x: float, y: float, texto: str):
    ponto = alt.Chart(pd.DataFrame({"x": [x], "y": [y], "t": [texto]}))
    marca = ponto.mark_point(filled=True, size=150, color=estilo.AZUL,
                             stroke=estilo.CARTAO, strokeWidth=2, opacity=1
                             ).encode(x="x:Q", y="y:Q")
    rotulo = ponto.mark_text(align="left", dx=10, dy=-10, fontWeight="bold",
                             color=estilo.AZUL).encode(x="x:Q", y="y:Q", text="t:N")
    return marca + rotulo


def _linha_piso(piso: float, rotulo: str):
    base = alt.Chart(pd.DataFrame({"y": [piso], "t": [rotulo]}))
    regra = base.mark_rule(strokeDash=[5, 4], color=estilo.TEXTO_SECUNDARIO).encode(y="y:Q")
    texto = base.mark_text(align="right", dy=-7, x="width", fontSize=11,
                           color=estilo.TEXTO_SECUNDARIO).encode(y="y:Q", text="t:N")
    return regra + texto


def curva_roc(curva: pd.DataFrame, fpr: float, recall: float, titulo: str,
              rotulo_ponto: str) -> alt.Chart:
    """Curva ROC da célula, diagonal do acaso e ponto de operação."""
    linha = alt.Chart(curva[["fpr", "tpr"]]).mark_line(
        color=estilo.ROXO, strokeWidth=2.5, interpolate="step-after"
    ).encode(x=alt.X("fpr:Q", title="1 − especificidade"),
             y=alt.Y("tpr:Q", title="recall (sensibilidade)"))
    acaso = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
        strokeDash=[5, 4], color=estilo.TEXTO_SECUNDARIO
    ).encode(x="x:Q", y="y:Q")
    return _configurar(
        (acaso + linha + _ponto(fpr, recall, rotulo_ponto))
        .properties(title=titulo, height=360)
    )


def curva_pr(curva: pd.DataFrame, recall: float, precisao: float, piso: float,
             rotulo_piso: str, titulo: str, rotulo_ponto: str) -> alt.Chart:
    """Curva precisão × recall, linha do piso e ponto de operação."""
    linha = alt.Chart(curva[["recall", "precisao"]]).mark_line(
        color=estilo.ROXO, strokeWidth=2.5, interpolate="step-after"
    ).encode(x=alt.X("recall:Q", title="recall (sensibilidade)"),
             y=alt.Y("precisao:Q", title="precisão"))
    return _configurar(
        (_linha_piso(piso, rotulo_piso) + linha + _ponto(recall, precisao, rotulo_ponto))
        .properties(title=titulo, height=360)
    )


def decis(df: pd.DataFrame) -> alt.Chart:
    """Frequência observada contra escore médio por decil, com a diagonal."""
    dados = df[["decil", "prob_media", "freq_observada"]]
    diagonal = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
        strokeDash=[5, 4], color=estilo.TEXTO_SECUNDARIO
    ).encode(x="x:Q", y="y:Q")
    base = alt.Chart(dados).encode(
        x=alt.X("prob_media:Q", title="escore de ordenação médio do decil",
                scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("freq_observada:Q", title="frequência observada de HbA1c ≥ 6,5%",
                scale=alt.Scale(domain=[0, 1])),
    )
    linha = base.mark_line(color=estilo.AZUL, strokeWidth=2)
    pontos = base.mark_point(filled=True, size=70, color=estilo.AZUL)
    return _configurar((diagonal + linha + pontos).properties(height=380))


def busca(df: pd.DataFrame, piso: float, rotulo_piso: str) -> alt.Chart:
    """PR-AUC média de cada candidato por posição, vencedor destacado, piso marcado."""
    dados = df[["conjunto", "posicao", "pr_auc_media"]]
    vencedores = dados.sort_values("posicao").groupby("conjunto").head(1)
    escala = alt.Scale(domain=list(dados["conjunto"].unique()),
                       range=[estilo.AZUL, estilo.ROXO])
    pontos = alt.Chart(dados).mark_point(filled=True, size=40, opacity=.7).encode(
        x=alt.X("posicao:Q", title="posição no ranking da busca"),
        y=alt.Y("pr_auc_media:Q", title="PR-AUC média entre folds",
                scale=alt.Scale(zero=True)),
        color=alt.Color("conjunto:N", title="conjunto", scale=escala),
    )
    destaque = alt.Chart(vencedores).mark_point(
        shape="diamond", size=260, filled=True, stroke=estilo.TEXTO, strokeWidth=1.5,
        opacity=1
    ).encode(x="posicao:Q", y="pr_auc_media:Q",
             color=alt.Color("conjunto:N", legend=None, scale=escala))
    return _configurar(
        (_linha_piso(piso, rotulo_piso) + pontos + destaque).properties(height=320)
    )


def waterfall(caso: pd.DataFrame, ressalva: str, casas: int,
              rotulos: dict | None = None) -> alt.Chart:
    """Waterfall de um caso local na escala do escore de ordenação.

    A barra parte de `valor_base`, soma as contribuições na ordem do artefato e
    termina em `escore`. O ponto final vem da coluna `escore`, não da soma. A linha
    de referência fica no `limiar`, com o rótulo "limiar calibrado". A ressalva vai
    no subtítulo, dentro da figura. Dois tons de azul separam o sinal.
    """
    rotulos = rotulos or {}
    valor_base = caso["valor_base"].iloc[0]
    escore = caso["escore"].iloc[0]
    limiar = caso["limiar"].iloc[0]

    acumulado = np.cumsum([valor_base, *caso["contribuicao"]])
    barras = pd.DataFrame({
        "rotulo": [f"{rotulos.get(a, a)} (observado {observado(v)})"
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
                   axis=alt.Axis(labelLimit=340, labelOverlap=False, labelFontSize=12))
    eixo_x = alt.X("inicio:Q", title="escore de ordenação",
                   scale=alt.Scale(zero=False))

    camada_barras = alt.Chart(barras).mark_bar(height=18, cornerRadius=2).encode(
        y=eixo_y, x=eixo_x, x2="fim:Q",
        color=alt.Color("sinal:N", title="contribuição",
                        scale=alt.Scale(domain=["positiva", "negativa"],
                                        range=[estilo.AZUL_ESCURO, estilo.AZUL_CLARO]),
                        legend=alt.Legend(orient="bottom")),
    )
    texto_pos = alt.Chart(barras[barras["sinal"] == "positiva"]).mark_text(
        align="left", dx=4, fontSize=11, color=estilo.TEXTO
    ).encode(y=eixo_y, x="fim:Q", text="contribuicao:N")
    texto_neg = alt.Chart(barras[barras["sinal"] == "negativa"]).mark_text(
        align="right", dx=-4, fontSize=11, color=estilo.TEXTO
    ).encode(y=eixo_y, x="fim:Q", text="contribuicao:N")
    pontos = alt.Chart(extremos).mark_point(
        filled=True, size=90, color=estilo.TEXTO, shape="diamond", opacity=1
    ).encode(y=eixo_y, x="valor:Q")

    linha = pd.DataFrame({"limiar": [limiar], "texto": ["limiar calibrado"]})
    regra = alt.Chart(linha).mark_rule(strokeDash=[6, 4], color=estilo.TEXTO).encode(
        x="limiar:Q"
    )
    rotulo_regra = alt.Chart(linha).mark_text(
        align="left", dx=4, baseline="top", fontSize=11, fontWeight="bold",
        color=estilo.TEXTO
    ).encode(x="limiar:Q", y=alt.value(2), text="texto:N")

    return _configurar(
        (camada_barras + texto_pos + texto_neg + pontos + regra + rotulo_regra)
        .properties(
            title=alt.TitleParams(
                "Decomposição do escore de ordenação",
                subtitle=_frases(ressalva),
                subtitleFontSize=13,
                subtitleColor=estilo.TEXTO,
            ),
            height=34 * len(ordem),
        )
    )


def pipeline_dot(etapas: list, por_linha: int = 5) -> str:
    """Diagrama do pipeline em DOT, em linhas de `por_linha` etapas.

    Cada etapa é (rótulo, destaque). Etapas com destaque marcam o que se ajusta só
    no treino.
    """
    nos = []
    for i, (rotulo, destaque) in enumerate(etapas):
        cor = estilo.AZUL if destaque else estilo.BORDA
        fundo = estilo.AZUL_FUNDO if destaque else estilo.CARTAO
        # Quebra de linha do rótulo vira o escape de nova linha do DOT.
        texto = rotulo.replace('"', "'").replace("\n", "\\n")
        nos.append(f'  n{i} [label="{texto}", color="{cor}", fillcolor="{fundo}"];')
    linhas = [list(range(i, min(i + por_linha, len(etapas))))
              for i in range(0, len(etapas), por_linha)]
    mesmas = "\n".join("  { rank=same; " + " ".join(f"n{i}" for i in linha) + " }"
                       for linha in linhas)
    arestas = []
    for linha in linhas:
        arestas.append("  " + " -> ".join(f"n{i}" for i in linha) + ";")
    for anterior, seguinte in zip(linhas, linhas[1:]):
        # A aresta invisível alinha o começo de cada linha sob o da anterior; a
        # visível liga o fim de uma ao começo da outra sem mexer no alinhamento.
        arestas.append(f"  n{anterior[0]} -> n{seguinte[0]} [style=invis];")
        arestas.append(f"  n{anterior[-1]} -> n{seguinte[0]} [constraint=false];")
    arestas = "\n".join(arestas)
    return (
        "digraph pipeline {\n"
        '  rankdir=TB; bgcolor="transparent"; nodesep=0.35; ranksep=0.45;\n'
        f'  node [shape=box, style="rounded,filled", fontname="Segoe UI", fontsize=13,'
        f' fontcolor="{estilo.TEXTO}", penwidth=1.6, margin="0.22,0.14", width=2.3];\n'
        f'  edge [color="{estilo.TEXTO_SECUNDARIO}", arrowsize=0.8];\n'
        + "\n".join(nos) + f"\n{mesmas}\n{arestas}\n}}"
    )
