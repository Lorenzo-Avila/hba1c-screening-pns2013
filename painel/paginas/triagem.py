"""Triagem na atenção primária: regimes, carga de encaminhamento e critério."""

import streamlit as st

import componentes
import dados
import graficos
import selecao
import textos

selecao.abrir_pagina("triagem", "clinica")
g = textos.GLOSSARIO

carga = dados.carregar("carga_operacional")
casas = dados.casas("carga_operacional")

st.subheader(textos.REGIMES_TITULO)
for coluna, (_, regime) in zip(st.columns(len(carga)), carga.iterrows()):
    with coluna:
        componentes.bloco(regime["rotulo"], [
            componentes.metrica(
                "Encaminhados", graficos.numero(regime["encaminhados"], casas["encaminhados"]),
                ajuda=g["encaminhados_n"],
                nota=graficos.numero(regime["taxa_encaminhamento"],
                                     casas["taxa_encaminhamento"]),
                tom="azul"),
            componentes.metrica(
                "Casos detectados",
                graficos.numero(regime["casos_detectados"], casas["casos_detectados"]),
                ajuda=g["casos_detectados"]),
            componentes.metrica("Recall", graficos.numero(regime["recall"], casas["recall"]),
                                ajuda=g["recall"]),
            componentes.metrica("NNS", graficos.numero(regime["nns"], casas["nns"]),
                                ajuda=g["nns"]),
        ])

st.subheader(textos.ENCAMINHADOS_TITULO)
st.altair_chart(graficos.barras_encaminhados(carga, casas["taxa_encaminhamento"]),
                width="stretch")
componentes.quadro(textos.LEITURA_TITULO, textos.LEITURA_TRIAGEM)

st.subheader(textos.EQUIPARADO_TITULO)
componentes.nota(textos.EQUIPARADO_NOTA)
equiparado = dados.carregar("equiparado_por_modelo")
exibida = graficos.tabela(
    equiparado, dados.casas("equiparado_por_modelo"),
    colunas=["estrategia", "recall_obtido", "encaminhados", "taxa_encaminhamento",
             "casos_detectados", "nns"],
    nomes={"estrategia": "estratégia", "recall_obtido": "recall",
           "taxa_encaminhamento": "encaminhados (%)", "casos_detectados": "detectados",
           "nns": "NNS"},
)
# Selo mapeado da coluna booleana; a regra não tem critério e fica sem selo.
exibida["critério"] = equiparado["cumpre_criterio"].map({
    True: componentes.selo(*textos.CUMPRE),
    False: componentes.selo(*textos.NAO_CUMPRE),
}).fillna("").reset_index(drop=True)
componentes.tabela(exibida, colunas_html=("critério",))

st.subheader(textos.CUSTO_MARGINAL_TITULO)
custo = dados.carregar("custo_marginal_pareado")
# O artefato traz a chave do regime; o nome que aparece na tela vem de `textos`.
custo = custo.assign(regime=custo["regime"].map(textos.REGIMES_CUSTO))
componentes.tabela(graficos.tabela(
    custo, dados.casas("custo_marginal_pareado"),
    colunas=["conjunto", "regime", "encaminhados_floresta", "detectados_floresta",
             "encaminhados_logistica", "detectados_logistica", "delta_encaminhados",
             "delta_detectados", "custo_marginal", "leitura"],
    nomes={"encaminhados_floresta": "enc. floresta",
           "detectados_floresta": "det. floresta",
           "encaminhados_logistica": "enc. logística",
           "detectados_logistica": "det. logística",
           "delta_encaminhados": "Δ enc.", "delta_detectados": "Δ det.",
           "custo_marginal": "enc. por caso a mais"},
))
componentes.nota(textos.CUSTO_MARGINAL_NOTA)
