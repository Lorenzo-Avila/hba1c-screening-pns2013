"""Visão geral do projeto: amostra, desbalanceamento, pipeline e escopo."""

import streamlit as st

import componentes
import dados
import graficos
import selecao
import textos

selecao.abrir_pagina("visao_geral", "projeto")
casas = dados.casas("resumo_amostra")


def n(item: str) -> str:
    return graficos.numero(selecao.resumo(item), casas["n"])


st.subheader(textos.FUNIL_TITULO)
componentes.funil([
    ("linhas no CSV bruto", n("Linhas no CSV bruto"), ""),
    ("sem HbA1c válida", n("Excluídas: HbA1c ausente"), "exclusao"),
    ("sem bloco de medição", n("Excluídas: bloco de medição integralmente ausente"),
     "exclusao"),
    ("amostra analítica", n("Amostra analítica"), "destaque"),
    ("treino", n("Treino"), ""),
    ("teste", n("Teste"), ""),
])

# Largura total: dentro de coluna o Streamlit reduz o gráfico como imagem.
st.subheader(textos.DESBALANCEAMENTO_TITULO)
positivos = selecao.resumo("Positivos (HbA1c ≥ 6,5%)")
negativos = selecao.resumo("Negativos")
st.altair_chart(
    graficos.desbalanceamento([
        (f"HbA1c ≥ 6,5% · {graficos.numero(positivos, casas['n'])}", positivos),
        (f"HbA1c < 6,5% · {graficos.numero(negativos, casas['n'])}", negativos),
    ]),
    width="stretch",
)
componentes.grade([
    componentes.metrica(
        "Prevalência na amostra",
        graficos.numero(selecao.resumo("Positivos (HbA1c ≥ 6,5%)", "proporcao"),
                        casas["proporcao"]),
        ajuda=textos.GLOSSARIO["prevalencia"],
    ),
    componentes.metrica("Pessoas com HbA1c ≥ 6,5%", n("Positivos (HbA1c ≥ 6,5%)"),
                        ajuda=textos.GLOSSARIO["positivos"]),
    componentes.metrica("Pessoas com HbA1c < 6,5%", n("Negativos"),
                        ajuda=textos.GLOSSARIO["negativos"]),
])

st.subheader(textos.PIPELINE_TITULO)
st.graphviz_chart(graficos.pipeline_dot(textos.PIPELINE), width="stretch")
componentes.nota(textos.PIPELINE_LEGENDA)

st.subheader(textos.SMOTE_TITULO)
componentes.duas_colunas(*textos.SMOTE)

st.subheader(textos.CRISP_DM_TITULO)
componentes.linha_tempo(textos.CRISP_DM)

st.subheader(textos.NAO_FAZ_TITULO)
componentes.selos([(item, "cinza") for item in textos.NAO_FAZ])
