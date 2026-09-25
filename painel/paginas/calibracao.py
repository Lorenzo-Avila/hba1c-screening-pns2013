"""Calibração: curva por decis da célula selecionada e índices comparados."""

import streamlit as st

import componentes
import dados
import graficos
import selecao
import textos

linha = selecao.abrir_pagina("calibracao", "tecnica")
componentes.alerta(textos.RESTRICAO_ESCORE)

decis = dados.carregar("calibracao_decis")
recorte = decis[(decis["modelo"] == linha["modelo"])
                & (decis["conjunto"] == linha["conjunto"])]

curva, tabela = st.columns([3, 2])
with curva:
    st.subheader(textos.DECIS_TITULO)
    st.altair_chart(graficos.decis(recorte), width="stretch")
with tabela:
    st.subheader("Decis")
    componentes.tabela(graficos.tabela(
        recorte, dados.casas("calibracao_decis"),
        colunas=["decil", "n", "prob_media", "freq_observada"],
        nomes={"prob_media": "escore médio", "freq_observada": "frequência observada"},
    ))

st.subheader(textos.INDICES_CALIBRACAO)
componentes.tabela(graficos.tabela(
    dados.carregar("calibracao_comparada"), dados.casas("calibracao_comparada"),
    nomes={"brier_trivial": "Brier trivial", "brier_skill": "Brier skill",
           "prevalencia": "prevalência", "proba_media": "escore médio",
           "razao_inflacao": "inflação", "ece": "ECE", "brier": "Brier"},
))
