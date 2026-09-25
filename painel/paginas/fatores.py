"""Fatores associados: ranking de contribuição com rótulo clínico."""

import streamlit as st

import componentes
import dados
import graficos
import selecao
import textos

linha = selecao.abrir_pagina("fatores", "clinica")

atributos = dados.carregar("shap_global_atributo")
# Regime principal, na célula selecionada. A explicação só existe para a floresta.
recorte = atributos[
    (atributos["modelo"] == linha["modelo"])
    & (atributos["celula"] == linha["conjunto"])
    & (atributos["regime"] == "tree_path_dependent")
].sort_values("posicao")

if recorte.empty:
    componentes.alerta(textos.SHAP_SO_FLORESTA)
else:
    casas = dados.casas("shap_global_atributo")["shap_medio_abs"]
    st.altair_chart(
        graficos.fatores(recorte.head(5), textos.ROTULOS_CLINICOS, casas,
                         textos.FATORES_TITULO_EIXO),
        width="stretch",
    )
    with st.expander(textos.FATORES_DEMAIS):
        st.altair_chart(
            graficos.fatores(recorte.iloc[5:], textos.ROTULOS_CLINICOS, casas,
                             textos.FATORES_TITULO_EIXO),
            width="stretch",
        )
    componentes.nota(textos.NOTA_RACA_COR)
    componentes.quadro(textos.LEITURA_TITULO, textos.LEITURA_FATORES)
