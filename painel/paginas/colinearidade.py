"""Colinearidade e escolha do conjunto: Pearson, VIF, ablação e escore."""

import streamlit as st

import componentes
import dados
import graficos
import selecao
import textos

selecao.abrir_pagina("colinearidade", "tecnica")

vif = dados.carregar("vif_comparativo")
resumo = dados.carregar("vif_resumo")
casas_vif = dados.casas("vif_comparativo")
casas_resumo = dados.casas("vif_resumo")

# VIF de cada conjunto, com o limiar vindo de vif_resumo.
st.subheader("VIF por conjunto")
componentes.tabela(graficos.tabela(
    resumo, casas_resumo,
    nomes={"feature_set": "conjunto", "vif_maximo": "VIF máximo",
           "vif_mediano": "VIF mediano", "acima_do_limiar": "colunas acima do limiar",
           "limiar_vif": "limiar de VIF"},
))
conjuntos = list(resumo["feature_set"])
for aba, nome in zip(st.tabs(conjuntos), conjuntos):
    with aba:
        linha = resumo[resumo["feature_set"] == nome].iloc[0]
        rotulo = ("limiar de VIF "
                  + graficos.numero(linha["limiar_vif"], casas_resumo["limiar_vif"]))
        st.altair_chart(
            graficos.barras_vif(vif[vif["feature_set"] == nome], linha["limiar_vif"],
                                rotulo, casas_vif["vif"]),
            width="stretch",
        )
componentes.nota(textos.LIMIAR_VIF)

pearson_col, ablacao_col = st.columns(2)
with pearson_col:
    st.subheader("Pearson entre contínuas")
    pearson = dados.carregar("pearson_coeficientes")
    conjunto = st.selectbox("Conjunto", list(pearson["feature_set"].unique()),
                            key="pearson_conjunto")
    pares = pearson[pearson["feature_set"] == conjunto].sort_values(
        "abs_pearson", ascending=False
    )
    st.dataframe(
        graficos.tabela(pares, dados.casas("pearson_coeficientes"),
                        colunas=["var_1", "var_2", "pearson", "abs_pearson"],
                        nomes={"var_1": "variável 1", "var_2": "variável 2",
                               "abs_pearson": "|pearson|"}),
        hide_index=True, height=420,
    )
with ablacao_col:
    st.subheader("Ablação de rcest")
    componentes.nota(textos.ABLACAO_RCEST)
    ablacao = dados.carregar("vif_ablacao_rcest")
    st.dataframe(
        graficos.tabela(ablacao, dados.casas("vif_ablacao_rcest"),
                        colunas=["variavel", "vif_antes", "vif_depois", "delta_vif",
                                 "acima_antes", "acima_depois"],
                        nomes={"variavel": "coluna", "vif_antes": "VIF antes",
                               "vif_depois": "VIF depois", "delta_vif": "Δ VIF",
                               "acima_antes": "acima antes",
                               "acima_depois": "acima depois"}),
        hide_index=True, height=420,
    )

st.subheader("Decomposição do escore de multimorbidade")
componentes.nota(textos.DECOMPOSICAO_ESCORE)
componentes.tabela(graficos.tabela(
    dados.carregar("decomposicao_escore"), dados.casas("decomposicao_escore"),
    nomes={"feature_set": "conjunto", "r2_identidade": "R² da identidade",
           "r2_vif": "R² da auxiliar do VIF", "vif": "VIF"},
))
