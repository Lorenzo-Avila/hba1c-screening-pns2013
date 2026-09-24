"""Colinearidade e conjunto: Pearson, VIF dos três conjuntos, ablação e escore."""

import streamlit as st

import dados
import graficos
import textos

st.header("Colinearidade e conjunto")

# Pearson entre contínuas, por conjunto. A lista de conjuntos vem do artefato.
st.subheader("Correlação de Pearson entre contínuas")
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
    hide_index=True, height="content",
)

# VIF de cada conjunto, com o limiar vindo de vif_resumo.
st.subheader("VIF por conjunto")
vif = dados.carregar("vif_comparativo")
resumo = dados.carregar("vif_resumo")
casas_vif = dados.casas("vif_comparativo")
casas_resumo = dados.casas("vif_resumo")
conjuntos = list(resumo["feature_set"])
for aba, nome in zip(st.tabs(conjuntos), conjuntos):
    with aba:
        linha = resumo[resumo["feature_set"] == nome].iloc[0]
        rotulo = f"limiar de VIF {graficos.numero(linha['limiar_vif'], casas_resumo['limiar_vif'])}"
        st.altair_chart(
            graficos.barras_vif(vif[vif["feature_set"] == nome], linha["limiar_vif"],
                                rotulo, casas_vif["vif"]),
            width="stretch",
        )
st.caption(textos.LIMIAR_VIF)
st.dataframe(
    graficos.tabela(resumo, casas_resumo,
                    nomes={"feature_set": "conjunto", "vif_maximo": "VIF máximo",
                           "vif_mediano": "VIF mediano",
                           "acima_do_limiar": "colunas acima do limiar",
                           "limiar_vif": "limiar de VIF"}),
    hide_index=True, height="content",
)

st.subheader("Ablação de rcest")
st.markdown(textos.ABLACAO_RCEST)
ablacao = dados.carregar("vif_ablacao_rcest")
st.dataframe(
    graficos.tabela(ablacao, dados.casas("vif_ablacao_rcest"),
                    nomes={"variavel": "coluna", "vif_antes": "VIF antes",
                           "r2_antes": "R² antes", "acima_antes": "acima antes",
                           "vif_depois": "VIF depois", "r2_depois": "R² depois",
                           "acima_depois": "acima depois", "delta_vif": "Δ VIF"}),
    hide_index=True, height="content",
)

st.subheader("Decomposição do escore de multimorbidade")
st.markdown(textos.DECOMPOSICAO_ESCORE)
decomposicao = dados.carregar("decomposicao_escore")
st.dataframe(
    graficos.tabela(decomposicao, dados.casas("decomposicao_escore"),
                    nomes={"feature_set": "conjunto",
                           "r2_identidade": "R² da identidade",
                           "r2_vif": "R² da auxiliar do VIF", "vif": "VIF"}),
    hide_index=True, height="content",
)
