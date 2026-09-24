"""Amostra analítica: exclusões, ausentes e as duas prevalências."""

import streamlit as st

import dados
import graficos
import textos

st.header("Amostra")

resumo = dados.carregar("resumo_amostra")
casas_resumo = dados.casas("resumo_amostra")

st.subheader("Exclusões e amostra analítica")
st.dataframe(
    graficos.tabela(resumo, casas_resumo,
                    nomes={"item": "etapa", "n": "n", "proporcao": "proporção"}),
    hide_index=True, height="content",
)

# As duas linhas que carregam proporção: a amostral, na linha dos positivos, e a
# ponderada por peso_lab.
com_proporcao = resumo.dropna(subset=["proporcao"])
colunas = st.columns(len(com_proporcao))
for coluna, (_, linha) in zip(colunas, com_proporcao.iterrows()):
    coluna.metric(linha["item"], graficos.numero(linha["proporcao"],
                                                 casas_resumo["proporcao"]))
st.info(textos.PREVALENCIA_PONDERADA)

st.subheader("Ausentes por variável ao fim da limpeza")
ausentes = dados.carregar("ausentes_por_variavel")
st.dataframe(
    graficos.tabela(ausentes, dados.casas("ausentes_por_variavel"),
                    nomes={"variavel": "variável", "proporcao": "proporção",
                           "no_treino": "no treino"}),
    hide_index=True, height="content",
)
st.caption(
    "Os ausentes das variáveis classificatórias são imputados por KNNImputer "
    "ajustado só no treino (R3)."
)
