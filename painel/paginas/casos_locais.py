"""Casos locais: waterfall de cada um dos oito casos, com o valor observado."""

import streamlit as st

import dados
import graficos
import textos

st.header("Casos locais")
st.markdown(textos.CASOS_LOCAIS)

casos = dados.carregar("shap_casos_locais")
casas_casos = dados.casas("shap_casos_locais")

# Um caso por combinação de célula, caso e índice, na ordem do artefato.
chaves = casos[["celula", "caso", "indice"]].drop_duplicates()
opcoes = [tuple(linha) for linha in chaves.itertuples(index=False)]
escolhido = st.selectbox(
    "Caso",
    opcoes,
    format_func=lambda c: f"{c[0]} · {textos.CASOS[c[1]]} · índice {c[2]}",
    key="caso_local",
)
celula, caso, indice = escolhido
recorte = casos[(casos["celula"] == celula) & (casos["caso"] == caso)
                & (casos["indice"] == indice)]
primeira = recorte.iloc[0]

# Índice presente em mais de uma célula: mesmo indivíduo.
celulas_do_indice = list(chaves[chaves["indice"] == indice]["celula"])
if len(celulas_do_indice) > 1:
    st.info(textos.MESMO_INDIVIDUO.format(indice=indice,
                                          celulas=" e ".join(celulas_do_indice)))

metricas = st.columns(4)
metricas[0].metric("escore de ordenação",
                   graficos.numero(primeira["escore"], casas_casos["escore"]))
metricas[1].metric("limiar calibrado",
                   graficos.numero(primeira["limiar"], casas_casos["limiar"]))
metricas[2].metric("distância ao limiar",
                   graficos.numero(primeira["distancia_ao_limiar"],
                                   casas_casos["distancia_ao_limiar"]))
metricas[3].metric("valor de base",
                   graficos.numero(primeira["valor_base"], casas_casos["valor_base"]))
st.caption(textos.ESCORE_NAO_E_RISCO)

st.altair_chart(
    graficos.waterfall(recorte, textos.RESSALVA_LOCAL, casas_casos["contribuicao"]),
    width="stretch",
)
st.caption(textos.WATERFALL)
st.warning(textos.RESSALVA_LOCAL)

exibida = graficos.tabela(recorte, casas_casos,
                          colunas=["atributo", "contribuicao"],
                          nomes={"contribuicao": "contribuição"})
exibida["valor observado"] = [graficos.observado(v)
                              for v in recorte["valor_observado"]]
st.dataframe(exibida, hide_index=True, height="content")
