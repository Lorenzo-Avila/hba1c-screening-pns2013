"""Floresta × logística: McNemar pareado no teste, com o recorte nos positivos."""

import pandas as pd
import streamlit as st

import dados
import graficos
import textos

st.header("Floresta × logística")

mcnemar = dados.carregar("mcnemar_pareado")
casas = dados.casas("mcnemar_pareado")
linha = mcnemar.iloc[0]
modelo_a, modelo_b = linha["modelo_a"], linha["modelo_b"]


def _n(coluna: str) -> str:
    return graficos.numero(linha[coluna], casas[coluna])


st.markdown(
    f"Teste de McNemar pareado entre **{modelo_a}** e **{modelo_b}**, caso a caso "
    "no conjunto de teste, cada modelo no seu limiar calibrado. O p-valor vem do "
    "teste binomial exato sobre os casos discordantes (D-043)."
)

st.subheader("Matriz de acertos pareados")
matriz = pd.DataFrame(
    {
        f"{modelo_b} acerta": [_n("ambos_acertam"), _n("so_b_acerta")],
        f"{modelo_b} erra": [_n("so_a_acerta"), _n("ambos_erram")],
    },
    index=[f"{modelo_a} acerta", f"{modelo_a} erra"],
)
st.table(matriz)

metricas = st.columns(4)
metricas[0].metric("casos no teste", _n("n"))
metricas[1].metric("concordantes", _n("concordam"))
metricas[2].metric("discordantes", _n("discordancias"))
metricas[3].metric("p-valor", _n("p_valor"))

st.subheader("Recorte nos positivos do teste")
positivos = st.columns(3)
positivos[0].metric("positivos no teste", _n("positivos_teste"))
positivos[1].metric(f"detectados só por {modelo_a}", _n("positivos_so_a_detecta"))
positivos[2].metric(f"detectados só por {modelo_b}", _n("positivos_so_b_detecta"))

st.info(f"Leitura, por D-043: {modelo_a} {textos.LEITURA_MCNEMAR}.")
