"""Floresta × logística: McNemar pareado no teste, com o recorte nos positivos."""

import streamlit as st

import componentes
import dados
import graficos
import selecao
import textos

selecao.abrir_pagina("floresta_logistica", "tecnica")
g = textos.GLOSSARIO

mcnemar = dados.carregar("mcnemar_pareado")
casas = dados.casas("mcnemar_pareado")
linha = mcnemar.iloc[0]
modelo_a, modelo_b = linha["modelo_a"], linha["modelo_b"]


def n(coluna: str) -> str:
    return graficos.numero(linha[coluna], casas[coluna])


esquerda, direita = st.columns([3, 2])
with esquerda:
    st.subheader(textos.MATRIZ_MCNEMAR)
    componentes.matriz_confusao([
        componentes.metrica(f"{modelo_a} acerta · {modelo_b} acerta", n("ambos_acertam"),
                            ajuda=g["concordantes"], tom="acerto"),
        componentes.metrica(f"{modelo_a} acerta · {modelo_b} erra", n("so_a_acerta"),
                            ajuda=g["discordantes"], tom="azul"),
        componentes.metrica(f"{modelo_a} erra · {modelo_b} acerta", n("so_b_acerta"),
                            ajuda=g["discordantes"], tom="azul"),
        componentes.metrica(f"{modelo_a} erra · {modelo_b} erra", n("ambos_erram"),
                            ajuda=g["concordantes"], tom="erro"),
    ])
with direita:
    st.subheader("Teste")
    componentes.grade([
        componentes.metrica("Casos no teste", n("n"), ajuda=g["n_teste"]),
        componentes.metrica("Discordantes", n("discordancias"), ajuda=g["discordantes"]),
        componentes.metrica("p-valor", n("p_valor"), ajuda=g["p_valor"]),
    ])

st.subheader(textos.RECORTE_POSITIVOS)
componentes.grade([
    componentes.metrica("Positivos no teste", n("positivos_teste"),
                        ajuda=g["positivos_teste"]),
    componentes.metrica(f"Detectados só por {modelo_a}", n("positivos_so_a_detecta"),
                        ajuda=g["positivos_so"], tom="azul"),
    componentes.metrica(f"Detectados só por {modelo_b}", n("positivos_so_b_detecta"),
                        ajuda=g["positivos_so"]),
])
componentes.quadro(textos.LEITURA_TITULO, f"{modelo_a} {textos.LEITURA_MCNEMAR}.")
