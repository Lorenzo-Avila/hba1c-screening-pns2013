"""Amostra analítica: exclusões, ausentes e as duas prevalências."""

import streamlit as st

import componentes
import dados
import graficos
import selecao
import textos

selecao.abrir_pagina("amostra", "tecnica")
g = textos.GLOSSARIO
resumo = dados.carregar("resumo_amostra")
casas = dados.casas("resumo_amostra")


def n(item: str) -> str:
    return graficos.numero(selecao.resumo(item), casas["n"])


def p(item: str) -> str:
    return graficos.numero(selecao.resumo(item, "proporcao"), casas["proporcao"])


componentes.grade([
    componentes.metrica("Linhas no CSV bruto", n("Linhas no CSV bruto"),
                        ajuda=g["linhas_brutas"]),
    componentes.metrica("Amostra analítica", n("Amostra analítica"),
                        ajuda=g["amostra_analitica"]),
    componentes.metrica("Treino", n("Treino"), ajuda=g["treino"]),
    componentes.metrica("Teste", n("Teste"), ajuda=g["teste"]),
])
componentes.grade([
    componentes.metrica("Prevalência amostral", p("Positivos (HbA1c ≥ 6,5%)"),
                        ajuda=g["prevalencia"],
                        nota=f"{n('Positivos (HbA1c ≥ 6,5%)')} positivos"),
    componentes.metrica("Prevalência no treino", p("Prevalência no treino"),
                        ajuda=g["prevalencia"]),
    componentes.metrica("Prevalência no teste", p("Prevalência no teste"),
                        ajuda=g["prevalencia"]),
    componentes.metrica("Prevalência ponderada",
                        p("Prevalência ponderada por peso_lab (descritiva)"),
                        ajuda=g["prevalencia_ponderada"], tom="azul"),
])
componentes.quadro(textos.LEITURA_TITULO, textos.PREVALENCIA_PONDERADA)

tabela, ausentes = st.columns(2)
with tabela:
    st.subheader(textos.FUNIL_TITULO)
    componentes.tabela(graficos.tabela(resumo, casas,
                                       nomes={"item": "etapa", "proporcao": "proporção"}))
with ausentes:
    st.subheader(textos.AUSENTES_TITULO)
    componentes.tabela(graficos.tabela(
        dados.carregar("ausentes_por_variavel"), dados.casas("ausentes_por_variavel"),
        nomes={"variavel": "variável", "proporcao": "proporção", "no_treino": "no treino"},
    ))
    componentes.nota(textos.AUSENTES_NOTA)
