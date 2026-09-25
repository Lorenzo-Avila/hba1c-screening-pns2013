"""Casos exemplares: decisão, desfecho e waterfall de oito casos do teste."""

import streamlit as st

import componentes
import dados
import graficos
import selecao
import textos

selecao.abrir_pagina("casos_exemplares", "clinica")
componentes.alerta(textos.FAIXA_CASOS)

casos = dados.carregar("shap_casos_locais")
casas = dados.casas("shap_casos_locais")
g = textos.GLOSSARIO

# Um caso por combinação de célula, caso e índice, na ordem do artefato.
chaves = casos[["celula", "caso", "indice"]].drop_duplicates()
opcoes = [tuple(linha) for linha in chaves.itertuples(index=False)]
celula, caso, indice = st.selectbox(
    textos.ROTULO_CASO,
    opcoes,
    format_func=lambda c: f"{c[0]} · {textos.CASOS[c[1]]} · índice {c[2]}",
    key="caso_exemplar",
)
recorte = casos[(casos["celula"] == celula) & (casos["caso"] == caso)
                & (casos["indice"] == indice)]
primeira = recorte.iloc[0]

if primeira["indice_nas_duas_celulas"]:
    celulas = list(chaves[chaves["indice"] == indice]["celula"])
    componentes.nota(textos.MESMO_INDIVIDUO.format(indice=indice,
                                                  celulas=" e ".join(celulas)))

campos, decisao = st.columns([3, 2])
with campos:
    st.subheader(textos.VALORES_OBSERVADOS)
    componentes.campos([
        (textos.ROTULOS_CLINICOS.get(a, a), graficos.observado(v))
        for a, v in zip(recorte["atributo"], recorte["valor_observado"])
    ])
with decisao:
    st.subheader(textos.DECISAO_TITULO)
    componentes.selos([textos.DECISAO[caso]])
    componentes.nota(textos.DESFECHO_OBSERVADO)
    componentes.selos([(textos.DESFECHO[caso], "cinza"), textos.ACERTO[caso]])
    componentes.grade([
        componentes.metrica("Escore de ordenação",
                            graficos.numero(primeira["escore"], casas["escore"]),
                            ajuda=g["escore"]),
        componentes.metrica("Limiar calibrado",
                            graficos.numero(primeira["limiar"], casas["limiar"]),
                            ajuda=g["limiar"]),
        componentes.metrica(textos.ROTULO_DISTANCIA,
                            graficos.numero(primeira["distancia_ao_limiar"],
                                            casas["distancia_ao_limiar"]),
                            ajuda=g["distancia"]),
    ])

st.altair_chart(
    graficos.waterfall(recorte, textos.RESSALVA_LOCAL, casas["contribuicao"],
                       textos.ROTULOS_CLINICOS),
    width="stretch",
)
componentes.nota(textos.WATERFALL)
componentes.quadro(textos.LEITURA_TITULO, textos.RESSALVA_LOCAL)
