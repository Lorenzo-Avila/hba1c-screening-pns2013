"""Casos exemplares: decisão, desfecho e waterfall dos casos da célula selecionada."""

import pandas as pd
import streamlit as st

import componentes
import dados
import graficos
import selecao
import textos

linha = selecao.abrir_pagina("casos_exemplares", "clinica")
componentes.alerta(textos.FAIXA_CASOS)

casos = dados.carregar("shap_casos_locais")
casas = dados.casas("shap_casos_locais")
g = textos.GLOSSARIO

# Só os casos da célula da barra lateral. A explicação existe só para a floresta.
da_celula = casos[(casos["modelo"] == linha["modelo"])
                  & (casos["celula"] == linha["conjunto"])]


def mostrar(da_celula: pd.DataFrame) -> None:
    """Seletor dos quatro casos da célula, valores observados, decisão e waterfall."""
    chaves = da_celula[["caso", "indice"]].drop_duplicates()
    opcoes = [tuple(c) for c in chaves.itertuples(index=False)]
    # Chave por célula: trocar a célula na barra lateral troca as opções do seletor.
    caso, indice = st.selectbox(
        textos.ROTULO_CASO,
        opcoes,
        format_func=lambda c: f"{textos.CASOS[c[0]]} · índice {c[1]}",
        key=f"caso_exemplar {selecao.nome(linha)}",
    )
    recorte = da_celula[(da_celula["caso"] == caso) & (da_celula["indice"] == indice)]
    primeira = recorte.iloc[0]

    if primeira["indice_nas_duas_celulas"]:
        celulas = list(casos[casos["indice"] == indice]["celula"].unique())
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


if da_celula.empty:
    componentes.alerta(textos.SHAP_SO_FLORESTA)
else:
    mostrar(da_celula)
