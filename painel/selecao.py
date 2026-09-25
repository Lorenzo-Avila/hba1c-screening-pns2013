"""Célula selecionada na barra lateral e cabeçalho das páginas.

Os seletores de modelo e conjunto leem as opções de `comparativo_consolidado`,
só entre as células com limiar calibrado, e abrem na célula marcada em
`referencia_d030`.
"""

import pandas as pd
import streamlit as st

import componentes
import dados
import graficos
import textos

CHAVE_MODELO = "selecao_modelo"
CHAVE_CONJUNTO = "selecao_conjunto"


def _calibraveis() -> pd.DataFrame:
    tabela = dados.carregar("comparativo_consolidado")
    return tabela[~tabela["corte_unico"]]


def renderizar() -> None:
    """Desenha os seletores na barra lateral, abrindo na célula de referência."""
    celulas = _calibraveis()
    referencia = celulas[celulas["referencia_d030"]].iloc[0]
    modelos = list(celulas["modelo"].unique())
    if st.session_state.get(CHAVE_MODELO) not in modelos:
        st.session_state[CHAVE_MODELO] = referencia["modelo"]
    modelo = st.sidebar.selectbox(textos.ROTULO_MODELO, modelos, key=CHAVE_MODELO)

    conjuntos = list(celulas[celulas["modelo"] == modelo]["conjunto"])
    if st.session_state.get(CHAVE_CONJUNTO) not in conjuntos:
        padrao = (referencia["conjunto"] if referencia["conjunto"] in conjuntos
                  else conjuntos[0])
        st.session_state[CHAVE_CONJUNTO] = padrao
    st.sidebar.selectbox(textos.ROTULO_CONJUNTO, conjuntos, key=CHAVE_CONJUNTO)


def atual() -> pd.Series:
    """Linha de `comparativo_consolidado` da célula selecionada."""
    celulas = _calibraveis()
    modelo = st.session_state.get(CHAVE_MODELO)
    conjunto = st.session_state.get(CHAVE_CONJUNTO)
    escolhida = celulas[(celulas["modelo"] == modelo) & (celulas["conjunto"] == conjunto)]
    if escolhida.empty:
        escolhida = celulas[celulas["referencia_d030"]]
    return escolhida.iloc[0]


def nome(linha: pd.Series) -> str:
    """Nome da célula como aparece nos artefatos: modelo · conjunto."""
    return f"{linha['modelo']} · {linha['conjunto']}"


def resumo(item: str, coluna: str = "n"):
    """Valor de uma linha de `resumo_amostra`, pelo rótulo do item."""
    tabela = dados.carregar("resumo_amostra").set_index("item")
    return tabela.loc[item, coluna]


def abrir_pagina(chave: str, secao: str) -> pd.Series:
    """Cabeçalho da página com os selos da célula. Devolve a linha da célula."""
    titulo, subtitulo, abertura = textos.PAGINAS[chave]
    linha = atual()
    casas = dados.casas("resumo_amostra")
    itens = [
        (textos.SELO_CELULA.format(celula=nome(linha)), "azul"),
        (textos.SELO_TESTE.format(n=graficos.numero(resumo("Teste"), casas["n"])),
         "cinza"),
    ]
    if linha["referencia_d030"]:
        rotulo = (textos.SELO_REFERENCIA_TECNICA if secao == "tecnica"
                  else textos.SELO_REFERENCIA_CLINICA)
        itens.append((rotulo, "verde"))
    componentes.cabecalho(titulo, subtitulo, itens)
    componentes.nota(abertura)
    return linha

