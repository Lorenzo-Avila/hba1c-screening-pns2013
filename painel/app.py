"""Ponto de entrada do painel: configura a página, monta a navegação e o rodapé.

Rode com `streamlit run painel/app.py`. O painel lê `reports/tabelas/` e não
calcula nada (D-054).
"""

import streamlit as st

import textos

st.set_page_config(page_title=textos.TITULO, layout="wide")

PAGINAS = [
    st.Page("paginas/amostra.py", title="Amostra", default=True),
    st.Page("paginas/colinearidade.py", title="Colinearidade e conjunto"),
    st.Page("paginas/floresta_logistica.py", title="Floresta × logística"),
    st.Page("paginas/explicabilidade.py", title="Explicabilidade global"),
    st.Page("paginas/casos_locais.py", title="Casos locais"),
]

navegacao = st.navigation(PAGINAS)
st.title(textos.TITULO)
navegacao.run()
st.divider()
st.caption(textos.RODAPE)
