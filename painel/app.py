"""Ponto de entrada do painel: estilo, barra lateral e navegação.

Rode com `streamlit run painel/app.py`. O painel lê `reports/tabelas/` e não
calcula nada. A navegação padrão fica escondida e a barra lateral monta os links
por seção, para o título vir antes deles.
"""

from html import escape

import streamlit as st

import componentes
import selecao
import textos

st.set_page_config(page_title=textos.TITULO_APP, layout="wide")
componentes.aplicar_estilo()

SECOES = {
    "projeto": [
        st.Page("paginas/visao_geral.py", title="Visão geral", default=True),
        st.Page("paginas/reproducao.py", title="Reprodução ao vivo"),
    ],
    "clinica": [
        st.Page("paginas/desempenho.py", title="Desempenho"),
        st.Page("paginas/triagem.py", title="Triagem na atenção primária"),
        st.Page("paginas/fatores.py", title="Fatores associados"),
        st.Page("paginas/casos_exemplares.py", title="Casos exemplares"),
    ],
    "tecnica": [
        st.Page("paginas/amostra.py", title="Amostra"),
        st.Page("paginas/colinearidade.py", title="Colinearidade e escolha do conjunto"),
        st.Page("paginas/modelagem.py", title="Modelagem"),
        st.Page("paginas/floresta_logistica.py", title="Floresta × logística"),
        st.Page("paginas/calibracao.py", title="Calibração"),
        st.Page("paginas/explicabilidade_tecnica.py", title="Explicabilidade técnica"),
    ],
}

navegacao = st.navigation([p for paginas in SECOES.values() for p in paginas],
                          position="hidden")

with st.sidebar:
    st.markdown(
        f'<p class="marca-titulo">{escape(textos.TITULO_APP)}</p>'
        f'<p class="marca-sub">{escape(textos.SUBTITULO_APP)}</p>',
        unsafe_allow_html=True,
    )
    for chave, paginas in SECOES.items():
        st.markdown(f'<div class="secao-nav">{escape(textos.SECOES[chave])}</div>',
                    unsafe_allow_html=True)
        for pagina in paginas:
            if pagina.url_path == navegacao.url_path:
                # Contêiner com chave vira a classe CSS do link ativo.
                with st.container(key="nav-ativo"):
                    st.page_link(pagina)
            else:
                st.page_link(pagina)
    st.divider()
selecao.renderizar()

navegacao.run()
