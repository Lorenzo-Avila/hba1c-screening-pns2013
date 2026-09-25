"""Explicabilidade técnica: dois regimes, topo coincidente, blocos e árvores."""

import streamlit as st

import componentes
import dados
import graficos
import selecao
import textos

linha = selecao.abrir_pagina("explicabilidade_tecnica", "tecnica")
g = textos.GLOSSARIO

atributos = dados.carregar("shap_global_atributo")
da_celula = atributos[(atributos["modelo"] == linha["modelo"])
                      & (atributos["celula"] == linha["conjunto"])]

if da_celula.empty:
    componentes.alerta(textos.SHAP_SO_FLORESTA)
else:
    celula = linha["conjunto"]
    casas_atr = dados.casas("shap_global_atributo")
    componentes.nota(textos.DOIS_REGIMES)

    # Topo coincidente: a comparação pré-registrada primeiro, a publicada depois.
    st.subheader(textos.CONCORDANCIA_TITULO)
    concordancia = dados.carregar("shap_concordancia_topo")
    concordancia = concordancia[concordancia["celula"] == celula]
    casas_conc = dados.casas("shap_concordancia_topo")
    cartoes = []
    for comparacao, (titulo, explicacao) in textos.COMPARACOES.items():
        registro = concordancia[concordancia["comparacao"] == comparacao].iloc[0]
        cartoes.append(componentes.metrica(
            titulo,
            f"{graficos.numero(registro['topo_coincidente'], casas_conc['topo_coincidente'])}"
            f" de {graficos.numero(registro['n_atributos'], casas_conc['n_atributos'])}",
            ajuda=g["topo_coincidente"],
            nota=f"{explicacao} Topo: {registro['atributos_no_topo']}.",
            tom="azul" if not cartoes else "neutro",
        ))
    componentes.grade(cartoes)

    st.subheader("Ranking por variável conceitual")
    regimes = list(da_celula["regime"].unique())
    recortes = []
    for regime in regimes:
        recorte = da_celula[da_celula["regime"] == regime]
        recortes.append((f"{regime} · {recorte['objeto_explicado'].iloc[0]}", recorte))
    st.altair_chart(graficos.rankings_lado_a_lado(recortes, casas_atr["shap_medio_abs"]),
                    width="content")

    with st.expander("Por coluna da matriz de design"):
        componentes.nota(textos.CATEGORIA_OMITIDA)
        colunas_design = dados.carregar("shap_global_coluna")
        da_celula_col = colunas_design[colunas_design["celula"] == celula]
        casas_col = dados.casas("shap_global_coluna")
        for coluna, regime in zip(st.columns(len(regimes)), regimes):
            with coluna:
                st.caption(regime)
                st.dataframe(
                    graficos.tabela(da_celula_col[da_celula_col["regime"] == regime],
                                    casas_col,
                                    colunas=["posicao", "coluna", "atributo",
                                             "shap_medio_abs"],
                                    nomes={"posicao": "posição",
                                           "shap_medio_abs": "|SHAP| médio"}),
                    hide_index=True, height="content",
                )

    st.subheader(textos.BLOCOS_TITULO)
    pares = dados.carregar("shap_pares_colineares")
    pares = pares[pares["celula"] == celula]
    casas_pares = dados.casas("shap_pares_colineares")
    for bloco in pares["bloco"].unique():
        do_bloco = pares[pares["bloco"] == bloco]
        componentes.nota(f"{bloco} · {do_bloco['evidencia'].iloc[0]}")
        componentes.tabela(graficos.tabela(
            do_bloco, casas_pares,
            colunas=["regime", "membro", "shap_medio_abs", "posicao",
                     "shap_bloco_somado", "fracao_do_bloco", "razao_maior_menor"],
            nomes={"shap_medio_abs": "|SHAP| individual", "posicao": "posição",
                   "shap_bloco_somado": "|SHAP| do bloco somado",
                   "fracao_do_bloco": "fração do bloco",
                   "razao_maior_menor": "razão maior/menor"},
        ))
    componentes.quadro(textos.LEITURA_TITULO,
                       f"{textos.PREDICAO_D053} {textos.DESFECHO_D053}")

    st.subheader(textos.ARVORES_TITULO)
    componentes.alerta(textos.FLORESTA_REDUZIDA)
    arvores = dados.carregar("shap_arvores_excluidas")
    arvores = arvores[arvores["celula"] == celula]
    casas_arv = dados.casas("shap_arvores_excluidas")
    componentes.grade([
        componentes.metrica(
            "Árvores excluídas",
            f"{graficos.numero(arvores['arvores_excluidas'].iloc[0], 0)} de "
            f"{graficos.numero(arvores['arvores_ajustadas'].iloc[0], 0)}",
            ajuda=g["arvores_excluidas"]),
    ])
    exibida = graficos.tabela(
        arvores, casas_arv,
        colunas=["regime_principal", "regime_contraste", "arvores_mantidas", "corte",
                 "piso_por_arvore", "erro_min_mantidas", "erro_max_mantidas",
                 "erro_min_excluidas", "erro_max_excluidas",
                 "erro_aditividade_principal", "erro_aditividade_contraste",
                 "delta_arvores", "delta_regime"],
        nomes={"regime_principal": "regime principal",
               "regime_contraste": "regime de contraste",
               "arvores_mantidas": "árvores mantidas", "corte": "corte de exclusão",
               "piso_por_arvore": "piso por árvore",
               "erro_min_mantidas": "erro mínimo das mantidas",
               "erro_max_mantidas": "erro máximo das mantidas",
               "erro_min_excluidas": "erro mínimo das excluídas",
               "erro_max_excluidas": "erro máximo das excluídas",
               "erro_aditividade_principal": "erro de aditividade, principal",
               "erro_aditividade_contraste": "erro de aditividade, contraste",
               "delta_arvores": "deslocamento por remoção",
               "delta_regime": "deslocamento por regime"},
    )
    transposta = exibida.T.reset_index()
    transposta.columns = ["", celula]
    componentes.tabela(transposta)
