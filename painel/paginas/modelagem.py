"""Modelagem: busca de hiperparâmetros da floresta e a configuração congelada."""

import pandas as pd
import streamlit as st

import componentes
import dados
import graficos
import selecao
import textos

selecao.abrir_pagina("modelagem", "tecnica")
g = textos.GLOSSARIO

busca = dados.carregar("busca_rf")
casas = dados.casas("busca_rf")
casas_resumo = dados.casas("resumo_amostra")
piso = selecao.resumo("Prevalência no treino", "proporcao")
texto_piso = textos.PISO.format(piso=graficos.numero(piso, casas_resumo["proporcao"]))

st.subheader(textos.BUSCA_TITULO)
st.altair_chart(graficos.busca(busca, piso, texto_piso), width="stretch")

# Vencedor de cada conjunto: a primeira posição do ranking.
vencedores = busca.sort_values("posicao").groupby("conjunto").head(1)
st.subheader(textos.HIPERPARAMETROS_TITULO)
for coluna, (_, venc) in zip(st.columns(len(vencedores)), vencedores.iterrows()):
    with coluna:
        componentes.grade([
            componentes.metrica(
                f"PR-AUC do vencedor · {venc['conjunto']}",
                graficos.numero(venc["pr_auc_media"], casas["pr_auc_media"]),
                ajuda=g["pr_auc_busca"],
                nota=(f"± {graficos.numero(venc['pr_auc_desvio'], casas['pr_auc_desvio'])}"
                      f" entre folds · {texto_piso}"),
            ),
        ])
        componentes.campos([
            ("max_depth", "sem limite" if pd.isna(venc["max_depth"])
             else graficos.numero(venc["max_depth"], casas["max_depth"])),
            ("min_samples_leaf",
             graficos.numero(venc["min_samples_leaf"], casas["min_samples_leaf"])),
            # Mistura fração e nome ("sqrt"): passa como texto, com vírgula decimal.
            ("max_features", str(venc["max_features"]).replace(".", ",")),
            ("class_weight", str(venc["class_weight"])),
            ("n_estimators", graficos.numero(venc["n_estimators"], casas["n_estimators"])),
        ])
componentes.nota(textos.HIPERPARAMETROS_NOTA)
componentes.quadro(textos.LEITURA_TITULO, textos.D034)

with st.expander(textos.BUSCA_TITULO):
    st.dataframe(
        graficos.tabela(busca, casas,
                        nomes={"posicao": "posição", "pr_auc_media": "PR-AUC média",
                               "pr_auc_desvio": "dp entre folds"}),
        hide_index=True, height=420,
    )
