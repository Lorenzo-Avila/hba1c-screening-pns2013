"""Desempenho da célula selecionada contra a regra clínica ADA."""

import streamlit as st

import componentes
import dados
import graficos
import selecao
import textos

linha = selecao.abrir_pagina("desempenho", "clinica")
comparativo = dados.carregar("comparativo_consolidado")
casas = dados.casas("comparativo_consolidado")
g = textos.GLOSSARIO


def v(registro, coluna: str) -> str:
    return graficos.numero(registro[coluna], casas[coluna])


def dp(coluna: str) -> str:
    return textos.DP_FOLDS.format(dp=v(linha, coluna))


# Faixa do comparador, antes dos cartões da célula.
ada = comparativo[comparativo["modelo"] == "Regra ADA"].iloc[0]
componentes.faixa(textos.COMPARADOR_ADA, [
    componentes.metrica("Recall", v(ada, "recall"), ajuda=g["recall"]),
    componentes.metrica("Especificidade", v(ada, "especificidade"),
                        ajuda=g["especificidade"]),
    componentes.metrica("Encaminhados", v(ada, "taxa_encaminhamento"),
                        ajuda=g["encaminhados"]),
    componentes.metrica("NNS", v(ada, "nns"), ajuda=g["nns"]),
])

calibracao = dados.carregar("calibracao_comparada")
piso = calibracao[(calibracao["modelo"] == linha["modelo"])
                  & (calibracao["conjunto"] == linha["conjunto"])].iloc[0]["prevalencia"]
texto_piso = textos.PISO.format(
    piso=graficos.numero(piso, dados.casas("calibracao_comparada")["prevalencia"]))

componentes.grade([
    componentes.metrica("Recall (sensibilidade)", v(linha, "recall"),
                        ajuda=g["recall"], nota=dp("recall_cv_dp")),
    componentes.metrica("Especificidade", v(linha, "especificidade"),
                        ajuda=g["especificidade"], nota=dp("especificidade_cv_dp")),
    componentes.metrica("F1", v(linha, "f1"), ajuda=g["f1"], nota=dp("f1_cv_dp")),
    componentes.metrica("NNS", v(linha, "nns"), ajuda=g["nns"]),
    componentes.metrica("Encaminhados (%)", v(linha, "taxa_encaminhamento"),
                        ajuda=g["encaminhados"],
                        nota=f"{v(linha, 'encaminhados')} pessoas"),
    componentes.metrica("AUC-ROC", v(linha, "auc_roc"), ajuda=g["auc_roc"],
                        nota=dp("auc_roc_cv_dp")),
    componentes.metrica("PR-AUC", v(linha, "pr_auc"), ajuda=g["pr_auc"],
                        nota=texto_piso),
])
componentes.nota(textos.LEGENDA_DESEMPENHO.format(recall_alvo=v(linha, "recall_alvo")))

matriz, curvas = st.columns([2, 3])
with matriz:
    st.subheader(textos.MATRIZ_TITULO)
    componentes.matriz_confusao([
        componentes.metrica(textos.MATRIZ["vn"], v(linha, "vn"), ajuda=g["vn"],
                            nota="verdadeiro negativo", tom="acerto"),
        componentes.metrica(textos.MATRIZ["fp"], v(linha, "fp"), ajuda=g["fp"],
                            nota="falso positivo", tom="erro"),
        componentes.metrica(textos.MATRIZ["fn"], v(linha, "fn"), ajuda=g["fn"],
                            nota="falso negativo", tom="erro"),
        componentes.metrica(textos.MATRIZ["vp"], v(linha, "vp"), ajuda=g["vp"],
                            nota="verdadeiro positivo", tom="acerto"),
    ])
with curvas:
    roc, pr = st.tabs([textos.ABA_ROC, textos.ABA_PR])
    celula = {"modelo": linha["modelo"], "conjunto": linha["conjunto"]}
    with roc:
        curva = dados.carregar("curva_roc")
        curva = curva[(curva["modelo"] == celula["modelo"])
                      & (curva["conjunto"] == celula["conjunto"])]
        st.altair_chart(
            graficos.curva_roc(curva, linha["fpr"], linha["recall"],
                               f"AUC-ROC {v(linha, 'auc_roc')}", textos.ROTULO_PONTO),
            width="stretch",
        )
    with pr:
        curva = dados.carregar("curva_pr")
        curva = curva[(curva["modelo"] == celula["modelo"])
                      & (curva["conjunto"] == celula["conjunto"])]
        st.altair_chart(
            graficos.curva_pr(curva, linha["recall"], linha["precisao"], piso,
                              texto_piso, f"PR-AUC {v(linha, 'pr_auc')}",
                              textos.ROTULO_PONTO),
            width="stretch",
        )

with st.expander(textos.COMPARAR_CELULAS):
    componentes.tabela(graficos.tabela(
        comparativo, casas,
        colunas=["modelo", "conjunto", "limiar", "recall", "especificidade", "f1",
                 "auc_roc", "pr_auc", "nns", "encaminhados", "taxa_encaminhamento",
                 "corte_unico"],
        nomes={"especificidade": "especif.", "auc_roc": "AUC-ROC", "pr_auc": "PR-AUC",
               "nns": "NNS", "f1": "F1", "taxa_encaminhamento": "encaminhados (%)",
               "corte_unico": "corte único"},
    ))
    componentes.nota(textos.NOTA_CORTE_UNICO)
