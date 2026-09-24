"""Explicabilidade global: ranking nos dois regimes, blocos colineares, árvores."""

import streamlit as st

import dados
import graficos
import textos

st.header("Explicabilidade global")

atributos = dados.carregar("shap_global_atributo")
colunas_design = dados.carregar("shap_global_coluna")
pares = dados.carregar("shap_pares_colineares")
arvores = dados.carregar("shap_arvores_excluidas")
casas_atr = dados.casas("shap_global_atributo")

celula = st.selectbox("Célula", list(atributos["celula"].unique()),
                      key="explicabilidade_celula")
st.markdown(textos.DOIS_REGIMES)

# Os dois regimes lado a lado, na ordem em que aparecem no artefato.
st.subheader("Ranking por variável conceitual")
da_celula = atributos[atributos["celula"] == celula]
regimes = list(da_celula["regime"].unique())
for coluna, regime in zip(st.columns(len(regimes)), regimes):
    recorte = da_celula[da_celula["regime"] == regime]
    titulo = f"{regime} · {recorte['objeto_explicado'].iloc[0]}"
    with coluna:
        st.altair_chart(
            graficos.ranking(recorte, titulo, casas_atr["shap_medio_abs"]),
            width="stretch",
        )
        st.dataframe(
            graficos.tabela(recorte, casas_atr,
                            colunas=["posicao", "atributo", "shap_medio_abs",
                                     "n_colunas"],
                            nomes={"posicao": "posição",
                                   "shap_medio_abs": "|SHAP| médio",
                                   "n_colunas": "colunas"}),
            hide_index=True, height="content",
        )

with st.expander("Por coluna da matriz de design"):
    st.markdown(textos.CATEGORIA_OMITIDA)
    casas_col = dados.casas("shap_global_coluna")
    da_celula_col = colunas_design[colunas_design["celula"] == celula]
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

# Blocos colineares, individual e somado, nos dois regimes.
st.subheader("Blocos colineares, individual e somado")
casas_pares = dados.casas("shap_pares_colineares")
pares_celula = pares[pares["celula"] == celula]
for bloco in pares_celula["bloco"].unique():
    do_bloco = pares_celula[pares_celula["bloco"] == bloco]
    st.markdown(f"**{bloco}** · {do_bloco['evidencia'].iloc[0]}")
    st.dataframe(
        graficos.tabela(do_bloco, casas_pares,
                        colunas=["regime", "membro", "shap_medio_abs", "posicao",
                                 "shap_bloco_somado", "fracao_do_bloco",
                                 "membro_dominante", "razao_maior_menor"],
                        nomes={"shap_medio_abs": "|SHAP| médio individual",
                               "posicao": "posição",
                               "shap_bloco_somado": "|SHAP| médio do bloco somado",
                               "fracao_do_bloco": "fração do bloco",
                               "membro_dominante": "membro dominante",
                               "razao_maior_menor": "razão maior/menor"}),
        hide_index=True, height="content",
    )
st.markdown(textos.PREDICAO_D053)
st.markdown(textos.DESFECHO_D053)

st.subheader("Árvores excluídas do regime de contraste")
st.warning(textos.FLORESTA_REDUZIDA)
# Uma linha com quinze colunas: transposta, cabe na tela sem rolagem lateral.
tabela_arvores = graficos.tabela(
    arvores[arvores["celula"] == celula], dados.casas("shap_arvores_excluidas"),
    colunas=["regime_principal", "regime_contraste", "arvores_ajustadas",
             "arvores_mantidas", "arvores_excluidas", "corte", "piso_por_arvore",
             "erro_min_mantidas", "erro_max_mantidas", "erro_min_excluidas",
             "erro_max_excluidas", "erro_aditividade_principal",
             "erro_aditividade_contraste", "delta_arvores", "delta_regime"],
    nomes={"regime_principal": "regime principal",
           "regime_contraste": "regime de contraste",
           "arvores_ajustadas": "árvores ajustadas",
           "arvores_mantidas": "árvores mantidas",
           "arvores_excluidas": "árvores excluídas",
           "corte": "corte de exclusão",
           "piso_por_arvore": "piso por árvore",
           "erro_min_mantidas": "erro mínimo das mantidas",
           "erro_max_mantidas": "erro máximo das mantidas",
           "erro_min_excluidas": "erro mínimo das excluídas",
           "erro_max_excluidas": "erro máximo das excluídas",
           "erro_aditividade_principal": "erro de aditividade, regime principal",
           "erro_aditividade_contraste": "erro de aditividade, regime de contraste",
           "delta_arvores": "deslocamento por remoção",
           "delta_regime": "deslocamento por regime"},
)
st.table(tabela_arvores.T.rename(columns={0: celula}))
