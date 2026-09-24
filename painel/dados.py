"""Leitura dos artefatos de `reports/tabelas/`.

Único módulo do painel que abre arquivo. Cada artefato declara as colunas que as
telas usam e as casas decimais do `.tex` correspondente. Quando existir a API da
fase 6, este arquivo vira o cliente HTTP e as telas não mudam (D-054).
"""

from pathlib import Path

import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parent.parent / "reports" / "tabelas"

# Colunas esperadas por artefato e casas decimais de cada coluna numérica,
# copiadas do `.tex` irmão. "e3" indica notação científica com três casas.
ARTEFATOS = {
    "resumo_amostra": {
        "colunas": ["item", "n", "proporcao"],
        "casas": {"n": 0, "proporcao": 3},
    },
    "ausentes_por_variavel": {
        "colunas": ["variavel", "ausentes", "proporcao", "no_treino"],
        "casas": {"ausentes": 0, "proporcao": 3, "no_treino": 0},
    },
    "pearson_coeficientes": {
        "colunas": ["feature_set", "var_1", "var_2", "pearson", "abs_pearson"],
        "casas": {"pearson": 2, "abs_pearson": 2},
    },
    "vif_comparativo": {
        "colunas": ["feature_set", "variavel", "vif", "r2_auxiliar",
                    "acima_do_limiar"],
        "casas": {"vif": 2, "r2_auxiliar": 2},
    },
    "vif_resumo": {
        "colunas": ["feature_set", "colunas", "vif_maximo", "vif_mediano",
                    "acima_do_limiar", "limiar_vif"],
        "casas": {"colunas": 0, "vif_maximo": 2, "vif_mediano": 2,
                  "acima_do_limiar": 0, "limiar_vif": 0},
    },
    "vif_ablacao_rcest": {
        "colunas": ["variavel", "vif_antes", "r2_antes", "acima_antes",
                    "vif_depois", "r2_depois", "acima_depois", "delta_vif"],
        "casas": {"vif_antes": 2, "r2_antes": 2, "vif_depois": 2,
                  "r2_depois": 2, "delta_vif": 2},
    },
    "decomposicao_escore": {
        "colunas": ["feature_set", "r2_identidade", "r2_vif", "vif", "parcelas"],
        "casas": {"r2_identidade": 3, "r2_vif": 3, "vif": 3},
    },
    "mcnemar_pareado": {
        "colunas": ["modelo_a", "modelo_b", "n", "concordam", "so_a_acerta",
                    "so_b_acerta", "positivos_so_a_detecta",
                    "positivos_so_b_detecta", "p_valor", "ambos_acertam",
                    "ambos_erram", "discordancias", "positivos_teste"],
        "casas": {"n": 0, "concordam": 0, "so_a_acerta": 0, "so_b_acerta": 0,
                  "positivos_so_a_detecta": 0, "positivos_so_b_detecta": 0,
                  "p_valor": 3, "ambos_acertam": 0, "ambos_erram": 0,
                  "discordancias": 0, "positivos_teste": 0},
    },
    "shap_global_atributo": {
        "colunas": ["celula", "regime", "objeto_explicado", "atributo",
                    "shap_medio_abs", "posicao", "n_colunas",
                    "categoria_referencia_omitida"],
        "casas": {"shap_medio_abs": 6, "posicao": 0, "n_colunas": 0},
    },
    "shap_global_coluna": {
        "colunas": ["celula", "regime", "objeto_explicado", "coluna", "atributo",
                    "shap_medio_abs", "posicao", "categoria_referencia_omitida"],
        "casas": {"shap_medio_abs": 6, "posicao": 0},
    },
    "shap_pares_colineares": {
        "colunas": ["celula", "regime", "objeto_explicado", "bloco", "evidencia",
                    "n_membros", "membro", "shap_medio_abs", "posicao",
                    "shap_bloco_somado", "fracao_do_bloco", "membro_dominante",
                    "razao_maior_menor"],
        "casas": {"n_membros": 0, "shap_medio_abs": 6, "posicao": 0,
                  "shap_bloco_somado": 6, "fracao_do_bloco": 6,
                  "razao_maior_menor": 6},
    },
    "shap_arvores_excluidas": {
        "colunas": ["celula", "regime_principal", "regime_contraste",
                    "arvores_ajustadas", "arvores_mantidas", "arvores_excluidas",
                    "corte", "erro_min_mantidas", "erro_max_mantidas",
                    "erro_min_excluidas", "erro_max_excluidas", "piso_por_arvore",
                    "erro_aditividade_principal", "erro_aditividade_contraste",
                    "delta_arvores", "delta_regime"],
        "casas": {"arvores_ajustadas": 0, "arvores_mantidas": 0,
                  "arvores_excluidas": 0, "corte": "e3",
                  "erro_min_mantidas": "e3", "erro_max_mantidas": "e3",
                  "erro_min_excluidas": "e3", "erro_max_excluidas": "e3",
                  "piso_por_arvore": "e3", "erro_aditividade_principal": "e3",
                  "erro_aditividade_contraste": "e3", "delta_arvores": 0,
                  "delta_regime": 0},
    },
    "shap_casos_locais": {
        "colunas": ["celula", "regime", "objeto_explicado", "caso", "indice",
                    "escore", "limiar", "distancia_ao_limiar", "valor_base",
                    "atributo", "contribuicao", "valor_observado"],
        "casas": {"indice": 0, "escore": 6, "limiar": 6,
                  "distancia_ao_limiar": 6, "valor_base": 6, "contribuicao": 6,
                  "valor_observado": 6},
    },
}


class ArtefatoInvalido(Exception):
    """Artefato ausente, não declarado ou sem uma coluna esperada."""


def ler(nome: str, raiz: Path = RAIZ) -> pd.DataFrame:
    """Lê `<raiz>/<nome>.csv` e confere as colunas declaradas em `ARTEFATOS`.

    Falha nomeando o artefato e cada coluna ausente. Colunas a mais passam: o
    painel só se compromete com as que declara.
    """
    if nome not in ARTEFATOS:
        raise ArtefatoInvalido(f"artefato '{nome}' não declarado em painel/dados.py")
    caminho = Path(raiz) / f"{nome}.csv"
    if not caminho.is_file():
        raise ArtefatoInvalido(f"artefato '{nome}': arquivo {caminho} não encontrado")
    tabela = pd.read_csv(caminho)
    ausentes = [c for c in ARTEFATOS[nome]["colunas"] if c not in tabela.columns]
    if ausentes:
        raise ArtefatoInvalido(
            f"artefato '{nome}': coluna(s) esperada(s) ausente(s): {', '.join(ausentes)}"
        )
    return tabela


@st.cache_data(show_spinner=False)
def carregar(nome: str) -> pd.DataFrame:
    """Versão com cache de `ler`, usada pelas telas."""
    return ler(nome)


def casas(nome: str) -> dict:
    """Casas decimais do `.tex` de cada coluna numérica do artefato."""
    return ARTEFATOS[nome]["casas"]
