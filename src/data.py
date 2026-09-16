"""Carga, limpeza e exclusões da Subamostra Laboratorial da PNS 2013.

Somente operações row-wise: nada neste módulo olha para mais de uma
linha ao mesmo tempo. Nenhuma imputação acontece aqui — imputação é pós-split
e vive em `src/features.py`.

Registros sem HbA1c válida são excluído, jamais imputados:
imputar o alvo comprometeria a validade do desfecho.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


def _para_numero(serie: pd.Series) -> pd.Series:
    """Converte série de texto para numérico tratando o decimal brasileiro.

    Nos microdados da PNS os valores são gravados com vírgula decimal
    (`'38,00'`, `'0,58'`). Ler sem tratar faria o Pandas manter tudo como texto
    e quebrar o pipeline silenciosamente. Células vazias viram `NaN`.
    """
    limpa = serie.astype("string").str.strip().str.replace(",", ".", regex=False)
    limpa = limpa.replace("", pd.NA)
    return pd.to_numeric(limpa, errors="coerce")


def carregar_bruto(path=None) -> pd.DataFrame:
    """Lê o recorte de 16 variáveis + alvo + peso amostral do CSV bruto.

    Leitura seletiva, `dtype=str` e `keep_default_na=False` para que a conversão
    de tipo seja explícita e auditável na limpeza.

    Args:
        path: caminho do CSV bruto. Se omitido, usa `config.PATH_RAW`.

    Returns:
        DataFrame com as colunas já renomeadas pelo `DE_PARA`, ainda como texto.
    """
    path = cfg.PATH_RAW if path is None else path
    df = pd.read_csv(path, usecols=cfg.COLS_ORIGEM, dtype=str, keep_default_na=False)
    return df.rename(columns=cfg.DE_PARA)


def limpar(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica a limpeza row-wise na ordem obrigatória da §3 da spec.

    1. Conversão de tipos — contínuas `float64`, categóricas `Int64` nullable.
    2. Decodificação de ausência — `raca_cor == 9` ("Ignorado") vira `NaN`.
    3. `drop_duplicates()`.
    4. Exclusões: alvo ausente e bloco de medição 100% ausente.
    5. Derivadas row-wise: `imc` (Quetelet) e `rcest` (§6.4.1).
    6. Binarização do desfecho pelo critério ADA 2010.

    Nenhuma imputação: toda imputação é pós-split (R2).
    """
    df = df.copy()

    for col in cfg.CONTINUAS + [cfg.ALVO_CONTINUO, cfg.PESO]:
        df[col] = _para_numero(df[col]).astype("float64")
    for col in cfg.CATEGORICAS:
        df[col] = _para_numero(df[col]).astype("Int64")

    # "Ignorado" não é categoria, é não resposta.
    df["raca_cor"] = df["raca_cor"].replace(cfg.COD_IGNORADO_RACA, pd.NA)

    df = df.drop_duplicates().reset_index(drop=True)

    # Sem HbA1c válida, o registro sai da base (R3).
    sem_alvo = df[cfg.ALVO_CONTINUO].isna()
    sem_medicao = df[cfg.MEDICAO].isna().all(axis=1)
    df = df[~sem_alvo & ~sem_medicao].reset_index(drop=True)

    df["imc"] = df["peso_kg"] / (df["altura_cm"] / 100.0) ** 2   # Quetelet
    df["rcest"] = df["circ_cintura"] / df["altura_cm"]

    # Desfecho: hiperglicemia crônica clinicamente significativa (§6.3.1).
    df[cfg.ALVO] = (df[cfg.ALVO_CONTINUO] >= cfg.LIMIAR_HBA1C).astype(int)

    return df


def separar(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Separa preditoras, desfecho e peso amostral.

    `peso_lab` sai de `X` e é devolvido em separado: pondera exclusivamente a EDA
    descritiva e não entra no treino de nenhum estimador.

    Returns:
        Tupla `(X, y, peso_lab)`.
    """
    X = df[cfg.PREDITORAS].copy()
    y = df[cfg.ALVO].copy()
    peso_lab = df[cfg.PESO].copy()
    return X, y, peso_lab


def carregar_analitico(path=None) -> pd.DataFrame:
    """Atalho de orquestração: `carregar_bruto` seguido de `limpar`."""
    return limpar(carregar_bruto(path))
