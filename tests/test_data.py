"""Testes de contrato da etapa de dados (§3 da spec da Sprint 1).

Validam shape, prevalência e padrão de ausência após a limpeza. São testes de
contrato, não de unidade: qualquer mudança em `src/data.py` que altere a amostra
analítica precisa ser uma decisão consciente, registrada em
`docs/RASTREABILIDADE.md`.
"""

import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src.data import carregar_bruto, limpar, separar

# Contrato de saída da §3 da spec.
N_ESPERADO = 8451
PREVALENCIA_ESPERADA = 0.0702
TOLERANCIA_PREVALENCIA = 0.001
AUSENTES_ESPERADOS = {
    "raca_cor": 2,
    "regiao": 4,
    "has_autorrelato": 261,
    "colesterol_alto": 1297,
}


@pytest.fixture(scope="module")
def df():
    """Amostra analítica limpa, carregada uma única vez por módulo."""
    if not cfg.PATH_RAW.exists():
        pytest.skip(f"CSV bruto ausente em {cfg.PATH_RAW} (não versionado).")
    return limpar(carregar_bruto())


def test_tamanho_da_amostra_analitica(df):
    assert len(df) == N_ESPERADO


def test_prevalencia_do_desfecho(df):
    assert df["target"].mean() == pytest.approx(
        PREVALENCIA_ESPERADA, abs=TOLERANCIA_PREVALENCIA
    )


def test_continuas_sem_ausentes(df):
    assert df[cfg.CONTINUAS].isna().sum().sum() == 0


def test_ausentes_remanescentes_nas_categoricas(df):
    observados = df[cfg.CATEGORICAS].isna().sum()
    assert observados[list(AUSENTES_ESPERADOS)].to_dict() == AUSENTES_ESPERADOS
    sem_ausentes = [c for c in cfg.CATEGORICAS if c not in AUSENTES_ESPERADOS]
    assert observados[sem_ausentes].sum() == 0


def test_faixa_etaria(df):
    assert df["idade"].min() == 18
    assert df["idade"].max() == 101


def test_alvo_sem_ausentes_e_binario(df):
    """R3: registros sem HbA1c válida são excluídos, jamais imputados."""
    assert df[cfg.ALVO_CONTINUO].isna().sum() == 0
    assert set(df["target"].unique()) == {0, 1}


def test_decimal_brasileiro_convertido(df):
    """Conversão numérica: nenhuma contínua pode sobrar como texto."""
    for col in cfg.CONTINUAS + [cfg.ALVO_CONTINUO, cfg.PESO, "imc", "rcest"]:
        assert pd.api.types.is_float_dtype(df[col]), col
    # Valores fracionários existem de fato — prova que a vírgula foi tratada.
    assert (df["hba1c"] % 1 != 0).any()


def test_raca_cor_ignorado_decodificado(df):
    """Código 9 ('Ignorado') é não resposta, não categoria."""
    assert (df["raca_cor"] == cfg.COD_IGNORADO_RACA).sum() == 0
    assert set(df["raca_cor"].dropna().unique()) <= {1, 2, 3, 4, 5}


def test_sem_duplicados(df):
    assert df.duplicated().sum() == 0


def test_derivadas_row_wise(df):
    """`imc` e `rcest` são funções apenas da própria linha (R1)."""
    esperado_imc = df["peso_kg"] / (df["altura_cm"] / 100.0) ** 2
    esperado_rcest = df["circ_cintura"] / df["altura_cm"]
    assert np.allclose(df["imc"], esperado_imc)
    assert np.allclose(df["rcest"], esperado_rcest)


def test_nenhuma_imputacao_em_data(df):
    """Imputação é pós-split (R2). `data.py` não pode ter preenchido nada."""
    assert df[list(AUSENTES_ESPERADOS)].isna().sum().sum() == sum(
        AUSENTES_ESPERADOS.values()
    )


def test_separar_isola_peso_lab(df):
    """R10: `peso_lab` pondera a EDA e não entra em X."""
    X, y, peso_lab = separar(df)
    assert cfg.PESO not in X.columns
    assert list(X.columns) == cfg.PREDITORAS
    assert len(X) == len(y) == len(peso_lab) == N_ESPERADO
    assert peso_lab.notna().all()


def test_hba1c_nao_e_preditora(df):
    """R6: HbA1c é exclusivamente desfecho, jamais preditor."""
    X, _, _ = separar(df)
    assert cfg.ALVO_CONTINUO not in X.columns
    assert cfg.ALVO not in X.columns


def test_variaveis_comportamentais_ausentes(df):
    """R5: dieta, atividade física e tabagismo permanecem fora do recorte."""
    proibidas = {"dieta", "atividade_fisica", "tabagismo", "fumo"}
    assert proibidas.isdisjoint(df.columns)
