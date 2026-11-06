"""Testes de contrato da etapa de dados.

Validam shape, prevalência e padrão de ausência depois da limpeza. São contrato, não
unidade: qualquer mudança em `src/data.py` que mexa na amostra analítica tem de ser
decisão consciente, registrada em `docs/RASTREABILIDADE.md`.
"""

import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src.data import (ausentes_por_variavel, carregar_bruto, derivar, limpar,
                      resumo_amostra, separar)

# Contrato de saída da etapa de dados.
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
    """Registro sem HbA1c válida é excluído, nunca imputado."""
    assert df[cfg.ALVO_CONTINUO].isna().sum() == 0
    assert set(df["target"].unique()) == {0, 1}


def test_decimal_brasileiro_convertido(df):
    """Conversão numérica: nenhuma contínua pode sobrar como texto."""
    for col in cfg.CONTINUAS + [cfg.ALVO_CONTINUO, cfg.PESO, "imc", "rcest"]:
        assert pd.api.types.is_float_dtype(df[col]), col
    # Há valores fracionários de fato, o que prova que a vírgula foi tratada.
    assert (df["hba1c"] % 1 != 0).any()


def test_raca_cor_ignorado_decodificado(df):
    """Código 9 ('Ignorado') é não resposta, não categoria."""
    assert (df["raca_cor"] == cfg.COD_IGNORADO_RACA).sum() == 0
    assert set(df["raca_cor"].dropna().unique()) <= {1, 2, 3, 4, 5}


def test_sem_duplicados(df):
    assert df.duplicated().sum() == 0


def test_derivadas_row_wise(df):
    """`imc` e `rcest` dependem só da própria linha."""
    esperado_imc = df["peso_kg"] / (df["altura_cm"] / 100.0) ** 2
    esperado_rcest = df["circ_cintura"] / df["altura_cm"]
    assert np.allclose(df["imc"], esperado_imc)
    assert np.allclose(df["rcest"], esperado_rcest)


def test_derivar_em_linha_isolada_reproduz_limpar(df):
    """A API chama `derivar` sobre um registro só, com valores numéricos.

    O IMC e a RCEst desse registro têm de ser **idênticos** aos que `limpar` produziu
    para a mesma pessoa na base, sem tolerância: é o mesmo número que o modelo viu no
    treino.
    """
    for i in df.sample(n=50, random_state=cfg.RANDOM_STATE).index:
        linha = derivar(df.loc[[i], cfg.CONTINUAS + cfg.CATEGORICAS])
        assert linha.at[i, "imc"] == df.at[i, "imc"], i
        assert linha.at[i, "rcest"] == df.at[i, "rcest"], i


def test_derivar_decodifica_raca_ignorada_em_linha_isolada():
    """`raca_cor = 9` vira ausente também na inferência, como no treino."""
    registro = pd.DataFrame([{
        "idade": 50, "peso_kg": 80.0, "altura_cm": 160.0, "circ_cintura": 96.0,
        "pas": 130.0, "pad": 85.0, "sexo": 2, "raca_cor": 9, "regiao": 3,
        "escolaridade": 5, "saude_autoaval": 2, "plano_saude": 2,
        "has_autorrelato": 3, "colesterol_alto": 2,
    }])
    linha = derivar(registro).iloc[0]
    assert pd.isna(linha["raca_cor"])
    assert linha["imc"] == 80.0 / 1.6 ** 2
    assert linha["rcest"] == 96.0 / 160.0


def test_nenhuma_imputacao_em_data(df):
    """Imputação é pós-split: `data.py` não pode ter preenchido nada."""
    assert df[list(AUSENTES_ESPERADOS)].isna().sum().sum() == sum(
        AUSENTES_ESPERADOS.values()
    )


def test_separar_isola_peso_lab(df):
    """`peso_lab` pondera a EDA e não entra em X."""
    X, y, peso_lab = separar(df)
    assert cfg.PESO not in X.columns
    assert list(X.columns) == cfg.PREDITORAS
    assert len(X) == len(y) == len(peso_lab) == N_ESPERADO
    assert peso_lab.notna().all()


def test_hba1c_nao_e_preditora(df):
    """HbA1c é só desfecho, nunca preditor."""
    X, _, _ = separar(df)
    assert cfg.ALVO_CONTINUO not in X.columns
    assert cfg.ALVO not in X.columns


def test_variaveis_comportamentais_ausentes(df):
    """Dieta, atividade física e tabagismo seguem fora do recorte."""
    proibidas = {"dieta", "atividade_fisica", "tabagismo", "fumo"}
    assert proibidas.isdisjoint(df.columns)


# ---------------------------------------------------------------------------
# Quadro descritivo da amostra (insumo do notebook de resultados)
# ---------------------------------------------------------------------------

def test_exclusoes_registradas_na_limpeza(df):
    """As contagens vêm de onde o corte acontece, não de recontagem posterior."""
    exclusoes = df.attrs["exclusoes"]
    assert exclusoes["linhas_brutas"] == 8952
    assert exclusoes["sem_hba1c"] == 411
    assert exclusoes["sem_bloco_medicao"] == 90
    # As exclusões não se sobrepõem: a soma tem de fechar com a amostra analítica.
    total_excluido = (exclusoes["duplicadas"] + exclusoes["sem_hba1c"]
                      + exclusoes["sem_bloco_medicao"])
    assert exclusoes["linhas_brutas"] - total_excluido == N_ESPERADO


def test_resumo_amostra_fecha_a_aritmetica(df):
    resumo = resumo_amostra(df)
    # As quatro primeiras linhas somam a amostra analítica.
    assert int(resumo["n"].iloc[:4].sum()) == N_ESPERADO
    assert int(resumo.loc[resumo["item"] == "Amostra analítica", "n"].iloc[0]) == N_ESPERADO


def test_resumo_amostra_reporta_as_duas_prevalencias(df):
    """Amostral e ponderada por `peso_lab` medem coisas diferentes."""
    resumo = resumo_amostra(df)
    proporcoes = resumo.dropna(subset=["proporcao"])
    assert len(proporcoes) == 2
    amostral, ponderada = proporcoes["proporcao"].tolist()
    assert amostral == pytest.approx(df["target"].mean())
    peso = df[cfg.PESO]
    assert ponderada == pytest.approx((peso * df["target"]).sum() / peso.sum())
    # Têm de ser distintas: iguais significaria ponderação não aplicada.
    assert amostral != pytest.approx(ponderada, abs=1e-6)


def test_resumo_amostra_exige_attrs_da_limpeza():
    """Sem os `attrs`, o quadro falha em vez de recalcular as exclusões por conta."""
    sem_attrs = pd.DataFrame({cfg.ALVO: [0, 1], cfg.PESO: [1.0, 2.0]})
    with pytest.raises(KeyError):
        resumo_amostra(sem_attrs)


def test_ausentes_por_variavel_bate_com_o_contrato(df):
    tabela = ausentes_por_variavel(df)
    assert dict(zip(tabela["variavel"], tabela["ausentes"])) == AUSENTES_ESPERADOS
    # Ordem decrescente: quem mais pesa na imputação aparece primeiro.
    assert tabela["ausentes"].is_monotonic_decreasing


def test_ausentes_por_variavel_com_treino(df):
    """A coluna do treino é a que importa: é onde o KNN é ajustado."""
    X, _, _ = separar(df)
    treino = X.iloc[:6760]
    tabela = ausentes_por_variavel(df, treino)
    assert "no_treino" in tabela.columns
    assert (tabela["no_treino"] <= tabela["ausentes"]).all()


def test_resumo_amostra_com_particao_acrescenta_as_linhas_no_fim(df):
    """As linhas da partição entram depois das de sempre, que saem idênticas."""
    sem = resumo_amostra(df)
    y = df[cfg.ALVO]
    y_tr, y_te = y.iloc[:6760], y.iloc[6760:]
    com = resumo_amostra(df, y_tr=y_tr, y_te=y_te)

    pd.testing.assert_frame_equal(com.iloc[:len(sem)], sem)
    novas = com.iloc[len(sem):].set_index("item")
    assert list(novas.index) == ["Treino", "Teste", "Negativos",
                                 "Prevalência no treino", "Prevalência no teste"]
    assert novas.loc["Treino", "n"] + novas.loc["Teste", "n"] == N_ESPERADO
    assert novas.loc["Negativos", "n"] == N_ESPERADO - int(y.sum())
    assert novas.loc["Prevalência no treino", "proporcao"] == pytest.approx(y_tr.mean())
    assert novas.loc["Prevalência no teste", "proporcao"] == pytest.approx(y_te.mean())
