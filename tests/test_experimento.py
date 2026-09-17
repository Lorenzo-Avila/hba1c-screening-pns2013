"""Contratos da fábrica de estimadores e do grid.

Dois invariantes sustentam a reprodutibilidade e nenhum dos dois é óbvio pela
leitura do código:

1. `models.estimadores` **delega** os comparadores ao `benchmark`. Se algum dia
   alguém os reinstanciar aqui, passam a existir duas configurações para o mesmo
   objeto e elas divergem em silêncio.
2. O produto cartesiano **não é cheio**. Estimador invariante ao conjunto rende
   uma linha, não três. Cartesiano cheio faria a tabela deixar de bater com o
   `ESTADO.md` por artefato do grid: divergência falsa em cima da verificação que
   existe justamente para dar confiança.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src import benchmark as bm
from src import config as cfg
from src import experimento as xp
from src import models as md


# ---------------------------------------------------------------------------
# Fábrica
# ---------------------------------------------------------------------------

def test_estimadores_traz_os_quatro():
    """Logística, Random Forest e os dois pisos não-ML."""
    grid = md.estimadores()
    assert set(grid) == {"LogReg", "Random Forest", *bm.construir_comparadores()}


def test_comparadores_sao_delegados_e_nao_reconstruidos():
    """A configuração dos pisos continua sendo do `benchmark`, não desta fábrica."""
    do_benchmark = bm.construir_comparadores()
    grid = md.estimadores()
    for nome, referencia in do_benchmark.items():
        montado = md.montar(grid[nome], conjunto=None)
        assert type(montado) is type(referencia)
        assert montado.get_params() == referencia.get_params()


def test_pisos_marcados_como_invariantes_e_com_limiar_fixo():
    """A regra lê idade e IMC crus; o Dummy ignora X. Nenhum depende do conjunto."""
    grid = md.estimadores()
    for nome in bm.construir_comparadores():
        assert grid[nome].invariante_ao_conjunto
        assert grid[nome].limiar_fixo == bm.LIMIAR_INDICADORA


def test_modelos_dependem_do_conjunto():
    """LogReg e RF saem embrulhados no preprocessador."""
    grid = md.estimadores()
    for nome in ("LogReg", "Random Forest"):
        assert not grid[nome].invariante_ao_conjunto
        assert grid[nome].limiar_fixo is None
        montado = md.montar(grid[nome], "B_composto")
        assert isinstance(montado, Pipeline)
        assert list(dict(montado.steps)) == ["pre", "clf"]


def test_montar_sem_conjunto_falha_para_quem_depende_dele():
    """Erro explícito em vez de conjunto silenciosamente errado."""
    with pytest.raises(ValueError, match="depende do conjunto"):
        md.montar(md.estimadores()["LogReg"], conjunto=None)


def test_piso_nao_recebe_preprocessador():
    """O corte de IMC 30 da regra ADA não sobrevive a uma padronização."""
    montado = md.montar(md.estimadores()["Regra ADA"], conjunto="B_composto")
    assert not isinstance(montado, Pipeline)


def test_parametros_metodologicos_dos_modelos():
    """Cost-sensitive e semente fixa valem para os dois algoritmos, placeholder ou não."""
    for construir in (md.logistica, md.random_forest):
        params = construir().get_params()
        assert params["class_weight"] == "balanced"
        assert params["random_state"] == cfg.RANDOM_STATE


def test_dummy_nao_e_compartilhado_entre_celulas():
    """Cada célula recebe objeto novo; nada carrega ajuste de outra."""
    grid = md.estimadores()
    a, b = md.montar(grid["Dummy (maioria)"]), md.montar(grid["Dummy (maioria)"])
    assert a is not b
    assert isinstance(a, DummyClassifier)


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def particao():
    if not cfg.PATH_RAW.exists():
        pytest.skip(f"CSV bruto ausente em {cfg.PATH_RAW} (não versionado).")
    return xp.preparar_particao()


@pytest.fixture(scope="module")
def grid_dois_conjuntos(particao):
    """LogReg sobre dois conjuntos mais os dois pisos: 4 linhas, não 6."""
    return xp.rodar_grid(conjuntos=["A_atomico", "B_composto"],
                         particao=particao, salvar_figuras=False)


def test_produto_cartesiano_nao_e_cheio(grid_dois_conjuntos):
    """Dois modelos calibráveis × 2 conjuntos + 2 pisos × 1 = 4 linhas."""
    linhas = [(r["modelo"], r["conjunto"]) for r in grid_dois_conjuntos]
    assert linhas == [
        ("LogReg", "A_atomico"),
        ("LogReg", "B_composto"),
        ("Regra ADA", xp.SEM_CONJUNTO),
        ("Dummy (maioria)", xp.SEM_CONJUNTO),
    ]


def test_pisos_aparecem_uma_vez_so(grid_dois_conjuntos):
    """O sintoma do cartesiano cheio seria linhas idênticas repetidas."""
    for nome in bm.construir_comparadores():
        assert sum(r["modelo"] == nome for r in grid_dois_conjuntos) == 1


def test_pisos_nao_passam_por_calibracao(grid_dois_conjuntos):
    """Calibrar sobre indicadora converteria o Dummy em "sempre positivo"."""
    for r in grid_dois_conjuntos:
        if r["modelo"] in bm.construir_comparadores():
            assert r["limiar_calibrado"] is False
            assert r["limiar"] == bm.LIMIAR_INDICADORA
    dummy = next(r for r in grid_dois_conjuntos if r["modelo"] == "Dummy (maioria)")
    assert dummy["recall"] == 0.0  # o piso continua sendo piso


def test_random_forest_fora_do_padrao():
    """Placeholder declarado não vira número reportado antes da Sprint 2."""
    assert "Random Forest" not in xp.MODELOS_PADRAO


def test_modelo_desconhecido_falha_cedo(particao):
    with pytest.raises(ValueError, match="desconhecido"):
        xp.rodar_grid(modelos=["XGBoost"], particao=particao)


def test_modelo_de_referencia_ignora_os_pisos(grid_dois_conjuntos):
    """O Dummy tem NNS infinito e a regra opera em limiar fixo: nenhum é candidato."""
    escolhido = xp.modelo_de_referencia(grid_dois_conjuntos)
    assert escolhido["modelo"] == "LogReg"
    assert escolhido["limiar_calibrado"] is True


# ---------------------------------------------------------------------------
# Conferência de reprodução
# ---------------------------------------------------------------------------

def test_referencia_da_sprint1_esta_versionada():
    """Sem ela não há âncora: a conferência passaria por vacuidade."""
    assert (cfg.PATH_TABELAS / xp.REFERENCIA).exists()


def test_conferencia_acusa_divergencia_injetada(grid_dois_conjuntos):
    """Uma conferência que nunca reprova não verifica nada."""
    from src import evaluate as ev

    tabela = ev.comparar(grid_dois_conjuntos).copy()
    tabela.loc[tabela["conjunto"] == "A_atomico", "recall"] += 0.01
    divergencias = xp.conferir_reproducao(tabela)
    assert not divergencias.empty
    assert (divergencias["coluna"] == "recall").any()


def test_conferencia_tolera_divergencia_abaixo_da_terceira_casa(grid_dois_conjuntos):
    """O contrato é três casas decimais, não igualdade de ponto flutuante."""
    from src import evaluate as ev

    tabela = ev.comparar(grid_dois_conjuntos).copy()
    tabela["recall"] += 1e-6
    assert xp.conferir_reproducao(tabela).empty


def test_conferencia_compara_so_o_que_a_referencia_cobre(grid_dois_conjuntos):
    """A referência é anterior às colunas novas e só tem as linhas da LogReg."""
    from src import evaluate as ev

    divergencias = xp.conferir_reproducao(ev.comparar(grid_dois_conjuntos))
    assert isinstance(divergencias, pd.DataFrame)
    assert divergencias.empty


# ---------------------------------------------------------------------------
# Carga operacional
# ---------------------------------------------------------------------------

def test_carga_operacional_tem_tres_regimes(particao, grid_dois_conjuntos):
    tabela = xp.carga_operacional(grid_dois_conjuntos, particao)
    assert len(tabela) == 3
    assert tabela["regime"].iloc[0] == "Regra ADA"
    assert "recall equiparado" in tabela["regime"].iloc[1]
    assert "ponto padrão" in tabela["regime"].iloc[2]


def test_regime_equiparado_supera_a_regra_em_nns(particao, grid_dois_conjuntos):
    """O argumento é esse: mais casos por encaminhamento, não menos volume."""
    tabela = xp.carga_operacional(grid_dois_conjuntos, particao)
    regra, equiparado = tabela.iloc[0], tabela.iloc[1]
    assert equiparado["nns"] < regra["nns"]
    assert equiparado["casos_detectados"] > regra["casos_detectados"]


def test_carga_operacional_carrega_a_advertencia(particao, grid_dois_conjuntos):
    """Nenhuma das duas tabelas deve circular sem a nota de volume absoluto."""
    tabela = xp.carga_operacional(grid_dois_conjuntos, particao)
    assert tabela.attrs["nota"] == bm.NOTA_VOLUME_ABSOLUTO
    assert "recall equiparado" in tabela.attrs["leitura"]


def test_carga_operacional_exige_a_regra(particao, grid_dois_conjuntos):
    sem_regra = [r for r in grid_dois_conjuntos if r["modelo"] != "Regra ADA"]
    with pytest.raises(ValueError, match="ausente"):
        xp.carga_operacional(sem_regra, particao)


def test_executar_aceita_e_devolve_a_particao(particao, tabelas_preservadas):
    """O notebook diagnostica colinearidade sobre `X_tr` antes de rodar o grid.

    Sem este parâmetro ele releria o CSV para refazer exatamente o mesmo split, ou,
    pior, reimplementaria a orquestração de `executar` célula a célula.

    `tabelas_preservadas` não é zelo: `executar` **exporta**, e este teste roda um
    recorte do grid (dois modelos, um conjunto). Sem a fixture, ele deixaria
    `comparativo_consolidado.tex` com duas linhas no lugar das cinco, e a guarda do
    notebook falharia por rastro de teste anterior, não por defeito real. Foi
    exatamente o que aconteceu na primeira execução da suíte completa.
    """
    saida = xp.executar(modelos=["LogReg", "Regra ADA"], conjuntos=["B_composto"],
                        particao=particao, conferir=False)
    assert saida["particao"] is particao
