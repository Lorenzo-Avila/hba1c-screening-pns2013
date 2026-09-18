"""Contratos da busca de hiperparâmetros da Random Forest.

Quatro guardas, e a primeira é a que paga o preço de uma escolha deliberada.
`models.random_forest` traz o vencedor da busca como literal copiado à mão, em vez de
ler `busca_rf.csv` em tempo de execução, porque configuração publicada não pode depender
de um arquivo que não é código (D-029). A contrapartida é que literal e artefato podem
sair de sincronia em silêncio: quem rerodar a busca e esquecer de atualizar o módulo não
vê nada quebrar. Esta suíte é o alarme, e é barato.

As outras três cobrem o que a busca precisa acertar para o resultado valer alguma coisa:
o pipeline inteiro dentro da validação cruzada, para o imputador por KNN não enxergar os
cinco folds de uma vez (R2, R3); o piso do tamanho de folha, que mantém a curva
precision-recall com resolução suficiente para a seleção de limiar; e o determinismo da
semente.

Nenhuma delas escreve em `reports/tabelas/`, então a fixture `tabelas_preservadas` de
D-032 não aparece aqui. A primeira guarda só lê o artefato; a de determinismo roda a
busca com `n_iter` de brinquedo e não exporta nada.
"""

from __future__ import annotations

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src import config as cfg
from src import features as ft
from src import models as md
from src import tuning as tn

# Só os hiperparâmetros que a busca de fato escolheu. `n_jobs` fica de fora porque não é
# capacidade: a floresta usa um núcleo dentro da busca, para não aninhar paralelismo, e
# todos fora dela. `random_state` fica de fora porque é R10, não resultado de busca.
BUSCADOS = ["max_depth", "min_samples_leaf", "max_features", "class_weight"]

CONJUNTO_CONGELADO = "B_composto"


def _texto(valor) -> str:
    """Normaliza para comparar literal do código com célula do CSV.

    O artefato grava `max_depth` sem limite como a string `"None"` e devolve os inteiros
    como `int64` do numpy. Comparar por texto evita ter que reconstruir tipo por tipo.
    """
    return str(valor)


@pytest.fixture(scope="module")
def vencedor_no_artefato():
    """A linha de melhor PR-AUC do conjunto cuja configuração foi congelada."""
    caminho = cfg.PATH_TABELAS / f"{tn.NOME_ARTEFATO}.csv"
    if not caminho.exists():
        pytest.skip(f"Artefato da busca ausente em {caminho}.")
    return tn.vencedor(tn.carregar_busca(caminho), CONJUNTO_CONGELADO)


def test_literal_congelado_bate_com_o_vencedor_da_busca(vencedor_no_artefato):
    """O que `models.random_forest` devolve é o que a busca escolheu.

    Falha aqui significa uma de duas coisas, e as duas pedem decisão humana: ou a busca
    foi rerodada e o literal ficou para trás, ou alguém editou o literal sem passar pela
    busca. Nos dois casos a configuração publicada deixou de ter lastro no artefato.
    """
    congelado = md.random_forest().get_params()

    for parametro in BUSCADOS:
        esperado = _texto(vencedor_no_artefato[parametro])
        obtido = _texto(congelado[parametro])
        assert obtido == esperado, (
            f"{parametro}: o artefato da busca diz {esperado!r} e "
            f"models.random_forest() devolve {obtido!r}. Ou o literal ficou para trás "
            f"de uma reexecução da busca, ou foi editado fora dela."
        )

    assert congelado["n_estimators"] == tn.N_ESTIMATORS, (
        f"n_estimators: esperado {tn.N_ESTIMATORS} (convergência, valor fixo fora do "
        f"espaço de busca) e obtido {congelado['n_estimators']}."
    )


def test_busca_roda_sobre_o_pipeline_inteiro():
    """A busca ajusta o preprocessador dentro de cada fold, não antes deles.

    O sintoma de uma busca sobre matriz pré-transformada seria o espaço apontar direto
    para a floresta, sem estágio de preprocessamento pela frente. Aí o `ImputadorPNS`
    veria os cinco folds de uma vez e o score sairia otimista por vazamento.
    """
    pipeline = tn.montar_pipeline(CONJUNTO_CONGELADO)
    assert isinstance(pipeline, Pipeline)

    nomes = [nome for nome, _ in pipeline.steps]
    assert nomes == ["pre", "clf"]

    preprocessador = pipeline.named_steps["pre"]
    referencia = ft.construir_preprocessador(CONJUNTO_CONGELADO)
    assert type(preprocessador) is type(referencia)
    assert [n for n, _ in preprocessador.steps] == [n for n, _ in referencia.steps]

    assert isinstance(pipeline.named_steps["clf"], RandomForestClassifier)
    assert all(chave.startswith("clf__") for chave in tn.ESPACO_BUSCA), (
        "Todo parâmetro do espaço tem que endereçar o estágio da floresta. Chave sem "
        "prefixo é sinal de busca montada sobre o estimador cru."
    )


def test_folha_unitaria_esta_fora_do_espaco():
    """O piso de 5 no tamanho da folha não é detalhe, é resolução do limiar.

    Com 474 positivos no treino, folha unitária produz probabilidade quase degenerada e
    curva precision-recall em degraus grossos, o que estraga a seleção de limiar por
    recall-alvo que roda depois (D-004, D-033).
    """
    valores = tn.ESPACO_BUSCA["clf__min_samples_leaf"]
    assert 1 not in valores, f"folha unitária de volta ao espaço: {valores}"
    assert min(valores) >= 5


@pytest.mark.lento
def test_mesma_semente_devolve_o_mesmo_vencedor():
    """Duas execuções da busca com a mesma semente escolhem a mesma configuração.

    Roda com `n_iter` de brinquedo: o que está sob teste é a semente governar sorteio,
    embaralhamento dos folds e bootstrap da floresta, e isso aparece com quatro
    candidatos tão bem quanto com sessenta.
    """
    if not cfg.PATH_RAW.exists():
        pytest.skip(f"CSV bruto ausente em {cfg.PATH_RAW} (não versionado).")

    from src import experimento as xp

    particao = xp.preparar_particao()
    tabelas = [
        tn.tabela_da_busca(
            tn.buscar(CONJUNTO_CONGELADO, particao, n_iter=4), CONJUNTO_CONGELADO
        )
        for _ in range(2)
    ]

    vencedores = [tn.vencedor(tabela, CONJUNTO_CONGELADO) for tabela in tabelas]
    assert ([_texto(vencedores[0][p]) for p in BUSCADOS]
            == [_texto(vencedores[1][p]) for p in BUSCADOS])
    assert vencedores[0]["pr_auc_media"] == pytest.approx(vencedores[1]["pr_auc_media"])
