"""Fábrica dos estimadores avaliados, em fonte única.

Antes deste módulo, cada notebook e cada teste montava a sua própria
`LogisticRegression`. Configuração replicada é divergência silenciosa esperando
acontecer: basta um `max_iter` diferente entre dois lugares para duas execuções do
mesmo experimento darem números distintos sem que nada acuse.

Cada parâmetro não-default traz, no comentário ao lado, a origem metodológica que o
justifica. Parâmetro sem lastro é parâmetro a questionar.

**Os comparadores não-ML não são construídos aqui.** A fonte deles é
`benchmark.construir_comparadores`, e duplicá-los neste módulo reintroduziria
justamente o problema que ele resolve. `estimadores()` delega.

Cada estimador declara se depende do conjunto de atributos. `RegraClinicaADA` lê
idade e IMC direto das preditoras cruas e o `DummyClassifier` ignora `X`: rodá-los
sobre A, B e C daria três linhas idênticas numa tabela que reporta uma. A marca
`invariante_ao_conjunto` é o que `experimento.rodar_grid` usa para não inflar o
produto cartesiano.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import benchmark as bm
from . import config as cfg
from . import features as ft


@dataclass(frozen=True)
class Estimador:
    """Um estimador do grid, com o que o harness precisa saber sobre ele.

    Attributes:
        nome: identificação na tabela comparativa. É a chave de junção com o
            `ESTADO.md`, então mudar aqui muda o resultado publicado.
        construir: fábrica sem argumentos. Função, e não instância pronta, para
            cada célula do grid receber um objeto novo. `avaliar` clona antes de
            ajustar, mas depender disso seria confiar em detalhe interno.
        invariante_ao_conjunto: `True` quando a saída não muda com o conjunto de
            atributos. Esses estimadores rodam **uma única vez**, com
            `conjunto="—"`, e não recebem preprocessador.
        limiar_fixo: quando presente, dispensa a calibração por recall-alvo. Vale
            para saídas indicadoras em {0, 1}, onde a curva precision-recall tem
            dois pontos e não define ponto de operação.
    """

    nome: str
    construir: Callable[[], BaseEstimator]
    invariante_ao_conjunto: bool = False
    limiar_fixo: float | None = None


def logistica() -> LogisticRegression:
    """Baseline interpretável do projeto.

    O produto metodológico aqui são os coeficientes, não só o classificador: é
    sobre eles que o `LinearExplainer` do SHAP opera na Sprint 3.
    """
    return LogisticRegression(
        class_weight="balanced",       # Cost-Sensitive Learning no lugar do SMOTE
        max_iter=1000,                 # convergência do lbfgs sobre a matriz padronizada
        random_state=cfg.RANDOM_STATE,
    )


def random_forest() -> RandomForestClassifier:
    """Segundo algoritmo do projeto, configurado pela busca de hiperparâmetros.

    Os valores saíram da busca aleatória de `tuning.py`, vencedor de `B_composto`
    aplicado também a `A_atomico` por D-036. Estão aqui como literais copiados à
    mão: ler o vencedor do CSV em tempo de execução faria a configuração publicada
    depender de um arquivo que não é código, e uma reexecução da busca mudaria o
    modelo sem commit e sem diff. `tests/test_tuning.py` é quem confere que o
    literal e o artefato continuam de acordo.
    """
    return RandomForestClassifier(
        class_weight="balanced_subsample",   # Cost-Sensitive Learning por R4, variante de D-034
        n_estimators=500,                    # convergência, não capacidade: fixo fora do espaço, ver D-033
        max_depth=10,                        # vencedor da busca, ver busca_rf.csv e D-036
        min_samples_leaf=20,                 # vencedor da busca; piso de 5 no espaço, ver D-033
        max_features="sqrt",                 # vencedor da busca, ver busca_rf.csv e D-036
        n_jobs=-1,                           # roda fora da busca, então não aninha paralelismo
        random_state=cfg.RANDOM_STATE,
    )


def estimadores() -> dict[str, Estimador]:
    """Os quatro estimadores do grid, configurados e marcados.

    Returns:
        Dicionário `nome -> Estimador`, na ordem em que as linhas aparecem na
        tabela consolidada: os dois modelos, depois os dois pisos.

    Os comparadores vêm de `benchmark.construir_comparadores()`. Este módulo só os
    embrulha no `Estimador` e acrescenta as marcas que o grid consome; a
    configuração deles continua sendo do `benchmark`.
    """
    comparadores = bm.construir_comparadores()

    grid = {
        "LogReg": Estimador("LogReg", logistica),
        "Random Forest": Estimador("Random Forest", random_forest),
    }
    for nome, pronto in comparadores.items():
        grid[nome] = Estimador(
            nome=nome,
            # `type(pronto)(**params)` reconstrói com a configuração que o
            # benchmark definiu, sem congelar a mesma instância entre células.
            construir=(lambda p=pronto: type(p)(**p.get_params())),
            invariante_ao_conjunto=True,
            limiar_fixo=bm.LIMIAR_INDICADORA,
        )
    return grid


def montar(estimador: Estimador, conjunto: str | dict | None = None):
    """Devolve o objeto pronto para `evaluate.avaliar`.

    Quem depende do conjunto sai embrulhado no preprocessador de três estágios; os
    invariantes saem crus, porque operam sobre as preditoras em escala clínica: o
    corte de IMC 25 da regra ADA não sobrevive a uma padronização.

    Raises:
        ValueError: se um estimador dependente do conjunto for montado sem um.
    """
    if estimador.invariante_ao_conjunto:
        return estimador.construir()
    if conjunto is None:
        raise ValueError(
            f"{estimador.nome!r} depende do conjunto de atributos; informe um."
        )
    return Pipeline([
        ("pre", ft.construir_preprocessador(conjunto)),
        ("clf", estimador.construir()),
    ])
