"""Guarda do documento de resultados.

`notebooks/03_resultados.ipynb` é o artefato mais exposto do projeto e era o único
sem proteção. Ele é commitado **com outputs**, de propósito, porque é a fonte das
figuras do LaTeX e o documento que sustenta a arguição. É exatamente essa propriedade
que o torna perigoso: os outputs ficam congelados no arquivo enquanto `src/` continua
mudando, e um notebook que mostra números de duas semanas atrás não acusa nada, só
parece certo.

Três guardas, cada uma falhando por um motivo diferente:

1. **Outputs gravados** (`test_outputs_versionados_*`): barata, lê o `.ipynb` do
   disco sem executar nada. Pega notebook commitado com erro dentro, sem outputs ou
   com a conferência de reprodução acusando divergência.
2. **Reexecução** (`test_reexecucao_reproduz_a_tabela_consolidada`, marcada `lento`):
   cara, roda o notebook inteiro via nbconvert e compara o que ele emite com
   `reports/tabelas/comparativo_consolidado.tex` **como está versionado**. É a que
   pega notebook desatualizado em relação ao `src/` atual.
3. **Ausência de modelagem** (`test_notebook_nao_implementa_modelagem`): barata,
   sustenta a convenção que separa orquestração de implementação.

A segunda lê o `.tex` antes de executar, porque a execução o regenera: comparar o
notebook com um arquivo que ele mesmo acabou de escrever passaria sempre. Ela também
confere a **forma** da tabela emitida antes de compará-la, já que
`conferir_reproducao` só olha o que existe nos dois lados, e uma tabela com menos
linhas passaria por vacuidade.

O conteúdo de `reports/tabelas/` é restaurado pela fixture `tabelas_preservadas`, do
`conftest.py`. Não é zelo: `experimento.executar` exporta, e um teste que roda um
recorte do grid deixa a tabela consolidada com menos linhas para quem vier depois.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from src import config as cfg
from src import experimento as xp

NOTEBOOK = cfg.RAIZ / "notebooks" / "03_resultados.ipynb"
CONSOLIDADO = cfg.PATH_TABELAS / f"{xp.NOME_CONSOLIDADO}.tex"

# Folga generosa: a execução completa leva alguns minutos e um timeout apertado
# transformaria máquina lenta em falha de contrato.
TIMEOUT_CELULA = 900
TIMEOUT_PROCESSO = 1800


nbformat = pytest.importorskip("nbformat", reason="nbformat ausente")


def _ler_notebook():
    if not NOTEBOOK.exists():
        pytest.skip(f"Notebook ausente em {NOTEBOOK}.")
    return nbformat.read(NOTEBOOK, as_version=4)


def _celulas_de_codigo(nb):
    return [c for c in nb.cells if c.cell_type == "code"]


def _texto_dos_outputs(celula) -> str:
    """Concatena o que a célula imprimiu ou devolveu, ignorando imagens."""
    partes = []
    for saida in celula.get("outputs", []):
        if saida.output_type == "stream":
            partes.append(saida.get("text", ""))
        elif saida.output_type in ("execute_result", "display_data"):
            partes.append(saida.get("data", {}).get("text/plain", ""))
        elif saida.output_type == "error":
            partes.append("\n".join(saida.get("traceback", [])))
    return "".join(partes)


# ---------------------------------------------------------------------------
# Guarda barata: o que está gravado no arquivo versionado
# ---------------------------------------------------------------------------

def test_outputs_versionados_nao_contem_erro():
    """Notebook commitado com traceback dentro é notebook que ninguém executou."""
    nb = _ler_notebook()
    com_erro = [
        (i, c.source.splitlines()[0] if c.source else "")
        for i, c in enumerate(nb.cells)
        if c.cell_type == "code"
        and any(s.output_type == "error" for s in c.get("outputs", []))
    ]
    assert not com_erro, f"Células com erro nos outputs versionados: {com_erro}"


def test_outputs_versionados_estao_preenchidos():
    """Notebook sem outputs não é o documento de arguição, é o esqueleto dele."""
    nb = _ler_notebook()
    codigo = _celulas_de_codigo(nb)
    assert codigo, "Notebook sem células de código."
    sem_saida = [c for c in codigo if not c.get("outputs")]
    # A célula de imports não produz saída; qualquer outra sem output é sinal de
    # que o notebook foi commitado sem ter sido executado por inteiro.
    assert len(sem_saida) <= 1, (
        f"{len(sem_saida)} células de código sem output. Rode `make notebook` "
        f"antes de commitar."
    )


def test_outputs_versionados_nao_acusam_divergencia():
    """A célula de conferência tem de mostrar DataFrame vazio."""
    nb = _ler_notebook()
    celulas = [c for c in _celulas_de_codigo(nb) if "divergencias" in c.source]
    assert celulas, "Notebook não exibe mais `saida['divergencias']`."
    texto = _texto_dos_outputs(celulas[-1])
    assert "Empty DataFrame" in texto, (
        f"A conferência de reprodução gravada no notebook não está vazia:\n{texto}"
    )


# ---------------------------------------------------------------------------
# Guarda cara: reexecução contra o artefato versionado
# ---------------------------------------------------------------------------

@pytest.mark.lento
def test_reexecucao_reproduz_a_tabela_consolidada(tabelas_preservadas):
    """O notebook, com o `src/` de hoje, emite a tabela que está versionada.

    Falha aqui é uma de duas coisas, e as duas pedem decisão humana: ou `src/` mudou
    e o notebook commitado está desatualizado (rode `make notebook` e confira o diff
    antes de aceitar), ou o resultado mudou de verdade, e aí o `ESTADO.md` precisa
    ser atualizado junto.
    """
    if not cfg.PATH_RAW.exists():
        pytest.skip(f"CSV bruto ausente em {cfg.PATH_RAW} (não versionado).")
    if not CONSOLIDADO.exists():
        pytest.skip(f"Artefato de referência ausente em {CONSOLIDADO}.")
    pytest.importorskip("nbconvert", reason="nbconvert ausente")

    # Lido ANTES da execução: o notebook regrava este arquivo, e comparar contra o
    # que ele acabou de escrever passaria sempre.
    with tempfile.TemporaryDirectory() as tmp:
        referencia = Path(tmp) / "referencia.tex"
        referencia.write_text(CONSOLIDADO.read_text(encoding="utf-8"),
                              encoding="utf-8")

        processo = subprocess.run(
            [sys.executable, "-m", "jupyter", "nbconvert",
             "--execute", "--to", "notebook",
             f"--ExecutePreprocessor.timeout={TIMEOUT_CELULA}",
             "--output-dir", tmp, "--output", "executado.ipynb",
             str(NOTEBOOK)],
            capture_output=True, text=True, timeout=TIMEOUT_PROCESSO,
            cwd=str(cfg.RAIZ),
        )
        assert processo.returncode == 0, (
            f"O notebook não executou de ponta a ponta:\n{processo.stderr[-3000:]}"
        )

        emitida = xp._ler_referencia(CONSOLIDADO)

        # Conferido antes da comparação e sobre a tabela recém-emitida: uma tabela
        # com menos linhas passaria por vacuidade, porque `conferir_reproducao` só
        # confere o que existe nos dois lados. Isso é o comportamento certo para a
        # âncora da Sprint 1 e o errado para esta guarda.
        assert set(emitida["modelo"]) == set(xp.MODELOS_PADRAO), (
            f"Modelos emitidos: {sorted(set(emitida['modelo']))}; "
            f"esperados: {sorted(xp.MODELOS_PADRAO)}"
        )
        # Logística nos três conjuntos, floresta nos dois que sobreviveram ao
        # diagnóstico de colinearidade, mais uma linha por comparador invariante. A
        # contagem vem da composição declarada, e não de um número escrito aqui: número
        # mágico em teste envelhece calado quando o grid muda.
        esperada = xp.composicao_esperada()
        assert len(emitida) == len(esperada), (
            f"Esperadas {len(esperada)} linhas, obtidas {len(emitida)}"
        )
        faltando = xp.conferir_composicao(emitida, esperada)
        assert faltando.empty, (
            f"Composição errada na tabela do notebook:\n"
            f"{faltando.to_string(index=False)}"
        )
        invariantes = emitida[emitida["conjunto"] == xp.SEM_CONJUNTO]
        assert len(invariantes) == 2, (
            "Comparadores invariantes deveriam render uma linha cada."
        )

        divergencias = xp.conferir_reproducao(emitida, referencia=referencia)

    assert divergencias.empty, (
        "A tabela emitida pelo notebook diverge da versionada em "
        f"{CONSOLIDADO.name}:\n{divergencias.to_string(index=False)}"
    )


def test_notebook_nao_implementa_modelagem():
    """notebooks orquestram, `src/` implementa.

    Não é questão de estilo. Lógica de modelagem escrita na célula não tem teste, não
    dá para chamar pela CLI e some quando o notebook é reexecutado. Foi assim que os
    resultados da Sprint 1 deixaram de ser reproduzíveis.
    """
    nb = _ler_notebook()
    proibidos = [
        "train_test_split", "LogisticRegression", "RandomForestClassifier",
        "StratifiedKFold", "cross_val_predict", ".fit(", "predict_proba",
        "class_weight", "KNNImputer", "ColumnTransformer",
    ]
    encontrados = [
        (i, termo)
        for i, c in enumerate(nb.cells) if c.cell_type == "code"
        for termo in proibidos if termo in c.source
    ]
    assert not encontrados, (
        f"Lógica de modelagem no notebook: {encontrados}. "
        f"Se não puder ser chamado de `src/`, o problema está em `src/`."
    )
