"""Guarda de consistência interna do `docs/RASTREABILIDADE.md`.

O documento carrega o Status de cada decisão em dois lugares: na tabela de índice e no
corpo da própria decisão. Duas fontes para o mesmo dado divergem, e divergiram: ao
fechar a Sprint 2 apareceram cinco rótulos errados no índice, entre eles D-034 listada
como divergência quando o corpo a classifica como execução de protocolo. Os cinco foram
corrigidos à mão, e correção à mão sem guarda volta a errar (D-049).

**Isto não congela resultado experimental.** Nenhum número do arquivo muda por causa de
medição; o que o teste verifica é que o documento concorda consigo mesmo. É a mesma
natureza da guarda do notebook, que compara um artefato versionado com outro (D-032).

A comparação é pela **classificação**, que é o que os dois lados escrevem igual. O
qualificador depois dela não entra: o índice resume ("Execução de protocolo, não
acionado") e o corpo detalha ("EXECUÇÃO DE PROTOCOLO"), e exigir coincidência literal
transformaria a guarda num teste de redação que falha a cada ajuste de frase.

O arquivo está no `.gitignore` por opção do autor, então o módulo inteiro pula com
motivo quando ele não existe, em vez de reprovar uma cópia limpa do repositório.
"""

from __future__ import annotations

import re

import pytest

from src import config as cfg

CAMINHO = cfg.RAIZ / "docs" / "RASTREABILIDADE.md"

pytestmark = pytest.mark.skipif(
    not CAMINHO.exists(),
    reason=(f"{CAMINHO.name} está no .gitignore e não existe nesta cópia; "
            f"nada a conferir."),
)

# O cabeçalho da seção de índice. Delimitar a varredura importa: o corpo de D-041 traz
# uma tabela de conferência que também começa por `| D-0NN |`, e ela não é índice.
TITULO_INDICE = "## Índice e pendências"

# Linha de índice: exatamente três células, `| ID | assunto | Status |`.
LINHA_INDICE = re.compile(r"^\|\s*(D-\d{3})\s*\|[^|]*\|\s*([^|]+?)\s*\|\s*$")

TITULO_CORPO = re.compile(r"^###\s+(D-\d{3})\b")
STATUS_CORPO = re.compile(r"^\*\*Status:\*\*\s*(.+?)\s*$")


def _classificacao(status: str) -> str:
    """A classificação, sem o qualificador que cada lado escreve do seu jeito.

    Corta no primeiro travessão ou parêntese e normaliza a caixa: o índice escreve
    "Conforme" e o corpo "CONFORME (correção de implementação)", e os dois dizem a
    mesma coisa.
    """
    return re.split(r"\s*[—–(]\s*", status, maxsplit=1)[0].strip().upper()


@pytest.fixture(scope="module")
def linhas() -> list[str]:
    return CAMINHO.read_text(encoding="utf-8").splitlines()


@pytest.fixture(scope="module")
def indice(linhas) -> dict[str, str]:
    """Os Status como a tabela de índice os declara."""
    try:
        inicio = next(i for i, l in enumerate(linhas) if l.startswith(TITULO_INDICE))
    except StopIteration:
        pytest.fail(f"Seção {TITULO_INDICE!r} ausente de {CAMINHO.name}.")

    encontrados = {}
    for linha in linhas[inicio + 1:]:
        if linha.startswith("## "):   # acabou a seção
            break
        achado = LINHA_INDICE.match(linha)
        if achado:
            encontrados[achado.group(1)] = achado.group(2)
    assert encontrados, f"Nenhuma linha de índice reconhecida em {CAMINHO.name}."
    return encontrados


@pytest.fixture(scope="module")
def corpos(linhas) -> dict[str, str]:
    """Os Status como cada decisão os declara, logo abaixo do próprio título."""
    encontrados = {}
    atual = None
    for linha in linhas:
        titulo = TITULO_CORPO.match(linha)
        if titulo:
            atual = titulo.group(1)
            continue
        if atual is None:
            continue
        status = STATUS_CORPO.match(linha)
        if status:
            encontrados[atual] = status.group(1)
            atual = None
    assert encontrados, f"Nenhum corpo de decisão reconhecido em {CAMINHO.name}."
    return encontrados


def test_indice_e_corpos_cobrem_as_mesmas_decisoes(indice, corpos):
    """Decisão sem linha de índice some da leitura rápida; linha sem corpo mente."""
    sem_corpo = sorted(set(indice) - set(corpos))
    sem_indice = sorted(set(corpos) - set(indice))
    assert not sem_corpo, f"No índice e sem corpo: {', '.join(sem_corpo)}"
    assert not sem_indice, f"Com corpo e fora do índice: {', '.join(sem_indice)}"


def test_toda_decisao_do_corpo_declara_status(linhas, corpos):
    """Corpo sem `**Status:**` escapa da conferência inteira, calado."""
    titulados = [t.group(1) for t in map(TITULO_CORPO.match, linhas) if t]
    sem_status = sorted(set(titulados) - set(corpos))
    assert not sem_status, f"Sem linha de Status: {', '.join(sem_status)}"


def test_status_do_indice_bate_com_o_do_corpo(indice, corpos):
    """As duas fontes do mesmo dado têm que dizer a mesma classificação.

    A mensagem nomeia os três campos porque é o que o autor precisa para decidir qual
    dos dois lados está errado. Corrigir pelo índice e corrigir pelo corpo levam a
    documentos diferentes, e a escolha é dele.
    """
    divergencias = [
        (identificador, indice[identificador], corpos[identificador])
        for identificador in sorted(set(indice) & set(corpos))
        if _classificacao(indice[identificador])
        != _classificacao(corpos[identificador])
    ]
    assert not divergencias, "Índice e corpo divergem:\n" + "\n".join(
        f"  {identificador}: índice diz {do_indice!r}, corpo diz {do_corpo!r}"
        for identificador, do_indice, do_corpo in divergencias
    )
