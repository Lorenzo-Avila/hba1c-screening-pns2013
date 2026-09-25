"""Reprodução ao vivo: roda o experimento a partir do CSV bruto e mostra o log.

Única página que dispara processo e única com botão. Chama
`python -m src.experimento --saida <temporário>` como processo separado, com a
raiz do repositório como diretório de trabalho, e exibe o log como texto. O
painel não interpreta número do log: o selo final sai do código de saída do
processo, e as linhas de conferência são destacadas pelo começo do texto. O
diretório temporário é apagado no fim, e `reports/tabelas/` não é tocado
(D-054, D-055).

Antes de disparar, confere se o interpretador do painel consegue importar o
experimento. O painel não depende de sklearn e sobe com qualquer Python que tenha
streamlit; sem a conferência, um Python fora da `.venv` só apareceria como
traceback no meio do log.
"""

import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pandas as pd
import streamlit as st

import componentes
import selecao
import textos

RAIZ = Path(__file__).resolve().parents[2]
TIMEOUT_MINUTOS = 5
CHAVE_RESULTADO = "reproducao_resultado"

# Começo das linhas que o experimento imprime ao conferir composição e âncoras, e o
# tom de cada uma. O destaque copia a linha inteira do log.
MARCAS_CONFERENCIA = {
    "Composição conferida": "verde",
    "Reprodução conferida": "verde",
    "COMPOSIÇÃO ERRADA": "vermelho",
    "DIVERGÊNCIA": "vermelho",
    "Âncora ainda não congelada": "cinza",
}


def _ler_saida(fluxo, fila: queue.Queue) -> None:
    """Passa cada linha do processo para a fila; `None` marca o fim do fluxo."""
    for linha in fluxo:
        fila.put(linha.rstrip("\n"))
    fila.put(None)


def _artefatos(diretorio: Path) -> pd.DataFrame:
    """Arquivos gravados no diretório de saída, agrupados pelo nome do artefato."""
    grupos = {}
    for arquivo in sorted(p for p in diretorio.rglob("*") if p.is_file()):
        grupos.setdefault(arquivo.stem, []).append(arquivo.suffix)
    return pd.DataFrame(
        [(nome, " · ".join(extensoes)) for nome, extensoes in grupos.items()],
        columns=list(textos.TABELA_ARTEFATO),
    )


def conferir_ambiente() -> str | None:
    """Tenta importar o experimento com o interpretador do painel, em processo separado.

    Devolve `None` se o import passa, ou a última linha do erro. O import acontece no
    processo filho, e o painel continua sem importar `src/` nem sklearn.
    """
    try:
        processo = subprocess.run(
            [sys.executable, "-c", "import src.experimento"],
            cwd=RAIZ, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "o import passou de 120 s"
    if processo.returncode == 0:
        return None
    linhas = processo.stderr.strip().splitlines()
    return linhas[-1] if linhas else f"código de saída {processo.returncode}"


def mostrar_falha_de_ambiente(erro: str) -> None:
    """Aviso de ambiente com o interpretador em uso e o comando de correção."""
    componentes.selos([(textos.AMBIENTE_TITULO, "vermelho")])
    componentes.alerta(textos.AMBIENTE_FALHA.format(interpretador=sys.executable,
                                                   erro=erro))
    componentes.nota(textos.AMBIENTE_CORRECAO)
    st.code(textos.COMANDO_PAINEL_VENV, language="bash")


def reproduzir(area) -> dict:
    """Roda o experimento num diretório temporário, com tempo e log ao vivo em `area`.

    Encerra o processo se passar de `TIMEOUT_MINUTOS`. Devolve o código de saída, o
    log, os segundos decorridos, os artefatos gravados e se houve estouro de tempo.
    """
    with area.container():
        st.subheader(textos.LOG_AO_VIVO)
        tempo, log = st.empty(), st.empty()
    ambiente = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    linhas, estourou, fim = [], False, False
    inicio = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="reproducao_") as tmp:
        processo = subprocess.Popen(
            [sys.executable, "-m", "src.experimento", "--saida", tmp],
            cwd=RAIZ, env=ambiente, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        fila = queue.Queue()
        threading.Thread(target=_ler_saida, args=(processo.stdout, fila),
                         daemon=True).start()
        while not fim:
            time.sleep(0.5)
            # Esvazia a fila a cada meio segundo e redesenha uma vez só.
            while not fim:
                try:
                    linha = fila.get_nowait()
                except queue.Empty:
                    break
                if linha is None:
                    fim = True
                else:
                    linhas.append(linha)
            segundos = int(time.monotonic() - inicio)
            tempo.markdown(textos.REPRODUCAO_RODANDO.format(segundos=segundos))
            log.code("\n".join(linhas[-40:]), language=None)
            if not fim and segundos > TIMEOUT_MINUTOS * 60:
                processo.kill()
                estourou = True
                break
        codigo = processo.wait()
        artefatos = _artefatos(Path(tmp))
    area.empty()
    return {
        "codigo": codigo,
        "log": "\n".join(linhas),
        "segundos": int(time.monotonic() - inicio),
        "artefatos": artefatos,
        "estourou": estourou,
    }


def mostrar(resultado: dict) -> None:
    """Selo de status, tempo, linhas de conferência, artefatos e log completo."""
    componentes.selos([textos.REPRODUCAO_STATUS[resultado["codigo"] == 0]])
    componentes.nota(textos.REPRODUCAO_DECORRIDO.format(segundos=resultado["segundos"]))
    if resultado["estourou"]:
        componentes.alerta(textos.REPRODUCAO_TIMEOUT.format(minutos=TIMEOUT_MINUTOS))

    conferencia = [(linha, tom) for linha in resultado["log"].splitlines()
                   for marca, tom in MARCAS_CONFERENCIA.items()
                   if linha.startswith(marca)]
    if conferencia:
        st.subheader(textos.CONFERENCIA_TITULO)
        for linha, tom in conferencia:
            componentes.linha_de_log(linha, tom)

    st.subheader(textos.REPRODUCAO_ARTEFATOS_TITULO)
    componentes.tabela(resultado["artefatos"])

    with st.expander(textos.LOG_TITULO, expanded=resultado["codigo"] != 0):
        st.code(resultado["log"], language=None)


selecao.abrir_pagina("reproducao", "projeto")

st.subheader(textos.REPRODUCAO_COMANDO)
st.code(textos.COMANDO_REPRODUCAO, language="bash")
componentes.nota(textos.REPRODUCAO_TEMPO)
componentes.quadro(textos.REPRODUCAO_ARTEFATOS_TITULO,
                   f"{textos.REPRODUCAO_ARTEFATOS} {textos.REPRODUCAO_FORA}")

if st.button(textos.BOTAO_REPRODUZIR, type="primary", key="reproduzir"):
    erro = conferir_ambiente()
    if erro is None:
        st.session_state[CHAVE_RESULTADO] = reproduzir(st.empty())
    else:
        # Resultado de clique anterior ficaria na tela como se fosse deste.
        st.session_state.pop(CHAVE_RESULTADO, None)
        mostrar_falha_de_ambiente(erro)

if CHAVE_RESULTADO in st.session_state:
    mostrar(st.session_state[CHAVE_RESULTADO])
