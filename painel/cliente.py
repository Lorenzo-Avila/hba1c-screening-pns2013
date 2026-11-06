"""Cliente HTTP da API: o único caminho do painel até o modelo.

Só a página de classificação individual usa este módulo. O painel continua sem
importar `src/`, `api/` ou biblioteca de modelo (guarda em `tests/test_painel.py`): o
que a página mostra chega por estas duas chamadas.
"""

import os

import httpx

URL_API = os.environ.get("URL_API", "http://127.0.0.1:8000")


def ler_metadados() -> dict:
    """Modelo, limiar, campos, codificação das categóricas e aviso do escore."""
    resposta = httpx.get(f"{URL_API}/metadados", timeout=10)
    resposta.raise_for_status()
    return resposta.json()


def classificar(registro: dict) -> httpx.Response:
    """Envia o registro e devolve a resposta crua, para a tela mostrar o 422."""
    return httpx.post(f"{URL_API}/classificacao", json=registro, timeout=60)
