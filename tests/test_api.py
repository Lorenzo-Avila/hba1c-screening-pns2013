"""Guardas da API do protótipo.

Tudo pelo cliente de teste do FastAPI, sem servidor rodando. Os registros enviados
saem da base limpa, onde peso e altura ainda existem, nos índices do conjunto de
teste; é a única forma de mandar à API, no contrato clínico, a mesma pessoa que o
experimento avaliou.

O artefato é gitignorado e o CSV bruto não é versionado; sem um dos dois, o módulo
pula com motivo.
"""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import textos
from api.app import criar_app
from src import config as cfg
from src import data as dt
from src import experimento as xp
from src import servir as sv
from tests.test_painel import VOCABULARIO_PROIBIDO

pytestmark = pytest.mark.skipif(
    not cfg.PATH_RAW.exists()
    or not (sv.PATH_MODELO / sv.ARQUIVO_METADADOS).exists(),
    reason="CSV bruto ou artefato ausente; rode python -m src.servir.",
)

TOLERANCIA = 1e-9
TOLERANCIA_SOMA_PARALELA = 1e-12
N_AMOSTRA = 30
# O brief veta "probabilidade" sozinha, não só "probabilidade de" como o painel.
VETADOS = VOCABULARIO_PROIBIDO + ["probabilidade"]
ENCAMINHADOS = {"VP": True, "FP": True, "FN": False, "VN": False}


def _payload(linha: pd.Series) -> dict:
    """Linha da base limpa -> corpo da requisição, com ausente como `null`."""
    corpo = {}
    for campo in sv.CAMPOS_ENTRADA:
        valor = linha[campo]
        if pd.isna(valor):
            corpo[campo] = None
        elif campo in cfg.CONTINUAS:
            corpo[campo] = float(valor)
        else:
            corpo[campo] = int(valor)
    return corpo


@pytest.fixture(scope="module")
def cliente():
    return TestClient(criar_app())


@pytest.fixture(scope="module")
def analitico():
    return dt.carregar_analitico()


@pytest.fixture(scope="module")
def particao():
    return xp.preparar_particao()


@pytest.fixture(scope="module")
def casos():
    """Os quatro casos de `B_composto`, uma linha por variável."""
    tabela = pd.read_csv(cfg.PATH_TABELAS / "shap_casos_locais.csv", encoding="utf-8")
    return tabela[(tabela["celula"] == sv.CONJUNTO)
                  & (tabela["regime"] == "tree_path_dependent")]


@pytest.fixture(scope="module")
def respostas(cliente, analitico, particao, casos):
    """Respostas da API para uma amostra do teste que inclui os quatro casos."""
    indices = set(particao.X_te.sample(n=N_AMOSTRA,
                                       random_state=cfg.RANDOM_STATE).index)
    indices |= set(casos["indice"])
    saida = {}
    for indice in sorted(indices):
        resposta = cliente.post("/classificacao",
                                json=_payload(analitico.loc[indice]))
        assert resposta.status_code == 200, (indice, resposta.text)
        saida[indice] = resposta.json()
    return saida


# --- Equivalência ------------------------------------------------------------

def test_escore_coincide_com_o_artefato(respostas, particao):
    """Coincide até a ordem de soma das árvores.

    Com `n_jobs=-1`, a floresta soma as 500 árvores na ordem em que as threads
    terminam: duas chamadas em lote do próprio artefato já divergem em até 3e-16, e
    com `n_jobs=1` lote e linha a linha coincidem bit a bit. A matriz de design que
    a API monta é idêntica à do lote. 1e-12 fica três ordens acima desse ruído e
    três abaixo da tolerância de 1e-9 das contribuições.
    """
    pipeline, _ = sv.carregar()
    indices = list(respostas)
    esperado = pipeline.predict_proba(particao.X_te.loc[indices])[:, 1]
    obtido = [respostas[i]["escore_de_ordenacao"] for i in indices]
    assert obtido == pytest.approx(list(esperado), abs=TOLERANCIA_SOMA_PARALELA)


# --- Casos exemplares --------------------------------------------------------

def test_casos_exemplares_reproduzem_escore_e_decisao(respostas, casos):
    for (indice, caso), linhas in casos.groupby(["indice", "caso"]):
        resposta = respostas[indice]
        assert round(resposta["escore_de_ordenacao"], 9) == linhas["escore"].iloc[0]
        assert round(resposta["limiar"], 9) == linhas["limiar"].iloc[0]
        assert resposta["encaminhar"] is ENCAMINHADOS[caso], (indice, caso)
        assert resposta["decisao"] == (textos.ENCAMINHAR if ENCAMINHADOS[caso]
                                       else textos.NAO_ENCAMINHAR)


# --- Validação de domínio ----------------------------------------------------

@pytest.mark.parametrize("campo, valor", [
    ("sexo", 3), ("raca_cor", 6), ("regiao", 0), ("escolaridade", 8),
    ("saude_autoaval", 6), ("plano_saude", 3), ("has_autorrelato", 4),
    ("colesterol_alto", 3), ("altura_cm", 1.70), ("peso_kg", 80000.0),
    ("idade", 12.0),
])
def test_valor_fora_do_dominio_devolve_422(cliente, analitico, casos, campo, valor):
    corpo = _payload(analitico.loc[int(casos["indice"].iloc[0])])
    corpo[campo] = valor
    assert cliente.post("/classificacao", json=corpo).status_code == 422


def test_campo_desconhecido_ou_faltando_devolve_422(cliente, analitico, casos):
    corpo = _payload(analitico.loc[int(casos["indice"].iloc[0])])
    assert cliente.post("/classificacao",
                        json={**corpo, "hba1c": 7.0}).status_code == 422
    del corpo["idade"]
    assert cliente.post("/classificacao", json=corpo).status_code == 422


# --- Vocabulário -------------------------------------------------------------

def _textos(objeto):
    """Chaves e valores de texto de uma resposta JSON, em qualquer profundidade."""
    if isinstance(objeto, dict):
        for chave, valor in objeto.items():
            yield chave
            yield from _textos(valor)
    elif isinstance(objeto, list):
        for item in objeto:
            yield from _textos(item)
    elif isinstance(objeto, str):
        yield objeto


def test_resposta_sem_vocabulario_vetado(respostas, cliente):
    """A restrição de leitura é a única constante liberada: ela nega os termos."""
    corpos = list(respostas.values()) + [cliente.get("/metadados").json()]
    achados = sorted({
        (termo, texto) for corpo in corpos for texto in _textos(corpo)
        if texto != textos.RESTRICAO_DE_LEITURA
        for termo in VETADOS if termo in texto.lower()
    })
    assert not achados, achados


def test_restricao_de_leitura_acompanha_toda_classificacao(respostas):
    assert all(r["restricao_de_leitura"] == textos.RESTRICAO_DE_LEITURA
               for r in respostas.values())


# --- Versões -----------------------------------------------------------------

def test_api_recusa_subir_com_metadados_adulterados(tmp_path):
    _, metadados = sv.carregar()
    adulterado = {**metadados, "versoes": {**metadados["versoes"], "numpy": "0.0.0"}}
    (tmp_path / sv.ARQUIVO_METADADOS).write_text(json.dumps(adulterado),
                                                 encoding="utf-8")
    with pytest.raises(RuntimeError, match="numpy 0.0.0"):
        criar_app(tmp_path)


# --- Contribuições publicadas ------------------------------------------------

def test_contribuicoes_reproduzem_as_publicadas(respostas, casos):
    for indice, linhas in casos.groupby("indice"):
        resposta = respostas[indice]
        obtidas = {c["atributo"]: c for c in resposta["contribuicoes"]}
        assert set(obtidas) == set(linhas["atributo"]), indice
        assert resposta["valor_de_base"] == pytest.approx(
            linhas["valor_base"].iloc[0], abs=TOLERANCIA)
        for linha in linhas.itertuples():
            obtida = obtidas[linha.atributo]
            assert obtida["contribuicao"] == pytest.approx(
                linha.contribuicao, abs=TOLERANCIA), (indice, linha.atributo)
            assert obtida["valor"] == pytest.approx(
                linha.valor_observado, abs=TOLERANCIA), (indice, linha.atributo)


# --- Aditividade -------------------------------------------------------------

def _erro_de_aditividade(resposta: dict) -> float:
    soma = math.fsum(c["contribuicao"] for c in resposta["contribuicoes"])
    return abs(resposta["valor_de_base"] + soma - resposta["escore_de_ordenacao"])


def test_base_mais_contribuicoes_reconstroi_o_escore(respostas):
    erros = {i: _erro_de_aditividade(r) for i, r in respostas.items()}
    assert max(erros.values()) <= TOLERANCIA, erros


# --- Imputadas ---------------------------------------------------------------

def test_ausente_aceito_volta_marcado_como_imputado(cliente, analitico, casos):
    corpo = _payload(analitico.loc[int(casos["indice"].iloc[0])])
    corpo["colesterol_alto"] = None
    resposta = cliente.post("/classificacao", json=corpo)
    assert resposta.status_code == 200, resposta.text
    corpo = resposta.json()
    assert corpo["campos_imputados"] == ["colesterol_alto"]
    marcadas = {c["atributo"] for c in corpo["contribuicoes"] if c["imputada"]}
    # Colesterol entra no modelo B só pela multimorbidade.
    assert marcadas == {"multimorbidade"}
    assert _erro_de_aditividade(corpo) <= TOLERANCIA


# --- Ausentes: só onde o treino teve ausente ---------------------------------

@pytest.fixture(scope="module")
def completo(analitico, casos):
    """Um caso exemplar sem nenhum ausente, base das variações abaixo."""
    for indice in casos["indice"].unique():
        corpo = _payload(analitico.loc[int(indice)])
        if None not in corpo.values():
            return corpo
    pytest.fail("Nenhum caso exemplar completo para servir de base.")


def test_registro_completo_passa_sem_imputadas(cliente, completo):
    resposta = cliente.post("/classificacao", json=completo)
    assert resposta.status_code == 200, resposta.text
    assert resposta.json()["campos_imputados"] == []
    assert not any(c["imputada"] for c in resposta.json()["contribuicoes"])


def test_bloco_de_medicao_inteiro_ausente_e_recusado(cliente, completo):
    corpo = {**completo, **{campo: None for campo in cfg.MEDICAO}}
    resposta = cliente.post("/classificacao", json=corpo)
    assert resposta.status_code == 422
    assert "Bloco de medição inteiro ausente" in resposta.json()["detail"]


@pytest.mark.parametrize("campo", ["pas", "peso_kg", "idade", "sexo", "escolaridade"])
def test_ausencia_em_campo_sem_ausente_no_treino_e_recusada(cliente, completo, campo):
    resposta = cliente.post("/classificacao", json={**completo, campo: None})
    assert resposta.status_code == 422
    assert campo in resposta.json()["detail"]


def test_raca_ignorada_e_tratada_como_ausente(cliente, completo):
    resposta = cliente.post("/classificacao",
                            json={**completo, "raca_cor": cfg.COD_IGNORADO_RACA})
    assert resposta.status_code == 200, resposta.text
    corpo = resposta.json()
    assert corpo["campos_imputados"] == ["raca_cor"]
    assert {c["atributo"] for c in corpo["contribuicoes"] if c["imputada"]} == {
        "raca_cor"}
    # Raça 9 e raça nula são o mesmo registro para o modelo.
    nula = cliente.post("/classificacao", json={**completo, "raca_cor": None}).json()
    assert nula["escore_de_ordenacao"] == pytest.approx(
        corpo["escore_de_ordenacao"], abs=TOLERANCIA_SOMA_PARALELA)


def test_ausencia_parcial_aceita_e_marcada(cliente, completo):
    corpo = {**completo, "has_autorrelato": None, "regiao": None}
    resposta = cliente.post("/classificacao", json=corpo)
    assert resposta.status_code == 200, resposta.text
    corpo = resposta.json()
    assert corpo["campos_imputados"] == ["regiao", "has_autorrelato"]
    # HAS autorrelatada entra em B só pela multimorbidade; região é nominal própria.
    assert {c["atributo"] for c in corpo["contribuicoes"] if c["imputada"]} == {
        "regiao", "multimorbidade"}
    assert _erro_de_aditividade(corpo) <= TOLERANCIA


def test_lista_de_aceitos_vem_dos_metadados(cliente):
    _, metadados = sv.carregar()
    assert cliente.get("/metadados").json()["campos_que_aceitam_ausencia"] == (
        metadados["campos_que_aceitam_ausencia"])


def test_metadados_trazem_a_codificacao_de_cada_categorica(cliente):
    """O formulário do protótipo monta as opções daqui, sem importar `src/`."""
    dominios = cliente.get("/metadados").json()["dominios"]
    assert set(dominios) == set(cfg.CATEGORICAS)
    assert dominios["raca_cor"] == {str(k): v for k, v in cfg.COD_RACA.items()}
