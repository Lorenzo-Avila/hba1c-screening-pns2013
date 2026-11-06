"""Guarda principal do protótipo: o artefato servido é a célula de referência publicada.

O artefato, aplicado ao conjunto de teste, tem de reproduzir da linha marcada em
`referencia_d030` de `comparativo_consolidado.csv` o limiar em nove casas e a matriz
de confusão exata. Se reproduz, a API serve o modelo da tabela, não outro com a mesma
cara.

O artefato é gitignorado e o CSV bruto não é versionado; sem um dos dois, o módulo
pula com motivo, como as outras guardas que dependem de `data/raw/`.
"""

from __future__ import annotations

import json

import pytest

from src import config as cfg
from src import experimento as xp
from src import servir as sv

pytestmark = pytest.mark.skipif(
    not cfg.PATH_RAW.exists()
    or not (sv.PATH_MODELO / sv.ARQUIVO_METADADOS).exists(),
    reason="CSV bruto ou artefato ausente; rode python -m src.servir.",
)


@pytest.fixture(scope="module")
def artefato():
    return sv.carregar()


@pytest.fixture(scope="module")
def referencia():
    return sv.celula_de_referencia()


@pytest.fixture(scope="module")
def particao():
    return xp.preparar_particao()


def test_artefato_e_a_celula_de_referencia(artefato, referencia):
    _, metadados = artefato
    assert (metadados["modelo"], metadados["conjunto"]) == (
        referencia["modelo"], referencia["conjunto"])
    assert round(metadados["limiar"], sv.CASAS_LIMIAR) == round(
        float(referencia["limiar"]), sv.CASAS_LIMIAR)


def test_artefato_reproduz_a_matriz_publicada(artefato, referencia, particao):
    """VP, FP, FN, VN e encaminhados exatos, no limiar gravado nos metadados."""
    pipeline, metadados = artefato
    escore = pipeline.predict_proba(particao.X_te)[:, 1]
    encaminhado = escore >= metadados["limiar"]
    y = particao.y_te.to_numpy() == 1
    obtido = {
        "vp": int((encaminhado & y).sum()),
        "fp": int((encaminhado & ~y).sum()),
        "fn": int((~encaminhado & y).sum()),
        "vn": int((~encaminhado & ~y).sum()),
        "encaminhados": int(encaminhado.sum()),
    }
    assert obtido == {chave: int(referencia[chave]) for chave in obtido}


def test_conferir_celula_recusa_outra_celula(referencia):
    resultado = {"modelo": referencia["modelo"], "conjunto": "A_atomico",
                 "limiar": float(referencia["limiar"])}
    with pytest.raises(ValueError, match="célula de referência"):
        sv.conferir_celula(resultado, referencia)


def test_conferir_celula_recusa_limiar_deslocado(referencia):
    resultado = {"modelo": referencia["modelo"], "conjunto": referencia["conjunto"],
                 "limiar": float(referencia["limiar"]) + 1e-6}
    with pytest.raises(ValueError, match="célula de referência"):
        sv.conferir_celula(resultado, referencia)


def test_campos_que_aceitam_ausencia_sao_os_do_treino(artefato):
    """Só as quatro categóricas têm ausente no treino; nenhuma contínua tem."""
    _, metadados = artefato
    assert set(metadados["campos_que_aceitam_ausencia"]) == {
        "raca_cor", "regiao", "has_autorrelato", "colesterol_alto"}
    assert metadados["campos_entrada"] == sv.CAMPOS_ENTRADA


def test_carregar_recusa_versao_divergente(tmp_path, artefato):
    """Metadados adulterados: a carga falha antes de desserializar o binário."""
    _, metadados = artefato
    adulterado = {**metadados,
                  "versoes": {**metadados["versoes"], "scikit-learn": "0.0.0"}}
    (tmp_path / sv.ARQUIVO_METADADOS).write_text(json.dumps(adulterado),
                                                 encoding="utf-8")
    with pytest.raises(RuntimeError, match="scikit-learn 0.0.0"):
        sv.carregar(tmp_path)
