"""API do protótipo de classificação.

Serve a célula de referência gravada por `python -m src.servir`. Na subida, carrega o
artefato e confere as versões; se divergirem, `criar_app` levanta e a API não sobe.
Por isso a aplicação sai de uma fábrica, e não de uma variável de módulo: importar
este arquivo não carrega modelo nenhum.

    uvicorn api.app:criar_app --factory

A resposta é decisão de triagem, escore de ordenação contra o limiar e a
decomposição SHAP local, nunca risco individual: o `class_weight` desloca a escala do
escore, que ordena bem mas não estima chance de HbA1c ≥ 6,5%. Todo cálculo vem de
`src/`: a derivação de `data.derivar`, o escore do pipeline gravado e a atribuição
das mesmas funções que produziram `shap_casos_locais`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from src import colinearidade as col
from src import config as cfg
from src import data as dt
from src import explicabilidade as ex
from src import features as ft
from src import servir as sv

from . import textos
from .esquema import DOMINIOS, Classificacao, Contribuicao, Metadados, Registro


def preparar(registro: Registro) -> tuple[pd.DataFrame, list[str]]:
    """Registro clínico -> preditoras do modelo, mais os campos que vieram ausentes.

    A ausência é lida depois de `derivar`, para `raca_cor = 9` contar como ausente.
    """
    bruto = pd.DataFrame([registro.model_dump()])[sv.CAMPOS_ENTRADA]
    derivado = dt.derivar(bruto)
    ausentes = [c for c in sv.CAMPOS_ENTRADA if pd.isna(derivado.at[0, c])]
    return derivado[cfg.PREDITORAS], ausentes


def conferir_ausencias(ausentes: list[str], aceitos: list[str]) -> None:
    """Recusa as ausências que o modelo não conheceu no treino.

    O bloco de medição inteiro ausente é recusado mesmo que a lista mude: o treino
    excluiu esses registros em `data.limpar`, então eles estão fora do domínio do
    modelo. Fora isso, só é aceita ausência em campo que teve ausente no treino,
    lista gravada nos metadados por `servir.gravar`.

    Raises:
        HTTPException: 422, nomeando a regra e os campos.
    """
    if set(cfg.MEDICAO) <= set(ausentes):
        raise HTTPException(status_code=422, detail=(
            f"Bloco de medição inteiro ausente ({', '.join(cfg.MEDICAO)}): registro "
            f"fora do domínio do modelo, que o treino excluiu."
        ))
    recusados = [campo for campo in ausentes if campo not in aceitos]
    if recusados:
        raise HTTPException(status_code=422, detail=(
            f"Campos sem ausência no treino não podem vir ausentes: "
            f"{', '.join(recusados)}. Aceitam ausência: {', '.join(aceitos)}."
        ))


def criar_app(origem: Path | None = None) -> FastAPI:
    """Monta a API sobre o artefato de `origem` (padrão: `modelo/`).

    Raises:
        FileNotFoundError: se o artefato não foi gravado.
        RuntimeError: se as versões instaladas divergirem das gravadas.
    """
    pipeline, metadados = sv.carregar(origem)
    preprocessador = pipeline.named_steps["pre"]
    floresta = pipeline.named_steps["clf"]
    mapa = ex.proveniencia(preprocessador)
    omitidas = (mapa.groupby("atributo", sort=False)["categoria_referencia_omitida"]
                .first())
    limiar = metadados["limiar"]

    app = FastAPI(title="Protótipo de classificação · HbA1c ≥ 6,5%")

    @app.get("/metadados", response_model=Metadados)
    def ler_metadados() -> Metadados:
        return Metadados(**{campo: metadados[campo] for campo in (
            "modelo", "conjunto", "limiar", "recall_alvo", "versoes",
            "campos_entrada", "campos_que_aceitam_ausencia")},
            dominios=DOMINIOS, aviso=textos.AVISO_ESCORE)

    @app.post("/classificacao", response_model=Classificacao)
    def classificar(registro: Registro) -> Classificacao:
        X, ausentes = preparar(registro)
        conferir_ausencias(ausentes, metadados["campos_que_aceitam_ausencia"])
        escore = float(pipeline.predict_proba(X)[0, 1])

        # Mesmo regime e mesmo objeto da explicação local publicada: o
        # `tree_path_dependent` sobre a floresta completa é exato e explica o escore
        # que o limiar corta.
        design = col.matriz_de_design(pipeline, X)
        valores, base = ex.explicar(floresta, design, ex.REGIME_PRINCIPAL)
        por_atributo = ex.contribuicoes_locais(valores, mapa, 0)

        # Escala clínica, depois da imputação e das compostas, como em
        # `shap_casos_locais.valor_observado`.
        clinicos = preprocessador[:2].transform(X).iloc[0]

        contribuicoes = [
            Contribuicao(
                atributo=linha.atributo,
                contribuicao=float(linha.contribuicao),
                valor=float(clinicos[linha.atributo]),
                imputada=bool(ft.FONTES[linha.atributo] & set(ausentes)),
                categoria_omitida=omitidas[linha.atributo] or None,
            )
            for linha in por_atributo.itertuples()
        ]
        encaminhar = escore >= limiar
        return Classificacao(
            decisao=textos.ENCAMINHAR if encaminhar else textos.NAO_ENCAMINHAR,
            encaminhar=encaminhar,
            escore_de_ordenacao=escore,
            limiar=limiar,
            aviso=textos.AVISO_ESCORE,
            valor_de_base=float(base[0]),
            contribuicoes=contribuicoes,
            campos_imputados=ausentes,
            restricao_de_leitura=textos.RESTRICAO_DE_LEITURA,
        )

    return app
