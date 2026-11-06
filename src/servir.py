"""Grava e carrega o modelo que a API do protótipo serve.

A API carrega um artefato gravado em vez de treinar na subida: assim ela sobe rápido
e não depende do CSV bruto. O modelo servido é a célula de referência,
`Random Forest · B_composto`, a de menor NNS no ponto de operação principal. Ele é
ajustado pelo mesmo caminho que produz a tabela consolidada: `models.montar` seguido
de `evaluate.avaliar`. Montar pipeline próprio aqui daria outro modelo com a mesma
cara.

A célula fica fixada neste módulo, mas não como verdade solta: `gravar` confere que
ela é a linha marcada em `referencia_d030` de `comparativo_consolidado.csv` e que o
limiar coincide em nove casas, e falha se não for.

Ao lado do binário vai um `metadados.json`. `carregar` recusa o artefato quando as
versões instaladas divergem das gravadas: o binário é acoplado às versões de
scikit-learn e NumPy, e um erro alto na carga é melhor que uma desserialização que
funciona e devolve outro número. Carregar pickle só é seguro com artefato próprio,
então `carregar` só lê o diretório que `gravar` produz.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import date
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd

from . import config as cfg
from . import data as dt
from . import evaluate as ev
from . import experimento as xp
from . import models as md

CONJUNTO = "B_composto"
PATH_MODELO = cfg.RAIZ / "modelo"   # no .gitignore; regenerável por `make modelo`
ARQUIVO_MODELO = "modelo.joblib"
ARQUIVO_METADADOS = "metadados.json"
CASAS_LIMIAR = 9

# Contrato de entrada da API, no nível de um prontuário: o que `data.derivar`
# consome. Peso e altura chegam ao modelo só pelas derivadas `imc` e `rcest`.
CAMPOS_ENTRADA = cfg.CONTINUAS + cfg.CATEGORICAS

# Bibliotecas cuja versão entra nos metadados e é conferida na carga.
BIBLIOTECAS = ["scikit-learn", "numpy", "pandas", "scipy", "shap", "joblib"]


def versoes() -> dict[str, str]:
    """Versões instaladas de Python e das bibliotecas que o binário depende."""
    return {"python": platform.python_version(),
            **{nome: version(nome) for nome in BIBLIOTECAS}}


def sha256(path: Path) -> str:
    """Hash do CSV bruto: identifica a base de que o modelo saiu."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def celula_de_referencia(path: Path | None = None) -> pd.Series:
    """A linha da tabela consolidada marcada em `referencia_d030`.

    Raises:
        ValueError: se nenhuma ou mais de uma linha estiver marcada.
    """
    if path is None:
        path = cfg.PATH_TABELAS / f"{xp.NOME_CONSOLIDADO}.csv"
    tabela = pd.read_csv(path, encoding="utf-8")
    marcadas = tabela[tabela["referencia_d030"]]
    if len(marcadas) != 1:
        raise ValueError(
            f"{Path(path).name} marca {len(marcadas)} linhas em referencia_d030; "
            f"esperada uma."
        )
    return marcadas.iloc[0]


def conferir_celula(resultado: dict, referencia: pd.Series) -> None:
    """Confere que o modelo ajustado é a célula de referência publicada.

    Raises:
        ValueError: se modelo, conjunto ou limiar em nove casas divergirem.
    """
    obtido = (resultado["modelo"], resultado["conjunto"],
              round(resultado["limiar"], CASAS_LIMIAR))
    esperado = (referencia["modelo"], referencia["conjunto"],
                round(float(referencia["limiar"]), CASAS_LIMIAR))
    if obtido != esperado:
        raise ValueError(
            f"O modelo ajustado {obtido} não é a célula de referência {esperado} de "
            f"{xp.NOME_CONSOLIDADO}.csv."
        )


def campos_que_aceitam_ausencia(analitico: pd.DataFrame,
                                indices_treino: pd.Index) -> list[str]:
    """Campos do contrato com ao menos um ausente no treino.

    A API só aceita ausência nesses campos: neles a imputação foi exercitada no
    treino, e fora deles o registro sai do domínio que o modelo conheceu. Calcula
    sobre as linhas de treino da base limpa, onde peso e altura ainda existem como
    coluna própria. É contagem que descreve a partição, sem ajuste.
    """
    ausentes = analitico.loc[indices_treino, CAMPOS_ENTRADA].isna().any()
    return [campo for campo in CAMPOS_ENTRADA if ausentes[campo]]


def gravar(destino: Path | None = None) -> dict:
    """Ajusta a célula de referência, confere contra a tabela e grava binário e metadados.

    Returns:
        Os metadados gravados.
    """
    destino = PATH_MODELO if destino is None else Path(destino)

    analitico = dt.carregar_analitico()
    particao = xp.preparar_particao()
    resultado = ev.avaliar(
        md.montar(md.estimadores()[xp.NOME_FLORESTA], CONJUNTO),
        particao.X_tr, particao.y_tr, particao.X_te, particao.y_te,
        nome=xp.NOME_FLORESTA, conjunto=CONJUNTO, recall_alvo=cfg.RECALL_ALVO,
        salvar_figuras=False,
    )
    conferir_celula(resultado, celula_de_referencia())

    metadados = {
        "modelo": resultado["modelo"],
        "conjunto": resultado["conjunto"],
        # `json` grava o float pelo `repr`, que volta idêntico na leitura.
        "limiar": resultado["limiar"],
        "recall_alvo": cfg.RECALL_ALVO,
        "random_state": cfg.RANDOM_STATE,
        "campos_entrada": CAMPOS_ENTRADA,
        "campos_que_aceitam_ausencia": campos_que_aceitam_ausencia(
            analitico, particao.X_tr.index),
        "versoes": versoes(),
        "sha256_csv_bruto": sha256(cfg.PATH_RAW),
        "gerado_em": date.today().isoformat(),
    }

    destino.mkdir(parents=True, exist_ok=True)
    joblib.dump(resultado["estimador"], destino / ARQUIVO_MODELO)
    (destino / ARQUIVO_METADADOS).write_text(
        json.dumps(metadados, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadados


def conferir_versoes(metadados: dict) -> None:
    """Recusa o artefato se alguma versão instalada divergir da gravada.

    Raises:
        RuntimeError: com cada biblioteca divergente, gravada contra instalada.
    """
    instaladas = versoes()
    divergentes = {nome: (gravada, instaladas.get(nome))
                   for nome, gravada in metadados["versoes"].items()
                   if instaladas.get(nome) != gravada}
    if divergentes:
        detalhe = ", ".join(f"{nome} {gravada} gravada, {instalada} instalada"
                            for nome, (gravada, instalada) in divergentes.items())
        raise RuntimeError(
            f"Versões divergem do artefato: {detalhe}. Regenere com make modelo."
        )


def carregar(origem: Path | None = None):
    """Lê binário e metadados, conferindo as versões antes de desserializar.

    Returns:
        Tupla `(pipeline, metadados)`.

    Raises:
        FileNotFoundError: se o artefato não foi gravado.
        RuntimeError: se as versões divergirem.
    """
    origem = PATH_MODELO if origem is None else Path(origem)
    caminho_metadados = origem / ARQUIVO_METADADOS
    if not caminho_metadados.exists():
        raise FileNotFoundError(
            f"Artefato ausente em {origem}. Rode python -m src.servir antes."
        )
    metadados = json.loads(caminho_metadados.read_text(encoding="utf-8"))
    conferir_versoes(metadados)
    return joblib.load(origem / ARQUIVO_MODELO), metadados


def main(argv: list[str] | None = None) -> int:
    """Entrada de linha de comando. É o que `make modelo` chama de fato."""
    # O console do Windows abre em cp1252 e quebra com acentuação; sem isto o
    # comando sai com status 1 por codificação.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        prog="python -m src.servir",
        description="Ajusta a célula de referência e grava o modelo servido pela API.",
    )
    parser.add_argument("--saida", type=Path, default=None, metavar="DIR",
                        help=f"padrão: {PATH_MODELO}")
    args = parser.parse_args(argv)

    metadados = gravar(args.saida)
    print(f"Modelo servido: {metadados['modelo']} · {metadados['conjunto']}, "
          f"limiar {metadados['limiar']:.9f}, conferido contra "
          f"{xp.NOME_CONSOLIDADO}.csv.")
    print(f"Campos que aceitam ausência: "
          f"{', '.join(metadados['campos_que_aceitam_ausencia'])}")
    print(f"Gravado em {args.saida or PATH_MODELO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
