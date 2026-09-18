"""Ponto único de reprodução dos resultados.

Até a Sprint 1, as tabelas publicadas no `ESTADO.md` saíram de código de sessão que
não sobreviveu à sessão. Os números estavam certos e passavam nos testes, mas das
cinco linhas da tabela consolidada só três tinham artefato rastreado
(`reports/tabelas/comparativo_logreg.tex`), e os três regimes de carga operacional
não tinham nenhum. Este módulo fecha a lacuna: o que ele emite passa a ser a fonte e
o `ESTADO.md` passa a ser a transcrição, não o contrário.

Dois artefatos, ambos em `reports/tabelas/`:

- `comparativo_consolidado.tex`, uma linha por célula do grid.
- `carga_operacional.tex`, os três regimes que sustentam o argumento de
  aplicabilidade.

O produto cartesiano **não é cheio**. `RegraClinicaADA` lê idade e IMC das preditoras
cruas e o `DummyClassifier` ignora `X`: rodá-los sobre A, B e C daria três linhas
idênticas cada, e a tabela deixaria de bater com o `ESTADO.md` por artefato do próprio
grid, ou seja, uma divergência falsa em cima da verificação que existe para dar
confiança. Quem governa isso é a marca `Estimador.invariante_ao_conjunto`.

Uso:

    python -m src.experimento              # emite os dois artefatos e confere
    python -m src.experimento --figuras    # idem, com as figuras do teste
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import benchmark as bm
from . import config as cfg
from . import data as dt
from . import evaluate as ev
from . import features as ft
from . import models as md

# Modelos com resultado publicável hoje. A Random Forest fica de fora até a Sprint 2
# definir seus hiperparâmetros: os valores em `models.random_forest` são placeholder
# declarado, e reportar número saído de placeholder é pior que não reportar. Quem
# quiser rodá-la passa `--modelos`.
MODELOS_PADRAO = ["LogReg", "Regra ADA", "Dummy (maioria)"]

# Referência da reprodução: único artefato da Sprint 1 rastreado em git com os
# números da tabela consolidada.
REFERENCIA = "comparativo_logreg.tex"

NOME_CONSOLIDADO = "comparativo_consolidado"
NOME_CARGA = "carga_operacional"
NOME_RESUMO = "resumo_amostra"
NOME_AUSENTES = "ausentes_por_variavel"
NOME_CALIBRACAO = "calibracao_comparada"

# Conjunto vazio na coluna, para as linhas invariantes. É o mesmo símbolo que
# `benchmark.avaliar_comparadores` usa, senão as tabelas não se juntam.
SEM_CONJUNTO = "—"


@dataclass(frozen=True)
class Particao:
    """As quatro partes do split 80/20, viajando juntas.

    Existe para nenhuma função do módulo receber `X_tr, y_tr, X_te, y_te` soltos e
    poder trocá-los de ordem sem que nada acuse.
    """

    X_tr: pd.DataFrame
    X_te: pd.DataFrame
    y_tr: pd.Series
    y_te: pd.Series


def preparar_particao(path: Path | None = None) -> Particao:
    """Carga, limpeza e split 80/20 estratificado pelo alvo.

    O split é a primeira operação entre linhas do projeto: nada que olhe para mais de
    um registro acontece antes dele. `peso_lab` é descartado aqui, porque pondera a
    EDA descritiva e não o treino.
    """
    X, y, _ = dt.separar(dt.carregar_analitico(path))
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=cfg.TEST_SIZE,
        stratify=y,
        random_state=cfg.RANDOM_STATE,
    )
    return Particao(X_tr, X_te, y_tr, y_te)


def rodar_grid(modelos: list[str] | None = None,
               conjuntos: list[str] | None = None,
               particao: Particao | None = None,
               recall_alvo: float = cfg.RECALL_ALVO,
               salvar_figuras: bool = False) -> list[dict]:
    """Executa o grid modelo × conjunto e devolve os resultados brutos.

    Args:
        modelos: nomes em `models.estimadores()`. Omitido, usa `MODELOS_PADRAO`.
        conjuntos: chaves de `features.FEATURE_SETS`. Omitido, usa as três.
        particao: split pronto. Omitido, `preparar_particao` constrói um, o que
            implica reler o CSV bruto.
        recall_alvo: ponto de operação dos estimadores calibráveis. Os invariantes
            ignoram, porque operam com limiar fixo.

    Returns:
        Lista de dicionários de `evaluate.avaliar`, na ordem das linhas da tabela.

    Estimador invariante ao conjunto roda **uma vez**, com `conjunto="—"`. O produto
    cartesiano vale só para o que depende do conjunto.
    """
    disponiveis = md.estimadores()
    modelos = list(MODELOS_PADRAO) if modelos is None else list(modelos)
    conjuntos = list(ft.FEATURE_SETS) if conjuntos is None else list(conjuntos)
    particao = preparar_particao() if particao is None else particao

    desconhecidos = [m for m in modelos if m not in disponiveis]
    if desconhecidos:
        raise ValueError(
            f"Modelo(s) desconhecido(s): {desconhecidos}. "
            f"Disponíveis: {sorted(disponiveis)}"
        )

    resultados = []
    for nome in modelos:
        estimador = disponiveis[nome]
        celulas = [SEM_CONJUNTO] if estimador.invariante_ao_conjunto else conjuntos
        for conjunto in celulas:
            alvo = None if estimador.invariante_ao_conjunto else conjunto
            resultados.append(ev.avaliar(
                md.montar(estimador, alvo),
                particao.X_tr, particao.y_tr, particao.X_te, particao.y_te,
                nome=nome,
                conjunto=conjunto,
                recall_alvo=recall_alvo,
                limiar=estimador.limiar_fixo,   # presente só nos invariantes
                salvar_figuras=salvar_figuras,
            ))
    return resultados


def _rotulo(resultado: dict) -> str:
    """Nome da estratégia como `benchmark.tabela_nns` o escreve."""
    conjunto = resultado["conjunto"]
    return (f"{resultado['modelo']} · {conjunto}"
            if conjunto not in ("", SEM_CONJUNTO) else resultado["modelo"])


def modelo_de_referencia(resultados: list[dict]) -> dict:
    """O resultado de menor NNS entre os estimadores calibráveis.

    Critério declarado e reprodutível, o mesmo que `benchmark.leitura_operacional` já
    aplica ao escolher a estratégia que enfrenta a regra: menor NNS é menos
    encaminhamento por caso detectado, que é a pergunta operacional. Hoje devolve
    `LogReg · B_composto`, com NNS 7,26.

    Raises:
        ValueError: se não houver nenhum estimador calibrável nos resultados.
    """
    calibraveis = [r for r in resultados if r.get("limiar_calibrado", True)]
    if not calibraveis:
        raise ValueError("Nenhum estimador calibrável nos resultados.")
    return min(calibraveis, key=lambda r: r["nns"])


def carga_operacional(resultados: list[dict],
                      particao: Particao,
                      nome_regra: str = "Regra ADA") -> pd.DataFrame:
    """Os três regimes de encaminhamento.

    É o resultado que sustenta a conclusão do trabalho: a regra clínica encaminha
    57,6% da população adulta e opera em ponto único, enquanto o modelo escolhe onde
    operar. Sem esta tabela, o argumento de aplicabilidade não tem lastro.

    O regime equiparado exige uma **segunda avaliação** do modelo de referência, com
    `recall_alvo = benchmark.recall_da_regra_no_treino`. É o cálculo mais frágil da
    Sprint 1, o único que atravessa treino e teste de propósito, e por isso o que mais
    precisa estar versionado: o recall da regra é medido **no treino**, porque
    calibrar o limiar pelo recall que ela obtém no teste vazaria o teste para dentro
    da decisão de limiar.

    Returns:
        DataFrame de três linhas com `regime`, `encaminhados`, `taxa_encaminhamento`,
        `casos_detectados`, `casos_perdidos`, `recall` e `nns`. A advertência sobre
        volume absoluto fica em `attrs["nota"]`.
    """
    regra = next((r for r in resultados if r["modelo"] == nome_regra), None)
    if regra is None:
        raise ValueError(f"Comparador {nome_regra!r} ausente dos resultados.")

    padrao = modelo_de_referencia(resultados)
    alvo_da_regra = bm.recall_da_regra_no_treino(particao.X_tr, particao.y_tr)

    equiparado = ev.avaliar(
        md.montar(md.estimadores()[padrao["modelo"]], padrao["conjunto"]),
        particao.X_tr, particao.y_tr, particao.X_te, particao.y_te,
        nome=padrao["modelo"],
        conjunto=padrao["conjunto"],
        recall_alvo=alvo_da_regra,
        salvar_figuras=False,
    )

    rotulos = {
        id(regra): nome_regra,
        id(equiparado): (f"{_rotulo(padrao)}, recall equiparado "
                         f"(alvo {alvo_da_regra:.4f} no treino)"),
        id(padrao): f"{_rotulo(padrao)}, ponto padrão (recall-alvo {padrao['recall_alvo']:.2f})",
    }

    linhas = []
    for r in (regra, equiparado, padrao):
        total = r["vp"] + r["vn"] + r["fp"] + r["fn"]
        encaminhados = r["vp"] + r["fp"]
        linhas.append({
            "regime": rotulos[id(r)],
            "encaminhados": encaminhados,
            "taxa_encaminhamento": encaminhados / total,
            "casos_detectados": r["vp"],
            "casos_perdidos": r["fn"],
            "recall": r["recall"],
            "nns": r["nns"],
        })

    tabela = pd.DataFrame(linhas)
    tabela.attrs["nota"] = bm.NOTA_VOLUME_ABSOLUTO
    tabela.attrs["leitura"] = bm.leitura_operacional(
        resultados, nome_regra=nome_regra, equiparados=[equiparado, regra]
    )
    return tabela


def exportar(tabela: pd.DataFrame, nome: str, nota: str | None = None) -> dict[str, Path]:
    """Grava a tabela como `.tex` e `.csv` em `reports/tabelas/`.

    O `.csv` não é cópia redundante do `.tex`: é sobre ele que a conferência de
    reprodução trabalha, sem ter de reparsear LaTeX.
    """
    cfg.PATH_TABELAS.mkdir(parents=True, exist_ok=True)
    destino_csv = cfg.PATH_TABELAS / f"{nome}.csv"
    tabela.to_csv(destino_csv, index=False, encoding="utf-8")
    if nota is None:
        nota = tabela.attrs.get("nota")
    destino_tex = ev.exportar_latex(tabela, f"{nome}.tex", nota=nota)
    return {"tex": destino_tex, "csv": destino_csv}


def exportar_descritivas(particao: Particao,
                         analitico: pd.DataFrame | None = None) -> dict[str, dict]:
    """Emite o quadro da amostra e as contagens de ausentes.

    Os dois números saíam de `data` direto para o texto, sem passar por arquivo
    nenhum. É a mesma lacuna que a tabela consolidada tinha antes de existir ponto de
    reprodução: quem transcreve não tem contra o que conferir, e o valor no documento
    envelhece sem que nada acuse.

    O quadro da amostra traz as duas prevalências, a amostral e a ponderada por
    `peso_lab`. A ponderada é estimativa de parâmetro populacional e vive só na
    descrição da amostra; nenhum estimador a enxerga (D-009, D-031). As contagens de
    ausentes são as que o `KNNImputer` vai tratar, e a coluna do treino é a que vale,
    porque é onde a imputação ajusta (D-011).

    Args:
        analitico: base limpa. Omitida, relê o CSV bruto. Passe a sua se já tiver uma
            em mãos, para não pagar a leitura duas vezes.
    """
    analitico = dt.carregar_analitico() if analitico is None else analitico
    return {
        NOME_RESUMO: exportar(dt.resumo_amostra(analitico), NOME_RESUMO, nota=""),
        NOME_AUSENTES: exportar(
            dt.ausentes_por_variavel(analitico, treino=particao.X_tr),
            NOME_AUSENTES, nota="",
        ),
    }


# ---------------------------------------------------------------------------
# Conferência de reprodução
# ---------------------------------------------------------------------------

def _ler_referencia(path: Path) -> pd.DataFrame:
    """Lê o `tabular` de um `.tex` gerado por `DataFrame.to_latex`."""
    linhas = path.read_text(encoding="utf-8").splitlines()
    inicio = next(i for i, l in enumerate(linhas) if l.startswith(r"\toprule"))
    fim = next(i for i, l in enumerate(linhas) if l.startswith(r"\bottomrule"))
    corpo = [l for l in linhas[inicio + 1:fim] if not l.startswith(r"\midrule")]
    registros = [[c.strip() for c in l.rstrip("\\ ").split("&")] for l in corpo]
    tabela = pd.DataFrame(registros[1:], columns=registros[0])
    for coluna in tabela.columns:
        convertida = pd.to_numeric(tabela[coluna], errors="coerce")
        if convertida.notna().all():
            tabela[coluna] = convertida
    return tabela


def conferir_reproducao(tabela: pd.DataFrame,
                        referencia: Path | None = None,
                        casas: int = 3) -> pd.DataFrame:
    """Compara a tabela emitida com o artefato rastreado da Sprint 1.

    Confere só as colunas e as linhas presentes nos dois lados: a referência é
    anterior às colunas de dispersão entre folds e às de calibração, e cobre apenas
    as três linhas da LogReg. O que ela cobre tem de bater até a terceira casa; o que
    ela não cobre fica estabelecido agora.

    Returns:
        DataFrame das divergências, com `linha`, `coluna`, `referencia` e `obtido`.
        Vazio significa reprodução exata no escopo coberto.
    """
    referencia = (cfg.PATH_TABELAS / REFERENCIA) if referencia is None else Path(referencia)
    if not referencia.exists():
        raise FileNotFoundError(f"Referência ausente: {referencia}")

    esperado = _ler_referencia(referencia)
    chave = ["modelo", "conjunto"]
    numericas = [c for c in esperado.columns
                 if c in tabela.columns and c not in chave
                 and pd.api.types.is_numeric_dtype(esperado[c])]

    obtido = tabela.set_index(chave)
    esperado = esperado.set_index(chave)
    comuns = esperado.index.intersection(obtido.index)

    divergencias = []
    for linha in comuns:
        for coluna in numericas:
            a = round(float(esperado.loc[linha, coluna]), casas)
            b = round(float(obtido.loc[linha, coluna]), casas)
            if a != b:
                divergencias.append({
                    "linha": " · ".join(linha), "coluna": coluna,
                    "referencia": a, "obtido": b,
                })
    return pd.DataFrame(divergencias,
                        columns=["linha", "coluna", "referencia", "obtido"])


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------

def executar(modelos: list[str] | None = None,
             conjuntos: list[str] | None = None,
             salvar_figuras: bool = False,
             conferir: bool = True,
             particao: Particao | None = None,
             descritivas: bool = True) -> dict:
    """Roda o grid, emite os artefatos e confere a reprodução.

    Args:
        particao: split pronto. Omitido, `preparar_particao` constrói um. O parâmetro
            existe para quem já tem a partição em mãos: o notebook de resultados
            diagnostica colinearidade sobre `X_tr` antes de chamar aqui, e reler o CSV
            só para refazer o mesmo split seria gasto sem ganho. A partição volta no
            retorno para o chamador não ter de reconstruí-la depois.
        descritivas: emite também o quadro da amostra e as contagens de ausentes.
            Desligue em teste que só precisa do grid.

    Returns:
        Dicionário com `resultados`, `consolidado`, `carga`, `artefatos`,
        `divergencias` e `particao`.
    """
    particao = preparar_particao() if particao is None else particao
    resultados = rodar_grid(modelos, conjuntos, particao=particao,
                            salvar_figuras=salvar_figuras)

    consolidado = ev.comparar(resultados)
    carga = carga_operacional(resultados, particao)

    # Slope e intercepto de calibração saem de `diagnostico_calibracao` e não têm
    # coluna em `comparar`, então sem esta linha eles não existem em arquivo nenhum.
    # São a evidência que sustenta D-022 e D-040.
    calibracao = ev.tabela_calibracao_comparada(resultados)

    artefatos = {
        NOME_CONSOLIDADO: exportar(consolidado, NOME_CONSOLIDADO),
        NOME_CARGA: exportar(carga, NOME_CARGA),
        NOME_CALIBRACAO: exportar(calibracao, NOME_CALIBRACAO,
                                  nota=ev.nota_corte_unico(consolidado)),
    }
    if descritivas:
        artefatos.update(exportar_descritivas(particao))

    divergencias = pd.DataFrame()
    if conferir:
        divergencias = conferir_reproducao(consolidado)

    return {
        "resultados": resultados,
        "consolidado": consolidado,
        "carga": carga,
        "artefatos": artefatos,
        "divergencias": divergencias,
        "particao": particao,
    }


def main(argv: list[str] | None = None) -> int:
    """Entrada de linha de comando. É o que o `make resultados` chama de fato."""
    parser = argparse.ArgumentParser(
        prog="python -m src.experimento",
        description="Reproduz as tabelas consolidadas a partir do CSV bruto.",
    )
    parser.add_argument("--modelos", nargs="+", default=None,
                        help=f"padrão: {' '.join(MODELOS_PADRAO)}")
    parser.add_argument("--conjuntos", nargs="+", default=None,
                        help=f"padrão: {' '.join(ft.FEATURE_SETS)}")
    parser.add_argument("--figuras", action="store_true",
                        help="salva as figuras do teste em reports/figuras/")
    parser.add_argument("--sem-conferencia", action="store_true",
                        help="não compara com o artefato da Sprint 1")
    args = parser.parse_args(argv)

    saida = executar(modelos=args.modelos, conjuntos=args.conjuntos,
                     salvar_figuras=args.figuras,
                     conferir=not args.sem_conferencia)

    print(ev.comparar_formatado(saida["resultados"]).to_string(index=False))
    print()
    print(saida["carga"].to_string(index=False))
    print()
    print(saida["carga"].attrs["leitura"])
    print()
    for nome, caminhos in saida["artefatos"].items():
        print(f"{nome}: {caminhos['tex'].name}, {caminhos['csv'].name}")

    divergencias = saida["divergencias"]
    if args.sem_conferencia:
        return 0
    if divergencias.empty:
        print(f"\nReprodução conferida contra {REFERENCIA}: sem divergência "
              f"na terceira casa.")
        return 0

    print(f"\nDIVERGÊNCIA contra {REFERENCIA}:")
    print(divergencias.to_string(index=False))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
