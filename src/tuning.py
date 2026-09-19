"""Busca de hiperparâmetros da Random Forest.

Uma responsabilidade só: achar a configuração da floresta no conjunto de treino e deixar
o resultado num artefato versionado. Não avalia no teste, não calibra limiar, não exporta
tabela de resultado.

A busca roda sobre o pipeline inteiro, com o preprocessador como primeiro estágio e a
floresta como segundo. Transformar os dados uma vez e só então cruzar validação faria o
imputador por KNN enxergar os cinco folds de uma vez, que é o vazamento que a metodologia
proíbe: imputação, escalonamento e codificação ajustam no treino de cada fold e só
transformam o resto.

O pipeline recebe cache de transformador. Sem cache o preprocessador refita uma vez por
candidato, e o KNN é quadrático no número de linhas; com cache o ajuste cai para uma vez
por partição, porque a chave inclui os dados e os parâmetros do estágio, que não mudam
entre candidatos. O diretório de cache fica fora do repositório e some ao final.

Paralelismo em um nível só: a busca usa todos os núcleos e a floresta usa um. Os dois
disputando os mesmos núcleos ficam mais lentos que qualquer um sozinho.

Os dois conjuntos recebem a mesma lista de candidatos, porque o sorteio depende da
semente e do espaço, não dos dados. É isso que deixa a condição de revisão dos
hiperparâmetros compartilhados legível direto do artefato: o vencedor de `B_composto`
está entre os candidatos de `A_atomico`, com score medido nos mesmos folds, sem precisar
de um ajuste extra.

Uso:

    python -m src.tuning                       # cronometra, projeta e busca
    python -m src.tuning --apenas-cronometragem
    python -m src.tuning --n-iter 30           # corte de orçamento, a registrar
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from . import artefatos as art
from . import config as cfg
from . import evaluate as ev
from . import experimento as xp
from . import features as ft

# `C_completo` fica fora: o diagnóstico de multicolinearidade já o descartou, e buscar
# hiperparâmetros para um conjunto que não vai a resultado é orçamento jogado fora.
CONJUNTOS = ["A_atomico", "B_composto"]

N_ESTIMATORS = 500
N_ITER = 60
SCORING = "average_precision"
ORCAMENTO_MINUTOS = 60.0

NOME_ARTEFATO = "busca_rf"
TOPO_TEX = 10

# Espaço da busca, com o prefixo do estágio classificador do pipeline.
#
# `min_samples_leaf` tem piso em 5 de propósito. São 474 positivos no treino, e folha
# unitária produz probabilidade quase degenerada, com curva precision-recall em degraus
# grossos, o que piora a resolução da seleção de limiar que vem depois.
#
# `class_weight` traz os dois sabores porque numa floresta com bootstrap eles não são a
# mesma coisa: `balanced` pesa pela distribuição do treino inteiro e aplica dentro de uma
# árvore ajustada sobre outra distribuição, enquanto `balanced_subsample` recalcula o peso
# dentro de cada reamostragem. A métrica decide qual fica.
ESPACO_BUSCA = {
    "clf__max_depth": [None, 6, 10, 14, 20],
    "clf__min_samples_leaf": [5, 10, 20, 40],
    "clf__max_features": ["sqrt", "log2", 0.3, 0.5],
    "clf__class_weight": ["balanced", "balanced_subsample"],
}

# Nomes dos hiperparâmetros sem o prefixo, na ordem das colunas do artefato.
PARAMS = [nome.removeprefix("clf__") for nome in ESPACO_BUSCA]

COLUNAS_FIXAS = ["conjunto", "posicao", "pr_auc_media", "pr_auc_desvio"]

NOTA_TEX = (
    "As dez melhores configurações de cada conjunto de atributos, por PR-AUC média entre "
    "os cinco folds da validação cruzada no treino. A busca sorteia a mesma lista de "
    "candidatos para os dois conjuntos, o que torna os scores diretamente comparáveis "
    "linha a linha. A lista completa está em \\texttt{busca\\_rf.csv}."
)


def floresta() -> RandomForestClassifier:
    """A floresta que a busca configura, com o que não está em disputa."""
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,   # convergência, não capacidade: buscar aqui seleciona ruído
        n_jobs=1,                    # o paralelismo mora na busca, não na floresta
        random_state=cfg.RANDOM_STATE,
    )


def particionador() -> StratifiedKFold:
    """A validação cruzada do projeto, embaralhada e com semente fixa."""
    return StratifiedKFold(
        n_splits=cfg.CV_FOLDS,
        shuffle=True,
        random_state=cfg.RANDOM_STATE,
    )


def montar_pipeline(conjunto: str, memory: str | Path | None = None) -> Pipeline:
    """Preprocessador mais floresta, com cache opcional do primeiro estágio.

    Args:
        conjunto: chave de `features.FEATURE_SETS`.
        memory: diretório de cache do joblib. Passado, o ajuste do preprocessador é
            reaproveitado entre candidatos que compartilham a mesma partição.
    """
    return Pipeline(
        [("pre", ft.construir_preprocessador(conjunto)), ("clf", floresta())],
        memory=None if memory is None else str(memory),
    )


def candidatos(n_iter: int = N_ITER) -> list[dict]:
    """A lista de candidatos que a busca vai percorrer, na ordem do sorteio.

    Mesma semente e mesmo espaço que a busca usa internamente, então esta lista é a
    dela. Serve para cronometrar um ajuste que a busca de fato vai fazer, em vez de um
    ajuste inventado para a medição.
    """
    return list(ParameterSampler(ESPACO_BUSCA, n_iter=n_iter,
                                 random_state=cfg.RANDOM_STATE))


# ---------------------------------------------------------------------------
# Cronometragem
# ---------------------------------------------------------------------------

def cronometrar_ajuste(particao: xp.Particao,
                       conjunto: str = "B_composto",
                       n_iter: int = N_ITER,
                       n_conjuntos: int = len(CONJUNTOS)) -> dict:
    """Ajusta o pipeline uma vez num fold e projeta o custo da busca inteira.

    Mede os dois estágios separados, porque eles escalam de formas diferentes: com cache,
    o preprocessador roda uma vez por partição e a floresta uma vez por candidato.

    Returns:
        Dicionário com `segundos_pre`, `segundos_clf`, `segundos_ajuste`, as duas
        projeções em segundos e o candidato cronometrado.

    As projeções ignoram o paralelismo da busca, que divide o tempo de parede pelo número
    de núcleos. Erram para cima, que é o lado certo de errar num orçamento.
    """
    treino, _ = next(iter(particionador().split(particao.X_tr, particao.y_tr)))
    X = particao.X_tr.iloc[treino]
    y = particao.y_tr.iloc[treino]

    candidato = candidatos(n_iter)[0]

    pre = ft.construir_preprocessador(conjunto)
    marca = time.perf_counter()
    Z = pre.fit_transform(X, y)
    segundos_pre = time.perf_counter() - marca

    clf = floresta().set_params(
        **{nome.removeprefix("clf__"): valor for nome, valor in candidato.items()}
    )
    marca = time.perf_counter()
    clf.fit(Z, y)
    segundos_clf = time.perf_counter() - marca

    ajuste = segundos_pre + segundos_clf
    total = n_iter * cfg.CV_FOLDS * n_conjuntos
    return {
        "conjunto": conjunto,
        "candidato": {nome.removeprefix("clf__"): valor
                      for nome, valor in candidato.items()},
        "segundos_pre": segundos_pre,
        "segundos_clf": segundos_clf,
        "segundos_ajuste": ajuste,
        "ajustes_previstos": total,
        "projecao_sem_cache": ajuste * total,
        "projecao_com_cache": (segundos_pre * cfg.CV_FOLDS * n_conjuntos
                               + segundos_clf * total),
    }


def formatar_cronometragem(medida: dict, orcamento_minutos: float = ORCAMENTO_MINUTOS) -> str:
    """Texto de uma linha por número, para a saída da CLI."""
    def minutos(segundos: float) -> str:
        return f"{segundos / 60:.1f} min"

    return "\n".join([
        f"Ajuste frio em {medida['conjunto']}, candidato {medida['candidato']}",
        f"  preprocessador      {medida['segundos_pre']:.1f} s",
        f"  floresta            {medida['segundos_clf']:.1f} s",
        f"  ajuste completo     {medida['segundos_ajuste']:.1f} s",
        f"Projeção para {medida['ajustes_previstos']} ajustes:",
        f"  sem cache           {minutos(medida['projecao_sem_cache'])}",
        f"  com cache           {minutos(medida['projecao_com_cache'])}",
        f"Orçamento             {orcamento_minutos:.0f} min",
    ])


# ---------------------------------------------------------------------------
# Busca
# ---------------------------------------------------------------------------

def buscar(conjunto: str, particao: xp.Particao,
           cache: str | Path | None = None,
           n_iter: int = N_ITER) -> RandomizedSearchCV:
    """Roda a busca aleatória num conjunto e devolve o objeto já ajustado.

    Busca aleatória em vez de exaustiva: o produto cartesiano passa de 160 combinações, e
    amostrar cobre melhor o espaço por unidade de tempo quando poucos hiperparâmetros
    dominam o resultado, que é o caso da floresta.
    """
    busca = RandomizedSearchCV(
        montar_pipeline(conjunto, memory=cache),
        param_distributions=ESPACO_BUSCA,
        n_iter=n_iter,
        scoring=SCORING,       # ordenação ao longo da curva inteira, porque o limiar sai depois
        cv=particionador(),
        n_jobs=-1,             # paralelismo em um nível só, a floresta fica em 1
        refit=False,           # o vencedor vira literal em models.py, não sai daqui ajustado
        return_train_score=False,
        error_score="raise",   # candidato que quebra tem de aparecer, não virar score vazio
        random_state=cfg.RANDOM_STATE,
    )
    busca.fit(particao.X_tr, particao.y_tr)
    return busca


def tabela_da_busca(busca: RandomizedSearchCV, conjunto: str) -> pd.DataFrame:
    """Uma linha por candidato, ordenada pela posição no ranking.

    `max_depth` sem limite vira a string `"None"` de propósito: célula vazia no CSV se
    confunde com valor ausente, e quem for copiar o vencedor para a configuração precisa
    ler o que estava lá.
    """
    resultados = busca.cv_results_
    linhas = []
    for posicao, media, desvio, params in zip(resultados["rank_test_score"],
                                              resultados["mean_test_score"],
                                              resultados["std_test_score"],
                                              resultados["params"]):
        limpos = {nome.removeprefix("clf__"): valor for nome, valor in params.items()}
        linha = {
            "conjunto": conjunto,
            "posicao": int(posicao),
            "pr_auc_media": float(media),
            "pr_auc_desvio": float(desvio),
        }
        linha.update({nome: ("None" if limpos[nome] is None else limpos[nome])
                      for nome in PARAMS})
        linha["n_estimators"] = N_ESTIMATORS
        linhas.append(linha)

    tabela = pd.DataFrame(linhas, columns=COLUNAS_FIXAS + PARAMS + ["n_estimators"])
    return tabela.sort_values("posicao").reset_index(drop=True)


def vencedor(tabela: pd.DataFrame, conjunto: str) -> pd.Series:
    """A linha de melhor PR-AUC média de um conjunto."""
    do_conjunto = tabela[tabela["conjunto"] == conjunto]
    if do_conjunto.empty:
        raise ValueError(f"Conjunto {conjunto!r} ausente da tabela da busca.")
    return do_conjunto.sort_values("posicao").iloc[0]


def condicao_revisao(tabela: pd.DataFrame,
                     base: str = "B_composto",
                     alvo: str = "A_atomico") -> dict:
    """Confere se o vencedor de um conjunto se sustenta no outro.

    Compartilhar hiperparâmetros entre os dois conjuntos só se justifica enquanto eles
    forem indistinguíveis em desempenho. A conferência é direta porque os dois receberam a
    mesma lista de candidatos: o vencedor de `base` já foi avaliado em `alvo`, nos mesmos
    folds, e basta procurar a linha.

    Returns:
        Dicionário com a PR-AUC do vencedor próprio de `alvo`, a do vencedor emprestado
        de `base`, a queda entre as duas, o desvio entre folds do vencedor próprio e o
        sinalizador `revisar`, verdadeiro quando a queda passa desse desvio.
    """
    proprio = vencedor(tabela, alvo)
    emprestador = vencedor(tabela, base)

    do_alvo = tabela[tabela["conjunto"] == alvo]
    mascara = pd.Series(True, index=do_alvo.index)
    for nome in PARAMS:
        mascara &= do_alvo[nome] == emprestador[nome]
    emprestado = do_alvo[mascara]
    if emprestado.empty:
        raise ValueError(
            f"O vencedor de {base!r} não foi avaliado em {alvo!r}. As duas buscas "
            f"deveriam sortear a mesma lista de candidatos."
        )
    emprestado = emprestado.iloc[0]

    queda = float(proprio["pr_auc_media"]) - float(emprestado["pr_auc_media"])
    desvio = float(proprio["pr_auc_desvio"])
    return {
        "base": base,
        "alvo": alvo,
        "pr_auc_proprio": float(proprio["pr_auc_media"]),
        "pr_auc_emprestado": float(emprestado["pr_auc_media"]),
        "posicao_emprestado": int(emprestado["posicao"]),
        "queda": queda,
        "desvio_entre_folds": desvio,
        "revisar": bool(queda > desvio),
    }


def formatar_revisao(revisao: dict) -> str:
    """Leitura da condição de revisão, em texto corrido."""
    veredito = ("acima do desvio entre folds: a decisão de compartilhar volta à mesa"
                if revisao["revisar"] else
                "dentro do desvio entre folds: compartilhar segue valendo")
    return (
        f"Vencedor de {revisao['base']} aplicado em {revisao['alvo']}: "
        f"PR-AUC {revisao['pr_auc_emprestado']:.4f} "
        f"(posição {revisao['posicao_emprestado']} no ranking de {revisao['alvo']}), "
        f"contra {revisao['pr_auc_proprio']:.4f} do vencedor próprio. "
        f"Queda de {revisao['queda']:.4f}, desvio entre folds "
        f"{revisao['desvio_entre_folds']:.4f}. {veredito.capitalize()}."
    )


# ---------------------------------------------------------------------------
# Artefato
# ---------------------------------------------------------------------------

def exportar(tabela: pd.DataFrame, topo: int = TOPO_TEX) -> dict[str, Path]:
    """Grava a busca como `.csv` com todos os candidatos e `.tex` com os primeiros.

    Sessenta linhas por conjunto não entram num documento acadêmico, e cortar na
    exportação é melhor que cortar na fonte: quem quiser auditar o espaço inteiro abre o
    CSV.
    """
    destino_csv = art.salvar_csv(tabela,
                                 cfg.PATH_TABELAS / f"{NOME_ARTEFATO}.csv")

    recorte = (tabela.sort_values(["conjunto", "posicao"])
                     .groupby("conjunto", sort=True)
                     .head(topo)
                     .reset_index(drop=True))
    destino_tex = ev.exportar_latex(recorte, f"{NOME_ARTEFATO}.tex",
                                    nota=NOTA_TEX, float_format="%.4f")
    return {"tex": destino_tex, "csv": destino_csv}


def carregar_busca(path: Path | None = None) -> pd.DataFrame:
    """Lê o artefato da busca de volta, com os tipos que ele foi gravado.

    Existe porque `read_csv` no padrão engole a string `"None"` e devolve valor ausente
    no lugar, que é justamente a leitura errada de floresta sem limite de profundidade.
    Quem for conferir o vencedor contra a configuração congelada lê por aqui.
    """
    path = (cfg.PATH_TABELAS / f"{NOME_ARTEFATO}.csv") if path is None else Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artefato da busca ausente: {path}")
    return pd.read_csv(path, keep_default_na=False, na_values=[""], encoding="utf-8")


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------

def executar(conjuntos: list[str] | None = None,
             n_iter: int = N_ITER,
             particao: xp.Particao | None = None,
             orcamento_minutos: float = ORCAMENTO_MINUTOS,
             forcar: bool = False,
             salvar: bool = True) -> dict:
    """Cronometra, decide se cabe no orçamento, roda a busca e emite o artefato.

    Args:
        orcamento_minutos: teto da projeção sem cache. Estourado, a busca não dispara.
        forcar: dispara mesmo com a projeção acima do teto.
        salvar: desligado, roda a busca sem escrever em `reports/tabelas/`.

    Returns:
        Dicionário com `cronometragem`, `tabela`, `vencedores`, `revisao`, `artefatos`,
        `segundos` e `estourou`. Com a projeção acima do teto e sem `forcar`, volta só
        com `cronometragem` e `estourou` verdadeiro.

    O diretório de cache é criado fora do repositório e removido em qualquer desfecho:
    cache sobrevivente entre execuções esconde mudança de dado atrás de um score antigo.
    """
    conjuntos = list(CONJUNTOS) if conjuntos is None else list(conjuntos)
    desconhecidos = [c for c in conjuntos if c not in ft.FEATURE_SETS]
    if desconhecidos:
        raise ValueError(
            f"Conjunto(s) desconhecido(s): {desconhecidos}. "
            f"Disponíveis: {sorted(ft.FEATURE_SETS)}"
        )

    particao = xp.preparar_particao() if particao is None else particao
    medida = cronometrar_ajuste(particao, conjunto=conjuntos[-1], n_iter=n_iter,
                                n_conjuntos=len(conjuntos))
    estourou = medida["projecao_sem_cache"] > orcamento_minutos * 60
    if estourou and not forcar:
        return {"cronometragem": medida, "estourou": True}

    cache = Path(tempfile.mkdtemp(prefix="cache_busca_rf_"))
    marca = time.perf_counter()
    try:
        tabelas = [tabela_da_busca(buscar(conjunto, particao, cache=cache, n_iter=n_iter),
                                   conjunto)
                   for conjunto in conjuntos]
    finally:
        shutil.rmtree(cache, ignore_errors=True)
    segundos = time.perf_counter() - marca

    tabela = pd.concat(tabelas, ignore_index=True)
    artefatos = exportar(tabela) if salvar else {}
    revisao = (condicao_revisao(tabela) if set(CONJUNTOS) <= set(conjuntos) else None)

    return {
        "cronometragem": medida,
        "estourou": estourou,
        "tabela": tabela,
        "vencedores": {conjunto: vencedor(tabela, conjunto) for conjunto in conjuntos},
        "revisao": revisao,
        "artefatos": artefatos,
        "segundos": segundos,
    }


def main(argv: list[str] | None = None) -> int:
    """Entrada de linha de comando.

    Sai com 2 quando a projeção estoura o orçamento e nada foi buscado: o corte de
    `n_iter` é decisão de quem toca o projeto, e registrada, não improviso do módulo.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.tuning",
        description="Busca os hiperparâmetros da Random Forest no conjunto de treino.",
    )
    parser.add_argument("--conjuntos", nargs="+", default=None,
                        help=f"padrão: {' '.join(CONJUNTOS)}")
    parser.add_argument("--n-iter", type=int, default=N_ITER,
                        help=f"candidatos sorteados por conjunto (padrão: {N_ITER})")
    parser.add_argument("--orcamento", type=float, default=ORCAMENTO_MINUTOS,
                        help=f"teto da projeção, em minutos (padrão: {ORCAMENTO_MINUTOS:.0f})")
    parser.add_argument("--apenas-cronometragem", action="store_true",
                        help="mede e projeta sem disparar a busca")
    parser.add_argument("--forcar", action="store_true",
                        help="dispara mesmo com a projeção acima do teto")
    parser.add_argument("--sem-artefato", action="store_true",
                        help="não escreve em reports/tabelas/")
    args = parser.parse_args(argv)

    if args.apenas_cronometragem:
        particao = xp.preparar_particao()
        conjuntos = args.conjuntos or list(CONJUNTOS)
        medida = cronometrar_ajuste(particao, conjunto=conjuntos[-1],
                                    n_iter=args.n_iter, n_conjuntos=len(conjuntos))
        print(formatar_cronometragem(medida, args.orcamento))
        return 0

    saida = executar(conjuntos=args.conjuntos, n_iter=args.n_iter,
                     orcamento_minutos=args.orcamento, forcar=args.forcar,
                     salvar=not args.sem_artefato)

    print(formatar_cronometragem(saida["cronometragem"], args.orcamento))
    print()
    if saida["estourou"] and "tabela" not in saida:
        print("Projeção acima do orçamento. Nada foi buscado.")
        print("Reduza --n-iter e registre o corte, ou repita com --forcar.")
        return 2

    print(f"Busca concluída em {saida['segundos'] / 60:.1f} min.")
    print()
    for conjunto, linha in saida["vencedores"].items():
        params = ", ".join(f"{nome}={linha[nome]}" for nome in PARAMS)
        print(f"{conjunto}: PR-AUC {linha['pr_auc_media']:.4f} "
              f"± {linha['pr_auc_desvio']:.4f} | {params}")
    if saida["revisao"] is not None:
        print()
        print(formatar_revisao(saida["revisao"]))
    if saida["artefatos"]:
        print()
        print(f"{NOME_ARTEFATO}: {saida['artefatos']['tex'].name}, "
              f"{saida['artefatos']['csv'].name} "
              f"({len(saida['tabela'])} candidatos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
