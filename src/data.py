"""Carga, limpeza e exclusões da Subamostra Laboratorial da PNS 2013.

Só operações row-wise: nada aqui olha para mais de uma linha ao mesmo tempo.
Imputação não acontece neste módulo; ela é pós-split e vive em `src/features.py`.

Registro sem HbA1c válida sai da base, nunca é imputado. Imputar o alvo
comprometeria a validade do desfecho.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


def _para_numero(serie: pd.Series) -> pd.Series:
    """Converte uma série de texto para numérico, tratando o decimal brasileiro.

    Os microdados da PNS gravam os valores com vírgula decimal (`'38,00'`,
    `'0,58'`). Sem tratar, o Pandas mantém tudo como texto e o pipeline quebra sem
    avisar. Célula vazia vira `NaN`.
    """
    limpa = serie.astype("string").str.strip().str.replace(",", ".", regex=False)
    limpa = limpa.replace("", pd.NA)
    return pd.to_numeric(limpa, errors="coerce")


def carregar_bruto(path=None) -> pd.DataFrame:
    """Lê do CSV bruto o recorte de 16 variáveis, mais o alvo e o peso amostral.

    Leitura seletiva, com `dtype=str` e `keep_default_na=False`: a conversão de
    tipo fica toda na limpeza, explícita e auditável.

    Args:
        path: caminho do CSV bruto. Omitido, usa `config.PATH_RAW`.

    Returns:
        DataFrame com as colunas já renomeadas pelo `DE_PARA`, ainda como texto.
    """
    path = cfg.PATH_RAW if path is None else path
    df = pd.read_csv(path, usecols=cfg.COLS_ORIGEM, dtype=str, keep_default_na=False)
    return df.rename(columns=cfg.DE_PARA)


def derivar(df: pd.DataFrame) -> pd.DataFrame:
    """Passos da limpeza que valem para uma linha isolada, nesta ordem:

    1. Conversão de tipos: contínuas em `float64`, categóricas em `Int64` nullable.
    2. Decodificação de ausência: `raca_cor == 9` ("Ignorado") vira `NaN`.
    3. Derivadas: `imc` (Quetelet) e `rcest`.

    É a única implementação dessas fórmulas. `limpar` a chama no treino e a API a
    chama na inferência: duas cópias do IMC seriam duas fórmulas prontas para
    divergir. Não exige `hba1c` nem `peso_lab`, que a inferência não tem.
    """
    df = df.copy()

    for col in cfg.CONTINUAS:
        df[col] = _para_numero(df[col]).astype("float64")
    for col in cfg.CATEGORICAS:
        df[col] = _para_numero(df[col]).astype("Int64")

    # "Ignorado" não é categoria, é não resposta.
    df["raca_cor"] = df["raca_cor"].replace(cfg.COD_IGNORADO_RACA, pd.NA)

    df["imc"] = df["peso_kg"] / (df["altura_cm"] / 100.0) ** 2   # Quetelet
    df["rcest"] = df["circ_cintura"] / df["altura_cm"]
    return df


def limpar(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica a limpeza row-wise, nesta ordem, que não é negociável:

    1. `derivar`: tipos, `raca_cor == 9` para `NaN`, `imc` e `rcest`.
    2. `drop_duplicates()`.
    3. Exclusões: alvo ausente e bloco de medição inteiramente ausente.
    4. Binarização do desfecho pelo critério ADA 2010.

    As derivadas vêm antes das duplicadas sem mudar o conjunto de duplicadas: são
    função de colunas que já entram na comparação.

    Nada de imputação aqui: ela é toda pós-split.
    """
    df = df.copy()

    for col in [cfg.ALVO_CONTINUO, cfg.PESO]:
        df[col] = _para_numero(df[col]).astype("float64")
    df = derivar(df)

    n_brutas = len(df)
    df = df.drop_duplicates().reset_index(drop=True)

    # Sem HbA1c válida, o registro sai da base.
    sem_alvo = df[cfg.ALVO_CONTINUO].isna()
    sem_medicao = df[cfg.MEDICAO].isna().all(axis=1)

    # As contagens ficam registradas onde o corte acontece. Recalculá-las depois,
    # a partir do resultado, exigiria repetir a conversão de tipo e os critérios
    # acima: duas implementações da mesma exclusão, prontas para divergir.
    exclusoes = {
        "linhas_brutas": n_brutas,
        "duplicadas": n_brutas - len(df),
        "sem_hba1c": int(sem_alvo.sum()),
        # Contado só entre quem tem alvo válido, para as duas exclusões não se
        # sobreporem e a soma fechar com o total.
        "sem_bloco_medicao": int((~sem_alvo & sem_medicao).sum()),
    }

    df = df[~sem_alvo & ~sem_medicao].reset_index(drop=True)

    # Desfecho: hiperglicemia crônica clinicamente significativa.
    df[cfg.ALVO] = (df[cfg.ALVO_CONTINUO] >= cfg.LIMIAR_HBA1C).astype(int)

    df.attrs["exclusoes"] = exclusoes
    return df


def separar(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Separa preditoras, desfecho e peso amostral.

    `peso_lab` sai de `X` e volta em separado: ele pondera a EDA descritiva e não
    entra no treino de estimador nenhum.

    Returns:
        Tupla `(X, y, peso_lab)`.
    """
    X = df[cfg.PREDITORAS].copy()
    y = df[cfg.ALVO].copy()
    peso_lab = df[cfg.PESO].copy()
    return X, y, peso_lab


def carregar_analitico(path=None) -> pd.DataFrame:
    """Atalho de orquestração: `carregar_bruto` seguido de `limpar`."""
    return limpar(carregar_bruto(path))


def resumo_amostra(analitico: pd.DataFrame,
                   y_tr: pd.Series | None = None,
                   y_te: pd.Series | None = None) -> pd.DataFrame:
    """Quadro descritivo: do CSV bruto ao conjunto analítico.

    Lê as contagens que `limpar` gravou em `attrs["exclusoes"]` em vez de
    recalculá-las, para cada exclusão ter uma definição só.

    A prevalência aparece duas vezes e as duas medem coisas diferentes. A **não
    ponderada** é a da amostra: é o desbalanceamento que `class_weight='balanced'`
    corrige e o número que contextualiza todas as métricas do teste. A **ponderada
    por `peso_lab`** é estimativa de parâmetro populacional e só existe aqui, na
    EDA descritiva, único lugar do projeto onde o peso amostral opera. Nenhum
    estimador a enxerga.

    Args:
        analitico: saída de `limpar` ou `carregar_analitico`.
        y_tr, y_te: alvo de cada lado do split. Passados os dois, o quadro ganha no
            fim o tamanho do treino e do teste, os negativos da amostra e a
            prevalência de cada lado. Contar depois do split não fere R1: a contagem
            descreve a partição e não entra em nenhum ajuste.

    Returns:
        DataFrame com `item`, `n` (Int64, nulo nas linhas de proporção) e
        `proporcao` (float, nulo nas linhas de contagem).

    Raises:
        KeyError: se `attrs["exclusoes"]` sumiu. Sinal de que o DataFrame não veio
            de `limpar`, ou de que alguma operação no meio do caminho descartou os
            `attrs`.
    """
    if "exclusoes" not in analitico.attrs:
        raise KeyError(
            "attrs['exclusoes'] ausente: passe a saída de `limpar`/"
            "`carregar_analitico` sem operações que descartem os attrs."
        )
    contagens = analitico.attrs["exclusoes"]

    n_total = len(analitico)
    positivos = int(analitico[cfg.ALVO].sum())
    peso = analitico[cfg.PESO]
    prevalencia_ponderada = float(
        (peso * analitico[cfg.ALVO]).sum() / peso.sum()
    )

    linhas = [
        ("Linhas no CSV bruto", contagens["linhas_brutas"], None),
        ("Duplicadas removidas", -contagens["duplicadas"], None),
        ("Excluídas: HbA1c ausente", -contagens["sem_hba1c"], None),
        ("Excluídas: bloco de medição integralmente ausente",
         -contagens["sem_bloco_medicao"], None),
        ("Amostra analítica", n_total, None),
        ("Positivos (HbA1c ≥ 6,5%)", positivos, positivos / n_total),
        ("Prevalência ponderada por peso_lab (descritiva)",
         None, prevalencia_ponderada),
    ]
    if y_tr is not None and y_te is not None:
        positivos_tr, positivos_te = int(y_tr.sum()), int(y_te.sum())
        linhas += [
            ("Treino", len(y_tr), None),
            ("Teste", len(y_te), None),
            ("Negativos", n_total - positivos, None),
            ("Prevalência no treino", positivos_tr, positivos_tr / len(y_tr)),
            ("Prevalência no teste", positivos_te, positivos_te / len(y_te)),
        ]
    return pd.DataFrame(linhas, columns=["item", "n", "proporcao"]).astype(
        {"n": "Int64", "proporcao": "float64"}
    )


def ausentes_por_variavel(analitico: pd.DataFrame,
                          treino: pd.DataFrame | None = None) -> pd.DataFrame:
    """Ausentes que sobram nas preditoras, depois da limpeza e antes da imputação.

    São exatamente os ausentes que o `KNNImputer` vai tratar, ajustado só no
    treino. A coluna do treino é a que importa na prática: é sobre ela que a
    imputação é ajustada.

    Args:
        analitico: base completa, saída de `limpar`.
        treino: partição de treino das preditoras. Omitida, a coluna `no_treino`
            não aparece.

    Returns:
        DataFrame com `variavel`, `ausentes`, `proporcao` e, se `treino` vier,
        `no_treino`. Só as variáveis com ao menos um ausente, em ordem decrescente.
    """
    presentes = [c for c in cfg.PREDITORAS if c in analitico.columns]
    contagem = analitico[presentes].isna().sum()
    contagem = contagem[contagem > 0].sort_values(ascending=False)

    tabela = pd.DataFrame({
        "variavel": contagem.index,
        "ausentes": contagem.to_numpy(),
        "proporcao": contagem.to_numpy() / len(analitico),
    })
    if treino is not None:
        tabela["no_treino"] = [int(treino[v].isna().sum()) for v in tabela["variavel"]]
    return tabela.reset_index(drop=True)
