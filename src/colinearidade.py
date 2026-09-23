"""Diagnóstico de multicolinearidade: Pearson entre contínuas mais VIF.

O VIF roda sobre a matriz de design pós-transformação, dummies incluídas.

Tudo é ajustado no treino. `peso_lab` não entra em função nenhuma daqui.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted

from . import artefatos as art
from . import config as cfg
from . import features as ft

# Paleta divergente Clinical Teal: coral (correlação negativa) -> teal (positiva).
CMAP_CLINICAL_TEAL = LinearSegmentedColormap.from_list(
    "clinical_teal", [cfg.CORAL, "#FFFFFF", cfg.TEAL]
)


# ---------------------------------------------------------------------------
# Correlação de Pearson entre contínuas
# ---------------------------------------------------------------------------


def matriz_pearson(X_train: pd.DataFrame, cols_continuas: list[str]) -> pd.DataFrame:
    """Correlação de Pearson entre as variáveis contínuas do treino.

    Args:
        X_train: matriz de treino. Aceita as preditoras cruas ou a saída dos
            estágios de imputação e compostas; o que importa é ter as colunas
            pedidas.
        cols_continuas: contínuas a correlacionar.

    Returns:
        Matriz de correlação quadrada, índice e colunas em `cols_continuas`.

    Calculada **só no treino**: correlação é estatística entre linhas, e estimá-la
    sobre a base inteira vazaria o teste.
    """
    faltantes = [c for c in cols_continuas if c not in X_train.columns]
    if faltantes:
        raise KeyError(f"Colunas ausentes em X_train: {faltantes}")
    return X_train[cols_continuas].astype("float64").corr(method="pearson")


def pares_pearson(matriz: pd.DataFrame, feature_set: str = "") -> pd.DataFrame:
    """Converte a matriz de Pearson numa lista de pares, ordenada por |r|.

    O heatmap comunica o padrão, mas os coeficientes precisam estar em tabela para
    entrar no texto e serem citados um a um.

    Returns:
        DataFrame com `feature_set`, `var_1`, `var_2`, `pearson` e `abs_pearson`,
        uma linha por par não ordenado, sem a diagonal.
    """
    linhas = []
    colunas = list(matriz.columns)
    for i, a in enumerate(colunas):
        for b in colunas[i + 1:]:
            linhas.append({"feature_set": feature_set, "var_1": a, "var_2": b,
                           "pearson": float(matriz.at[a, b])})
    tabela = pd.DataFrame(linhas)
    tabela["abs_pearson"] = tabela["pearson"].abs()
    return tabela.sort_values("abs_pearson", ascending=False).reset_index(drop=True)


def heatmap_pearson(matriz: pd.DataFrame, feature_set: str,
                    path_figuras: Path | None = None) -> Path:
    """Salva o heatmap da matriz de Pearson e devolve o caminho do arquivo.

    A figura sai em `reports/figuras/pearson_{feature_set}.png`, na paleta Clinical
    Teal.
    """
    path_figuras = cfg.PATH_FIGURAS if path_figuras is None else Path(path_figuras)
    path_figuras.mkdir(parents=True, exist_ok=True)
    destino = path_figuras / f"pearson_{feature_set}.png"

    n = len(matriz)
    fig, ax = plt.subplots(figsize=(1.05 * n + 2, 1.05 * n + 1.2))
    imagem = ax.imshow(matriz.to_numpy(), cmap=CMAP_CLINICAL_TEAL, vmin=-1, vmax=1)

    ax.set_xticks(range(n), matriz.columns, rotation=45, ha="right")
    ax.set_yticks(range(n), matriz.index)
    ax.set_title(f"Correlação de Pearson entre contínuas — {feature_set}",
                 fontweight="bold", color=cfg.INK)

    for i in range(n):
        for j in range(n):
            valor = matriz.iat[i, j]
            # Texto claro sobre fundo saturado, escuro sobre fundo pálido.
            cor = "#FFFFFF" if abs(valor) > 0.55 else cfg.INK
            ax.text(j, i, f"{valor:.2f}", ha="center", va="center",
                    color=cor, fontsize=9)

    barra = fig.colorbar(imagem, ax=ax, shrink=0.82)
    barra.set_label("coeficiente de Pearson", color=cfg.INK)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color=cfg.GRID, linewidth=0.8)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)

    fig.tight_layout()
    fig.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return destino


# ---------------------------------------------------------------------------
# VIF sobre a matriz de design
# ---------------------------------------------------------------------------


def matriz_de_design(pipeline, X_train: pd.DataFrame) -> pd.DataFrame:
    """Aplica os estágios de pré-processamento e devolve a matriz de design nomeada.

    Aceita um preprocessador (último estágio é o `ColumnTransformer`) ou um pipeline
    de modelagem completo (último estágio é o estimador). Só no segundo caso o
    último estágio é descartado, via `pipeline[:-1]`.

    A distinção importa: aplicar `pipeline[:-1]` a um preprocessador devolveria a
    saída de `ConstrutorCompostas`, ou seja, as preditoras antes da codificação, sem
    dummy nenhuma. O VIF calculado ali é o diagnóstico incompleto que o baseline
    fazia.
    """
    estagios = pipeline[:-1] if hasattr(pipeline[-1], "predict") else pipeline
    try:
        check_is_fitted(pipeline[-1])
    except (NotFittedError, TypeError):
        raise NotFittedError(
            "O pipeline precisa estar ajustado no treino antes do diagnóstico de VIF."
        ) from None

    matriz = estagios.transform(X_train)
    nomes = estagios.get_feature_names_out()
    return pd.DataFrame(np.asarray(matriz), columns=list(nomes),
                        index=X_train.index).astype("float64")


def _vif_de_coluna(design: pd.DataFrame, coluna: str) -> float:
    """VIF de uma coluna: `1 / (1 - R²)` da regressão auxiliar contra as demais.

    `R² >= 1` é colinearidade perfeita, com a coluna sendo combinação linear exata
    das outras. Aí devolve infinito em vez de estourar na divisão por zero: VIF
    infinito é o diagnóstico correto, não um erro numérico a esconder. Coluna
    constante cai no mesmo caso, porque sem variância não há regressão auxiliar.
    """
    y = design[coluna].to_numpy()
    if np.isclose(y.var(), 0.0):
        return np.inf

    X = design.drop(columns=[coluna]).to_numpy()
    if X.shape[1] == 0:
        return 1.0

    r2 = LinearRegression().fit(X, y).score(X, y)
    if r2 >= 1.0 or np.isclose(1.0 - r2, 0.0):
        return np.inf
    return float(1.0 / (1.0 - r2))


def vif_matriz_design(pipeline, X_train: pd.DataFrame,
                      feature_set: str) -> pd.DataFrame:
    """VIF de cada coluna da matriz de design pós-transformação.

    Args:
        pipeline: preprocessador **já ajustado no treino**, ou pipeline completo de
            modelagem.
        X_train: preditoras de treino, na forma que o pipeline espera.
        feature_set: nome do conjunto, registrado na tabela para a comparação.

    Returns:
        DataFrame com uma linha por coluna da matriz de design, dummies incluídas,
        e as colunas `feature_set`, `variavel`, `vif` e `acima_do_limiar`, ordenado
        por VIF decrescente.
    """
    design = matriz_de_design(pipeline, X_train)
    vifs = {coluna: _vif_de_coluna(design, coluna) for coluna in design.columns}

    tabela = pd.DataFrame({
        "feature_set": feature_set,
        "variavel": list(vifs),
        "vif": list(vifs.values()),
    })
    # R² da regressão auxiliar: fração da variância da coluna que as demais já
    # explicam. O VIF sozinho engana a leitura: 8,60 soa moderado, mas são 88,4%
    # de variância redundante, ainda abaixo do limiar de 10.
    tabela["r2_auxiliar"] = r2_de_vif(tabela["vif"])
    tabela["acima_do_limiar"] = tabela["vif"] > cfg.VIF_LIMIAR
    return tabela.sort_values("vif", ascending=False).reset_index(drop=True)


def r2_de_vif(vif):
    """Converte VIF em R² da regressão auxiliar: `R² = 1 - 1/VIF`.

    Aceita escalar ou Série. VIF infinito, que é colinearidade perfeita, devolve
    R² = 1.
    """
    vif = pd.Series(vif) if not isinstance(vif, pd.Series) else vif
    return (1.0 - 1.0 / vif).where(np.isfinite(vif), 1.0)


# ---------------------------------------------------------------------------
# Decomposição do escore: o que o VIF não mede
# ---------------------------------------------------------------------------


def preditoras_pos_compostas(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Aplica os estágios até imediatamente antes do `ColumnTransformer`.

    Devolve as preditoras imputadas e compostas em **escala clínica original**, que
    é onde dá para verificar as identidades das variáveis compostas.
    """
    estagios = pipeline[:-1] if hasattr(pipeline[-1], "predict") else pipeline
    saida = X
    for _, etapa in estagios.steps:
        if isinstance(etapa, ColumnTransformer):
            break
        saida = etapa.transform(saida)
    return pd.DataFrame(saida)


def obesidade_reconstruida(preditoras: pd.DataFrame) -> pd.Series:
    """Indicadora de obesidade, exatamente como as compostas a definem.

    Disjuntiva e sensível ao sexo: IMC ≥ 30, ou cintura ≥ 80 cm em mulheres, ou
    ≥ 94 cm em homens (SBD 2019). Reconstruída aqui para deixar visível a terceira
    parcela de `multimorbidade`, que não existe como coluna do modelo.
    """
    fem = preditoras["sexo_fem"] == 1
    return (
        (preditoras["imc"] >= cfg.CORTE_IMC_OBESIDADE)
        | (fem & (preditoras["circ_cintura"] >= cfg.CORTE_CINTURA_F))
        | (~fem & (preditoras["circ_cintura"] >= cfg.CORTE_CINTURA_M))
    ).astype(int).rename("obesidade_reconstruida")


def decomposicao_do_escore(pipeline, X_train: pd.DataFrame,
                           feature_set: str = "C_completo") -> dict:
    """Contrasta a redundância algébrica do escore com o que o VIF detecta.

    `multimorbidade` é, por construção, a soma de três parcelas binárias:
    `has_combinada`, `colesterol_alto == 1` e a indicadora de obesidade. Em
    `C_completo` as duas primeiras são colunas do modelo; a terceira não é, mas
    deriva de `imc`, `circ_cintura` e `sexo_fem`, que são.

    Daí duas regressões auxiliares sobre o mesmo escore:

    - **identidade**: sobre as três parcelas exatas. R² é 1 por definição, porque
      não é estimativa, é a identidade algébrica escrita como regressão.
    - **VIF**: sobre as demais colunas da matriz de design, onde a obesidade só
      aparece de forma indireta, via `imc` e `circ_cintura` **contínuas**. O limiar
      que as binariza não é recuperável por combinação linear.

    A distância entre os dois R² é a parte da redundância que o diagnóstico linear
    não enxerga.

    Returns:
        Dicionário com `r2_identidade`, `r2_vif`, `vif` e `parcelas`.
    """
    preditoras = preditoras_pos_compostas(pipeline, X_train)
    parcelas = pd.DataFrame({
        "has_combinada": preditoras["has_combinada"].astype("float64"),
        "colesterol_sim": preditoras["colesterol_sim"].astype("float64"),
        "obesidade_reconstruida": obesidade_reconstruida(preditoras).astype("float64"),
    })
    escore = preditoras["multimorbidade"].astype("float64").to_numpy()
    r2_identidade = LinearRegression().fit(parcelas.to_numpy(), escore).score(
        parcelas.to_numpy(), escore
    )

    design = matriz_de_design(pipeline, X_train)
    vif = _vif_de_coluna(design, "multimorbidade")
    r2_vif = np.nan if np.isinf(vif) else 1.0 - 1.0 / vif

    return {
        "feature_set": feature_set,
        "r2_identidade": float(r2_identidade),
        "r2_vif": float(r2_vif),
        "vif": float(vif),
        "parcelas": list(parcelas.columns),
    }


def tabela_decomposicao(decomposicao: dict) -> pd.DataFrame:
    """A decomposição do escore em uma linha, pronta para virar artefato.

    Os dois R² eram impressos pelo notebook e não existiam em arquivo nenhum, e são o
    argumento do descarte de `C_completo`: 1,000 da identidade contra 0,884 do VIF.
    """
    return pd.DataFrame([{
        "feature_set": decomposicao["feature_set"],
        "r2_identidade": decomposicao["r2_identidade"],
        "r2_vif": decomposicao["r2_vif"],
        "vif": decomposicao["vif"],
        "parcelas": ", ".join(decomposicao["parcelas"]),
    }])


# ---------------------------------------------------------------------------
# Tabela comparativa A/B/C: resultado do TCC, não diagnóstico interno
# ---------------------------------------------------------------------------


def tabela_comparativa(X_train: pd.DataFrame, y_train=None,
                       feature_sets: list[str] | None = None) -> pd.DataFrame:
    """Roda o VIF para cada conjunto de atributos e empilha os resultados.

    Ajusta um preprocessador por conjunto, sempre **no treino**, e devolve a tabela
    única, ordenada por conjunto e VIF decrescente.
    """
    feature_sets = list(ft.FEATURE_SETS) if feature_sets is None else feature_sets
    partes = []
    for nome in feature_sets:
        pre = ft.construir_preprocessador(nome).fit(X_train, y_train)
        partes.append(vif_matriz_design(pre, X_train, nome))
    return pd.concat(partes, ignore_index=True).sort_values(
        ["feature_set", "vif"], ascending=[True, False]
    ).reset_index(drop=True)


def resumo_por_conjunto(tabela: pd.DataFrame) -> pd.DataFrame:
    """Condensa a tabela de VIF em uma linha por conjunto de atributos."""
    return (
        tabela.groupby("feature_set")
        .agg(
            colunas=("variavel", "count"),
            vif_maximo=("vif", "max"),
            vif_mediano=("vif", "median"),
            acima_do_limiar=("acima_do_limiar", "sum"),
        )
        .reset_index()
    )


def ablacao_vif(X_train: pd.DataFrame, y_train=None,
                feature_set: str = "C_completo",
                variavel: str = "rcest") -> pd.DataFrame:
    """Executa o protocolo de ablação: remove uma variável e recalcula o VIF.

    A metodologia autoriza testar a remoção da variável de maior VIF dentro de um
    mesmo bloco temático. Esta função executa o procedimento e devolve o antes e o
    depois lado a lado.

    Args:
        variavel: coluna a remover do conjunto. Ela mesma não aparece na saída,
            porque o que interessa é o efeito sobre as **demais**.

    Returns:
        DataFrame com `variavel`, `vif_antes`, `vif_depois`, `delta_vif`,
        `r2_antes`, `r2_depois` e as sinalizações de limiar, ordenado pelo VIF
        original.
    """
    pre_antes = ft.construir_preprocessador(feature_set).fit(X_train, y_train)
    antes = vif_matriz_design(pre_antes, X_train, feature_set)

    conjunto = ft.conjunto_sem(feature_set, variavel)
    pre_depois = ft.construir_preprocessador(conjunto).fit(X_train, y_train)
    depois = vif_matriz_design(pre_depois, X_train, f"{feature_set}_sem_{variavel}")

    comparativo = (
        antes[["variavel", "vif", "r2_auxiliar", "acima_do_limiar"]]
        .rename(columns={"vif": "vif_antes", "r2_auxiliar": "r2_antes",
                         "acima_do_limiar": "acima_antes"})
        .merge(
            depois[["variavel", "vif", "r2_auxiliar", "acima_do_limiar"]]
            .rename(columns={"vif": "vif_depois", "r2_auxiliar": "r2_depois",
                             "acima_do_limiar": "acima_depois"}),
            on="variavel", how="inner",
        )
    )
    comparativo["delta_vif"] = comparativo["vif_depois"] - comparativo["vif_antes"]
    return comparativo.sort_values("vif_antes", ascending=False).reset_index(drop=True)


def exportar_latex(tabela: pd.DataFrame, nome_arquivo: str,
                   path_reports: Path | None = None, **kwargs) -> Path:
    """Exporta a tabela para `.tex` em `reports/tabelas/` e devolve o caminho.

    A tabela comparativa de VIF é **resultado do TCC**, não rascunho: por isso vai
    para `reports/tabelas/` e não para um diretório de trabalho.
    """
    path_reports = cfg.PATH_TABELAS if path_reports is None else Path(path_reports)
    path_reports.mkdir(parents=True, exist_ok=True)
    destino = path_reports / nome_arquivo

    opcoes = {"index": False, "float_format": "%.2f"}
    opcoes.update(kwargs)
    destino.write_text(tabela.to_latex(**opcoes), encoding="utf-8")
    return destino


def diagnostico_completo(X_train: pd.DataFrame, y_train=None,
                         feature_sets: list[str] | None = None) -> dict:
    """Roda o diagnóstico de colinearidade inteiro e salva os artefatos.

    Por conjunto: matriz de Pearson entre as contínuas e o heatmap correspondente.
    Para os três juntos: tabela de VIF empilhada, resumo e exportação LaTeX.

    Returns:
        Dicionário com `pearson` (por conjunto), `figuras`, `vif`, `resumo` e os
        caminhos dos arquivos gerados.
    """
    feature_sets = list(ft.FEATURE_SETS) if feature_sets is None else feature_sets

    pearson, figuras, pares = {}, {}, []
    for nome in feature_sets:
        pre = ft.construir_preprocessador(nome).fit(X_train, y_train)
        # Contínuas na escala clínica, depois de imputação e compostas.
        preditoras = preditoras_pos_compostas(pre, X_train)
        continuas = [c for c in ft.FEATURE_SETS[nome]["num"] if c in preditoras.columns]
        pearson[nome] = matriz_pearson(preditoras, continuas)
        figuras[nome] = heatmap_pearson(pearson[nome], nome)
        pares.append(pares_pearson(pearson[nome], nome))

    tabela_pares = pd.concat(pares, ignore_index=True)
    vif = tabela_comparativa(X_train, y_train, feature_sets)
    resumo = resumo_por_conjunto(vif)
    ablacao = ablacao_vif(X_train, y_train, "C_completo", "rcest")
    decomposicao = decomposicao_do_escore(
        ft.construir_preprocessador("C_completo").fit(X_train, y_train), X_train
    )
    tabela_escore = tabela_decomposicao(decomposicao)

    # Todo `.tex` sai com o `.csv` ao lado. O painel só lê CSV (D-055), e um número
    # que só existe em LaTeX obriga quem lê a reparsear o documento.
    return {
        "pearson": pearson,
        "pares_pearson": tabela_pares,
        "figuras": figuras,
        "vif": vif,
        "resumo": resumo,
        "ablacao_rcest": ablacao,
        "decomposicao_escore": decomposicao,
        "tex_vif": exportar_latex(vif, "vif_comparativo.tex"),
        "tex_resumo": exportar_latex(resumo, "vif_resumo.tex"),
        "tex_pearson": exportar_latex(tabela_pares, "pearson_coeficientes.tex"),
        "tex_ablacao": exportar_latex(ablacao, "vif_ablacao_rcest.tex"),
        "tex_decomposicao": exportar_latex(tabela_escore, "decomposicao_escore.tex",
                                           float_format="%.3f"),
        "csv_vif": art.salvar_csv(vif, cfg.PATH_TABELAS / "vif_comparativo.csv"),
        "csv_resumo": art.salvar_csv(resumo, cfg.PATH_TABELAS / "vif_resumo.csv"),
        "csv_pearson": art.salvar_csv(
            tabela_pares, cfg.PATH_TABELAS / "pearson_coeficientes.csv"),
        "csv_ablacao": art.salvar_csv(
            ablacao, cfg.PATH_TABELAS / "vif_ablacao_rcest.csv"),
        "csv_decomposicao": art.salvar_csv(
            tabela_escore, cfg.PATH_TABELAS / "decomposicao_escore.csv"),
    }
