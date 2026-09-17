"""Conjuntos de atributos, variáveis compostas e preprocessador.

Tudo aqui roda pós-split, e o módulo de propósito não expõe nenhuma função que
transforme o DataFrame inteiro: as compostas dependem de variáveis imputadas, e a
imputação é ajustada só no treino. Daí elas serem um transformador
(`ConstrutorCompostas`) dentro do `Pipeline`, e não uma função aplicada a `df`
antes da divisão.

Ordem do pipeline:

    ImputadorPNS -> ConstrutorCompostas -> ColumnTransformer(seleção + escala + one-hot)

`peso_lab` não é parâmetro de nenhuma função deste módulo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as cfg

N_VIZINHOS = 5
PESO_VIZINHOS = "distance"

# ---------------------------------------------------------------------------
# Conjuntos de atributos
# ---------------------------------------------------------------------------

FEATURE_SETS = {
    # A, decomposto: componentes atômicos, sem escores agregados. Ortogonalidade
    #    máxima e a melhor legibilidade do SHAP.
    "A_atomico": {
        "num": ["idade", "imc", "circ_cintura", "pas", "pad",
                "escolaridade", "saude_autoaval"],
        "bin": ["sexo_fem", "plano_sim"],
        "nom": ["raca_cor", "regiao", "has_autorrelato", "colesterol_alto"],
    },
    # B, agregado: o escore ordinal SUBSTITUI seus componentes de diagnóstico e
    #    pressão. Só assim a hipótese de sinergia fica testável.
    "B_composto": {
        "num": ["idade", "imc", "circ_cintura",
                "escolaridade", "saude_autoaval", "multimorbidade"],
        "bin": ["sexo_fem", "plano_sim"],
        "nom": ["raca_cor", "regiao"],
    },
    # C, completo: as 16 variáveis conceituais do TCC I, com RCEst. Reproduz o
    #    baseline atual e serve para quantificar o custo da redundância.
    "C_completo": {
        "num": ["idade", "imc", "circ_cintura", "rcest", "pas", "pad",
                "escolaridade", "saude_autoaval", "multimorbidade"],
        "bin": ["sexo_fem", "plano_sim", "colesterol_sim", "has_combinada"],
        "nom": ["raca_cor", "regiao", "has_autorrelato"],
    },
}

# Proveniência: cada coluna do modelo -> variáveis **cruas** da PNS que a alimentam.
# É daqui que saem os invariantes de ortogonalidade testados em
# `tests/test_features.py`.
FONTES = {
    "idade": {"idade"},
    "imc": {"peso_kg", "altura_cm"},
    "rcest": {"circ_cintura", "altura_cm"},
    "circ_cintura": {"circ_cintura"},
    "pas": {"pas"},
    "pad": {"pad"},
    "escolaridade": {"escolaridade"},
    "saude_autoaval": {"saude_autoaval"},
    "sexo_fem": {"sexo"},
    "plano_sim": {"plano_saude"},
    "colesterol_sim": {"colesterol_alto"},
    "raca_cor": {"raca_cor"},
    "regiao": {"regiao"},
    "has_autorrelato": {"has_autorrelato"},
    "colesterol_alto": {"colesterol_alto"},
    "has_combinada": {"pas", "pad", "has_autorrelato"},
    "multimorbidade": {"pas", "pad", "has_autorrelato", "colesterol_alto",
                       "peso_kg", "altura_cm", "circ_cintura", "sexo"},
}

# Colunas que embutem outras variáveis conceituais.
AGREGADAS = {"has_combinada", "multimorbidade"}

# Como cada fonte entra numa agregada. A distinção decide qual invariante de
# ortogonalidade cada conjunto cumpre.
#
# PARCELA_IDENTICA: a agregada soma exatamente o que a coluna própria já informa.
# `has_autorrelato == 1` dentro de `has_combinada` não acrescenta nada ao que o
# one-hot da nominal já representa. É o mesmo canal, duas vezes.
#
# COARSENING: a agregada usa a fonte binarizada por ponto de corte clínico. O
# gradiente contínuo carrega sinal que o limiar descarta, então coluna própria e
# agregada não são o mesmo canal. `sexo` entra aqui por escolher qual corte de
# cintura se aplica, não por ser parcela do escore.
PARCELA_IDENTICA = {
    "has_combinada": {"has_autorrelato"},
    "multimorbidade": {"has_autorrelato", "colesterol_alto"},
}
COARSENING = {
    "has_combinada": {"pas", "pad"},
    "multimorbidade": {"pas", "pad", "peso_kg", "altura_cm", "circ_cintura", "sexo"},
}

# Variáveis de diagnóstico autorrelatado: as que o baseline duplicava.
DIAGNOSTICO_AUTORRELATADO = {"has_autorrelato", "colesterol_alto"}

# Variáveis compostas produzidas por `ConstrutorCompostas`.
COMPOSTAS = ["sexo_fem", "plano_sim", "colesterol_sim", "has_combinada", "multimorbidade"]


def colunas_do_conjunto(nome: str) -> list[str]:
    """Devolve todas as colunas do conjunto de atributos, na ordem num/bin/nom."""
    cj = FEATURE_SETS[nome]
    return cj["num"] + cj["bin"] + cj["nom"]


def fontes_em_canal_duplo(nome: str) -> set[str]:
    """Variáveis cruas que alimentam o conjunto por mais de um canal.

    Canal duplo é quando a mesma variável crua chega ao modelo por duas colunas
    distintas, tipicamente como coluna própria e, ao mesmo tempo, dentro de uma
    agregada. É a quebra de ortogonalidade que o baseline tinha.

    Returns:
        Conjunto das variáveis cruas com mais de um canal. Vazio significa
        ortogonalidade plena em nível de variável fonte.
    """
    contagem: dict[str, int] = {}
    for coluna in colunas_do_conjunto(nome):
        for fonte in FONTES[coluna]:
            contagem[fonte] = contagem.get(fonte, 0) + 1
    return {fonte for fonte, n in contagem.items() if n > 1}


def _fontes_isoladas(nome: str) -> set[str]:
    """Fontes que chegam ao modelo por coluna não agregada."""
    return {
        fonte
        for coluna in colunas_do_conjunto(nome)
        if coluna not in AGREGADAS
        for fonte in FONTES[coluna]
    }


def _fontes_dentro_de_agregadas(nome: str, mapa: dict[str, set[str]]) -> set[str]:
    """Fontes que entram nas agregadas presentes no conjunto, pelo modo dado."""
    return {
        fonte
        for coluna in colunas_do_conjunto(nome)
        if coluna in AGREGADAS
        for fonte in mapa.get(coluna, set())
    }


def fontes_isoladas_e_agregadas(nome: str) -> set[str]:
    """Invariante estrito: fonte como coluna própria **e** parcela idêntica de agregada.

    É a forma exata do problema que o baseline tinha: `has_autorrelato` entrava
    como nominal isolada **e** dentro de `has_combinada` e `multimorbidade`;
    `colesterol_alto`, como binária isolada **e** dentro do escore. O SHAP não tem
    como sinalizar que a agregada é a soma de componentes já presentes no modelo, e
    o coeficiente de cada um se dilui.

    A e B cumprem (conjunto vazio). C viola de propósito: ele existe justamente
    para quantificar o custo dessa redundância.
    """
    return _fontes_isoladas(nome) & _fontes_dentro_de_agregadas(nome, PARCELA_IDENTICA)


def fontes_em_coarsening(nome: str) -> set[str]:
    """Fontes como coluna própria **e** binarizadas por corte clínico numa agregada.

    Sobreposição admitida, não violação: o limiar descarta o gradiente que a coluna
    contínua preserva, então os dois canais carregam informação distinta. Em B são
    `peso_kg`, `altura_cm`, `circ_cintura` e `sexo`, pelo componente de obesidade do
    escore. O tamanho dessa redundância se mede pelo VIF, nunca se presume.
    """
    return _fontes_isoladas(nome) & _fontes_dentro_de_agregadas(nome, COARSENING)


# ---------------------------------------------------------------------------
# Imputação: KNN como método primário, ajustado só no treino
# ---------------------------------------------------------------------------


class KNNImputerCategorico(BaseEstimator, TransformerMixin):
    """`KNNImputer` para variáveis categóricas, com arredondamento explícito.

    O `KNNImputer` é um método numérico: trata os códigos oficiais da PNS como se
    fossem quantidades contínuas e devolve médias ponderadas dos vizinhos. Para
    `raca_cor` isso daria valores como 2,7, que não são categoria nenhuma e
    quebrariam o one-hot logo adiante.

    Este transformador deixa a decisão explícita e auditável: o valor imputado é
    **arredondado para o inteiro mais próximo**, e o intervalo de códigos válidos é
    aprendido só no treino. O `clip` aos limites observados impede que uma
    extrapolação de borda invente um código que não existe na codificação oficial.

    Tratar a ausência como categoria própria é a alternativa, prevista como
    variante comparada, nunca como substituta do KNN.

    Args:
        colunas_arredondar: colunas a arredondar e conter. As demais voltam como o
            KNN as produziu. `None` arredonda todas. O parâmetro existe porque
            `ImputadorPNS` passa o espaço métrico contínuo junto da coluna
            categórica alvo: só a alvo é código, o resto é distância.
    """

    def __init__(self, n_neighbors: int = N_VIZINHOS, weights: str = PESO_VIZINHOS,
                 colunas_arredondar: list[str] | None = None):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.colunas_arredondar = colunas_arredondar

    def fit(self, X, y=None):
        X = pd.DataFrame(X).astype("float64")
        self.columns_ = list(X.columns)
        self.colunas_arredondar_ = (
            list(self.columns_) if self.colunas_arredondar is None
            else list(self.colunas_arredondar)
        )
        self.imputador_ = KNNImputer(n_neighbors=self.n_neighbors, weights=self.weights)
        self.imputador_.fit(X)
        # Códigos válidos observados no treino: são os limites do arredondamento.
        self.minimos_ = X[self.colunas_arredondar_].min()
        self.maximos_ = X[self.colunas_arredondar_].max()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).astype("float64")[self.columns_]
        imputado = pd.DataFrame(
            self.imputador_.transform(X), columns=self.columns_, index=X.index
        )
        alvo = self.colunas_arredondar_
        imputado[alvo] = imputado[alvo].round(0).clip(
            lower=self.minimos_, upper=self.maximos_, axis=1
        )
        return imputado

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_, dtype=object)


class ImputadorPNS(BaseEstimator, TransformerMixin):
    """Imputa as preditoras cruas por KNN, num espaço métrico controlado.

    Opera sobre **todas** as preditoras devolvidas por `data.separar`, não só sobre
    as do conjunto de atributos escolhido: as compostas precisam de
    `has_autorrelato` e `colesterol_alto` já imputados mesmo quando essas variáveis
    não entram no modelo como coluna própria, que é o caso do conjunto B.

    **Espaço métrico.** O `KNNImputer` escolhe vizinhos por distância euclidiana.
    Nas escalas brutas, quem tem maior amplitude numérica domina a distância: `pas`
    (desvio de uns 19 mmHg) e `circ_cintura` (uns 13 cm) decidem sozinhas quem é
    vizinho, enquanto `rcest` (uns 0,08) não muda a ordenação. A vizinhança vira
    artefato de unidade de medida, não de similaridade clínica. Por isso o espaço
    métrico usa as contínuas **padronizadas** por um `StandardScaler` interno,
    ajustado só no treino, mais `sexo_fem`.

    **Por que as nominais ficam fora da métrica.** `raca_cor`, `regiao` e
    `has_autorrelato` são códigos, não quantidades: a distância entre Norte (1) e
    Sul (4) não é três vezes a distância entre Norte e Nordeste (2), e nenhuma
    dessas diferenças quer dizer nada. Colocá-las na métrica injeta ruído com cara
    de informação. `escolaridade` e `saude_autoaval` são ordinais e até teriam
    ordem interpretável, mas o espaçamento entre níveis é arbitrário, então ficam
    de fora pelo mesmo motivo. Todas continuam sendo **alvos** de imputação; elas
    só não votam em quem é vizinho.

    `sexo_fem` é a única categórica na métrica: sendo binária, a distância entre
    seus dois valores é a única que existe e tem significado direto. Entra em 0/1,
    sem padronizar, o que lhe dá peso equivalente a cerca de meio desvio-padrão de
    uma contínua.

    **Escala de saída.** Depois de imputar, `inverse_transform` devolve as contínuas
    à escala original, porque as compostas comparam `imc` com 30 e `circ_cintura`
    com 80/94 cm, cortes clínicos em unidades reais (SBD 2019). A padronização é
    detalhe interno da imputação e não vaza para o resto do pipeline: quem padroniza
    para o modelo continua sendo o `StandardScaler` do `ColumnTransformer`.

    Ausentes reais a tratar: `colesterol_alto` (1.297), `has_autorrelato` (261),
    `regiao` (4), `raca_cor` (2).
    """

    # Contínuas disponíveis após a limpeza. Peso e altura entram só pelas
    # derivadas `imc` e `rcest`.
    CONTINUAS = ["idade", "imc", "circ_cintura", "rcest", "pas", "pad"]
    CATEGORICAS = ["sexo", "raca_cor", "regiao", "escolaridade", "saude_autoaval",
                   "plano_saude", "has_autorrelato", "colesterol_alto"]
    # Única categórica admitida na métrica, via recodificação binária.
    METRICA_BINARIA = "sexo_fem"

    def __init__(self, n_neighbors: int = N_VIZINHOS, weights: str = PESO_VIZINHOS):
        self.n_neighbors = n_neighbors
        self.weights = weights

    # -- espaço métrico ----------------------------------------------------

    def _sexo_fem(self, X: pd.DataFrame) -> pd.Series:
        """Monta `sexo_fem` preservando ausência: `NaN` em `sexo` não vira "masculino".

        A distância euclidiana com ausentes (`nan_euclidean`) descarta a dimensão
        ausente naquele par, então um `sexo` desconhecido deixa de votar em vez de
        votar errado.
        """
        sexo = pd.to_numeric(X["sexo"], errors="coerce").astype("float64")
        return sexo.where(sexo.isna(), (sexo == 2).astype("float64")).rename(
            self.METRICA_BINARIA
        )

    def espaco_metrico(self, X: pd.DataFrame) -> pd.DataFrame:
        """Monta o espaço de distância: contínuas padronizadas mais `sexo_fem`.

        Público de propósito: é o objeto que o diagnóstico de vizinhança precisa
        inspecionar para a escolha do espaço métrico ser auditável.
        """
        X = pd.DataFrame(X)
        continuas = pd.DataFrame(
            self.escalonador_.transform(X[self.continuas_].astype("float64")),
            columns=self.continuas_, index=X.index,
        )
        return pd.concat([continuas, self._sexo_fem(X)], axis=1)

    # -- sklearn API -------------------------------------------------------

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.continuas_ = [c for c in self.CONTINUAS if c in X.columns]
        self.categoricas_ = [c for c in self.CATEGORICAS if c in X.columns]

        # Escalonador interno do espaço métrico, ajustado só no treino.
        self.escalonador_ = StandardScaler().fit(
            X[self.continuas_].astype("float64")
        )

        # Contínuas: imputadas dentro do próprio espaço padronizado.
        self.imputador_continuas_ = KNNImputer(
            n_neighbors=self.n_neighbors, weights=self.weights
        ).fit(self.espaco_metrico(X))

        # Categóricas: uma imputação por coluna alvo, sempre sobre o mesmo espaço
        # métrico. A coluna alvo acompanha a matriz mas não contamina a distância:
        # onde ela está ausente, que são justamente as linhas a imputar, o
        # `nan_euclidean` descarta a dimensão. Assim dá para usar o KNN do
        # scikit-learn sem reimplementar nada, e a métrica segue sendo só a do
        # espaço declarado.
        metrico = self.espaco_metrico(self._imputar_continuas(X))
        self.imputadores_categoricos_ = {}
        for coluna in self.categoricas_:
            base = pd.concat([metrico, X[[coluna]].astype("float64")], axis=1)
            self.imputadores_categoricos_[coluna] = KNNImputerCategorico(
                n_neighbors=self.n_neighbors, weights=self.weights,
                colunas_arredondar=[coluna],
            ).fit(base)
        return self

    def _imputar_continuas(self, X: pd.DataFrame) -> pd.DataFrame:
        """Imputa as contínuas no espaço padronizado e devolve à escala original."""
        X = pd.DataFrame(X)
        imputado = pd.DataFrame(
            self.imputador_continuas_.transform(self.espaco_metrico(X)),
            columns=self.continuas_ + [self.METRICA_BINARIA], index=X.index,
        )
        # `inverse_transform`: as compostas usam cortes em unidades reais.
        continuas = pd.DataFrame(
            self.escalonador_.inverse_transform(imputado[self.continuas_]),
            columns=self.continuas_, index=X.index,
        )
        resultado = X.copy()
        resultado[self.continuas_] = continuas
        return resultado

    def transform(self, X):
        X = pd.DataFrame(X)[self.columns_]
        parcial = self._imputar_continuas(X)
        metrico = self.espaco_metrico(parcial)

        categoricas = {}
        for coluna, imputador in self.imputadores_categoricos_.items():
            base = pd.concat([metrico, X[[coluna]].astype("float64")], axis=1)
            categoricas[coluna] = imputador.transform(base)[coluna]

        resultado = parcial.copy()
        for coluna, valores in categoricas.items():
            resultado[coluna] = valores
        return resultado[self.columns_].astype("float64")

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_, dtype=object)


# ---------------------------------------------------------------------------
# Recodificações binárias e variáveis compostas
# ---------------------------------------------------------------------------


class ConstrutorCompostas(BaseEstimator, TransformerMixin):
    """Constrói as recodificações binárias e as variáveis compostas.

    Transformação row-wise e sem estado: `fit` não aprende nada. Ainda assim é um
    transformador de pipeline, e não uma função aplicada ao DataFrame completo,
    porque depende de `has_autorrelato` e `colesterol_alto` **já imputados**, e a
    imputação é pós-split. Construir as compostas antes da divisão faria as do
    teste carregarem informação estimada sobre a base inteira.
    """

    def fit(self, X, y=None):
        self.columns_in_ = list(pd.DataFrame(X).columns)
        return self

    def transform(self, X):
        f = pd.DataFrame(X).copy()

        # Recodificações binárias.
        f["sexo_fem"] = (f["sexo"] == 2).astype(int)
        f["plano_sim"] = (f["plano_saude"] == 1).astype(int)
        f["colesterol_sim"] = (f["colesterol_alto"] == 1).astype(int)

        # HAS Combinada: pressão aferida >= 140/90 (SBD 2019) OU autorrelato
        # afirmativo. Pega hipertensos diagnosticados e não diagnosticados. O
        # código 2 de Q002 ("apenas na gravidez") NÃO é HAS crônica.
        has_medida = (f["pas"] >= cfg.CORTE_PAS) | (f["pad"] >= cfg.CORTE_PAD)
        f["has_combinada"] = (has_medida | (f["has_autorrelato"] == 1)).astype(int)

        # Componente de obesidade: disjuntivo e sensível ao sexo, para contornar a
        # limitação conhecida do IMC isolado quanto a adiposidade central.
        fem = f["sexo_fem"] == 1
        obesidade = (
            (f["imc"] >= cfg.CORTE_IMC_OBESIDADE)
            | (fem & (f["circ_cintura"] >= cfg.CORTE_CINTURA_F))
            | (~fem & (f["circ_cintura"] >= cfg.CORTE_CINTURA_M))
        )

        # Multimorbidade Metabólica: escore ordinal de 0 a 3.
        f["multimorbidade"] = (
            f["has_combinada"]
            + (f["colesterol_alto"] == 1).astype(int)
            + obesidade.astype(int)
        )
        return f

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(self.columns_in_) + COMPOSTAS, dtype=object)


# ---------------------------------------------------------------------------
# Preprocessador
# ---------------------------------------------------------------------------


def conjunto_sem(feature_set: str, *variaveis: str) -> dict[str, list[str]]:
    """Devolve o conjunto de atributos sem as variáveis indicadas.

    Serve ao protocolo de ablação: remover experimentalmente a variável de maior
    VIF dentro de um mesmo bloco temático e recalcular o diagnóstico. O conjunto
    derivado não entra em `FEATURE_SETS`, porque é objeto de análise e não um
    quarto conjunto de modelagem.
    """
    base = FEATURE_SETS[feature_set]
    remover = set(variaveis)
    desconhecidas = remover - set(colunas_do_conjunto(feature_set))
    if desconhecidas:
        raise ValueError(
            f"{feature_set} não contém {sorted(desconhecidas)}"
        )
    return {bloco: [c for c in colunas if c not in remover]
            for bloco, colunas in base.items()}


def construir_preprocessador(feature_set: str | dict) -> Pipeline:
    """Monta o preprocessador completo de um conjunto de atributos.

    Args:
        feature_set: chave de `FEATURE_SETS` (`"A_atomico"`, `"B_composto"`,
            `"C_completo"`) ou um conjunto literal `{"num": [...], "bin": [...],
            "nom": [...]}`, como o que `conjunto_sem` devolve para as ablações.

    Returns:
        `Pipeline` de três estágios: imputação por KNN (ajustada só no treino),
        construção das compostas e o `ColumnTransformer` que seleciona o conjunto e
        aplica escala, passthrough e one-hot.

    A seleção do conjunto fica no último estágio (`remainder="drop"`) porque os
    estágios anteriores precisam de todas as preditoras cruas para imputar e para
    compor, inclusive as que não entram no modelo.
    """
    if isinstance(feature_set, dict):
        cj = feature_set
        faltando = {"num", "bin", "nom"} - set(cj)
        if faltando:
            raise ValueError(f"Conjunto literal sem os blocos {sorted(faltando)}")
    elif feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Conjunto desconhecido: {feature_set!r}. "
            f"Disponíveis: {sorted(FEATURE_SETS)}"
        )
    else:
        cj = FEATURE_SETS[feature_set]

    preparo = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), cj["num"]),
            ("bin", "passthrough", cj["bin"]),
            ("nom", OneHotEncoder(drop="first", handle_unknown="ignore",
                                  sparse_output=False), cj["nom"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline([
        ("imputacao", ImputadorPNS()),
        ("compostas", ConstrutorCompostas()),
        ("preparo", preparo),
    ])
