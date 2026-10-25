"""Atribuições SHAP sobre a Random Forest publicada.

Uma responsabilidade só: calcular atribuição e emitir tabela. O módulo não treina
modelo novo, não calibra limiar próprio, não mede desempenho e não toca no grid
publicado. O pipeline vem de `features.construir_preprocessador`, a floresta de
`models.random_forest` e o limiar do mesmo `evaluate.avaliar` que produz a tabela
consolidada, conferido contra ela antes de qualquer atribuição sair.

Fica fora de `experimento` pelo mesmo motivo que manteve `tuning` fora (D-033): o
regime de contraste é caro e nenhum número publicado depende dele. `make resultados`
precisa continuar barato, porque é ele que confere as âncoras.

**Dois regimes de perturbação, sobre dois objetos distintos** (D-053). O principal é
`tree_path_dependent` sobre as 500 árvores, e é ele que sustenta o SHAP global
publicado. O de contraste é `interventional`, que assume independência entre colunas e
é a formulação que a referência aponta como frágil. A divergência entre os dois mede a
exposição deste trabalho ao problema.

**A floresta reduzida é objeto de diagnóstico e nunca modelo publicado.** O `shap`
desta versão explica algumas árvores com erro sob `interventional`, e o erro é do
explicador, não do modelo: as mesmas árvores fecham exato sob `tree_path_dependent`. O
módulo identifica essas árvores dentro da execução, explicando cada uma isoladamente
sobre o teste completo, e roda o contraste sobre a floresta sem elas. A floresta
reduzida sai da mesma floresta ajustada, com a lista de árvores filtrada; reajustar
produziria outra floresta e invalidaria a comparação. O ramo que decidiu emitir o
contraste assim está registrado em D-053 e foi selecionado antes destes números.

**Portão de aditividade por regime.** O principal fecha em 1e-9. O de contraste fecha
quando o erro da floresta fica na ordem do erro de uma árvore isolada sem defeito,
medido na mesma execução: nenhuma tolerância se fixa abaixo do ruído do método que ela
mede.

**Restrição de leitura** (D-022, D-040, D-053). As atribuições ficam na escala de
fração de votos, que é a escala em que o limiar calibrado opera. A soma das
contribuições reconstrói o escore do indivíduo, nunca o risco dele, e o valor de base
fica perto de 0,5 por efeito do `class_weight`, não perto da prevalência de 7%.

Uso:

    python -m src.explicabilidade
    python -m src.explicabilidade --celulas A_atomico
    python -m src.explicabilidade --sem-artefato
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from matplotlib.colors import LinearSegmentedColormap

from . import artefatos as art
from . import colinearidade as col
from . import config as cfg
from . import evaluate as ev
from . import experimento as xp
from . import features as ft
from . import figuras as figu
from . import models as md

# `C_completo` ficou fora do grid publicado por D-016, então não há o que explicar nele.
CELULAS = ["A_atomico", "B_composto"]
MODELO = xp.NOME_FLORESTA

REGIME_PRINCIPAL = "tree_path_dependent"
REGIME_CONTRASTE = "interventional"
OBJETO_COMPLETA = "floresta completa"
OBJETO_REDUZIDA = "floresta reduzida (diagnostico)"

# Cada regime roda sobre o seu objeto, e o par viaja junto para nenhuma tabela deixar
# implícito o que foi explicado.
REGIMES = [(REGIME_PRINCIPAL, OBJETO_COMPLETA), (REGIME_CONTRASTE, OBJETO_REDUZIDA)]

# O regime principal sobre a floresta reduzida. Não entra nas tabelas globais, porque
# não é nem o que se publica nem o contraste, mas é o degrau do meio da regra de D-053:
# separa o que a remoção das árvores move do que a troca de regime move.
PRINCIPAL_REDUZIDA = (REGIME_PRINCIPAL, OBJETO_REDUZIDA)

# As duas comparações do topo coincidente. A primeira é a que a figura de contraste e o
# texto publicado descrevem. A segunda fixa o objeto e isola o efeito do regime.
COMPARACOES_TOPO = [
    ("principal completa x contraste reduzida",
     (REGIME_PRINCIPAL, OBJETO_COMPLETA), (REGIME_CONTRASTE, OBJETO_REDUZIDA)),
    ("principal reduzida x contraste reduzida",
     PRINCIPAL_REDUZIDA, (REGIME_CONTRASTE, OBJETO_REDUZIDA)),
]

# Linhas do treino que estimam a expectativa marginal do regime de contraste. Cem é a
# ordem de grandeza usual, e aumentar reduz variância de estimação sem mudar o que o
# regime assume. Saem do treino, nunca do teste, por R1 e R2.
N_BACKGROUND = 100

# Separa as árvores que o explicador erra das que ele acerta. Os dois grupos distam
# milhares de vezes, então qualquer valor entre eles serve; a execução reporta os
# extremos dos dois lados para a separação ficar demonstrada em vez de afirmada.
CORTE_ARVORE = 1e-4

TOL_PRINCIPAL = 1e-9
CASAS_CONFERENCIA = 3
TOPO_TEX = 10

# Algarismos significativos das colunas de erro que vêm de agregação sobre as árvores.
# Três basta para elas: são mínimos, máximos e média sobre centenas de explicações de
# árvore isolada, que rodam em um núcleo só e saem iguais entre execuções. Três é também
# a precisão em que o projeto confere tudo, e 5,90e-08 continua legível.
SIGNIFICATIVOS_ERRO = 3

# Colunas que passam pelo arredondamento acima.
COLUNAS_DE_ERRO = [
    "erro_min_mantidas", "erro_max_mantidas",
    "erro_min_excluidas", "erro_max_excluidas",
    "piso_por_arvore",
]

# Colunas gravadas como ordem de grandeza, a potência de dez logo abaixo do valor. As
# duas medem o erro de arredondamento da floresta, que roda com `n_jobs=-1`, e são
# máximos sobre o teste inteiro: a ordem em que os processos reduzem move o primeiro
# algarismo significativo delas, não os últimos. Estabilizar a mantissa não estava ao
# alcance, e arredondá-la a três algarismos deixava a guarda de byte falhando de vez em
# quando sem que nada tivesse mudado, que é o jeito de ensinar a ignorar vermelho. A
# fronteira passa a ser a virada de década: 5,52e-15 e 5,551e-15 saem as duas como
# 1e-15, e é isso que os documentos afirmam sobre elas. A mantissa nunca carregou
# informação aqui.
COLUNAS_DE_ORDEM = ["erro_aditividade_principal", "erro_aditividade_contraste"]

# Casas na gravação da mesma tabela. As nove de `artefatos.CASAS_CSV` zeram a coluna
# inteira, porque os erros de aditividade vivem entre 1e-15 e 1e-8 e um artefato de
# diagnóstico cheio de zeros não diagnostica nada. Aqui o número serve só para o
# arredondamento decimal não truncar o que os dois tratamentos acima já fixaram.
CASAS_ARVORES = 18

NOME_GLOBAL_ATRIBUTO = "shap_global_atributo"
NOME_GLOBAL_COLUNA = "shap_global_coluna"
NOME_PARES = "shap_pares_colineares"
NOME_CASOS = "shap_casos_locais"
NOME_ARVORES = "shap_arvores_excluidas"
NOME_CONCORDANCIA = "shap_concordancia_topo"

# As três estruturas de dependência que sobrevivem nas duas células publicadas, nomeadas
# em D-053 antes de qualquer atribuição existir. Nenhum limiar decide se agrupa: cada
# bloco sai com o membro e com a soma, lado a lado, nos dois regimes.
PARES_COLINEARES = [
    {
        "bloco": "imc x circ_cintura",
        "celulas": ["A_atomico", "B_composto"],
        "membros": ["imc", "circ_cintura"],
        "evidencia": "Pearson 0,793",
    },
    {
        "bloco": "pas x pad",
        "celulas": ["A_atomico"],
        "membros": ["pas", "pad"],
        "evidencia": "Pearson 0,743",
    },
    {
        "bloco": "imc, circ_cintura, sexo_fem x multimorbidade",
        "celulas": ["B_composto"],
        "membros": ["imc", "circ_cintura", "sexo_fem", "multimorbidade"],
        "evidencia": "canal de coarsening por ponto de corte clinico, D-012",
    },
]

CELULAS_CONFUSAO = ["VP", "FP", "FN", "VN"]

NOTA_ESCALA = (
    "Atribuições na escala de fração de votos da classe positiva, que é a escala em que "
    "o limiar calibrado opera. A soma das contribuições reconstrói o escore do "
    "indivíduo, nunca o risco dele, e o valor de base fica perto de 0,5 por efeito do "
    "class\\_weight, não perto da prevalência (D-022, D-040, D-053)."
)
NOTA_REGIMES = (
    "O regime principal é tree\\_path\\_dependent sobre as 500 árvores ajustadas, e é "
    "ele que sustenta o SHAP global publicado. O de contraste é interventional sobre a "
    "floresta reduzida, que exclui as árvores atribuídas com erro pelo explicador desta "
    "versão. A floresta reduzida é objeto de diagnóstico e nunca modelo publicado "
    "(D-053)."
)
NOTA_REFERENCIA = (
    "A codificação usa drop=\"first\", então a categoria de referência de cada nominal "
    "tem atribuição zero por construção, não por irrelevância. A coluna "
    "categoria\\_referencia\\_omitida registra qual ficou de fora."
)


# ---------------------------------------------------------------------------
# Proveniência das colunas
# ---------------------------------------------------------------------------

def proveniencia(preprocessador) -> pd.DataFrame:
    """Liga cada coluna da matriz de design à variável conceitual que a gerou.

    O mapa sai do estado do `ColumnTransformer` ajustado, nunca de parsing de prefixo
    de string: prefixo quebra em silêncio quando uma categoria tem underscore no nome.
    As dummies de uma nominal são particionadas pelas `categories_` do encoder, na
    mesma ordem em que ele as gera.

    Returns:
        DataFrame com `coluna`, `atributo` e `categoria_referencia_omitida`, uma linha
        por coluna da matriz de design, na ordem da matriz.

    Raises:
        KeyError: se alguma variável conceitual não estiver no mapa de proveniência de
            `features.FONTES`. Coluna sem proveniência declarada não entra em tabela.
    """
    transformador = preprocessador.named_steps["preparo"]
    linhas = []
    for nome, estagio, colunas in transformador.transformers_:
        if nome in ("num", "bin"):
            linhas += [{"coluna": c, "atributo": c,
                        "categoria_referencia_omitida": ""} for c in colunas]
        elif nome == "nom":
            gerados = list(estagio.get_feature_names_out(colunas))
            inicio = 0
            for atributo, categorias in zip(colunas, estagio.categories_):
                # drop="first": uma dummy a menos que categorias observadas, e a
                # primeira delas é a que ficou de fora.
                quantas = len(categorias) - 1
                omitida = _formatar_categoria(categorias[0])
                for gerado in gerados[inicio:inicio + quantas]:
                    linhas.append({"coluna": str(gerado), "atributo": atributo,
                                   "categoria_referencia_omitida": omitida})
                inicio += quantas
            if inicio != len(gerados):
                raise ValueError(
                    "A partição das dummies não fechou: o encoder gerou "
                    f"{len(gerados)} colunas e as categorias somam {inicio}."
                )

    mapa = pd.DataFrame(linhas)
    sem_fonte = sorted(set(mapa["atributo"]) - set(ft.FONTES))
    if sem_fonte:
        raise KeyError(
            f"Variáveis sem proveniência declarada em features.FONTES: {sem_fonte}"
        )
    return mapa


def _formatar_categoria(valor) -> str:
    """Escreve o código da categoria omitida sem a cauda decimal do float."""
    if isinstance(valor, (float, np.floating)) and float(valor).is_integer():
        return str(int(valor))
    return str(valor)


# ---------------------------------------------------------------------------
# Atribuições
# ---------------------------------------------------------------------------

def explicar(modelo, design, regime: str,
             background=None) -> tuple[np.ndarray, np.ndarray]:
    """Atribuições da classe positiva e valor de base, para um regime.

    Esta versão do `shap` devolve tensor `(n, colunas, 2)` para classificador binário,
    e a classe positiva é o índice 1 do último eixo. O `expected_value` também vem com
    dois elementos, e é o índice 1 que importa.

    `check_additivity` fica desligado: com background pequeno ele estoura por
    tolerância em vez de por erro real. A conferência é feita à mão, pelos portões que
    `erro_de_aditividade` mede.

    Args:
        regime: `REGIME_PRINCIPAL` dispensa background; `REGIME_CONTRASTE` exige um.

    Returns:
        Tupla `(valores, base)`, com `valores` de forma `(n, colunas)` e `base` de
        comprimento `n`.
    """
    if regime == REGIME_CONTRASTE:
        if background is None:
            raise ValueError(f"O regime {regime!r} precisa de background.")
        explicador = shap.TreeExplainer(modelo, data=np.asarray(background),
                                        feature_perturbation=regime)
    else:
        explicador = shap.TreeExplainer(modelo, feature_perturbation=regime)

    saida = explicador(np.asarray(design), check_additivity=False)
    valores = saida.values[:, :, 1]
    base = np.asarray(saida.base_values)
    base = base[:, 1] if base.ndim == 2 else np.full(len(valores), float(base[1]))
    return valores, base


def erro_de_aditividade(valores: np.ndarray, base: np.ndarray,
                        proba: np.ndarray) -> float:
    """Maior distância entre a soma das atribuições mais a base e o escore do modelo.

    É a conferência que diz se a atribuição descreve o modelo ou outra coisa.
    """
    return float(np.abs(valores.sum(axis=1) + base - proba).max())


def arvores_com_defeito(floresta, design, background,
                        corte: float = CORTE_ARVORE) -> dict:
    """Aponta as árvores que o explicador atribui com erro sob o regime de contraste.

    Explica cada árvore isoladamente sobre as mesmas linhas que depois são explicadas
    em conjunto, então a cobertura fica completa por construção. O defeito é esparso em
    pares de árvore e linha, e a contagem cresce com o número de linhas conferidas: ela
    vale para o conjunto passado aqui e só para ele (D-053).

    Returns:
        Dicionário com `mantidas` e `excluidas`, que são posições na lista de árvores,
        os extremos de erro dos dois grupos, o `piso`, que é o erro médio das mantidas,
        e o `corte` usado.

    Raises:
        ValueError: se o corte não separar dois grupos não vazios. A regra de D-053
            pressupõe a separação, e sem ela o diagnóstico não se sustenta.
    """
    matriz = np.asarray(design)
    erros = []
    for arvore in floresta.estimators_:
        valores, base = explicar(arvore, matriz, REGIME_CONTRASTE, background)
        erros.append(erro_de_aditividade(valores, base,
                                         arvore.predict_proba(matriz)[:, 1]))
    erros = np.asarray(erros)

    excluidas = np.flatnonzero(erros > corte)
    mantidas = np.flatnonzero(erros <= corte)
    if excluidas.size == 0 or mantidas.size == 0:
        raise ValueError(
            f"O corte {corte:g} não separou dois grupos: {mantidas.size} mantidas e "
            f"{excluidas.size} excluídas, com erro entre {erros.min():.3e} e "
            f"{erros.max():.3e}."
        )

    return {
        "erros": erros,
        "mantidas": mantidas,
        "excluidas": excluidas,
        "corte": float(corte),
        "erro_min_mantidas": float(erros[mantidas].min()),
        "erro_max_mantidas": float(erros[mantidas].max()),
        "erro_min_excluidas": float(erros[excluidas].min()),
        "erro_max_excluidas": float(erros[excluidas].max()),
        # Piso numérico do algoritmo sobre este modelo. É contra ele que o portão do
        # regime de contraste se mede, e não contra um valor escolhido por analogia com
        # o regime principal (D-053).
        "piso": float(erros[mantidas].mean()),
    }


def reduzir_floresta(floresta, mantidas: np.ndarray):
    """Devolve a mesma floresta ajustada, com a lista de árvores filtrada.

    Nada é reajustado. Reajustar sortearia outras amostras bootstrap e produziria uma
    floresta diferente, o que invalidaria a comparação com a publicada.
    """
    reduzida = copy.deepcopy(floresta)
    reduzida.estimators_ = [floresta.estimators_[i] for i in mantidas]
    reduzida.n_estimators = len(reduzida.estimators_)
    if len(reduzida.estimators_) != len(mantidas):
        raise ValueError("A floresta reduzida não ficou com o número de árvores pedido.")
    return reduzida


# ---------------------------------------------------------------------------
# Rankings
# ---------------------------------------------------------------------------

def ranking_por_coluna(valores: np.ndarray, mapa: pd.DataFrame) -> pd.DataFrame:
    """|SHAP| médio de cada coluna da matriz de design, com a posição no ranking."""
    tabela = mapa.copy()
    tabela["shap_medio_abs"] = np.abs(valores).mean(axis=0)
    return _posicionar(tabela)


def ranking_por_atributo(valores: np.ndarray, mapa: pd.DataFrame) -> pd.DataFrame:
    """|SHAP| médio por variável conceitual, somando as colunas de uma mesma nominal.

    A categoria de referência entra na soma com atribuição zero, porque o `drop="first"`
    a tirou da matriz. A coluna que declara qual ficou de fora acompanha a linha.
    """
    por_coluna = mapa.copy()
    por_coluna["shap_medio_abs"] = np.abs(valores).mean(axis=0)

    tabela = (por_coluna.groupby("atributo", sort=False)
              .agg(shap_medio_abs=("shap_medio_abs", "sum"),
                   n_colunas=("coluna", "count"),
                   categoria_referencia_omitida=("categoria_referencia_omitida",
                                                 "first"))
              .reset_index())
    return _posicionar(tabela)


def _posicionar(tabela: pd.DataFrame) -> pd.DataFrame:
    """Ordena por |SHAP| médio decrescente e numera as posições, de forma estável.

    Ordenação estável: empate exato de atribuição preserva a ordem da matriz de design,
    em vez de depender de como o algoritmo de ordenação desempata.
    """
    ordenada = tabela.sort_values("shap_medio_abs", ascending=False,
                                  kind="stable").reset_index(drop=True)
    ordenada["posicao"] = range(1, len(ordenada) + 1)
    return ordenada


def deslocamento(ranking_a: pd.DataFrame, ranking_b: pd.DataFrame) -> int:
    """Soma das mudanças absolutas de posição entre dois rankings de variável.

    Métrica única dos dois deslocamentos que a regra de D-053 compara, para que a
    comparação seja entre grandezas da mesma natureza.
    """
    a = ranking_a.set_index("atributo")["posicao"]
    b = ranking_b.set_index("atributo")["posicao"].reindex(a.index)
    if b.isna().any():
        faltando = sorted(a.index[b.isna()])
        raise ValueError(f"Rankings com variáveis distintas: {faltando}")
    return int((a - b).abs().sum())


def empates(ranking: pd.DataFrame) -> list[str]:
    """Variáveis com |SHAP| médio exatamente igual ao de outra, se houver alguma.

    Empate exato faz a posição depender da ordem de entrada, e quem lê a tabela precisa
    saber disso em vez de descobrir numa reexecução.
    """
    contagem = ranking["shap_medio_abs"].value_counts()
    repetidos = contagem[contagem > 1].index
    return sorted(ranking.loc[ranking["shap_medio_abs"].isin(repetidos), "atributo"])


# ---------------------------------------------------------------------------
# Casos locais
# ---------------------------------------------------------------------------

def selecionar_casos(y_te: pd.Series, proba: np.ndarray,
                     limiar: float) -> pd.DataFrame:
    """Um caso por célula da matriz de confusão, na fronteira do limiar.

    Cada caso é o de menor `|escore - limiar|` dentro da sua célula, com desempate pelo
    menor índice. A regra é um `argmin` de uma coluna: não admite julgamento e reproduz
    sozinha, que é o que separa escolha de caso de escolha por conveniência.

    A fronteira é onde a explicação tem valor operacional. Um encaminhado e um não
    encaminhado com escores quase iguais, e explicações diferentes, mostram que o modelo
    decide por variável.

    Returns:
        DataFrame com `caso`, `indice`, `escore` e `distancia_ao_limiar`, uma linha por
        célula da confusão.

    Raises:
        ValueError: se alguma célula da confusão ficar vazia no teste.
    """
    predito = (proba >= limiar).astype(int)
    base = pd.DataFrame({
        "indice": list(y_te.index),
        "observado": y_te.to_numpy(),
        "predito": predito,
        "escore": proba,
        "distancia_ao_limiar": np.abs(proba - limiar),
    })
    pertence = {
        "VP": (base["observado"] == 1) & (base["predito"] == 1),
        "FP": (base["observado"] == 0) & (base["predito"] == 1),
        "FN": (base["observado"] == 1) & (base["predito"] == 0),
        "VN": (base["observado"] == 0) & (base["predito"] == 0),
    }

    escolhidos = []
    for caso in CELULAS_CONFUSAO:
        candidatos = base[pertence[caso]]
        if candidatos.empty:
            raise ValueError(f"Nenhum caso na célula {caso!r} da matriz de confusão.")
        escolhido = candidatos.sort_values(["distancia_ao_limiar", "indice"],
                                           kind="stable").iloc[0]
        escolhidos.append({
            "caso": caso,
            "indice": int(escolhido["indice"]),
            "escore": float(escolhido["escore"]),
            "distancia_ao_limiar": float(escolhido["distancia_ao_limiar"]),
        })
    return pd.DataFrame(escolhidos)


def contribuicoes_locais(valores: np.ndarray, mapa: pd.DataFrame,
                         posicao: int) -> pd.DataFrame:
    """Contribuição de um indivíduo, com sinal, somada por variável conceitual.

    Aqui a soma é com sinal, e não em módulo: é ela que reconstrói o escore junto com o
    valor de base, e trocar por módulo quebraria a identidade que a explicação local
    afirma.
    """
    linha = mapa.copy()
    linha["contribuicao"] = valores[posicao, :]
    return linha.groupby("atributo", sort=False)["contribuicao"].sum().reset_index()


# ---------------------------------------------------------------------------
# Limiar publicado
# ---------------------------------------------------------------------------

def limiar_publicado(celula: str, path: Path | None = None) -> float:
    """Lê o limiar da célula na tabela consolidada versionada.

    Raises:
        FileNotFoundError: se a tabela consolidada não estiver emitida.
        KeyError: se a célula não estiver nela.
    """
    if path is None:
        path = cfg.PATH_TABELAS / f"{xp.NOME_CONSOLIDADO}.csv"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Tabela consolidada ausente: {path}. Rode python -m src.experimento antes."
        )
    tabela = pd.read_csv(path, encoding="utf-8")
    recorte = tabela[(tabela["modelo"] == MODELO) & (tabela["conjunto"] == celula)]
    if recorte.empty:
        raise KeyError(f"Célula ({MODELO!r}, {celula!r}) ausente de {path.name}.")
    return float(recorte.iloc[0]["limiar"])


def conferir_limiar(obtido: float, celula: str,
                    casas: int = CASAS_CONFERENCIA) -> dict:
    """Compara o limiar desta execução com o publicado, na casa que o projeto confere.

    Raises:
        ValueError: se divergirem. Parar aqui é o ponto: divergência significa que este
            módulo ajustou o modelo diferente do experimento, e a atribuição
            descreveria outro modelo.
    """
    esperado = limiar_publicado(celula)
    if round(obtido, casas) != round(esperado, casas):
        raise ValueError(
            f"Limiar de {celula} diverge do publicado: {obtido:.9f} contra "
            f"{esperado:.9f} em {xp.NOME_CONSOLIDADO}.csv. O modelo explicado não é o "
            f"da tabela consolidada."
        )
    return {"celula": celula, "limiar_obtido": obtido, "limiar_publicado": esperado}


# ---------------------------------------------------------------------------
# Medição de uma célula
# ---------------------------------------------------------------------------

def preparar(celula: str, particao: xp.Particao) -> dict:
    """Ajusta a célula pelo mesmo caminho de chamada que produz a tabela consolidada.

    `models.montar` embrulha a floresta no preprocessador do conjunto e
    `evaluate.avaliar` calibra o limiar por validação cruzada no treino. Montar
    pipeline próprio aqui daria outro modelo com a mesma cara.
    """
    resultado = ev.avaliar(
        md.montar(md.estimadores()[MODELO], celula),
        particao.X_tr, particao.y_tr, particao.X_te, particao.y_te,
        nome=MODELO, conjunto=celula, recall_alvo=cfg.RECALL_ALVO,
        salvar_figuras=False,
    )
    pipeline = resultado["estimador"]
    preprocessador = pipeline.named_steps["pre"]

    design_tr = col.matriz_de_design(pipeline, particao.X_tr)
    design_te = col.matriz_de_design(pipeline, particao.X_te)

    # Sorteio por índice, não por posição: sobrevive a qualquer reordenação da matriz.
    indices_background = design_tr.sample(n=N_BACKGROUND,
                                          random_state=cfg.RANDOM_STATE).index
    background = design_tr.loc[indices_background]

    # Valores em escala clínica, depois da imputação e antes da padronização. É o que a
    # tabela de casos locais mostra ao lado da contribuição, porque 94 cm de cintura se
    # lê e 0,83 desvio padrão não.
    clinicos = preprocessador[:2].transform(particao.X_te)
    if not isinstance(clinicos, pd.DataFrame):
        clinicos = pd.DataFrame(clinicos, index=particao.X_te.index)

    return {
        "celula": celula,
        "resultado": resultado,
        "pipeline": pipeline,
        "floresta": pipeline.named_steps["clf"],
        "design_tr": design_tr,
        "design_te": design_te,
        "background": background,
        "indices_background": list(indices_background),
        "mapa": proveniencia(preprocessador),
        "clinicos": clinicos,
        "limiar": float(resultado["limiar"]),
        "proba": resultado["proba_teste"],
        "y_te": particao.y_te,
    }


def medir_celula(celula: str, particao: xp.Particao) -> dict:
    """Roda os dois regimes na célula, confere os portões e devolve o que as tabelas usam.

    Raises:
        ValueError: se o limiar divergir do publicado ou se um portão de aditividade
            falhar. Os dois casos param a execução em vez de emitir tabela.
    """
    inicio = time.perf_counter()
    base = preparar(celula, particao)
    conferencia = conferir_limiar(base["limiar"], celula)

    floresta = base["floresta"]
    design = np.asarray(base["design_te"])
    background = base["background"]
    mapa = base["mapa"]

    defeito = arvores_com_defeito(floresta, design, background)
    reduzida = reduzir_floresta(floresta, defeito["mantidas"])

    proba_completa = floresta.predict_proba(design)[:, 1]
    proba_reduzida = reduzida.predict_proba(design)[:, 1]

    principal, base_principal = explicar(floresta, design, REGIME_PRINCIPAL)
    principal_reduzida, _ = explicar(reduzida, design, REGIME_PRINCIPAL)
    contraste, base_contraste = explicar(reduzida, design, REGIME_CONTRASTE,
                                         background)

    erro_principal = erro_de_aditividade(principal, base_principal, proba_completa)
    erro_contraste = erro_de_aditividade(contraste, base_contraste, proba_reduzida)
    if erro_principal > TOL_PRINCIPAL:
        raise ValueError(
            f"{celula}: aditividade do regime {REGIME_PRINCIPAL} não fecha em "
            f"{TOL_PRINCIPAL:g}. Erro máximo {erro_principal:.3e}."
        )
    if erro_contraste > defeito["piso"]:
        raise ValueError(
            f"{celula}: aditividade do regime {REGIME_CONTRASTE} acima do piso por "
            f"árvore. Erro máximo {erro_contraste:.3e} contra piso "
            f"{defeito['piso']:.3e}. A identificação das árvores está incompleta."
        )

    # Dois caminhos até o mesmo escore: o pipeline inteiro e a matriz de design com a
    # floresta. Conferir aqui fecha a ponte que a explicação local usa.
    if not np.allclose(proba_completa, base["proba"], atol=1e-12):
        raise ValueError(
            f"{celula}: o escore pela matriz de design não bate com o do pipeline."
        )

    ranking_principal = ranking_por_atributo(principal, mapa)
    ranking_reduzida = ranking_por_atributo(principal_reduzida, mapa)
    ranking_contraste = ranking_por_atributo(contraste, mapa)

    return {
        **base,
        "conferencia_limiar": conferencia,
        "defeito": defeito,
        "reduzida": reduzida,
        "proba_reduzida": proba_reduzida,
        "atribuicoes": {
            (REGIME_PRINCIPAL, OBJETO_COMPLETA): principal,
            (REGIME_CONTRASTE, OBJETO_REDUZIDA): contraste,
        },
        "bases": {
            (REGIME_PRINCIPAL, OBJETO_COMPLETA): base_principal,
            (REGIME_CONTRASTE, OBJETO_REDUZIDA): base_contraste,
        },
        "rankings": {
            (REGIME_PRINCIPAL, OBJETO_COMPLETA): ranking_principal,
            (REGIME_CONTRASTE, OBJETO_REDUZIDA): ranking_contraste,
            PRINCIPAL_REDUZIDA: ranking_reduzida,
        },
        "erro_principal": erro_principal,
        "erro_contraste": erro_contraste,
        # Os dois deslocamentos que a regra de D-053 comparou, recalculados aqui para a
        # tabela registrar com que margem o ramo selecionado se sustenta.
        "delta_arvores": deslocamento(ranking_principal, ranking_reduzida),
        "delta_regime": deslocamento(ranking_reduzida, ranking_contraste),
        "empates": {"principal": empates(ranking_principal),
                    "principal_reduzida": empates(ranking_reduzida),
                    "contraste": empates(ranking_contraste)},
        "segundos": time.perf_counter() - inicio,
    }


# ---------------------------------------------------------------------------
# Tabelas
# ---------------------------------------------------------------------------

def _com_modelo(tabela: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta ao fim a coluna `modelo`, com o estimador que as atribuições explicam.

    O módulo explica só a floresta publicada, e a coluna diz isso no próprio artefato:
    quem lê o CSV fora do contexto não precisa adivinhar de qual modelo é o ranking.
    """
    return tabela.assign(modelo=MODELO)


def tabela_global_atributo(medidas: list[dict],
                           chaves: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    """Uma linha por célula, regime e variável conceitual.

    Args:
        chaves: pares (regime, objeto) a incluir. Omitido, usa `REGIMES`, que é o que
            o artefato publica. `concordancia_topo` pede também `PRINCIPAL_REDUZIDA`.
    """
    chaves = list(REGIMES) if chaves is None else list(chaves)
    partes = []
    for medida in medidas:
        for chave in chaves:
            regime, objeto = chave
            ranking = medida["rankings"][chave].copy()
            ranking.insert(0, "objeto_explicado", objeto)
            ranking.insert(0, "regime", regime)
            ranking.insert(0, "celula", medida["celula"])
            partes.append(ranking)
    colunas = ["celula", "regime", "objeto_explicado", "atributo", "shap_medio_abs",
               "posicao", "n_colunas", "categoria_referencia_omitida"]
    return _com_modelo(pd.concat(partes, ignore_index=True)[colunas])


def concordancia_topo(tabela: pd.DataFrame,
                      comparacoes: list | None = None) -> pd.DataFrame:
    """Quantas posições do topo coincidem entre dois rankings, por célula.

    Para cada célula e comparação, devolve o maior `k` tal que as posições 1 a `k` têm
    a mesma variável nos dois rankings. É o número que o texto usa como resposta à
    multicolinearidade: sete em `A_atomico` e cinco em `B_composto` (D-053, D-055).

    Args:
        tabela: formato de `tabela_global_atributo`, com `celula`, `regime`,
            `objeto_explicado`, `atributo` e `posicao`. Pode ser o CSV versionado.
        comparacoes: triplas (nome, lado_a, lado_b), cada lado um par (regime,
            objeto). Omitido, usa `COMPARACOES_TOPO`.

    Returns:
        DataFrame com uma linha por célula e comparação: `celula`, `comparacao`, os
        dois lados, `topo_coincidente`, `n_atributos` e `atributos_no_topo`.

    Raises:
        KeyError: se um lado de alguma comparação não estiver na tabela. O CSV
            publicado não traz o regime principal sobre a floresta reduzida, e uma
            comparação sem um dos lados daria `k = 0` calado.
    """
    comparacoes = list(COMPARACOES_TOPO) if comparacoes is None else list(comparacoes)

    def ordem(recorte: pd.DataFrame, lado: tuple[str, str]) -> list[str]:
        regime, objeto = lado
        linhas = recorte[(recorte["regime"] == regime)
                         & (recorte["objeto_explicado"] == objeto)]
        if linhas.empty:
            raise KeyError(f"{recorte['celula'].iloc[0]}: ranking {regime} sobre "
                           f"{objeto} ausente da tabela.")
        return list(linhas.sort_values("posicao", kind="stable")["atributo"])

    linhas = []
    for celula in pd.unique(tabela["celula"]):
        recorte = tabela[tabela["celula"] == celula]
        for nome, lado_a, lado_b in comparacoes:
            esquerda, direita = ordem(recorte, lado_a), ordem(recorte, lado_b)
            k = _prefixo_comum(esquerda, direita)
            linhas.append({
                "celula": celula,
                "comparacao": nome,
                "regime_a": lado_a[0],
                "objeto_a": lado_a[1],
                "regime_b": lado_b[0],
                "objeto_b": lado_b[1],
                "topo_coincidente": k,
                "n_atributos": len(esquerda),
                "atributos_no_topo": ", ".join(esquerda[:k]),
                "modelo": MODELO,
            })
    return pd.DataFrame(linhas)


def tabela_global_coluna(medidas: list[dict]) -> pd.DataFrame:
    """Uma linha por célula, regime e coluna da matriz de design."""
    partes = []
    for medida in medidas:
        for chave in REGIMES:
            regime, objeto = chave
            ranking = ranking_por_coluna(medida["atribuicoes"][chave], medida["mapa"])
            ranking.insert(0, "objeto_explicado", objeto)
            ranking.insert(0, "regime", regime)
            ranking.insert(0, "celula", medida["celula"])
            partes.append(ranking)
    colunas = ["celula", "regime", "objeto_explicado", "coluna", "atributo",
               "shap_medio_abs", "posicao", "categoria_referencia_omitida"]
    return _com_modelo(pd.concat(partes, ignore_index=True)[colunas])


def tabela_pares_colineares(medidas: list[dict]) -> pd.DataFrame:
    """Os pares e blocos nomeados em D-053, membro a membro e em bloco somado.

    A tabela sai em formato longo, uma linha por membro, porque o bloco de `B_composto`
    tem quatro membros e uma coluna por membro não fecharia com o par de dois. Cada
    linha carrega a soma do bloco e a razão entre o maior e o menor membro, então a
    partição de crédito se lê sem sair da linha.
    """
    linhas = []
    for medida in medidas:
        celula = medida["celula"]
        for regime, objeto in REGIMES:
            ranking = medida["rankings"][(regime, objeto)].set_index("atributo")
            for par in PARES_COLINEARES:
                if celula not in par["celulas"]:
                    continue
                valores = {m: float(ranking.loc[m, "shap_medio_abs"])
                           for m in par["membros"]}
                soma = float(sum(valores.values()))
                maior = max(valores, key=valores.get)
                menor = min(valores, key=valores.get)
                razao = (valores[maior] / valores[menor]
                         if valores[menor] > 0 else float("inf"))
                for membro in par["membros"]:
                    linhas.append({
                        "celula": celula,
                        "regime": regime,
                        "objeto_explicado": objeto,
                        "bloco": par["bloco"],
                        "evidencia": par["evidencia"],
                        "n_membros": len(par["membros"]),
                        "membro": membro,
                        "shap_medio_abs": valores[membro],
                        "posicao": int(ranking.loc[membro, "posicao"]),
                        "shap_bloco_somado": soma,
                        "fracao_do_bloco": (valores[membro] / soma if soma
                                            else float("nan")),
                        "membro_dominante": maior,
                        "razao_maior_menor": razao,
                        "modelo": MODELO,
                    })
    return pd.DataFrame(linhas)


def tabela_casos_locais(medidas: list[dict]) -> pd.DataFrame:
    """Quatro casos por célula, com a contribuição de cada variável e o valor observado.

    Os casos saem do regime principal sobre a floresta completa, que é o modelo
    publicado: é o escore dele que o limiar calibrado corta, e explicar a fronteira com
    o escore de outro objeto descreveria uma decisão que ninguém tomou.
    """
    linhas = []
    for medida in medidas:
        chave = (REGIME_PRINCIPAL, OBJETO_COMPLETA)
        valores = medida["atribuicoes"][chave]
        valor_base = medida["bases"][chave]
        clinicos = medida["clinicos"]

        casos = selecionar_casos(medida["y_te"], medida["proba"], medida["limiar"])
        posicoes = {indice: p for p, indice in enumerate(medida["design_te"].index)}

        for _, caso in casos.iterrows():
            posicao = posicoes[int(caso["indice"])]
            contribuicoes = contribuicoes_locais(valores, medida["mapa"], posicao)
            for _, contribuicao in contribuicoes.iterrows():
                atributo = contribuicao["atributo"]
                linhas.append({
                    "celula": medida["celula"],
                    "regime": REGIME_PRINCIPAL,
                    "objeto_explicado": OBJETO_COMPLETA,
                    "caso": caso["caso"],
                    "indice": int(caso["indice"]),
                    "escore": float(caso["escore"]),
                    "limiar": medida["limiar"],
                    "distancia_ao_limiar": float(caso["distancia_ao_limiar"]),
                    "valor_base": float(valor_base[posicao]),
                    "atributo": atributo,
                    "contribuicao": float(contribuicao["contribuicao"]),
                    "valor_observado": float(
                        clinicos.loc[int(caso["indice"]), atributo]
                    ),
                    "modelo": MODELO,
                })

    # O mesmo índice nas duas células é o mesmo indivíduo: as duas usam a mesma
    # partição. A marca evita que o painel apresente como dois casos quem é um só.
    tabela = pd.DataFrame(linhas)
    celulas_do_indice = tabela.groupby("indice")["celula"].transform("nunique")
    tabela["indice_nas_duas_celulas"] = celulas_do_indice > 1
    return tabela


def _tres_significativos(valor: float) -> float:
    """Arredonda para `SIGNIFICATIVOS_ERRO` algarismos, preservando a ordem de grandeza.

    Arredondar por casa decimal não serve a uma coluna que vai de 1e-15 a 1e-1: ou zera
    o topo da escala ou grava o ruído do último bit. Por algarismo significativo, o
    valor fica legível e o arquivo para de se mover entre execuções.
    """
    return float(f"{float(valor):.{SIGNIFICATIVOS_ERRO - 1}e}")


def ordem_de_grandeza(valor: float) -> float:
    """Potência de dez logo abaixo do valor: 5,52e-15 e 5,551e-15 viram as duas 1e-15.

    É a forma estável de gravar um erro de arredondamento cujo primeiro algarismo se
    move entre execuções. Quem compara esses valores compara ordens, e comparar ordens
    é o que os documentos afirmam sobre eles.

    Erro exatamente zero não tem logaritmo e devolve zero: aditividade exata é a
    situação que o portão procura, não um caso a tratar por exceção.
    """
    valor = abs(float(valor))
    if valor == 0.0:
        return 0.0
    return float(10.0 ** np.floor(np.log10(valor)))


def tabela_arvores_excluidas(medidas: list[dict]) -> pd.DataFrame:
    """Quantas árvores saíram do regime de contraste, por que, e o que isso moveu.

    A floresta reduzida existe para o contraste fechar exato dos dois lados, e ela não é
    modelo publicado em momento nenhum. Os dois deslocamentos ficam na mesma linha
    porque é a comparação entre eles que sustenta o ramo escolhido em D-053.

    As colunas de erro agregadas sobre as árvores saem em `SIGNIFICATIVOS_ERRO`
    algarismos e as duas de aditividade da floresta saem em ordem de grandeza. Os dois
    tratamentos acontecem aqui e não na exportação: a tabela em memória e o arquivo
    versionado precisam trazer o mesmo número, senão a guarda que confere um passa a
    atestar o outro.
    """
    linhas = []
    for medida in medidas:
        defeito = medida["defeito"]
        linhas.append({
            "celula": medida["celula"],
            "regime_principal": f"{REGIME_PRINCIPAL} sobre {OBJETO_COMPLETA}",
            "regime_contraste": f"{REGIME_CONTRASTE} sobre {OBJETO_REDUZIDA}",
            "arvores_ajustadas": len(medida["floresta"].estimators_),
            "arvores_mantidas": int(defeito["mantidas"].size),
            "arvores_excluidas": int(defeito["excluidas"].size),
            "corte": defeito["corte"],
            "erro_min_mantidas": defeito["erro_min_mantidas"],
            "erro_max_mantidas": defeito["erro_max_mantidas"],
            "erro_min_excluidas": defeito["erro_min_excluidas"],
            "erro_max_excluidas": defeito["erro_max_excluidas"],
            "piso_por_arvore": defeito["piso"],
            "erro_aditividade_principal": medida["erro_principal"],
            "erro_aditividade_contraste": medida["erro_contraste"],
            "delta_arvores": medida["delta_arvores"],
            "delta_regime": medida["delta_regime"],
        })

    tabela = pd.DataFrame(linhas)
    for coluna in COLUNAS_DE_ERRO:
        tabela[coluna] = tabela[coluna].map(_tres_significativos)
    for coluna in COLUNAS_DE_ORDEM:
        tabela[coluna] = tabela[coluna].map(ordem_de_grandeza)
    return tabela


# ---------------------------------------------------------------------------
# Artefatos
# ---------------------------------------------------------------------------

def _recortar(tabela: pd.DataFrame, topo: int = TOPO_TEX) -> pd.DataFrame:
    """Os primeiros de cada célula e regime, para o `.tex` caber no documento.

    Sessenta linhas não entram em documento acadêmico, e cortar na exportação é melhor
    que cortar na fonte: quem quiser o ranking inteiro abre o CSV.
    """
    return (tabela.sort_values(["celula", "regime", "posicao"], kind="stable")
            .groupby(["celula", "regime"], sort=True)
            .head(topo).reset_index(drop=True))


def exportar(tabela: pd.DataFrame, nome: str, nota: str,
             recorte: pd.DataFrame | None = None,
             casas: int = art.CASAS_CSV, **kwargs) -> dict[str, Path]:
    """Grava a tabela como `.csv` completo e `.tex`, com o recorte quando houver um.

    O CSV passa por `artefatos.salvar_csv`, ponto único de gravação do projeto (D-050):
    valor gravado por caminho próprio escapa do arredondamento e devolve o diff
    permanente que D-048 fechou.

    Args:
        casas: precisão do CSV. O padrão de D-048 serve a tabela de resultado, onde
            nada vive abaixo de 1e-9. Quem grava erro numérico passa `CASAS_ARVORES`.
    """
    destino_csv = art.salvar_csv(tabela, cfg.PATH_TABELAS / f"{nome}.csv",
                                 casas=casas)
    para_tex = tabela if recorte is None else recorte
    destino_tex = ev.exportar_latex(para_tex, f"{nome}.tex", nota=nota, **kwargs)
    return {"tex": destino_tex, "csv": destino_csv}


def exportar_tudo(tabelas: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """Emite os seis pares de arquivos em `reports/tabelas/`."""
    nota_topo = (f"{NOTA_REGIMES} {NOTA_ESCALA} {NOTA_REFERENCIA} O .tex traz os "
                 f"{TOPO_TEX} primeiros de cada célula em cada regime; o .csv traz "
                 f"todos.")
    return {
        NOME_GLOBAL_ATRIBUTO: exportar(
            tabelas[NOME_GLOBAL_ATRIBUTO], NOME_GLOBAL_ATRIBUTO, nota=nota_topo,
            recorte=_recortar(tabelas[NOME_GLOBAL_ATRIBUTO]), float_format="%.6f",
        ),
        NOME_GLOBAL_COLUNA: exportar(
            tabelas[NOME_GLOBAL_COLUNA], NOME_GLOBAL_COLUNA, nota=nota_topo,
            recorte=_recortar(tabelas[NOME_GLOBAL_COLUNA]), float_format="%.6f",
        ),
        NOME_PARES: exportar(
            tabelas[NOME_PARES], NOME_PARES, float_format="%.6f",
            nota=("Pares e blocos nomeados em D-053 antes de qualquer atribuição "
                  "existir. Cada membro sai com a sua atribuição e com a soma do bloco, "
                  f"sem limiar que decida se agrupa. {NOTA_REGIMES} {NOTA_ESCALA}"),
        ),
        NOME_CASOS: exportar(
            tabelas[NOME_CASOS], NOME_CASOS, float_format="%.6f",
            nota=("Um verdadeiro positivo, um falso positivo, um falso negativo e um "
                  "verdadeiro negativo por célula, cada um o de menor distância entre "
                  "escore e limiar dentro da sua célula da confusão, com desempate pelo "
                  "menor índice. As contribuições saem do regime principal sobre a "
                  f"floresta completa, que é o modelo publicado. {NOTA_ESCALA}"),
        ),
        NOME_ARVORES: exportar(
            tabelas[NOME_ARVORES], NOME_ARVORES, float_format="%.3e",
            casas=CASAS_ARVORES,
            nota=("Árvores que o explicador desta versão atribui com erro sob o regime "
                  "de contraste, identificadas na própria execução sobre o teste "
                  "completo. A floresta reduzida é objeto de diagnóstico e nunca modelo "
                  "publicado: o SHAP global publicado é o tree\\_path\\_dependent sobre "
                  "as 500 árvores. O portão do regime de contraste é o piso por árvore, "
                  "medido na mesma execução (D-053)."),
        ),
        NOME_CONCORDANCIA: exportar_concordancia(tabelas[NOME_CONCORDANCIA]),
    }


def exportar_concordancia(tabela: pd.DataFrame) -> dict[str, Path]:
    """Grava `shap_concordancia_topo`, separado para poder sair sem o resto.

    O artefato se deriva dos rankings, e emiti-lo sozinho evita regravar as outras
    cinco tabelas quando só ele falta.
    """
    return exportar(
        tabela, NOME_CONCORDANCIA,
        nota=("Maior k tal que as posições 1 a k têm a mesma variável nos dois "
              "rankings. A primeira comparação é a do texto e da figura de contraste; "
              "a segunda fixa a floresta reduzida e isola o efeito do regime. "
              f"{NOTA_REGIMES}"),
    )


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

# Cor do valor observado no enxame: baixo no teal, alto no coral, na mesma paleta que
# o resto do projeto usa. O mapa vale só para variáveis com ordem; as nominais saem em
# cinza, porque colorir código de categoria sugeriria uma ordem que elas não têm.
CMAP_VALOR = LinearSegmentedColormap.from_list(
    "shap_valor", [cfg.TEAL, "#F1F5F9", cfg.CORAL]
)
CINZA_NOMINAL = "#94A3B8"

# Empilhamento do enxame. Os pontos se espalham na vertical pela densidade local em
# vez de por sorteio: sorteio faria a mesma execução desenhar figuras diferentes.
LARGURA_ENXAME = 0.40
BINS_ENXAME = 90
PONTO_ENXAME = 7
ALFA_ENXAME = 0.55

# Casos que viram figura. O falso positivo e o verdadeiro negativo ficam só na tabela:
# os dois que a triagem discute são o encaminhado e o que escapou.
CASOS_EM_FIGURA = ["VP", "FN"]
ROTULO_CASO = {
    "VP": "verdadeiro positivo",
    "FP": "falso positivo",
    "FN": "falso negativo",
    "VN": "verdadeiro negativo",
}

# O que toda explicação local precisa dizer na própria figura, e não na legenda
# (D-053). A primeira frase desarma a leitura de meio a meio; a segunda desarma a
# leitura de efeito protetor, que é o inverso da leitura clínica.
NOTA_LOCAL = (
    "O valor de base é a média das saídas da floresta sob reponderamento de classes, "
    "e fica perto de 0,5 por causa dele, não na prevalência de 7%.\n"
    "A referência das contribuições é o indivíduo médio sob esse reponderamento, e "
    "não a população: contribuição negativa significa abaixo dessa referência, nunca "
    "efeito protetor.\n"
    "A soma das contribuições reconstrói o escore do indivíduo, nunca o risco dele "
    "(D-022, D-040, D-053)."
)


def _eixo_figura(ax, xlabel: str, ylabel: str = "") -> None:
    """Rótulos e grade. O título da figura fica na legenda do LaTeX."""
    figu.eixo(ax, xlabel, ylabel, grade="x")
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)


def _destino(path_figuras, nome: str) -> Path:
    """Caminho do arquivo, com o diretório criado. `reports/figuras/` não é versionado."""
    path_figuras = cfg.PATH_FIGURAS if path_figuras is None else Path(path_figuras)
    path_figuras.mkdir(parents=True, exist_ok=True)
    return path_figuras / nome


def _formatar_valor(valor: float) -> str:
    """Valor observado como quem lê a variável o escreve: 87,3 e não 87,300000."""
    valor = float(valor)
    casas = 0 if valor.is_integer() else 1
    return figu.numero(valor, casas)


def atribuicao_por_atributo(valores: np.ndarray, mapa: pd.DataFrame) -> pd.DataFrame:
    """Contribuição com sinal de cada indivíduo, somada por variável conceitual.

    É a matriz que o enxame plota. A soma é com sinal, como na explicação local: em
    módulo, as colunas de uma nominal deixariam de se cancelar quando se cancelam.

    Returns:
        DataFrame de uma linha por indivíduo e uma coluna por variável conceitual, na
        ordem em que os atributos aparecem na matriz de design.
    """
    atributos = list(dict.fromkeys(mapa["atributo"]))
    colunas = {}
    for atributo in atributos:
        posicoes = np.flatnonzero((mapa["atributo"] == atributo).to_numpy())
        colunas[atributo] = valores[:, posicoes].sum(axis=1)
    return pd.DataFrame(colunas)


def _nominais(mapa: pd.DataFrame) -> set[str]:
    """Variáveis que passaram pelo one-hot, que são as que têm categoria omitida."""
    com_omitida = mapa[mapa["categoria_referencia_omitida"].astype(str) != ""]
    return set(com_omitida["atributo"])


def _empilhar(valores: np.ndarray, largura: float = LARGURA_ENXAME,
              bins: int = BINS_ENXAME) -> np.ndarray:
    """Deslocamento vertical de cada ponto, proporcional à densidade local.

    Ponto que cai numa faixa cheia se afasta do eixo, ponto isolado fica em cima dele.
    O resultado é determinístico, o que jitter sorteado não garante.
    """
    desloc = np.zeros(len(valores))
    if len(valores) == 0 or np.ptp(valores) == 0:
        return desloc

    cortes = np.linspace(valores.min(), valores.max(), bins + 1)[1:-1]
    faixas = np.clip(np.digitize(valores, cortes), 0, bins - 1)
    maior = max(int(np.bincount(faixas).max()), 1)
    for faixa in np.unique(faixas):
        dentro = np.flatnonzero(faixas == faixa)
        desloc[dentro] = np.arange(len(dentro)) - (len(dentro) - 1) / 2
    return desloc * (2 * largura / maior)


def figura_beeswarm(medida: dict, path_figuras: Path | None = None) -> Path:
    """Atribuição global da célula, um ponto por indivíduo e uma linha por variável.

    O enxame mostra o que o ranking esconde: a direção das contribuições, a assimetria
    da cauda e a dispersão dentro de cada variável. A cor traz o valor observado, então
    dá para ler se contribuição alta vem de valor alto ou de valor baixo. As nominais
    saem em cinza porque o código da categoria não tem ordem.

    Salva a figura e devolve o caminho.
    """
    celula = medida["celula"]
    chave = (REGIME_PRINCIPAL, OBJETO_COMPLETA)
    mapa = medida["mapa"]
    por_atributo = atribuicao_por_atributo(medida["atribuicoes"][chave], mapa)
    ordem = list(medida["rankings"][chave]["atributo"])
    nominais = _nominais(mapa)
    clinicos = medida["clinicos"].loc[medida["design_te"].index]

    fig, ax = figu.subplots(figu.LARGURA_INTEIRA, 1.4 + 0.42 * len(ordem))
    imagem = None

    for linha, atributo in enumerate(ordem):
        y = len(ordem) - 1 - linha
        contribuicoes = por_atributo[atributo].to_numpy()
        desloc = _empilhar(contribuicoes)

        if atributo in nominais:
            ax.scatter(contribuicoes, y + desloc, s=PONTO_ENXAME, c=CINZA_NOMINAL,
                       alpha=ALFA_ENXAME, linewidths=0)
            continue

        # Percentil no lugar do valor cru: uma cauda longa de circunferência abdominal
        # jogaria todo o resto da escala para a mesma cor.
        observado = clinicos[atributo].rank(pct=True).to_numpy()
        imagem = ax.scatter(contribuicoes, y + desloc, s=PONTO_ENXAME, c=observado,
                            cmap=CMAP_VALOR, vmin=0, vmax=1, alpha=ALFA_ENXAME,
                            linewidths=0)

    ax.axvline(0, color=cfg.INK, lw=1, ls=":")
    ax.set_yticks(range(len(ordem)), [figu.rotulo(a) for a in reversed(ordem)])
    ax.xaxis.set_major_formatter(figu.formatador(2))
    _eixo_figura(ax, "contribuição para o escore, em fração de votos")

    if imagem is not None:
        barra = fig.colorbar(imagem, ax=ax, pad=0.02, fraction=0.035, shrink=0.42)
        barra.set_ticks([0, 1], labels=["valor baixo", "valor alto"])
        barra.outline.set_visible(False)

    destino = _destino(path_figuras, f"shap_beeswarm_{celula}.png")
    return figu.salvar(
        fig, destino,
        "Nominais em cinza: o código da categoria não tem ordem para colorir. A "
        "categoria de referência de cada nominal tem atribuição zero por "
        "construção, efeito do drop=\"first\", e entra na soma da variável.\n"
        "Escala de fração de votos: a soma das contribuições reconstrói o escore "
        "do indivíduo, nunca o risco dele (D-022, D-040, D-053).",
    )


def ler_tabela(nome: str, path: Path | None = None) -> pd.DataFrame:
    """Lê um dos artefatos versionados de `reports/tabelas/`.

    O contraste entre regimes é a parte cara da execução e não muda de valor entre
    reexecuções do documento, então a figura o lê em vez de recalcular.

    Raises:
        FileNotFoundError: se o artefato não estiver emitido.
    """
    destino = (cfg.PATH_TABELAS / f"{nome}.csv") if path is None else Path(path)
    if not destino.exists():
        raise FileNotFoundError(
            f"Artefato ausente: {destino}. Rode python -m src.explicabilidade antes."
        )
    return pd.read_csv(destino, converters={"categoria_referencia_omitida": str})


def _prefixo_comum(esquerda: list[str], direita: list[str]) -> int:
    """Quantas posições do topo os dois rankings compartilham, na mesma ordem."""
    quantas = 0
    for um, outro in zip(esquerda, direita):
        if um != outro:
            break
        quantas += 1
    return quantas


def figura_contraste_de_regimes(celula: str, tabela: pd.DataFrame | None = None,
                                path_figuras: Path | None = None) -> Path:
    """Posições no ranking sob os dois regimes de perturbação, lado a lado.

    É a figura da mitigação: a divergência entre os dois rankings é a medida da
    exposição deste trabalho ao problema que a referência aponta (D-053). Quem não
    muda de posição sai em teal e quem muda sai em coral, com o tamanho do
    deslocamento anotado.

    Args:
        tabela: `shap_global_atributo` já lida. Omitida, sai do artefato versionado,
            que é o que evita recalcular o regime caro a cada reexecução.

    Salva a figura e devolve o caminho.
    """
    tabela = ler_tabela(NOME_GLOBAL_ATRIBUTO) if tabela is None else tabela
    recorte = tabela[tabela["celula"] == celula]
    if recorte.empty:
        raise KeyError(f"Célula {celula!r} ausente de {NOME_GLOBAL_ATRIBUTO}.")

    lados = {}
    for regime, _objeto in REGIMES:
        lado = recorte[recorte["regime"] == regime]
        if lado.empty:
            raise KeyError(f"Regime {regime!r} ausente de {NOME_GLOBAL_ATRIBUTO}.")
        lados[regime] = lado.set_index("atributo")["posicao"].astype(int).sort_values()

    esquerda, direita = lados[REGIME_PRINCIPAL], lados[REGIME_CONTRASTE]
    n = len(esquerda)

    fig, ax = figu.subplots(figu.LARGURA_INTEIRA, 1.4 + 0.40 * n)
    for atributo, posicao in esquerda.items():
        partida, chegada = int(posicao), int(direita[atributo])
        moveu = chegada != partida
        cor = cfg.CORAL if moveu else cfg.TEAL
        # Traço cheio e pontilhado: no cinza a cor sozinha não separa quem moveu.
        ax.plot([0, 1], [n - partida, n - chegada], color=cor,
                lw=2.0 if moveu else 1.2, ls="-" if moveu else ":",
                alpha=0.9 if moveu else 0.55, marker="o", markersize=5)
        nome = figu.rotulo(str(atributo))
        ax.text(-0.03, n - partida, f"{partida}. {nome}", ha="right", va="center",
                fontsize=9, color=cfg.INK, fontweight="bold" if moveu else "normal")
        deslocou = f" ({chegada - partida:+d})" if moveu else ""
        ax.text(1.03, n - chegada, f"{chegada}. {nome}{deslocou}", ha="left",
                va="center", fontsize=9, color=cor if moveu else cfg.INK,
                fontweight="bold" if moveu else "normal")

    ax.set_xlim(-0.62, 1.62)
    ax.set_ylim(-0.4, n + 0.6)
    ax.set_xticks([0, 1], [f"{REGIME_PRINCIPAL}\n{OBJETO_COMPLETA}",
                           f"{REGIME_CONTRASTE}\n{OBJETO_REDUZIDA}"],
                  fontsize=9, color=cfg.INK)
    ax.set_yticks([])
    ax.grid(visible=False)
    for lado in ("top", "right", "left", "bottom"):
        ax.spines[lado].set_visible(False)
    destino = _destino(path_figuras, f"shap_contraste_{celula}.png")
    return figu.salvar(
        fig, destino,
        "O regime de contraste assume independência entre as variáveis, que é a "
        "formulação apontada como frágil, e roda sobre a floresta reduzida, "
        "objeto de diagnóstico e nunca modelo publicado.\n"
        "A divergência entre os dois rankings é a medida da exposição deste "
        "trabalho ao problema (D-053).",
    )


def _painel_waterfall(ax, caso: pd.DataFrame) -> tuple[float, float]:
    """Uma explicação local: o acumulado do valor de base até o escore do indivíduo.

    Returns:
        Os extremos que o painel ocupa no eixo do escore. Quem desenha o par usa os
        dois para dar a mesma escala aos dois casos, já que eixo compartilhado fica
        com o limite do último painel desenhado.
    """
    linha = caso.iloc[0]
    valor_base = float(linha["valor_base"])
    limiar = float(linha["limiar"])
    escore = float(linha["escore"])

    ordenado = caso.reindex(
        caso["contribuicao"].abs().sort_values(ascending=False, kind="stable").index
    )
    inicio, esquerdas, tamanhos, cores, hachuras, rotulos = valor_base, [], [], [], [], []
    for _, contribuicao in ordenado.iterrows():
        valor = float(contribuicao["contribuicao"])
        esquerdas.append(inicio)
        tamanhos.append(valor)
        cores.append(cfg.CORAL if valor >= 0 else cfg.TEAL)
        # Hachura na contribuição negativa: no cinza a cor sozinha não separa o sinal.
        hachuras.append("" if valor >= 0 else "///")
        rotulos.append(f"{figu.rotulo(str(contribuicao['atributo']))} = "
                       f"{_formatar_valor(contribuicao['valor_observado'])}")
        inicio += valor

    y = np.arange(len(tamanhos))[::-1]
    for altura, comeco, tamanho, cor, hachura in zip(
            y, esquerdas, tamanhos, cores, hachuras):
        ax.barh(altura, tamanho, left=comeco, height=0.62, color=cor, alpha=0.9,
                hatch=hachura, edgecolor=cfg.INK, linewidth=0.4)
    for altura, (comeco, tamanho) in enumerate(zip(esquerdas, tamanhos)):
        fim = comeco + tamanho
        texto = figu.numero(abs(tamanho), 3)
        texto = f"+{texto}" if tamanho >= 0 else f"-{texto}"
        ax.text(fim + (0.003 if tamanho >= 0 else -0.003), y[altura], texto,
                ha="left" if tamanho >= 0 else "right", va="center", fontsize=8,
                color=cfg.INK)

    n = len(tamanhos)
    decisao = "encaminhado" if escore >= limiar else "não encaminhado"
    distancia = figu.numero(abs(escore - limiar), 3)
    # As retas param na faixa das barras. Se cruzarem o eixo inteiro, cortam o cabeçalho.
    ax.plot([valor_base, valor_base], [-1.8, n - 0.7], color=cfg.INK, lw=1.2, ls=":")
    ax.plot([limiar, limiar], [-1.8, n - 0.7], color=cfg.CORAL, lw=1.8, ls="--")
    ax.scatter([escore], [-1.0], color=cfg.INK, s=45, marker="D", zorder=5)
    ax.text(escore, -1.5, f"escore {figu.numero(escore, 3)}", ha="center",
            va="top", fontsize=8, fontweight="bold", color=cfg.INK)

    # Cabeçalho à esquerda, acima das retas. As duas marcas ficam em alturas
    # diferentes, cada uma sobre a sua reta, para não se cobrirem.
    ax.set_ylim(-2.3, n + 2.8)
    ax.text(0.0, n + 2.55,
            f"{ROTULO_CASO[linha['caso']]}, indivíduo {int(linha['indice'])}\n"
            f"{decisao}, a {distancia} do limiar",
            transform=ax.get_yaxis_transform(), ha="left", va="top", fontsize=9,
            color=cfg.INK, linespacing=1.25)
    ax.text(valor_base, n + 0.15, f"valor de base {figu.numero(valor_base, 3)}",
            ha="center", va="bottom", fontsize=8, color=cfg.INK)
    ax.text(limiar, n - 0.2, f"limiar {figu.numero(limiar, 3)}",
            ha="center", va="bottom", fontsize=8, fontweight="bold", color=cfg.CORAL)

    ax.set_yticks(y, rotulos)
    ax.xaxis.set_major_formatter(figu.formatador(2))
    _eixo_figura(ax, "escore, em fração de votos")

    fins = [e + t for e, t in zip(esquerdas, tamanhos)]
    return min(fins + [escore, limiar]), max(fins + [valor_base])


def figura_waterfall_par(medida: dict, casos: list[str] | None = None,
                         path_figuras: Path | None = None) -> Path:
    """O verdadeiro positivo e o falso negativo da célula, um sobre o outro.

    Os dois têm o mesmo desfecho observado e decisões opostas, e explicá-los juntos é
    o que mostra por qual variável a decisão passou. A marca do limiar diz onde a
    decisão acontece, e a nota diz contra qual referência as contribuições são lidas,
    que é o que D-053 manda pôr na figura em vez de na legenda.

    O falso positivo e o verdadeiro negativo ficam só na tabela: os quatro casos saem
    em `shap_casos_locais`, e a figura traz os dois que a triagem discute.

    Salva a figura e devolve o caminho.
    """
    casos = list(CASOS_EM_FIGURA) if casos is None else list(casos)
    tabela = tabela_casos_locais([medida])
    celula = medida["celula"]

    # Um painel por linha. Lado a lado, cada eixo fica com menos da metade dos 16 cm
    # e o rótulo por extenso não cabe: o tight_layout desiste e o texto estoura.
    n_variaveis = int(tabela["atributo"].nunique())
    por_painel = 0.26 * (n_variaveis + 5.1)
    fig, eixos = figu.subplots(
        figu.LARGURA_INTEIRA, por_painel * len(casos) / 0.82,
        len(casos), 1, sharex=True,
    )
    extremos = []
    for ax, caso in zip(np.atleast_1d(eixos), casos):
        recorte = tabela[tabela["caso"] == caso]
        if recorte.empty:
            raise KeyError(f"Caso {caso!r} ausente dos casos locais de {celula}.")
        extremos.append(_painel_waterfall(ax, recorte))

    # Folga à esquerda para o valor de cada contribuição negativa caber fora da barra
    # sem encostar no rótulo da variável.
    menor = min(e[0] for e in extremos)
    maior = max(e[1] for e in extremos)
    span = maior - menor or 1.0
    # Folga nos dois lados para o número da contribuição não encostar na borda.
    np.atleast_1d(eixos)[0].set_xlim(menor - 0.28 * span, maior + 0.16 * span)

    destino = _destino(path_figuras, f"shap_waterfall_{celula}.png")
    return figu.salvar(fig, destino, NOTA_LOCAL)


def material_das_figuras(celula: str, particao: xp.Particao) -> dict:
    """Ajusta a célula e calcula o regime principal, que é o que as figuras consomem.

    O documento recalcula este regime em vez de ler tabela porque o enxame precisa das
    atribuições linha a linha e o waterfall precisa do valor de base de cada
    indivíduo, e nenhum dos dois cabe num artefato tabular. O regime de contraste fica
    de fora: é a parte cara, não vira figura por indivíduo, e quem o exibe lê o
    artefato versionado.

    O limiar é conferido contra o publicado pelo mesmo caminho de `medir_celula`, de
    modo que a figura descreve o modelo da tabela consolidada e não um parecido com
    ele.

    Returns:
        Dicionário no formato que as funções de tabela já consomem, com o regime
        principal no lugar dos dois.

    Raises:
        ValueError: se o limiar divergir do publicado, se a aditividade não fechar ou
            se o escore pela matriz de design não bater com o do pipeline.
    """
    base = preparar(celula, particao)
    conferencia = conferir_limiar(base["limiar"], celula)

    design = np.asarray(base["design_te"])
    proba = base["floresta"].predict_proba(design)[:, 1]
    valores, valor_base = explicar(base["floresta"], design, REGIME_PRINCIPAL)

    erro = erro_de_aditividade(valores, valor_base, proba)
    if erro > TOL_PRINCIPAL:
        raise ValueError(
            f"{celula}: aditividade do regime {REGIME_PRINCIPAL} não fecha em "
            f"{TOL_PRINCIPAL:g}. Erro máximo {erro:.3e}."
        )
    if not np.allclose(proba, base["proba"], atol=1e-12):
        raise ValueError(
            f"{celula}: o escore pela matriz de design não bate com o do pipeline."
        )

    chave = (REGIME_PRINCIPAL, OBJETO_COMPLETA)
    return {
        **base,
        "conferencia_limiar": conferencia,
        "atribuicoes": {chave: valores},
        "bases": {chave: valor_base},
        "rankings": {chave: ranking_por_atributo(valores, base["mapa"])},
        "erro_principal": erro,
    }


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------

def executar(celulas: list[str] | None = None,
             particao: xp.Particao | None = None,
             salvar: bool = True) -> dict:
    """Mede as células, monta as seis tabelas e emite os artefatos.

    Args:
        celulas: chaves de `features.FEATURE_SETS`. Omitido, usa as duas do grid
            publicado da floresta.
        particao: split pronto. Omitido, `experimento.preparar_particao` constrói um, o
            que implica reler o CSV bruto.
        salvar: desligado, calcula tudo sem escrever em `reports/tabelas/`.

    Returns:
        Dicionário com `medidas`, `tabelas`, `artefatos`, `conferencias`, `particao` e
        `segundos`.
    """
    celulas = list(CELULAS) if celulas is None else list(celulas)
    desconhecidas = [c for c in celulas if c not in ft.FEATURE_SETS]
    if desconhecidas:
        raise ValueError(
            f"Conjunto(s) desconhecido(s): {desconhecidas}. "
            f"Disponíveis: {sorted(ft.FEATURE_SETS)}"
        )

    inicio = time.perf_counter()
    particao = xp.preparar_particao() if particao is None else particao
    medidas = [medir_celula(celula, particao) for celula in celulas]

    tabelas = {
        NOME_GLOBAL_ATRIBUTO: tabela_global_atributo(medidas),
        NOME_GLOBAL_COLUNA: tabela_global_coluna(medidas),
        NOME_PARES: tabela_pares_colineares(medidas),
        NOME_CASOS: tabela_casos_locais(medidas),
        NOME_ARVORES: tabela_arvores_excluidas(medidas),
        NOME_CONCORDANCIA: concordancia_topo(
            tabela_global_atributo(medidas, chaves=REGIMES + [PRINCIPAL_REDUZIDA])
        ),
    }
    artefatos = exportar_tudo(tabelas) if salvar else {}

    return {
        "medidas": medidas,
        "tabelas": tabelas,
        "artefatos": artefatos,
        "conferencias": [m["conferencia_limiar"] for m in medidas],
        "particao": particao,
        "segundos": time.perf_counter() - inicio,
    }


def formatar_portoes(medida: dict) -> str:
    """Os dois erros de aditividade e o piso por árvore, numa linha por célula."""
    defeito = medida["defeito"]
    return (
        f"{medida['celula']}: {REGIME_PRINCIPAL} {medida['erro_principal']:.3e} "
        f"(portão {TOL_PRINCIPAL:g}), {REGIME_CONTRASTE} "
        f"{medida['erro_contraste']:.3e} (piso por árvore {defeito['piso']:.3e}), "
        f"{defeito['excluidas'].size} árvores excluídas de "
        f"{len(medida['floresta'].estimators_)}, corte {defeito['corte']:g}, "
        f"mantidas entre {defeito['erro_min_mantidas']:.3e} e "
        f"{defeito['erro_max_mantidas']:.3e}, excluídas entre "
        f"{defeito['erro_min_excluidas']:.3e} e {defeito['erro_max_excluidas']:.3e}"
    )


def main(argv: list[str] | None = None) -> int:
    """Entrada de linha de comando. É o que `make shap` chama de fato."""
    # O console do Windows abre em cp1252 e a saída imprime acentuação. Sem isto a
    # execução sai com status 1 por codificação, que se lê como portão vermelho e não é
    # (D-052).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        prog="python -m src.explicabilidade",
        description=("Calcula as atribuições SHAP da floresta publicada e emite as "
                     "tabelas de explicabilidade."),
    )
    parser.add_argument("--celulas", nargs="+", default=None,
                        help=f"padrão: {' '.join(CELULAS)}")
    parser.add_argument("--sem-artefato", action="store_true",
                        help="calcula sem escrever em reports/tabelas/")
    args = parser.parse_args(argv)

    saida = executar(celulas=args.celulas, salvar=not args.sem_artefato)

    print("Limiar desta execução contra o publicado:")
    for conferencia in saida["conferencias"]:
        print(f"  {conferencia['celula']}: {conferencia['limiar_obtido']:.9f} contra "
              f"{conferencia['limiar_publicado']:.9f} em {xp.NOME_CONSOLIDADO}.csv")
    print()

    print("Aditividade e árvores:")
    for medida in saida["medidas"]:
        print(f"  {formatar_portoes(medida)}")
    for medida in saida["medidas"]:
        for nome, repetidos in medida["empates"].items():
            if repetidos:
                print(f"  empate de |SHAP| em {medida['celula']}, {nome}: "
                      f"{', '.join(repetidos)}")
    print()

    print("Deslocamento de ranking:")
    for medida in saida["medidas"]:
        print(f"  {medida['celula']}: delta_arvores {medida['delta_arvores']}, "
              f"delta_regime {medida['delta_regime']}")
    print()

    for nome, tabela in saida["tabelas"].items():
        print(f"{nome}: {len(tabela)} linhas")
    if saida["artefatos"]:
        print()
        for nome, caminhos in saida["artefatos"].items():
            print(f"{nome}: {caminhos['tex'].name}, {caminhos['csv'].name}")

    print()
    print(f"Concluído em {saida['segundos'] / 60:.1f} min.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
