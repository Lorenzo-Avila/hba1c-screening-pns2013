"""Ponto único de reprodução dos resultados.

Até a Sprint 1, as tabelas publicadas no `ESTADO.md` saíram de código de sessão que
não sobreviveu à sessão. Os números estavam certos e passavam nos testes, mas das
cinco linhas da tabela consolidada só três tinham artefato rastreado
(`reports/tabelas/comparativo_logreg.tex`), e os três regimes de carga operacional
não tinham nenhum. Este módulo fecha a lacuna: o que ele emite passa a ser a fonte e
o `ESTADO.md` passa a ser a transcrição, não o contrário.

Os artefatos que sustentam o argumento, todos em `reports/tabelas/`:

- `comparativo_consolidado`, uma linha por célula do grid.
- `carga_operacional`, os três regimes de encaminhamento.
- `equiparado_por_modelo`, o regime de recall equiparado de cada célula calibrável.
  A carga operacional equipara só o modelo de referência, e o contraste entre os dois
  modelos nesse regime tem dois lados (D-047).

O produto cartesiano **não é cheio**. `RegraClinicaADA` lê idade e IMC das preditoras
cruas e o `DummyClassifier` ignora `X`: rodá-los sobre A, B e C daria três linhas
idênticas cada, e a tabela deixaria de bater com o `ESTADO.md` por artefato do próprio
grid, ou seja, uma divergência falsa em cima da verificação que existe para dar
confiança. Quem governa isso é a marca `Estimador.invariante_ao_conjunto`.

Uso:

    python -m src.experimento              # emite os artefatos e confere
    python -m src.experimento --figuras    # idem, com as figuras do teste
    python -m src.experimento --saida DIR  # grava em DIR, confere contra reports/tabelas/
"""

from __future__ import annotations

import argparse
import numbers
import string
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import artefatos as art
from . import benchmark as bm
from . import config as cfg
from . import data as dt
from . import evaluate as ev
from . import features as ft
from . import models as md

# Modelos com resultado publicável. A Random Forest entra depois de a busca fixar seus
# hiperparâmetros: enquanto eles eram placeholder declarado, reportar número saído dali
# seria pior que não reportar.
MODELOS_PADRAO = ["LogReg", "Random Forest", "Regra ADA", "Dummy (maioria)"]

NOME_FLORESTA = "Random Forest"

# A floresta não roda em `C_completo`. O conjunto foi descartado pelo diagnóstico de
# multicolinearidade (D-016), e publicar desempenho de conjunto descartado sugere que ele
# ainda está em disputa. A logística continua aparecendo nos três porque a linha de
# `C_completo` é o lastro empírico do descarte: sem ela, D-016 vira afirmação sem número.
CONJUNTOS_FLORESTA = ["A_atomico", "B_composto"]

# Ordem das linhas na tabela publicada: os modelos primeiro, os pisos depois.
ORDEM_MODELOS = ["LogReg", NOME_FLORESTA, "Regra ADA", "Dummy (maioria)"]

# Referências da reprodução. A primeira cobre as três linhas da logística e vem da
# Sprint 1; a segunda cobre as duas da floresta. A antiga ganhou função nova ao lado da
# que já tinha: ela agora prova que acrescentar a floresta ao grid não moveu nenhum
# número da logística. Se mover, é defeito no grid, não resultado (D-039).
REFERENCIA = "comparativo_logreg.tex"
REFERENCIA_FLORESTA = "comparativo_rf.tex"
REFERENCIAS = [REFERENCIA, REFERENCIA_FLORESTA]

NOME_CONSOLIDADO = "comparativo_consolidado"
NOME_CARGA = "carga_operacional"
NOME_EQUIPARADO = "equiparado_por_modelo"
NOME_RESUMO = "resumo_amostra"
NOME_AUSENTES = "ausentes_por_variavel"
NOME_CALIBRACAO = "calibracao_comparada"
NOME_MCNEMAR = "mcnemar_pareado"
NOME_CUSTO_MARGINAL = "custo_marginal_pareado"
NOME_CURVA_ROC = "curva_roc"
NOME_CURVA_PR = "curva_pr"
NOME_DECIS = "calibracao_decis"

# O par do custo marginal: a floresta contra a logística, nos conjuntos em que as duas
# rodam. O sinal dos deltas é sempre floresta menos logística.
PAR_CUSTO = (NOME_FLORESTA, "LogReg")

# Par que o McNemar compara: os dois modelos no conjunto em que a floresta venceu.
PAR_MCNEMAR = ((NOME_FLORESTA, "B_composto"), ("LogReg", "B_composto"))

# Moldes do nome de cada regime de carga, pela chave que vai em `regime_id`. As colunas
# `regime` e `rotulo` saem do mesmo molde, a primeira com ponto e a segunda com vírgula
# decimal, para que as duas nunca descrevam regimes diferentes.
MOLDES_REGIME = {
    "regra": "{regra}",
    "equiparado": "{celula}, recall equiparado (alvo {alvo:.4f} no treino)",
    "padrao": "{celula}, ponto padrão (recall-alvo {alvo:.2f})",
}


class _FormatoDecimal(string.Formatter):
    """Formata os campos de um molde trocando o separador decimal dos números reais."""

    def __init__(self, separador: str):
        super().__init__()
        self.separador = separador

    def format_field(self, valor, especificacao):
        texto = super().format_field(valor, especificacao)
        if isinstance(valor, numbers.Real) and not isinstance(valor, numbers.Integral):
            return texto.replace(".", self.separador)
        return texto


def nome_do_regime(regime_id: str, separador: str = ".", **campos) -> str:
    """Preenche o molde de `MOLDES_REGIME` com o separador decimal pedido.

    Com ponto, devolve o texto da coluna `regime`; com vírgula, o da coluna `rotulo`.
    """
    return _FormatoDecimal(separador).format(MOLDES_REGIME[regime_id], **campos)


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
               salvar_figuras: bool = False,
               path_figuras: Path | None = None) -> list[dict]:
    """Executa o grid modelo × conjunto e devolve os resultados brutos.

    Args:
        modelos: nomes em `models.estimadores()`. Omitido, usa `MODELOS_PADRAO`.
        conjuntos: chaves de `features.FEATURE_SETS`. Omitido, usa as três.
        particao: split pronto. Omitido, `preparar_particao` constrói um, o que
            implica reler o CSV bruto.
        recall_alvo: ponto de operação dos estimadores calibráveis. Os invariantes
            ignoram, porque operam com limiar fixo.
        path_figuras: onde gravar as figuras. Omitido, `reports/figuras/`.

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
                path_figuras=path_figuras,
            ))
    return resultados


def compor_grid(modelos: list[str] | None = None,
                conjuntos: list[str] | None = None,
                particao: Particao | None = None,
                recall_alvo: float = cfg.RECALL_ALVO,
                salvar_figuras: bool = False,
                path_figuras: Path | None = None) -> list[dict]:
    """O grid publicado, onde nem todo modelo roda em todo conjunto.

    `rodar_grid` aplica os mesmos conjuntos a todos os modelos que dependem de um, e a
    tabela publicada não é assim: a logística aparece nos três conjuntos e a floresta só
    em `A_atomico` e `B_composto`, porque `C_completo` foi descartado por D-016. Daí duas
    passadas e uma concatenação, em vez de um cartesiano com uma célula recortada depois.

    A ordem final segue `ORDEM_MODELOS`, e dentro de cada modelo a ordem dos conjuntos
    que o chamador pediu. Ordenação estável, então nada se embaralha.
    """
    modelos = list(MODELOS_PADRAO) if modelos is None else list(modelos)
    conjuntos = list(ft.FEATURE_SETS) if conjuntos is None else list(conjuntos)
    particao = preparar_particao() if particao is None else particao

    demais = [nome for nome in modelos if nome != NOME_FLORESTA]
    resultados = rodar_grid(demais, conjuntos, particao=particao,
                            recall_alvo=recall_alvo, salvar_figuras=salvar_figuras,
                            path_figuras=path_figuras)

    if NOME_FLORESTA in modelos:
        da_floresta = [c for c in conjuntos if c in CONJUNTOS_FLORESTA]
        resultados += rodar_grid([NOME_FLORESTA], da_floresta, particao=particao,
                                 recall_alvo=recall_alvo,
                                 salvar_figuras=salvar_figuras,
                                 path_figuras=path_figuras)

    posicao = {nome: i for i, nome in enumerate(ORDEM_MODELOS)}
    resultados.sort(key=lambda r: posicao.get(r["modelo"], len(posicao)))
    return resultados


def _rotulo(resultado: dict) -> str:
    """Nome da estratégia como `benchmark.tabela_nns` o escreve."""
    conjunto = resultado["conjunto"]
    return (f"{resultado['modelo']} · {conjunto}"
            if conjunto not in ("", SEM_CONJUNTO) else resultado["modelo"])


def celula(resultados: list[dict], modelo: str,
           conjunto: str = SEM_CONJUNTO) -> dict:
    """Localiza uma célula do grid pelo par modelo e conjunto.

    Existe para o notebook não precisar montar filtro a cada gráfico que quer exibir.
    Falha alto quando a célula não está no grid, em vez de devolver vazio e deixar o
    erro aparecer três células adiante como `IndexError`.

    Raises:
        KeyError: se o par não estiver nos resultados.
    """
    for r in resultados:
        if r["modelo"] == modelo and r["conjunto"] == conjunto:
            return r
    disponiveis = sorted({(r["modelo"], r["conjunto"]) for r in resultados})
    raise KeyError(
        f"Célula ({modelo!r}, {conjunto!r}) ausente do grid. Disponíveis: {disponiveis}"
    )


def encaminhamento(resultado: dict) -> tuple[int, float]:
    """Encaminhados e taxa de encaminhamento de um resultado, sobre o teste inteiro.

    Encaminhado é quem o ponto de operação manda a exame, `vp + fp`. A carga
    operacional, o regime equiparado e a tabela consolidada leem daqui, para que as
    três tabelas publiquem o mesmo número.
    """
    total = resultado["vp"] + resultado["vn"] + resultado["fp"] + resultado["fn"]
    encaminhados = resultado["vp"] + resultado["fp"]
    return encaminhados, encaminhados / total


def consolidar(resultados: list[dict]) -> pd.DataFrame:
    """A tabela consolidada: `evaluate.comparar` e as colunas que o painel lê prontas.

    As colunas novas entram no fim, e as de `comparar` saem como sempre saíram:
    `encaminhados` e `taxa_encaminhamento` de `encaminhamento`; `referencia_d030`,
    verdadeira só na célula que `modelo_de_referencia` elege (D-030); `fpr`, a taxa
    de falsos positivos no ponto de operação; e `recall_alvo`, o alvo com que o limiar
    foi calibrado, vazio nos comparadores de limiar fixo, que não miram alvo nenhum.
    """
    tabela = ev.comparar(resultados)
    referencia = modelo_de_referencia(resultados)
    carga = [encaminhamento(r) for r in resultados]
    tabela["encaminhados"] = [enc for enc, _ in carga]
    tabela["taxa_encaminhamento"] = [taxa for _, taxa in carga]
    tabela["referencia_d030"] = [r is referencia for r in resultados]
    tabela["fpr"] = [r["fp"] / (r["fp"] + r["vn"]) for r in resultados]
    tabela["recall_alvo"] = [r["recall_alvo"] if r.get("limiar_calibrado", True)
                             else float("nan") for r in resultados]
    return tabela


def modelo_de_referencia(resultados: list[dict]) -> dict:
    """O resultado de menor NNS entre os estimadores calibráveis.

    Critério declarado e reprodutível, o mesmo que `benchmark.leitura_operacional` já
    aplica ao escolher a estratégia que enfrenta a regra: menor NNS é menos
    encaminhamento por caso detectado, que é a pergunta operacional. Quem ganha sai
    do grid da execução e está em `reports/tabelas/carga_operacional.csv`; escrever o
    vencedor aqui deixa o docstring mentir na primeira vez que o grid mudar, e ele já
    mudou uma vez (D-045).

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

    É o resultado que sustenta a conclusão do trabalho: a regra clínica encaminha mais
    da metade da população adulta e opera em ponto único, enquanto o modelo escolhe
    onde operar. Sem esta tabela, o argumento de aplicabilidade não tem lastro.

    O regime equiparado exige uma **segunda avaliação** do modelo de referência, com
    `recall_alvo = benchmark.recall_da_regra_no_treino`. É o cálculo mais frágil da
    Sprint 1, o único que atravessa treino e teste de propósito, e por isso o que mais
    precisa estar versionado: o recall da regra é medido **no treino**, porque
    calibrar o limiar pelo recall que ela obtém no teste vazaria o teste para dentro
    da decisão de limiar.

    Returns:
        DataFrame de três linhas com `regime`, `encaminhados`, `taxa_encaminhamento`,
        `casos_detectados`, `casos_perdidos`, `recall`, `nns`, e no fim `regime_id`
        (`regra`, `equiparado`, `padrao`) e `rotulo`, o nome do regime com vírgula
        decimal para o painel. A advertência sobre volume absoluto fica em
        `attrs["nota"]`.
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

    regimes = (
        ("regra", regra, {"regra": nome_regra}),
        ("equiparado", equiparado, {"celula": _rotulo(padrao), "alvo": alvo_da_regra}),
        ("padrao", padrao, {"celula": _rotulo(padrao), "alvo": padrao["recall_alvo"]}),
    )

    linhas = []
    for regime_id, r, campos in regimes:
        encaminhados, taxa = encaminhamento(r)
        linhas.append({
            "regime": nome_do_regime(regime_id, ".", **campos),
            "encaminhados": encaminhados,
            "taxa_encaminhamento": taxa,
            "casos_detectados": r["vp"],
            "casos_perdidos": r["fn"],
            "recall": r["recall"],
            "nns": r["nns"],
            "regime_id": regime_id,
            "rotulo": nome_do_regime(regime_id, ",", **campos),
        })

    tabela = pd.DataFrame(linhas)
    tabela.attrs["nota"] = bm.NOTA_VOLUME_ABSOLUTO
    tabela.attrs["leitura"] = bm.leitura_operacional(
        resultados, nome_regra=nome_regra, equiparados=[equiparado, regra]
    )
    tabela.attrs["criterio"] = veredito_do_criterio(tabela, nome_regra=nome_regra)
    return tabela


def equiparado_por_modelo(resultados: list[dict],
                          particao: Particao,
                          nome_regra: str = "Regra ADA") -> pd.DataFrame:
    """O regime de recall equiparado de **cada** célula calibrável do grid.

    `carga_operacional` equipara só o modelo de referência, e o formato dela está
    certo: três regimes, uma leitura, e é essa a tabela que a narrativa usa. O
    problema é o que fica de fora. O contraste que sustenta o achado da floresta tem
    dois lados, ela gasta mais encaminhamento por caso no regime equiparado do que a
    logística, e publicar a comparação emitindo só um lado repete, em escala menor, a
    lacuna que D-027 fechou (D-047).

    O alvo é `benchmark.recall_da_regra_no_treino`, medido **no treino**, o mesmo de
    `carga_operacional`. Calibrar pelo recall que a regra obtém no teste vazaria o
    teste para dentro da decisão de limiar.

    Custa uma avaliação por célula calibrável, cada uma com validação cruzada e ajuste
    final. A célula do modelo de referência é reavaliada aqui em vez de aproveitada da
    carga operacional: sai mais cara e paga, porque as duas tabelas passam a conferir
    uma à outra em vez de repetirem o mesmo cálculo.

    Returns:
        DataFrame com uma linha por célula calibrável mais a da regra, trazendo
        `recall_alvo`, `recall_obtido`, `recall_regra`, `encaminhados`,
        `casos_detectados`, `nns`, `cumpre_criterio` e `taxa_encaminhamento`. A linha
        da regra vai primeiro, e tem `recall_alvo` e `cumpre_criterio` vazios: ela
        opera em limiar fixo, não mira alvo nenhum e não se compara consigo mesma.
    """
    regra = next((r for r in resultados if r["modelo"] == nome_regra), None)
    if regra is None:
        raise ValueError(f"Comparador {nome_regra!r} ausente dos resultados.")

    alvo = bm.recall_da_regra_no_treino(particao.X_tr, particao.y_tr)
    disponiveis = md.estimadores()

    def linha(r: dict, recall_alvo: float) -> dict:
        # O critério é o mesmo de `veredito_do_criterio`, célula a célula. Na linha da
        # regra fica vazio.
        cumpre = (pd.NA if r is regra else
                  bm.cumpre_o_criterio(r["recall"] - regra["recall"],
                                       r["nns"] - regra["nns"]))
        return {
            "estrategia": _rotulo(r),
            "modelo": r["modelo"],
            "conjunto": r["conjunto"],
            # `recall_alvo` é onde a equiparação acontece, no treino;
            # `recall_obtido` é o que se realizou no teste. Duas colunas, senão a
            # tabela parece comparar recalls iguais quando o que está equiparado é o
            # alvo, não o resultado.
            "recall_alvo": recall_alvo,
            "recall_obtido": r["recall"],
            "recall_regra": regra["recall"],
            "encaminhados": r["vp"] + r["fp"],
            "casos_detectados": r["vp"],
            "nns": r["nns"],
            "cumpre_criterio": cumpre,
            "taxa_encaminhamento": encaminhamento(r)[1],
        }

    linhas = [linha(regra, float("nan"))]
    for r in resultados:
        if not r.get("limiar_calibrado", True):
            continue
        equiparado = ev.avaliar(
            md.montar(disponiveis[r["modelo"]], r["conjunto"]),
            particao.X_tr, particao.y_tr, particao.X_te, particao.y_te,
            nome=r["modelo"],
            conjunto=r["conjunto"],
            recall_alvo=alvo,
            salvar_figuras=False,
        )
        linhas.append(linha(equiparado, alvo))

    tabela = pd.DataFrame(linhas)
    tabela["cumpre_criterio"] = tabela["cumpre_criterio"].astype("boolean")
    tabela.attrs["nota"] = bm.NOTA_EQUIPARACAO
    return tabela


def _leitura_do_par(delta_encaminhados: int, delta_detectados: int) -> str:
    """Classifica o par pelo sinal dos dois deltas, floresta menos logística.

    Quem detecta ao menos o mesmo encaminhando no máximo o mesmo domina. Quando os
    dois deltas andam juntos, para cima ou para baixo, é troca: mais casos custam mais
    encaminhamento, e o preço é o custo marginal.
    """
    if delta_encaminhados == 0 and delta_detectados == 0:
        return "empate"
    if delta_detectados >= 0 and delta_encaminhados <= 0:
        return "floresta domina"
    if delta_detectados <= 0 and delta_encaminhados >= 0:
        return "logistica domina"
    return "troca"


def custo_marginal_pareado(resultados: list[dict],
                           equiparado: pd.DataFrame,
                           par: tuple[str, str] = PAR_CUSTO) -> pd.DataFrame:
    """Quanto encaminhamento a floresta gasta a mais por caso a mais, contra a logística.

    Uma linha por conjunto e regime. O ponto padrão sai do grid, o regime equiparado
    sai de `equiparado_por_modelo`, e nenhum modelo é reavaliado aqui: são os mesmos
    números que a carga operacional e o equiparado já publicam, subtraídos. É o que dá
    artefato à frase de 48 encaminhamentos por um caso (D-046, D-055).

    `custo_marginal` só existe quando a floresta detecta mais **e** encaminha mais.
    Sem caso a mais não há preço por caso a mais, e com caso a mais encaminhando menos
    a floresta domina: a razão sairia negativa e leria como desconto, que não é.

    Returns:
        DataFrame com `conjunto`, `regime`, encaminhados e detectados de cada modelo,
        `delta_encaminhados`, `delta_detectados`, `custo_marginal` e `leitura`.
    """
    floresta, logistica = par
    padrao = {(r["modelo"], r["conjunto"]): (r["vp"] + r["fp"], r["vp"])
              for r in resultados if r.get("limiar_calibrado", True)}
    equiparados = {(l["modelo"], l["conjunto"]): (int(l["encaminhados"]),
                                                   int(l["casos_detectados"]))
                   for _, l in equiparado.iterrows()}

    linhas = []
    for regime, fonte in (("padrao", padrao), ("equiparado", equiparados)):
        for conjunto in CONJUNTOS_FLORESTA:
            if (floresta, conjunto) not in fonte or (logistica, conjunto) not in fonte:
                continue
            enc_f, det_f = fonte[(floresta, conjunto)]
            enc_l, det_l = fonte[(logistica, conjunto)]
            delta_enc, delta_det = int(enc_f - enc_l), int(det_f - det_l)
            linhas.append({
                "conjunto": conjunto,
                "regime": regime,
                "encaminhados_floresta": int(enc_f),
                "detectados_floresta": int(det_f),
                "encaminhados_logistica": int(enc_l),
                "detectados_logistica": int(det_l),
                "delta_encaminhados": delta_enc,
                "delta_detectados": delta_det,
                "custo_marginal": (delta_enc / delta_det
                                   if delta_det > 0 and delta_enc > 0
                                   else float("nan")),
                "leitura": _leitura_do_par(delta_enc, delta_det),
            })

    tabela = pd.DataFrame(linhas)
    tabela.attrs["nota"] = (
        f"Deltas de {floresta} menos {logistica}, no mesmo conjunto e no mesmo regime. "
        f"O ponto padrão é o do grid publicado; o equiparado é o de "
        f"{NOME_EQUIPARADO}. Custo marginal é encaminhados a mais por caso a mais, "
        f"vazio quando a floresta não detecta mais ou quando detecta mais "
        f"encaminhando menos."
    )
    return tabela


def curvas(resultados: list[dict], particao: Particao) -> dict[str, pd.DataFrame]:
    """Pontos das curvas ROC, PR e de calibração por decis de cada célula calibrável.

    Os comparadores de corte único ficam fora: com saída em dois valores, a curva é uma
    quebra e não descreve desempenho (D-023). As funções são as mesmas que desenham as
    figuras, então PNG e CSV não divergem.

    Returns:
        Dicionário com `curva_roc`, `curva_pr` e `calibracao_decis`, cada um com
        `modelo` e `conjunto` à frente das colunas da curva.
    """
    construtores = {
        NOME_CURVA_ROC: ev.curva_roc,
        NOME_CURVA_PR: ev.curva_pr,
        NOME_DECIS: ev.calibracao_decis,
    }
    saida = {}
    for nome, construir in construtores.items():
        partes = []
        for r in resultados:
            if not r.get("limiar_calibrado", True):
                continue
            curva = construir(particao.y_te, r["proba_teste"])
            curva.insert(0, "conjunto", r["conjunto"])
            curva.insert(0, "modelo", r["modelo"])
            partes.append(curva)
        saida[nome] = pd.concat(partes, ignore_index=True)
    return saida


def veredito_do_criterio(carga: pd.DataFrame,
                         nome_regra: str = "Regra ADA") -> pd.DataFrame:
    """Aplica o critério de sucesso a cada regime da carga operacional.

    O critério do projeto é superar a regra clínica em recall **com NNS menor**, e as
    duas condições valem juntas. `benchmark.supera_a_regra_clinica` já o aplica ao
    grid; falta aplicá-lo aos regimes.

    **O resultado sai mecânico, e só um dos regimes decide o critério.** No ponto
    padrão o modelo opera em recall abaixo do da regra, então a condição de recall
    falha e o NNS ao lado dela foi medido em outro ponto de operação: a comparação
    registra, não decide. Quem decide é o regime de recall equiparado, e lá a célula
    de referência não cumpre. `leitura_do_criterio` carrega a qualificação junto com
    os números (D-024, D-046).

    Escrever esse veredito à mão no documento é o defeito que D-045 registra. Aqui ele
    sai da comparação, então quem trocar o modelo de referência vê o texto acompanhar.

    Returns:
        DataFrame com `regime`, `delta_recall`, `delta_nns` e `cumpre`, uma linha por
        regime que não é o da regra.
    """
    da_regra = carga[carga["regime"] == nome_regra]
    if da_regra.empty:
        raise ValueError(f"Regime {nome_regra!r} ausente da carga operacional.")
    referencia = da_regra.iloc[0]

    linhas = []
    for _, r in carga[carga["regime"] != nome_regra].iterrows():
        delta_recall = r["recall"] - referencia["recall"]
        delta_nns = r["nns"] - referencia["nns"]
        linhas.append({
            "regime": r["regime"],
            "recall": r["recall"],
            "recall_regra": referencia["recall"],
            "delta_recall": delta_recall,
            "nns": r["nns"],
            "nns_regra": referencia["nns"],
            "delta_nns": delta_nns,
            "cumpre": bm.cumpre_o_criterio(delta_recall, delta_nns),
        })
    return pd.DataFrame(linhas)


def _celula_avaliada(veredito: pd.DataFrame) -> str | None:
    """A estratégia que os regimes do veredito avaliam, quando é uma só.

    Os rótulos saem de `carga_operacional` como "<estratégia>, <regime>", e as duas
    linhas não invariantes da tabela são sempre da mesma célula, porque a carga
    equipara o modelo de referência e mais ninguém. Devolve `None` no dia em que isso
    deixar de valer, e aí o fecho não nomeia ninguém em vez de nomear errado.
    """
    nomes = {str(regime).split(",")[0].strip() for regime in veredito["regime"]}
    return nomes.pop() if len(nomes) == 1 else None


def _nota_de_recall_menor(veredito: pd.DataFrame) -> str:
    """Qualifica os regimes que operam em recall abaixo do da regra.

    O critério pede recall maior **e** NNS menor ao mesmo tempo. Onde o recall fica
    abaixo, o NNS ao lado dele foi medido em outro ponto de operação, e comparar NNS
    entre recalls diferentes não ordena ninguém: encaminhar menos gente para achar
    menos casos também baixa o NNS. O número continua saindo, com a ressalva colada
    nele, que é o mesmo tratamento que `evaluate.nota_corte_unico` dá às métricas de
    curva dos comparadores de corte único (D-024).

    Devolve string vazia quando todos os regimes operam em recall igual ou maior.
    """
    fora = [str(r["regime"]) for _, r in veredito.iterrows() if r["delta_recall"] <= 0]
    if not fora:
        return ""
    verbo = "operam" if len(fora) > 1 else "opera"
    return (
        f"Nota: {', '.join(fora)} {verbo} em recall abaixo do da regra. O critério "
        f"pede recall maior com NNS menor ao mesmo tempo, e NNS medido em recalls "
        f"diferentes não decide nada. Aí a comparação registra o resultado, não o "
        f"critério. Quem decide é o regime de recall equiparado, onde os dois lados "
        f"miram o mesmo recall."
    )


def leitura_do_criterio(carga: pd.DataFrame,
                        nome_regra: str = "Regra ADA") -> str:
    """Diz em voz alta, regime a regime, se o critério de sucesso foi cumprido.

    Um resultado negativo é legítimo e tem que ser reportado como tal. O texto sai do
    sinal dos deltas, nunca de afirmação fixa.

    **O fecho nomeia a célula, não o algoritmo.** A carga operacional avalia os dois
    regimes do modelo de referência e mais ninguém, então dizer "o critério não é
    cumprido" sem sujeito faz o leitor estender o veredito a todas as células, e o
    artefato diz o contrário: das cinco calibráveis, quatro cumprem no regime
    equiparado. Quem responde por todas é `equiparado_por_modelo` (D-046, D-047).

    **E nem todo regime decide o critério.** No ponto padrão o modelo opera em
    recall abaixo do da regra, e `_nota_de_recall_menor` cola a ressalva no
    resultado em vez de suprimi-lo. Regime que roda em outro ponto de operação
    entra como registro; o critério se decide no regime de recall equiparado
    (D-024).
    """
    veredito = carga.attrs.get("criterio")
    if veredito is None:
        veredito = veredito_do_criterio(carga, nome_regra=nome_regra)

    partes = []
    for _, r in veredito.iterrows():
        situacao = "cumpre" if r["cumpre"] else "NÃO cumpre"
        partes.append(
            f"{r['regime']}: {situacao} o critério, recall {r['recall']:.3f} contra "
            f"{r['recall_regra']:.3f} da regra e NNS {r['nns']:.2f} contra "
            f"{r['nns_regra']:.2f} (Δ recall {r['delta_recall']:+.3f}, "
            f"Δ NNS {r['delta_nns']:+.2f})."
        )

    quantos = int(veredito["cumpre"].sum())
    sujeito = _celula_avaliada(veredito) or "A estratégia avaliada"
    if quantos == len(veredito):
        fecho = f"{sujeito} cumpre o critério em todos os regimes avaliados aqui."
    elif quantos == 0:
        fecho = f"{sujeito} não cumpre o critério em nenhum regime avaliado aqui."
    else:
        fecho = (f"{sujeito} cumpre o critério em um regime e não no outro, então a "
                 f"escolha do ponto de operação decide se ela o cumpre.")
    fecho += (" O veredito vale para esta célula: as demais estão em "
              "equiparado_por_modelo.")
    nota = _nota_de_recall_menor(veredito)
    return "\n".join(partes + [fecho] + ([nota] if nota else []))


def exportar(tabela: pd.DataFrame, nome: str, nota: str | None = None,
             destino: Path | None = None, **kwargs) -> dict[str, Path]:
    """Grava a tabela como `.tex` e `.csv` em `reports/tabelas/`.

    O `.csv` não é cópia redundante do `.tex`: é sobre ele que a conferência de
    reprodução trabalha, sem ter de reparsear LaTeX. Ele sai arredondado, senão o
    diretório versionado acusa mudança a cada execução (D-048).

    Args:
        destino: diretório de gravação. Omitido, `reports/tabelas/`.
        **kwargs: repassados a `to_latex`, só para o `.tex`. O `.csv` não os vê,
            porque formatação de apresentação não pode chegar ao arquivo sobre o
            qual a conferência de reprodução trabalha.
    """
    destino = cfg.PATH_TABELAS if destino is None else Path(destino)
    destino_csv = art.salvar_csv(tabela, destino / f"{nome}.csv")
    if nota is None:
        nota = tabela.attrs.get("nota")
    destino_tex = ev.exportar_latex(tabela, f"{nome}.tex", path_reports=destino,
                                    nota=nota, **kwargs)
    return {"tex": destino_tex, "csv": destino_csv}


def exportar_curva(tabela: pd.DataFrame, nome: str,
                   destino: Path | None = None) -> dict[str, Path]:
    """Grava uma curva só como `.csv`.

    São milhares de pontos, que servem ao gráfico e não cabem em documento. O `.tex`
    de uma curva seria o maior arquivo do diretório e ninguém o leria.
    """
    destino = cfg.PATH_TABELAS if destino is None else Path(destino)
    return {"csv": art.salvar_csv(tabela, destino / f"{nome}.csv")}


def exportar_descritivas(particao: Particao,
                         analitico: pd.DataFrame | None = None,
                         destino: Path | None = None) -> dict[str, dict]:
    """Emite o quadro da amostra e as contagens de ausentes.

    Os dois números saíam de `data` direto para o texto, sem passar por arquivo
    nenhum. É a mesma lacuna que a tabela consolidada tinha antes de existir ponto de
    reprodução: quem transcreve não tem contra o que conferir, e o valor no documento
    envelhece sem que nada acuse.

    O quadro da amostra traz as duas prevalências, a amostral e a ponderada por
    `peso_lab`. A ponderada é estimativa de parâmetro populacional e vive só na
    descrição da amostra; nenhum estimador a enxerga (D-009, D-031). O quadro traz
    também o tamanho e a prevalência de cada lado da partição, a mesma que o grid usa.
    As contagens de ausentes são as que o `KNNImputer` vai tratar, e a coluna do
    treino é a que vale, porque é onde a imputação ajusta (D-011).

    Args:
        analitico: base limpa. Omitida, relê o CSV bruto. Passe a sua se já tiver uma
            em mãos, para não pagar a leitura duas vezes.
        destino: diretório de gravação. Omitido, `reports/tabelas/`.
    """
    analitico = dt.carregar_analitico() if analitico is None else analitico
    return {
        NOME_RESUMO: exportar(
            dt.resumo_amostra(analitico, y_tr=particao.y_tr, y_te=particao.y_te),
            NOME_RESUMO, nota="", destino=destino,
        ),
        NOME_AUSENTES: exportar(
            dt.ausentes_por_variavel(analitico, treino=particao.X_tr),
            NOME_AUSENTES, nota="", destino=destino,
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
                        casas: int = 3,
                        exigir_cobertura: bool = True) -> pd.DataFrame:
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
    # Linha da âncora que sumiu do emitido. Sem esta varredura, a conferência compara só
    # a interseção e uma tabela truncada passa: foi assim que a consolidada sobreviveu
    # com duas linhas e a suíte verde (D-044). Quem roda um recorte de propósito desliga
    # com `exigir_cobertura=False`, e aí a conferência volta a olhar só a interseção.
    if exigir_cobertura:
        for linha in esperado.index.difference(obtido.index):
            divergencias.append({
                "linha": " · ".join(linha), "coluna": "(linha inteira)",
                "referencia": "presente na âncora", "obtido": "ausente do emitido",
            })
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


def mcnemar_do_par(resultados: list[dict], particao: Particao,
                   par: tuple = PAR_MCNEMAR) -> dict | None:
    """Roda o McNemar entre as duas células do par, se as duas estiverem no grid.

    Retorna `None` quando o grid não traz as duas, que é o caso de qualquer execução
    com recorte. Exigir o par faria um teste de recorte falhar por ausência de célula
    que ele nunca pediu.
    """
    por_celula = {(r["modelo"], r["conjunto"]): r for r in resultados}
    if any(celula not in por_celula for celula in par):
        return None
    return ev.mcnemar_pareado(particao.y_te,
                              por_celula[par[0]], por_celula[par[1]])


def congelar_ancora(consolidado: pd.DataFrame,
                    modelo: str = NOME_FLORESTA,
                    nome: str = REFERENCIA_FLORESTA) -> Path:
    """Grava as linhas de um modelo como âncora de reprodução.

    Só depois de uma execução sair limpa: âncora congelada sobre resultado não conferido
    transforma um defeito em referência, e a partir daí a conferência passa a atestar o
    defeito.

    Raises:
        ValueError: se o modelo não aparece na tabela.
    """
    recorte = consolidado[consolidado["modelo"] == modelo]
    if recorte.empty:
        raise ValueError(f"{modelo!r} ausente da tabela; nada a congelar.")
    return ev.exportar_latex(recorte, nome, nota="")


def composicao_esperada(modelos: list[str] | None = None,
                        conjuntos: list[str] | None = None) -> list[tuple[str, str]]:
    """As células que a tabela publicada tem que trazer, na ordem em que saem."""
    modelos = list(MODELOS_PADRAO) if modelos is None else list(modelos)
    conjuntos = list(ft.FEATURE_SETS) if conjuntos is None else list(conjuntos)
    disponiveis = md.estimadores()

    celulas = []
    for nome in sorted(modelos, key=lambda n: ORDEM_MODELOS.index(n)
                       if n in ORDEM_MODELOS else len(ORDEM_MODELOS)):
        if disponiveis[nome].invariante_ao_conjunto:
            celulas.append((nome, SEM_CONJUNTO))
        elif nome == NOME_FLORESTA:
            celulas += [(nome, c) for c in conjuntos if c in CONJUNTOS_FLORESTA]
        else:
            celulas += [(nome, c) for c in conjuntos]
    return celulas


def conferir_composicao(tabela: pd.DataFrame,
                        esperada: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    """Confere quais células a tabela traz, e não só os valores das que ela traz.

    A conferência de reprodução compara a interseção dos índices, então ela flagra valor
    errado numa linha existente e deixa passar linha que sumiu. Uma tabela truncada por
    execução com recorte de grid tem números todos certos e conteúdo errado, e é
    exatamente esse o defeito que passou despercebido até agora.

    Returns:
        DataFrame com `linha` e `situacao`, uma entrada por célula ausente ou inesperada.
        Vazio significa composição exata.
    """
    esperada = composicao_esperada() if esperada is None else list(esperada)
    obtida = list(zip(tabela["modelo"], tabela["conjunto"]))

    faltando = [c for c in esperada if c not in obtida]
    sobrando = [c for c in obtida if c not in esperada]

    linhas = ([{"linha": " · ".join(c), "situacao": "ausente do emitido"}
               for c in faltando]
              + [{"linha": " · ".join(c), "situacao": "inesperada no emitido"}
                 for c in sobrando])
    return pd.DataFrame(linhas, columns=["linha", "situacao"])


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------

def executar(modelos: list[str] | None = None,
             conjuntos: list[str] | None = None,
             salvar_figuras: bool = False,
             conferir: bool = True,
             particao: Particao | None = None,
             descritivas: bool = True,
             saida: Path | None = None) -> dict:
    """Roda o grid, emite os artefatos e confere a reprodução.

    Args:
        saida: diretório onde gravar tabelas e figuras. Omitido, `reports/tabelas/` e
            `reports/figuras/`. A conferência lê as âncoras versionadas de
            `reports/tabelas/` em qualquer caso: é assim que o painel reproduz ao vivo
            sem regravar o diretório que ele mesmo lê (D-054, D-055).
        particao: split pronto. Omitido, `preparar_particao` constrói um. O parâmetro
            existe para quem já tem a partição em mãos: o notebook de resultados
            diagnostica colinearidade sobre `X_tr` antes de chamar aqui, e reler o CSV
            só para refazer o mesmo split seria gasto sem ganho. A partição volta no
            retorno para o chamador não ter de reconstruí-la depois.
        descritivas: emite também o quadro da amostra e as contagens de ausentes.
            Desligue em teste que só precisa do grid.

    Returns:
        Dicionário com `resultados`, `consolidado`, `carga`, `equiparado`,
        `artefatos`, `divergencias` e `particao`.
    """
    particao = preparar_particao() if particao is None else particao
    destino = None if saida is None else Path(saida)
    figuras = None if destino is None else destino / "figuras"
    resultados = compor_grid(modelos, conjuntos, particao=particao,
                             salvar_figuras=salvar_figuras, path_figuras=figuras)

    consolidado = consolidar(resultados)
    carga = carga_operacional(resultados, particao)
    equiparado = equiparado_por_modelo(resultados, particao)

    # Slope e intercepto de calibração saem de `diagnostico_calibracao` e não têm
    # coluna em `comparar`, então sem esta linha eles não existem em arquivo nenhum.
    # São a evidência que sustenta D-022 e D-040.
    calibracao = ev.tabela_calibracao_comparada(resultados)

    artefatos = {
        # `na_rep`: `recall_alvo` fica vazio nos comparadores de limiar fixo.
        NOME_CONSOLIDADO: exportar(consolidado, NOME_CONSOLIDADO, destino=destino,
                                   na_rep=SEM_CONJUNTO),
        NOME_CARGA: exportar(carga, NOME_CARGA, destino=destino),
        # `na_rep`: a linha da regra não tem alvo, e um "NaN" impresso numa tabela
        # que vai para o LaTeX lê como falha de geração, não como célula vazia.
        NOME_EQUIPARADO: exportar(equiparado, NOME_EQUIPARADO, destino=destino,
                                  na_rep=SEM_CONJUNTO),
        NOME_CALIBRACAO: exportar(calibracao, NOME_CALIBRACAO, destino=destino,
                                  nota=ev.nota_corte_unico(consolidado)),
    }
    if descritivas:
        artefatos.update(exportar_descritivas(particao, destino=destino))

    mcnemar = mcnemar_do_par(resultados, particao)
    if mcnemar is not None:
        artefatos[NOME_MCNEMAR] = exportar(
            ev.tabela_mcnemar(mcnemar), NOME_MCNEMAR, destino=destino,
            nota=(f"Teste de McNemar pareado entre {mcnemar['modelo_a']} e "
                  f"{mcnemar['modelo_b']}, nas predições do teste, cada modelo no seu "
                  f"limiar calibrado. O p-valor vem do teste binomial exato sobre os "
                  f"{mcnemar['discordantes']} casos discordantes."),
        )

    custo = custo_marginal_pareado(resultados, equiparado)
    if not custo.empty:
        artefatos[NOME_CUSTO_MARGINAL] = exportar(custo, NOME_CUSTO_MARGINAL,
                                                  destino=destino, na_rep="")
    if any(r.get("limiar_calibrado", True) for r in resultados):
        for nome, curva in curvas(resultados, particao).items():
            artefatos[nome] = exportar_curva(curva, nome, destino=destino)

    divergencias = pd.DataFrame()
    composicao = pd.DataFrame()
    if conferir:
        # Uma conferência por âncora, concatenadas. A da logística prova que a floresta
        # entrou no grid sem mover número dela; a da floresta cobre as linhas novas.
        # Âncora ainda não congelada é pulada, senão a primeira execução nunca sairia.
        partes = [conferir_reproducao(consolidado, cfg.PATH_TABELAS / nome)
                  for nome in REFERENCIAS
                  if (cfg.PATH_TABELAS / nome).exists()]
        divergencias = (pd.concat(partes, ignore_index=True) if partes
                        else pd.DataFrame(columns=["linha", "coluna",
                                                   "referencia", "obtido"]))
        composicao = conferir_composicao(
            consolidado, composicao_esperada(modelos, conjuntos)
        )

    return {
        "resultados": resultados,
        "consolidado": consolidado,
        "carga": carga,
        "equiparado": equiparado,
        "artefatos": artefatos,
        "divergencias": divergencias,
        "composicao": composicao,
        "mcnemar": mcnemar,
        "particao": particao,
    }


def main(argv: list[str] | None = None) -> int:
    """Entrada de linha de comando. É o que o `make resultados` chama de fato."""
    # O console do Windows abre em cp1252, e a reprodução imprime Δ, ·, ± e acentuação
    # em vários pontos. Sem isto ela sai com status 1 por codificação, que se lê como
    # divergência de número e não é. Trocar só o caractere que quebrou hoje resolveria
    # um caso e deixaria o próximo; o que precisa mudar é o canal de saída (D-052).
    #
    # Fica aqui e não no import: quem chama `executar` de dentro de teste ou notebook
    # não deve ter o stdout do processo trocado por baixo por causa de um import.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

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
                        help="não compara com as âncoras nem confere a composição")
    parser.add_argument("--congelar-ancora", action="store_true",
                        help=f"grava {REFERENCIA_FLORESTA} a partir desta execução")
    parser.add_argument("--saida", type=Path, default=None, metavar="DIR",
                        help="grava tabelas e figuras em DIR; a conferência continua "
                             "lendo as âncoras de reports/tabelas/")
    args = parser.parse_args(argv)

    if args.saida is not None and args.congelar_ancora:
        # A âncora vive em `reports/tabelas/`, e congelá-la a partir de uma execução
        # gravada em outro lugar misturaria as duas origens.
        parser.error("--congelar-ancora não combina com --saida.")

    saida = executar(modelos=args.modelos, conjuntos=args.conjuntos,
                     salvar_figuras=args.figuras,
                     conferir=not args.sem_conferencia,
                     saida=args.saida)

    print(ev.comparar_formatado(saida["resultados"]).to_string(index=False))
    print()
    print(saida["carga"].to_string(index=False))
    print()
    print(saida["carga"].attrs["leitura"])
    print()
    print(leitura_do_criterio(saida["carga"]))
    print()
    print(saida["equiparado"].to_string(index=False))
    print()
    if saida["mcnemar"] is not None:
        print(saida["mcnemar"]["tabela"].to_string())
        print(ev.leitura_do_mcnemar(saida["mcnemar"]))
        print()
    print(f"Artefatos em {args.saida or cfg.PATH_TABELAS}:")
    for nome, caminhos in saida["artefatos"].items():
        print(f"{nome}: {', '.join(p.name for p in caminhos.values())}")

    if args.sem_conferencia:
        return 0

    divergencias, composicao = saida["divergencias"], saida["composicao"]
    ancoras = [nome for nome in REFERENCIAS if (cfg.PATH_TABELAS / nome).exists()]
    status = 0

    if composicao.empty:
        print(f"\nComposição conferida: {len(saida['consolidado'])} linhas, "
              f"as esperadas.")
    else:
        print("\nCOMPOSIÇÃO ERRADA:")
        print(composicao.to_string(index=False))
        status = 1

    if divergencias.empty:
        print(f"Reprodução conferida contra {', '.join(ancoras)}: sem divergência "
              f"na terceira casa.")
    else:
        print(f"DIVERGÊNCIA contra {', '.join(ancoras)}:")
        print(divergencias.to_string(index=False))
        status = 1

    faltam = [nome for nome in REFERENCIAS if nome not in ancoras]
    if faltam:
        print(f"Âncora ainda não congelada, fora da conferência: {', '.join(faltam)}")

    if args.congelar_ancora:
        if status != 0:
            print("\nExecução com divergência: âncora não congelada.")
            return status
        destino = congelar_ancora(saida["consolidado"])
        print(f"\nÂncora congelada: {destino.name}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
