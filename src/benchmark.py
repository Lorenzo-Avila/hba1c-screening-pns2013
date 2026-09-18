"""Comparadores não-ML.

Dois pisos, avaliados pelo mesmo `evaluate.avaliar` e sobre as mesmas partições:

- `RegraClinicaADA`: `idade >= 45 OU IMC >= 30`, que é o que a Atenção Primária faz
  hoje, sem machine learning. Comparador honesto, com as mesmas variáveis não
  invasivas e nenhuma estimação.
- `DummyClassifier(strategy="most_frequent")`: piso absoluto. Com 7% de positivos,
  entrega uns 0,93 de acurácia e recall zero. Está aqui para expor a armadilha da
  acurácia, não para competir.

Critério de sucesso: o modelo precisa superar a regra clínica em Recall com NNS
menor. Resultado negativo é legítimo e deve ser reportado.

Sobre FINDRISC e BrDMrisc: não dá para reproduzi-los nesta base. Exigem histórico
familiar e variáveis comportamentais, excluídas por decisão metodológica. O
BrDMrisc, ainda por cima, é escore de incidência derivado de coorte longitudinal,
incompatível com o desenho transversal da PNS 2013. Documentar a impossibilidade é
resultado, não limitação varrida para baixo do tapete.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier

from . import config as cfg
from . import evaluate as ev

# Limiar dos comparadores. Ambos devolvem saídas em {0, 1}: qualquer valor em
# (0, 1] separa as duas classes do mesmo jeito, e 0,5 é o que deixa a intenção
# legível. Não é ponto de operação calibrado; ver `avaliar(limiar=...)`.
LIMIAR_INDICADORA = 0.5

# Cortes da regra de rastreamento da American Diabetes Association.
CORTE_IDADE_ADA = 45

# Acima desta fração da população adulta, a estratégia deixa de selecionar quem
# examinar e passa a empurrar a fila inteira para o laboratório. É o corte que a
# leitura operacional usa para julgar a carga, e ele fica explícito aqui para o
# veredito sair de uma comparação e não de uma frase escrita à mão.
CORTE_TRIAGEM_AMPLA = 0.5

NOTA_VOLUME_ABSOLUTO = (
    "Volume de encaminhamentos só é comparável entre estratégias que operam no "
    "mesmo recall. Quando os recalls realizados diferem, a comparação é por NNS, "
    "que normaliza por casos detectados — no regime de recall equiparado um modelo "
    "pode encaminhar mais pessoas que a regra em números absolutos e ainda assim "
    "ter NNS menor, por detectar mais casos."
)

NOTA_EQUIPARACAO = (
    "A equiparação de recall ocorre no TREINO, por construção: o limiar do modelo "
    "é calibrado para o recall que a regra clínica obtém no treino "
    "(`recall_alvo`). O valor realizado no teste (`recall_obtido`) desvia desse "
    "alvo, e o desvio é esperado — o limiar é fixado fora da amostra de avaliação, "
    "sob pena de vazamento. As colunas de recall NÃO são iguais entre modelo e "
    "regra no teste, e não deveriam ser: o que está equiparado é o regime de "
    "operação, não o resultado. O NNS permanece a métrica de comparação."
)


class RegraClinicaADA(BaseEstimator, ClassifierMixin):
    """Regra de rastreamento da ADA: `idade >= 45` **ou** `IMC >= 30`.

    Determinística e sem parâmetro estimado. `fit` não aprende nada: registra só
    `classes_` e o número de colunas, que a API do scikit-learn exige para o
    estimador atravessar `cross_val_predict` e `clone` como qualquer outro. Ajustar
    em 6.760 linhas ou em 10 dá exatamente o mesmo classificador.

    **Escala clínica original.** Lê `idade` e `imc` por nome, em anos e kg/m²,
    direto das preditoras que `data.separar` devolve, antes de qualquer
    padronização. Com IMC padronizado a comparação seria de z-score com 30, e a
    regra deixaria de ser a regra. Por isso este estimador **não** vai embrulhado no
    preprocessador: ele é o comparador de quem não tem pipeline nenhum.

    Args:
        corte_idade: idade mínima, em anos. Padrão 45 (ADA).
        corte_imc: IMC mínimo, em kg/m². Padrão 30, o corte de obesidade da SBD 2019
            que `features.ConstrutorCompostas` já usa.
    """

    def __init__(self, corte_idade: int = CORTE_IDADE_ADA,
                 corte_imc: float = cfg.CORTE_IMC_OBESIDADE):
        self.corte_idade = corte_idade
        self.corte_imc = corte_imc

    def fit(self, X, y=None):
        """Não estima nada. Registra apenas o que a API do scikit-learn exige."""
        X = self._validar(X)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.classes_ = np.array([0, 1])
        return self

    def _validar(self, X) -> pd.DataFrame:
        X = pd.DataFrame(X)
        faltantes = [c for c in ("idade", "imc") if c not in X.columns]
        if faltantes:
            raise KeyError(
                f"RegraClinicaADA opera sobre idade e imc em escala clínica; "
                f"colunas ausentes: {faltantes}. Não passe a matriz de design "
                f"padronizada — passe as preditoras de `data.separar`."
            )
        return X

    def predict(self, X) -> np.ndarray:
        """Indicadora da regra: 1 se `idade >= 45` ou `imc >= 30`, senão 0."""
        X = self._validar(X)
        idade = pd.to_numeric(X["idade"], errors="coerce")
        imc = pd.to_numeric(X["imc"], errors="coerce")
        return ((idade >= self.corte_idade) | (imc >= self.corte_imc)).astype(int).to_numpy()

    def predict_proba(self, X) -> np.ndarray:
        """Devolve a indicadora como 0/1 nas duas colunas, para o harness.

        **Isto não são probabilidades.** A regra não estima risco, ela responde sim
        ou não. Os valores são 0,0 e 1,0 exatos e a coluna positiva é a própria
        indicadora; a forma existe só para `evaluate.avaliar` poder tratar a regra
        como trata qualquer classificador.

        Consequências que precisam estar visíveis na leitura dos resultados:

        - **Não calibre limiar sobre esta saída.** Curva precision-recall de dois
          pontos não define ponto de operação, e a regra já é o ponto de operação.
          Passe `limiar=LIMIAR_INDICADORA` a `avaliar`.
        - **AUC-ROC e PR-AUC ficam degradadas por construção**, não por desempenho:
          com um único ponto de corte, o que se tem é uma quebra, não uma curva.
          Comparar a AUC da regra com a do modelo é comparar objetos diferentes. Aqui
          valem Recall, Especificidade e NNS.
        - **Brier e curva de calibração não fazem sentido** sobre 0/1 categórico.
        """
        indicadora = self.predict(X).astype("float64")
        return np.column_stack([1.0 - indicadora, indicadora])


def construir_comparadores() -> dict:
    """Os comparadores não-ML, prontos para `evaluate.avaliar`.

    Returns:
        Dicionário `nome -> estimador`. Os dois operam sobre as preditoras cruas,
        sem preprocessador: a regra por definição, o Dummy por ignorar `X`.
    """
    return {
        "Regra ADA": RegraClinicaADA(),
        "Dummy (maioria)": DummyClassifier(strategy="most_frequent",
                                           random_state=cfg.RANDOM_STATE),
    }


def avaliar_comparadores(X_tr, y_tr, X_te, y_te,
                         salvar_figuras: bool = True, **kwargs) -> list[dict]:
    """Avalia os comparadores pelo mesmo harness e nas mesmas partições.

    Os dois recebem `limiar=LIMIAR_INDICADORA`: as saídas são indicadoras, e deixar
    a calibração rodar sobre elas daria um ponto de operação sem significado. No
    Dummy seria pior que isso: a curva calibraria para o limiar 0,0, virando o
    "prevê sempre a maioria" em "prevê sempre positivo" e destruindo justamente o
    piso que ele existe para marcar.
    """
    return [
        ev.avaliar(estimador, X_tr, y_tr, X_te, y_te, nome=nome,
                   conjunto="—", limiar=LIMIAR_INDICADORA,
                   salvar_figuras=salvar_figuras, **kwargs)
        for nome, estimador in construir_comparadores().items()
    ]


def tabela_nns(resultados: list[dict], n_teste: int | None = None) -> pd.DataFrame:
    """Traduz cada estratégia em carga operacional para a Atenção Primária.

    O NNS responde a pergunta que decide se o modelo vale a implantação: quantas
    pessoas a UBS encaminha a exame laboratorial para achar um caso. As outras
    colunas dão a mesma informação em números absolutos, que é como o serviço
    enxerga.

    `taxa_encaminhamento` é a coluna que sustenta o argumento de aplicabilidade e
    costuma passar batida ao lado do NNS. Triagem que manda mais da metade da
    população adulta ao laboratório não tria, só desloca a fila, e é a taxa que diz
    de qual lado do corte cada estratégia cai. Os valores de cada execução saem em
    `reports/tabelas/carga_operacional.csv`; a leitura em texto fica em
    `leitura_operacional`, que deriva o veredito do número em vez de o afirmar.

    Returns:
        DataFrame com `estrategia`, `encaminhados`, `taxa_encaminhamento`,
        `casos_detectados`, `casos_perdidos`, `recall` e `nns`, ordenado por NNS.
    """
    linhas = []
    for r in resultados:
        total = r["vp"] + r["vn"] + r["fp"] + r["fn"] if n_teste is None else n_teste
        encaminhados = r["vp"] + r["fp"]
        nome = f"{r['modelo']} · {r['conjunto']}" if r["conjunto"] not in ("", "—") \
            else r["modelo"]
        linhas.append({
            "estrategia": nome,
            "encaminhados": encaminhados,
            "taxa_encaminhamento": encaminhados / total if total else float("nan"),
            "casos_detectados": r["vp"],
            "casos_perdidos": r["fn"],
            "recall": r["recall"],
            "nns": r["nns"],
        })
    tabela = pd.DataFrame(linhas).sort_values("nns").reset_index(drop=True)
    tabela.attrs["nota"] = NOTA_VOLUME_ABSOLUTO
    return tabela


def leitura_operacional(resultados: list[dict], nome_regra: str = "Regra ADA",
                        equiparados: list[dict] | None = None) -> str:
    """Escreve em uma frase a carga de encaminhamento de cada estratégia.

    A taxa de encaminhamento já está na tabela de NNS, mas número em célula não é
    argumento. Esta função escreve o argumento: quantos por cento da população cada
    estratégia manda ao laboratório, que é o que decide se a triagem cabe na
    capacidade instalada da rede.

    O veredito sobre a carga sai da taxa, não de uma afirmação fixa: a frase que
    fecha a leitura dizia sempre que a triagem manda mais da metade da população ao
    laboratório, e isso vale enquanto a taxa da regra estiver acima do corte. Mesmo
    defeito que D-045 corrigiu na comparação de NNS.

    Args:
        equiparados: resultados do regime de recall equiparado, quando houver.
            Acrescenta a linha que impede ler as duas tabelas isoladamente: nesse
            regime o modelo encaminha **mais** gente que a regra em números
            absolutos, e quem olhar só a tabela de carga operacional vê o volume sem
            perceber que os recalls realizados diferem. Ali a comparação válida é por
            NNS, que normaliza por casos detectados.
    """
    regra = next((r for r in resultados if r["modelo"] == nome_regra), None)
    if regra is None:
        raise ValueError(f"Comparador {nome_regra!r} ausente dos resultados.")

    tabela = tabela_nns(resultados)
    linha_regra = tabela[tabela["estrategia"] == nome_regra].iloc[0]
    modelos = tabela[~tabela["estrategia"].isin([nome_regra, "Dummy (maioria)"])]
    if modelos.empty:
        return (
            f"{nome_regra} encaminha {linha_regra['encaminhados']:.0f} de "
            f"{linha_regra['encaminhados'] / linha_regra['taxa_encaminhamento']:.0f} "
            f"pessoas ({100 * linha_regra['taxa_encaminhamento']:.1f}%)."
        )

    melhor = modelos.iloc[0]
    total = linha_regra["encaminhados"] / linha_regra["taxa_encaminhamento"]
    frase = (
        f"{nome_regra} encaminha {linha_regra['encaminhados']:.0f} de {total:.0f} "
        f"pessoas ({100 * linha_regra['taxa_encaminhamento']:.1f}%) para detectar "
        f"{linha_regra['casos_detectados']:.0f} casos. "
        f"{melhor['estrategia']} encaminha {melhor['encaminhados']:.0f} "
        f"({100 * melhor['taxa_encaminhamento']:.1f}%) para detectar "
        f"{melhor['casos_detectados']:.0f}. "
        + _veredito_da_carga(nome_regra, linha_regra["taxa_encaminhamento"])
    )
    if equiparados:
        # Mesma estratégia nas duas metades da frase: o que se compara são os dois
        # regimes do mesmo modelo, não dois modelos diferentes.
        frase += " " + _linha_do_regime_equiparado(equiparados, nome_regra,
                                                   melhor["estrategia"])
    return frase


def _veredito_da_carga(nome: str, taxa: float) -> str:
    """Julga a carga de encaminhamento comparando a taxa com `CORTE_TRIAGEM_AMPLA`.

    A frase que fecha a leitura operacional afirmava sempre que a triagem manda mais
    da metade da população ao laboratório. Vale para a regra clínica hoje e deixa de
    valer no dia em que a taxa cair, e aí o texto contradiz o número que ele mesmo
    acabou de citar. Mesmo defeito que D-045 corrigiu na comparação de NNS.
    """
    if taxa > CORTE_TRIAGEM_AMPLA:
        return (f"{nome} encaminha mais da metade da população adulta, o que não "
                f"seleciona quem examinar: apenas desloca a fila para o laboratório.")
    return (f"{nome} encaminha menos da metade da população adulta, então a carga "
            f"ainda responde à pergunta de quem examinar, em vez de deslocar a fila "
            f"inteira para o laboratório.")


def _diferenca(delta: float, singular: str, plural: str) -> str:
    """Escreve uma diferença em palavras, inclusive quando ela é zero.

    Sem o caso do zero, a frase sai como "0 casos a menos", que afirma desvantagem
    onde não há diferença nenhuma.
    """
    if delta == 0:
        return f"o mesmo número de {plural}"
    nome = singular if abs(delta) == 1 else plural
    return f"{abs(delta):.0f} {nome} {'a mais' if delta > 0 else 'a menos'}"


def _linha_do_regime_equiparado(equiparados: list[dict],
                                nome_regra: str = "Regra ADA",
                                estrategia: str | None = None) -> str:
    """Impede que a tabela de carga operacional seja lida sozinha.

    No regime equiparado o modelo encaminha mais gente que a regra em números
    absolutos. Sem esta linha, quem olha só o volume lê o sinal errado, porque os
    recalls realizados diferem e o que normaliza a comparação é o NNS.

    O sentido da comparação de NNS sai do número, e não de uma afirmação fixa na
    frase. Quando a logística era o modelo de referência, ela vencia a regra também
    neste regime, e o texto dizia isso direto; com outro modelo de referência o sinal
    pode inverter, e frase que afirma vitória por escrito passaria a mentir sozinha.
    """
    tabela = tabela_nns(equiparados)
    regra = tabela[tabela["estrategia"] == nome_regra].iloc[0]
    modelos = tabela[~tabela["estrategia"].isin([nome_regra, "Dummy (maioria)"])]
    if estrategia is not None:
        escolhida = modelos[modelos["estrategia"] == estrategia]
        modelos = escolhida if not escolhida.empty else modelos
    if modelos.empty:
        return ""

    alvo = modelos.iloc[0]
    recall_modelo = next(
        r["recall"] for r in equiparados
        if f"{r['modelo']} · {r['conjunto']}" == alvo["estrategia"]
        or r["modelo"] == alvo["estrategia"]
    )
    recall_regra = next(r["recall"] for r in equiparados if r["modelo"] == nome_regra)

    d_pessoas = alvo["encaminhados"] - regra["encaminhados"]
    d_casos = alvo["casos_detectados"] - regra["casos_detectados"]
    if alvo["nns"] < regra["nns"]:
        veredito_nns = "mantendo NNS inferior"
    elif alvo["nns"] > regra["nns"]:
        veredito_nns = "mas com NNS superior, ou seja, pior"
    else:
        veredito_nns = "com NNS idêntico"
    return (
        f"No regime de recall equiparado, {alvo['estrategia']} encaminha "
        f"{_diferenca(d_pessoas, 'pessoa', 'pessoas')} "
        f"({alvo['encaminhados']:.0f} contra {regra['encaminhados']:.0f}) e detecta "
        f"{_diferenca(d_casos, 'caso', 'casos')} "
        f"({alvo['casos_detectados']:.0f} contra {regra['casos_detectados']:.0f}), "
        f"{veredito_nns} ({alvo['nns']:.2f} contra {regra['nns']:.2f}): "
        f"a comparação válida é por NNS, não por volume absoluto, já que os recalls "
        f"realizados diferem ({recall_modelo:.3f} contra {recall_regra:.3f})."
    )


def recall_da_regra_no_treino(X_tr, y_tr, regra=None) -> float:
    """Recall da regra clínica no treino, alvo para comparar em pé de igualdade.

    Sem isso, a comparação fica enviesada pelo ponto de operação: a regra opera onde
    cai, o modelo opera onde o `RECALL_ALVO` o colocou, e comparar NNS entre dois
    recalls diferentes não responde nada. Todo classificador com escore contínuo
    troca recall por precisão ao longo da curva, então só há comparação quando os
    dois estão no mesmo ponto dessa troca.

    Medido **no treino**: usar o recall que a regra obtém no teste para calibrar o
    modelo vazaria o teste para dentro da decisão de limiar.

    Uso:

        alvo = recall_da_regra_no_treino(X_tr, y_tr)
        resultado = evaluate.avaliar(modelo, ..., recall_alvo=alvo)
    """
    from sklearn.metrics import recall_score

    regra = RegraClinicaADA() if regra is None else regra
    predito = regra.fit(X_tr, y_tr).predict(X_tr)
    return float(recall_score(y_tr, predito, zero_division=0))


def supera_a_regra_clinica(resultados: list[dict],
                           nome_regra: str = "Regra ADA") -> pd.DataFrame:
    """Aplica o critério de sucesso do projeto: mais recall **com** NNS menor.

    As duas condições valem juntas. Recall maior ao custo de encaminhar mais gente
    não é ganho, é o mesmo que baixar o limiar da regra.

    **Só é interpretável quando os estimadores comparados operam em regime de recall
    comparável**, ver `recall_da_regra_no_treino`. Aplicada a resultados calibrados
    em `RECALL_ALVO = 0,80` contra uma regra que opera em 0,87, a função reporta
    corretamente que nenhum modelo domina, e isso diz mais sobre os pontos de
    operação do que sobre os classificadores.

    Returns:
        DataFrame com `delta_recall`, `delta_nns` e `supera`, uma linha por
        estratégia comparada.
    """
    regra = next((r for r in resultados if r["modelo"] == nome_regra), None)
    if regra is None:
        raise ValueError(f"Comparador {nome_regra!r} ausente dos resultados.")

    linhas = []
    for r in resultados:
        if r["modelo"] == nome_regra:
            continue
        nome = f"{r['modelo']} · {r['conjunto']}" if r["conjunto"] not in ("", "—") \
            else r["modelo"]
        delta_recall = r["recall"] - regra["recall"]
        delta_nns = r["nns"] - regra["nns"]
        linhas.append({
            "estrategia": nome,
            # `recall_alvo` é onde a equiparação acontece, no treino, por
            # construção; `recall_obtido` é o que se realizou no teste. Separar as
            # duas colunas evita que a tabela pareça comparar recalls iguais quando
            # o que está equiparado é o alvo, não o resultado.
            "recall_alvo": r.get("recall_alvo", float("nan"))
                           if r.get("limiar_calibrado", True) else float("nan"),
            "recall_obtido": r["recall"],
            "recall_regra": regra["recall"],
            "delta_recall": delta_recall,
            "nns": r["nns"],
            "nns_regra": regra["nns"],
            "delta_nns": delta_nns,
            "supera": bool(delta_recall > 0 and delta_nns < 0),
        })
    tabela = pd.DataFrame(linhas)
    tabela.attrs["nota"] = NOTA_EQUIPARACAO
    return tabela
