"""Frases fixas do painel.

Nenhuma frase aqui carrega número de resultado: todo número exibido sai de coluna de
`reports/tabelas/`. As frases que citam termos vetados pelo vocabulário do painel
existem para negá-los, e a guarda de vocabulário as reconhece como literais. Frases
das seções de projeto e clínica não levam identificador de decisão; as da seção
técnica podem levar.
"""

# ---------------------------------------------------------------------------
# Moldura do app
# ---------------------------------------------------------------------------

TITULO_APP = "HbA1c ≥ 6,5%"
SUBTITULO_APP = "Classificação de perfis · PNS 2013"

RODAPE = (
    "Painel de resultados das Sprints 1 a 3. "
    "O protótipo de classificação com API é a fase 6."
)

SECOES = {
    "projeto": "Projeto",
    "clinica": "Visão clínica e de gestão",
    "tecnica": "Visão técnica",
}

SELO_CELULA = "Célula: {celula}"
SELO_TESTE = "PNS 2013 · teste n = {n}"
SELO_REFERENCIA_CLINICA = "Referência: menor NNS no ponto padrão"
SELO_REFERENCIA_TECNICA = "Referência por D-030, menor NNS no ponto padrão"
ROTULO_MODELO = "Modelo"
ROTULO_CONJUNTO = "Conjunto de variáveis"

# Título, subtítulo e abertura de cada página. Abertura com até duas frases.
PAGINAS = {
    "visao_geral": (
        "Visão geral",
        "Da amostra da PNS ao escore de ordenação",
        "Adultos da subamostra laboratorial da PNS 2013, com HbA1c medida. O modelo "
        "ordena quem encaminhar a exame usando só variáveis não invasivas.",
    ),
    "desempenho": (
        "Desempenho",
        "Quanto o modelo encontra e quanto encaminha",
        "Cada número compara a triagem do modelo com o desfecho medido no laboratório.",
    ),
    "triagem": (
        "Triagem na atenção primária",
        "Carga de encaminhamento e casos encontrados",
        "Três formas de decidir quem vai ao laboratório, lidas pela carga que geram na "
        "rede.",
    ),
    "fatores": (
        "Fatores associados",
        "O que mais move o escore de ordenação",
        "Contribuição média de cada variável ao escore, no conjunto de teste.",
    ),
    "casos_exemplares": (
        "Casos exemplares",
        "Como o escore de uma pessoa se compõe",
        "Para cada conjunto, o caso mais próximo do limiar em cada célula da matriz de "
        "confusão.",
    ),
    "amostra": (
        "Amostra",
        "Exclusões, ausentes e prevalências",
        "Registros sem HbA1c válida saem da amostra e nunca são imputados.",
    ),
    "colinearidade": (
        "Colinearidade e escolha do conjunto",
        "Pearson, VIF, ablação de rcest e o escore de multimorbidade",
        "O diagnóstico de colinearidade decide quais variáveis compostas convivem com "
        "seus componentes.",
    ),
    "modelagem": (
        "Modelagem",
        "Busca de hiperparâmetros da Random Forest",
        "Os candidatos são ordenados por PR-AUC média na validação cruzada do treino "
        "(D-035).",
    ),
    "floresta_logistica": (
        "Floresta × logística",
        "Teste de McNemar pareado no conjunto de teste",
        "Os dois modelos no mesmo conjunto, caso a caso, cada um no seu limiar "
        "calibrado (D-043).",
    ),
    "calibracao": (
        "Calibração",
        "Escala do escore contra a frequência observada",
        "O reponderamento de classes desloca a escala do escore sem estragar a "
        "ordenação (D-022, D-040).",
    ),
    "explicabilidade_tecnica": (
        "Explicabilidade técnica",
        "Dois regimes de perturbação, blocos colineares e árvores excluídas",
        "O contraste entre regimes mede a exposição da explicação à suposição de "
        "independência (D-053).",
    ),
}

# ---------------------------------------------------------------------------
# Glossário em linguagem de UBS, usado no `ajuda` de todo cartão
# ---------------------------------------------------------------------------

GLOSSARIO = {
    "recall": "Parcela das pessoas com HbA1c ≥ 6,5% que a triagem encaminha a exame.",
    "especificidade": "Parcela das pessoas sem HbA1c ≥ 6,5% que a triagem não encaminha.",
    "f1": "Equilíbrio entre encontrar os casos e não encaminhar à toa.",
    "nns": "Quantas pessoas são encaminhadas a exame para encontrar um caso.",
    "encaminhados": "Parcela das pessoas atendidas que a triagem manda ao laboratório.",
    "encaminhados_n": "Quantas pessoas a triagem manda ao laboratório.",
    "auc_roc": "Capacidade de pôr quem tem HbA1c ≥ 6,5% à frente de quem não tem, em "
               "qualquer ponto de corte.",
    "pr_auc": "Qualidade da fila quando os casos são raros. Ordenar ao acaso dá o piso, "
              "que é a prevalência.",
    "vp": "Encaminhados que de fato têm HbA1c ≥ 6,5%.",
    "fp": "Encaminhados sem HbA1c ≥ 6,5%: exame sem achado.",
    "fn": "Não encaminhados com HbA1c ≥ 6,5%: caso que a triagem deixa passar.",
    "vn": "Não encaminhados que de fato não têm HbA1c ≥ 6,5%.",
    "casos_detectados": "Pessoas com HbA1c ≥ 6,5% entre as encaminhadas.",
    "escore": "Posição na fila de encaminhamento. Não é chance de ter o desfecho.",
    "limiar": "Ponto de corte do escore, calibrado no treino. Acima dele, encaminha.",
    "distancia": "Quanto o escore de ordenação ficou longe do limiar.",
    "valor_base": "Escore médio do modelo, ponto de partida da decomposição.",
    "linhas_brutas": "Registros da subamostra laboratorial antes da limpeza.",
    "amostra_analitica": "Registros que entram na análise depois das exclusões.",
    "positivos": "Pessoas com HbA1c ≥ 6,5% na amostra analítica.",
    "negativos": "Pessoas com HbA1c abaixo de 6,5% na amostra analítica.",
    "prevalencia": "Parcela da amostra com HbA1c ≥ 6,5%.",
    "prevalencia_ponderada": "Estimativa para a população adulta brasileira, com o "
                             "peso amostral da PNS.",
    "treino": "Registros usados para ajustar o modelo e calibrar o limiar.",
    "teste": "Registros guardados para avaliar o modelo uma única vez.",
    "n_teste": "Pessoas no conjunto de teste.",
    "concordantes": "Pessoas em que os dois modelos acertam juntos ou erram juntos.",
    "discordantes": "Pessoas em que só um dos modelos acerta.",
    "p_valor": "Chance de uma diferença assim aparecer se os dois modelos fossem "
               "equivalentes.",
    "positivos_teste": "Pessoas com HbA1c ≥ 6,5% no conjunto de teste.",
    "positivos_so": "Casos com HbA1c ≥ 6,5% que só um dos modelos encaminha.",
    "pr_auc_busca": "PR-AUC média do vencedor na validação cruzada do treino.",
    "topo_coincidente": "Quantas primeiras posições do ranking coincidem nos dois "
                        "regimes.",
    "arvores_excluidas": "Árvores retiradas da floresta de diagnóstico por erro do "
                         "explicador.",
}

# ---------------------------------------------------------------------------
# Rótulos clínicos das variáveis
# ---------------------------------------------------------------------------

ROTULOS_CLINICOS = {
    "idade": "Idade",
    "sexo_fem": "Sexo feminino",
    "imc": "Índice de massa corporal",
    "circ_cintura": "Circunferência da cintura",
    "rcest": "Relação cintura-estatura",
    "pas": "Pressão arterial sistólica",
    "pad": "Pressão arterial diastólica",
    "has_autorrelato": "Hipertensão autorreferida",
    "has_combinada": "Hipertensão combinada",
    "colesterol_alto": "Colesterol alto autorreferido",
    "multimorbidade": "Multimorbidade metabólica",
    "raca_cor": "Raça/cor",
    "escolaridade": "Escolaridade",
    "plano_sim": "Plano de saúde",
    "regiao": "Região",
    "saude_autoaval": "Saúde autoavaliada",
}

# ---------------------------------------------------------------------------
# Projeto
# ---------------------------------------------------------------------------

FUNIL_TITULO = "Da base bruta ao conjunto de teste"
DESBALANCEAMENTO_TITULO = "Desbalanceamento do desfecho"
PIPELINE_TITULO = "Pipeline"
PIPELINE = [
    ("CSV bruto da PNS", False),
    ("Limpeza e exclusões", False),
    ("Divisão estratificada\ntreino e teste", False),
    ("Imputação KNN\najustada só no treino", True),
    ("Variáveis compostas", True),
    ("Codificação", True),
    ("Modelo com\nclass_weight", True),
    ("Limiar calibrado por\nvalidação cruzada no treino", True),
    ("Avaliação no teste", False),
    ("Explicabilidade", False),
]
PIPELINE_LEGENDA = "Em azul, as etapas ajustadas só com dados de treino."

SMOTE_TITULO = "Por que não SMOTE"
SMOTE = (
    ("Reamostragem sintética (SMOTE)", [
        "Cria pessoas artificiais interpolando positivos.",
        "Quebra a estrutura de ponderação amostral da PNS.",
        "Pode gerar perfis clínicos que não existem.",
    ]),
    ("Aprendizado sensível ao custo", [
        "Treina só com registros reais.",
        "Repondera as classes na função de perda.",
        "Preserva o desenho amostral dos dados.",
    ]),
)

CRISP_DM_TITULO = "Fases do CRISP-DM"
CRISP_DM = [
    ("Entendimento do negócio", "concluída", True),
    ("Entendimento dos dados", "concluída", True),
    ("Preparação dos dados", "concluída", True),
    ("Modelagem", "concluída", True),
    ("Avaliação", "concluída", True),
    ("Implantação", "pendente, após a pré-banca", False),
]

NAO_FAZ_TITULO = "O que o sistema não faz"
NAO_FAZ = [
    "Não classifica novos pacientes",
    "Não estima risco individual",
    "Não elege a célula principal",
]

# ---------------------------------------------------------------------------
# Visão clínica e de gestão
# ---------------------------------------------------------------------------

COMPARADOR_ADA = "Comparador: regra clínica ADA (idade ≥ 45 ou IMC ≥ 30)"
LEGENDA_DESEMPENHO = (
    "Métricas no conjunto de teste. Limiar calibrado no treino para recall-alvo "
    "{recall_alvo}."
)
DP_FOLDS = "± {dp} entre folds"
PISO = "piso {piso}"
ROTULO_PONTO = "ponto de operação"
MATRIZ_TITULO = "Matriz de confusão no teste"
MATRIZ = {
    "vn": "Não encaminhado · sem HbA1c ≥ 6,5%",
    "fp": "Encaminhado · sem HbA1c ≥ 6,5%",
    "fn": "Não encaminhado · com HbA1c ≥ 6,5%",
    "vp": "Encaminhado · com HbA1c ≥ 6,5%",
}
ABA_ROC = "Curva ROC"
ABA_PR = "Curva PR"
COMPARAR_CELULAS = "Comparar células"
NOTA_CORTE_UNICO = (
    "Regra ADA e Dummy decidem num corte único. AUC-ROC e PR-AUC deles não se "
    "comparam com as dos modelos."
)

REGIMES_TITULO = "Três regimes de encaminhamento"
ENCAMINHADOS_TITULO = "Percentual de encaminhados a exame"
EQUIPARADO_TITULO = "Regime equiparado ao recall da regra"
EQUIPARADO_NOTA = (
    "Cada célula calibrada para o recall que a regra obtém no treino. Cumpre quem "
    "supera a regra em recall com NNS menor."
)
CUMPRE = ("cumpre", "verde")
NAO_CUMPRE = ("não cumpre", "vermelho")
CUSTO_MARGINAL_TITULO = "Custo marginal da floresta sobre a logística"
CUSTO_MARGINAL_NOTA = (
    "enc. são encaminhados e det. são casos detectados. Quando a floresta encaminha "
    "menos e detecta mais, a leitura é dominância e não há custo a calcular."
)
LEITURA_TRIAGEM = (
    "A regra encaminha mais da metade da amostra num ponto único. O modelo permite "
    "escolher o ponto de operação conforme a capacidade da rede."
)

FATORES_TITULO_EIXO = "contribuição média ao escore"
FATORES_DEMAIS = "Demais variáveis"
LEITURA_TITULO = "Leitura"
LEITURA_FATORES = (
    "As variáveis no topo são as que mais movem o escore do modelo. Contribuição "
    "descreve o modelo e não um efeito causal, e não indica intervenção."
)
NOTA_RACA_COR = (
    "Raça/cor entra como marcador de iniquidade social e de acesso a serviço de "
    "saúde, não como fator biológico."
)
SHAP_SO_FLORESTA = (
    "O SHAP da regressão logística não foi executado. A explicação existe só para a "
    "Random Forest nos conjuntos atômico e composto: escolha uma dessas células na "
    "barra lateral."
)

FAIXA_CASOS = (
    "Casos reais do conjunto de teste, calculados previamente. Esta tela não "
    "classifica novos pacientes."
)
ROTULO_CASO = "Caso"
VALORES_OBSERVADOS = "Valores observados"
DECISAO_TITULO = "Decisão da triagem"
DECISAO = {
    "VP": ("Encaminhar a exame", "azul"),
    "FP": ("Encaminhar a exame", "azul"),
    "FN": ("Não encaminhar", "cinza"),
    "VN": ("Não encaminhar", "cinza"),
}
DESFECHO = {
    "VP": "HbA1c ≥ 6,5%: sim",
    "FP": "HbA1c ≥ 6,5%: não",
    "FN": "HbA1c ≥ 6,5%: sim",
    "VN": "HbA1c ≥ 6,5%: não",
}
ACERTO = {
    "VP": ("acerto", "verde"),
    "FP": ("erro", "vermelho"),
    "FN": ("erro", "vermelho"),
    "VN": ("acerto", "verde"),
}
CASOS = {
    "VP": "verdadeiro positivo",
    "FP": "falso positivo",
    "FN": "falso negativo",
    "VN": "verdadeiro negativo",
}
DESFECHO_OBSERVADO = "Desfecho observado no laboratório"
ROTULO_DISTANCIA = "distância do escore de ordenação ao limiar"
MESMO_INDIVIDUO = (
    "O índice {indice} aparece nas células {celulas}: é o mesmo indivíduo nas duas."
)

RESSALVA_LOCAL = (
    "Contribuição relativa ao indivíduo médio sob reponderamento de classes. "
    "Negativa significa abaixo dessa referência, nunca efeito protetor."
)

WATERFALL = (
    "A barra parte do valor de base, soma as contribuições na ordem da tabela e "
    "termina no escore de ordenação. A linha vertical é o limiar calibrado."
)

# ---------------------------------------------------------------------------
# Visão técnica
# ---------------------------------------------------------------------------

PREVALENCIA_PONDERADA = (
    "A prevalência ponderada por peso_lab é estimativa populacional e não "
    "contextualiza as métricas. As métricas do teste se leem contra a prevalência "
    "amostral, que é a que o class_weight='balanced' reequilibra (R9, D-031)."
)
AUSENTES_TITULO = "Ausentes por variável ao fim da limpeza"
AUSENTES_NOTA = (
    "Os ausentes das variáveis classificatórias são imputados por KNNImputer ajustado "
    "só no treino (R3)."
)

LIMIAR_VIF = "A linha vertical marca o limiar de VIF declarado em vif_resumo (R8)."
ABLACAO_RCEST = (
    "VIF de cada coluna do conjunto completo antes e depois de remover rcest. Sem "
    "rcest nenhuma coluna cruza o limiar (D-019)."
)
DECOMPOSICAO_ESCORE = (
    "multimorbidade é, por construção, a soma de três parcelas binárias, e o R² da "
    "identidade mostra isso. A indicadora de obesidade não é coluna do modelo: sai "
    "de imc e circ_cintura por ponto de corte, que a combinação linear não "
    "recupera. A distância entre os dois R² é a parte da redundância que o "
    "diagnóstico linear não enxerga (D-012)."
)

BUSCA_TITULO = "Candidatos da busca"
HIPERPARAMETROS_TITULO = "Hiperparâmetros congelados"
HIPERPARAMETROS_NOTA = (
    "O vencedor do conjunto composto é aplicado aos dois conjuntos (D-036)."
)
D034 = (
    "balanced_subsample reproduz na floresta o que balanced faz na logística, e a "
    "busca não separou os dois por métrica (D-034)."
)

LEITURA_MCNEMAR = (
    "supera em recall, com diferença de acerto global não significativa, "
    "detectando um superconjunto dos casos"
)
MATRIZ_MCNEMAR = "Matriz de acertos pareados"
RECORTE_POSITIVOS = "Recorte nos positivos do teste"

RESTRICAO_ESCORE = (
    "Escore de ordenação, não probabilidade de desfecho. Um escore de 0,60 não "
    "significa 60% de chance."
)
DECIS_TITULO = "Curva de calibração por decis"
INDICES_CALIBRACAO = "Índices de calibração"

DOIS_REGIMES = (
    "O regime tree_path_dependent sobre a floresta completa é o que vai ao texto e às "
    "figuras. O regime interventional roda sobre a floresta reduzida, como contraste "
    "diagnóstico."
)
CONCORDANCIA_TITULO = "Topo coincidente entre regimes"
COMPARACOES = {
    "principal reduzida x contraste reduzida": (
        "Comparação pré-registrada",
        "Principal e contraste sobre a mesma floresta reduzida (D-053).",
    ),
    "principal completa x contraste reduzida": (
        "Comparação publicada antes",
        "Principal na floresta completa, contraste na reduzida. Mistura remoção de "
        "árvores e troca de regime.",
    ),
}
CATEGORIA_OMITIDA = (
    "A codificação usa drop='first': a categoria omitida de cada nominal tem "
    "atribuição zero por construção, não por irrelevância."
)
BLOCOS_TITULO = "Blocos colineares, individual e somado"
PREDICAO_D053 = (
    "Predição registrada em D-053 antes da medição: dentro de cada bloco colinear, "
    "um membro concentraria a atribuição e o outro colapsaria."
)
DESFECHO_D053 = (
    "Desfecho: imc × circ_cintura não confirma, nos dois conjuntos e nos dois "
    "regimes, e a divisão do bloco fica estável. O bloco de coarsening de "
    "multimorbidade não confirma: o escore retém parte relevante do bloco por "
    "conta própria. pas × pad confirma como assimetria e não como colapso, e a "
    "assimetria se acentua no regime interventional."
)
ARVORES_TITULO = "Árvores excluídas do regime de contraste"
FLORESTA_REDUZIDA = (
    "A floresta reduzida é objeto de diagnóstico e nunca modelo publicado."
)
