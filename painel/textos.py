"""Frases fixas do painel.

Nenhuma frase aqui carrega número: todo número exibido sai de coluna de
`reports/tabelas/`. As frases que citam termos vetados pelo vocabulário do painel
existem para negá-los, e a guarda de vocabulário as reconhece como literais.
"""

TITULO = "Classificação de perfis associados a HbA1c ≥ 6,5%"

# D-054
RODAPE = (
    "Painel de resultados das Sprints 1 a 3. "
    "O protótipo de classificação com API é a fase 6."
)

# D-022 e D-040
ESCORE_NAO_E_RISCO = (
    "O escore de ordenação serve para priorizar encaminhamento. "
    "Ele não é risco individual e nunca aparece em porcentagem (D-022, D-040)."
)

# D-053
RESSALVA_LOCAL = (
    "Contribuição relativa ao indivíduo médio sob reponderamento de classes. "
    "Negativa significa abaixo dessa referência, nunca efeito protetor."
)

# D-043
LEITURA_MCNEMAR = (
    "supera em recall, com diferença de acerto global não significativa, "
    "detectando um superconjunto dos casos"
)

# D-053
FLORESTA_REDUZIDA = (
    "A floresta reduzida é objeto de diagnóstico e nunca modelo publicado."
)

PREVALENCIA_PONDERADA = (
    "A prevalência ponderada por peso_lab é estimativa populacional e não "
    "contextualiza as métricas. As métricas do teste se leem contra a prevalência "
    "amostral, que é a que o class_weight='balanced' reequilibra (R9, D-031)."
)

# D-053, pré-registro escrito antes de qualquer valor
PREDICAO_D053 = (
    "Predição registrada em D-053 antes da medição: dentro de cada bloco colinear, "
    "um membro concentraria a atribuição e o outro colapsaria."
)

DESFECHO_D053 = (
    "Desfecho: imc × circ_cintura não confirma, nos dois conjuntos e nos dois "
    "regimes, e a divisão do bloco fica estável. O bloco de coarsening de "
    "multimorbidade não confirma: o escore retém parte relevante do bloco por "
    "conta própria. pas × pad confirma como assimetria e não como colapso, e a "
    "assimetria se acentua no regime interventional. Os números estão na tabela "
    "acima, coluna fração do bloco."
)

DOIS_REGIMES = (
    "O regime tree_path_dependent sobre a floresta completa é o que vai ao texto "
    "e às figuras. O regime interventional roda sobre a floresta reduzida, como "
    "contraste diagnóstico. A divergência entre os dois rankings mede a exposição "
    "do trabalho à suposição de independência."
)

CATEGORIA_OMITIDA = (
    "A codificação usa drop='first': a categoria omitida de cada nominal tem "
    "atribuição zero por construção, não por irrelevância."
)

LIMIAR_VIF = (
    "A linha vertical marca o limiar de VIF declarado em vif_resumo (R8)."
)

ABLACAO_RCEST = (
    "Ablação de rcest no conjunto completo: VIF de cada coluna antes e depois da "
    "remoção. Sem rcest nenhuma coluna cruza o limiar (D-019)."
)

DECOMPOSICAO_ESCORE = (
    "multimorbidade é, por construção, a soma de três parcelas binárias, e o R² da "
    "identidade mostra isso. A indicadora de obesidade não é coluna do modelo: sai "
    "de imc e circ_cintura por ponto de corte, que a combinação linear não "
    "recupera. A distância entre o R² da identidade e o da auxiliar do VIF é a "
    "parte da redundância que o diagnóstico linear não enxerga (D-012)."
)

MESMO_INDIVIDUO = (
    "O índice {indice} aparece nas células {celulas}: é o mesmo indivíduo nas duas."
)

CASOS = {
    "VP": "verdadeiro positivo",
    "FP": "falso positivo",
    "FN": "falso negativo",
    "VN": "verdadeiro negativo",
}

CASOS_LOCAIS = (
    "Quatro casos por célula, cada um o de menor distância ao limiar calibrado "
    "dentro da sua célula da matriz de confusão."
)

WATERFALL = (
    "A barra parte do valor de base, soma as contribuições na ordem da tabela e "
    "termina no escore de ordenação. A linha vertical é o limiar calibrado."
)
