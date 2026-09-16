# RASTREABILIDADE METODOLÓGICA

Registro vivo das decisões de implementação e sua relação com o Projeto de TCC I
assinado. Atualizado ao final de cada sprint.

**Função:** insumo direto da seção de Metodologia do TCC II e antecipação das perguntas
de banca. Toda decisão que o código toma e o TCC I não previu precisa aparecer aqui
antes de aparecer nos resultados.

**Classificação:**

| Status | Significado |
|---|---|
| **CONFORME** | Implementa diretamente o que o TCC I estabelece |
| **EXECUÇÃO DE PROTOCOLO** | O TCC I prevê o procedimento; a decisão é a forma concreta de executá-lo |
| **EXTRAPOLAÇÃO** | Acréscimo legítimo, ausente do TCC I. Exige redação no TCC II |
| **DIVERGÊNCIA** | Contraria o TCC I. Exige justificativa formal e aval do orientador |
| **PENDÊNCIA DE REDAÇÃO** | Lacuna argumentativa no texto assinado, a fechar no LaTeX |

---

## Sprint 1 — Refatoração e harness de avaliação

### D-001 · Ablação de conjuntos de atributos (A/B/C)
**Status:** EXECUÇÃO DE PROTOCOLO
**Origem:** §6.3.2 princípio 4 · §6.5

O baseline alimentava `has_autorrelato` por três canais (one-hot, `has_combinada`,
`multimorbidade`) e `colesterol_alto` por dois. O Princípio 4 exige variáveis
conceitualmente ortogonais, justamente para não diluir a interpretabilidade das
contribuições individuais via SHAP. A §6.5 autoriza avaliar por experimentação a remoção
de variáveis correlacionadas do mesmo bloco temático.

**Redação no TCC II:** apresentar como execução do protocolo da §6.5, não como correção
de erro. A tabela comparativa A/B/C é resultado.

---

### D-002 · VIF sobre a matriz de design pós-transformação
**Status:** CONFORME (correção de implementação)
**Origem:** §6.5

O baseline calculava VIF apenas sobre o bloco numérico cru, deixando `has_combinada`
(binária) e `has_autorrelato` (nominal) fora do diagnóstico — razão pela qual a
redundância de D-001 não foi detectada. O VIF passa a rodar sobre a matriz completa
pós-`ColumnTransformer`, incluindo dummies.

**Redação no TCC II:** nenhuma justificativa adicional necessária. É o cumprimento
correto do que já está escrito.

---

### D-003 · Correlação de Pearson entre contínuas
**Status:** CONFORME (lacuna de implementação corrigida)
**Origem:** §6.5

A §6.5 exige Pearson **e** VIF. O baseline entregou apenas VIF.

---

### D-004 · Seleção de limiar por Recall-alvo
**Status:** EXTRAPOLAÇÃO
**Origem:** ausente no TCC I

O TCC I define Recall como métrica prioritária (§6.3 fase 5), mas não especifica o
ponto de operação. Usar o default 0,5 do scikit-learn é uma decisão silenciosa e
inadequada a 7% de prevalência. O limiar passa a ser parâmetro explícito, calibrado por
validação cruzada no treino.

**Valor provisório:** `RECALL_ALVO = 0.80`.
**Pendência:** confirmar com o orientador antes da etapa 4 da Sprint 1. Alteração
posterior obriga a regerar todas as figuras e tabelas.

**Redação no TCC II:** justificar o limiar como decisão de projeto derivada da assimetria
de custo entre falso negativo (caso não triado) e falso positivo (exame desnecessário)
no contexto da APS.

---

### D-005 · Métricas complementares — PR-AUC, Brier, NNS
**Status:** EXTRAPOLAÇÃO
**Origem:** ausente no TCC I

As cinco métricas assinadas (Recall, Especificidade, F1, AUC-ROC, Matriz de Confusão)
permanecem obrigatórias em toda tabela. As complementares se somam:

- **PR-AUC** — mais informativa que ROC sob prevalência de 7%
- **Brier score** — subsidia o backlog de calibração probabilística
- **NNS** (Number Needed to Screen) — tradução operacional para a APS: quantas pessoas
  encaminhar a exame para identificar um caso

**Redação no TCC II:** posicionar como enriquecimento do protocolo de avaliação, com
justificativa técnica para cada uma.

---

### D-006 · Benchmark por regra clínica ADA
**Status:** EXTRAPOLAÇÃO
**Origem:** ancorado em §2 e §5 (discussão de FINDRISC/BrDMrisc), não formalizado

O TCC I discute FINDRISC e BrDMrisc como estado da arte do qual o trabalho se
diferencia, mas não estabelece comparador executável. A regra `idade ≥ 45 OU IMC ≥ 30`
representa o que a APS faz hoje sem machine learning.

**Impossibilidade documentada:** FINDRISC exige histórico familiar e variáveis
comportamentais, excluídas por §6.3.2 princípio 3. BrDMrisc é escore de incidência
derivado de coorte longitudinal, incompatível com o desenho transversal da PNS 2013.

**Redação no TCC II:** a impossibilidade de reproduzir os escores existentes é resultado
a reportar, não limitação a omitir.

---

### D-007 · Agrupamento de `raca_cor`
**Status:** EXTRAPOLAÇÃO — **não implementada**
**Origem:** ausente no TCC I

No baseline, `raca_cor_3` (Amarela, n=81, 6 eventos) produziu o maior coeficiente do
modelo (OR 2,93), enquanto a prevalência marginal entre categorias é praticamente plana
(6,8%–9,2%). Indica instabilidade de célula esparsa, não achado.

Agrupamento candidato: Branca / Negra (Preta+Parda, n=5.208) / Outras (Amarela+Indígena,
n=146).

**Decisão:** rodar primeiro com as cinco categorias e IC bootstrap. Agrupar apenas se os
intervalos confirmarem a instabilidade. Agrupar antes de medir seria decidir sem
evidência.
**Reavaliar:** etapa 6 da Sprint 1.

**Redação no TCC II:** a §6.4.1 justifica raça/cor como proxy de iniquidades históricas e
acesso diferencial. O texto deve reforçar explicitamente a leitura como **determinação
social**, jamais biológica, e reportar IC sempre que o coeficiente for discutido.

---

### D-008 · Categoria "não sabe/não mediu" para `colesterol_alto`
**Status:** DIVERGÊNCIA — implementar como variante comparada
**Origem:** contraria §6.3 fase 3 (R3)

`colesterol_alto` tem 1.297 ausentes (≈15,4%). O TCC I estabelece KNN para variáveis
classificatórias. Hipótese concorrente: o padrão de ausência é MNAR — quem nunca mediu
colesterol tende a ter menos acesso a serviço de saúde, o que se correlaciona com o
próprio desfecho. Imputar destrói essa informação.

**Decisão:** KNN permanece como padrão (conformidade com R3). A categoria explícita entra
como variante comparada, com resultados reportados lado a lado.

**Redação no TCC II:** se a variante vencer, justificar a divergência formalmente e
submeter ao orientador. Se não vencer, reportar como análise de sensibilidade.

---

### D-009 · `peso_lab` fora do treinamento
**Status:** PENDÊNCIA DE REDAÇÃO
**Origem:** §5 · §6.3 fase 2

A §5 rejeita o SMOTE argumentando que observações sintéticas quebram a estrutura de
ponderação amostral usada para representatividade populacional. O texto não explicita
por que essa mesma ponderação não se aplica ao treinamento do modelo.

Lida em conjunto, a redação atual expõe o trabalho à leitura de que a ponderação foi
invocada para rejeitar uma técnica e ignorada na etapa seguinte.

**No código:** `peso_lab` pondera exclusivamente a EDA descritiva (§6.3 fase 2). Nenhuma
função de `features.py`, `evaluate.py` ou `benchmark.py` o recebe como parâmetro.

**Pendência:** parágrafo a redigir no LaTeX antes da defesa. Marcado pelo autor.

---

### D-010 · `peso_kg` e `altura_cm` fora do conjunto de preditoras
**Status:** CONFORME
**Origem:** §6.4.1

A §6.4.1 define `IMC` e `RCEst` como variáveis derivadas do bloco antropométrico. Peso e
altura brutos permanecem na base analítica (são usados no cálculo das derivadas e na EDA),
mas não são devolvidos por `data.separar`: entrariam no modelo por canal redundante com as
próprias derivadas, contrariando o Princípio 4 da §6.3.2.

**Redação no TCC II:** nenhuma justificativa adicional necessária.

---

### D-011 · Contagem de ausentes de `colesterol_alto` — divergência de documentação
**Status:** CONFORME (correção de documentação)
**Origem:** §6.4.1

O `CLAUDE.md` registra 1.319 ausentes (15,6%) para `colesterol_alto`; a
`docs/Spec sprint1.md` registra 1.297 (≈15,4%). A execução de `src/data.py` sobre a base
bruta confirma **1.297** na amostra analítica de 8.451 registros — a spec está correta.
A contagem maior corresponde à base antes das exclusões da §3.

**Ação:** corrigir o número no `CLAUDE.md` e em D-008 antes da pré-banca.

---

### D-012 · `has_combinada` removida de B e invariantes em dois níveis
**Status:** EXECUÇÃO DE PROTOCOLO
**Origem:** §6.3.2 princípio 4 · §6.5 · §4.2 da spec v2

A formulação da v1 — "nenhuma variável fonte alimenta o modelo por dois canais em A e
B" — era forte demais, e B não tinha como cumpri-la. Duas correções, ambas na spec v2:

**1. `has_combinada` sai de `B_composto`.** Ela é a primeira parcela literal de
`multimorbidade`. Mantê-la ao lado do escore reproduzia em B a mesma violação que a
§4.1 acusa no baseline, e o modelo recuperaria a parcela linearmente, esvaziando a
hipótese de sinergia que B existe para testar.

**2. Dois invariantes distintos, implementados e testados separadamente:**

| Predicado | Enunciado | Cumprido por |
|---|---|---|
| `fontes_isoladas_e_agregadas` | nenhuma fonte entra como coluna própria **e** parcela idêntica de agregada | A e B |
| `fontes_em_canal_duplo` | nenhuma fonte contribui por mais de um caminho, em qualquer forma | apenas A |

A sobreposição residual de B (`fontes_em_coarsening`) é `peso_kg`, `altura_cm`,
`circ_cintura` e `sexo`: entram como colunas próprias e alimentam o componente de
obesidade do escore em forma **binarizada por ponto de corte clínico**. Não é o mesmo
canal — o gradiente contínuo carrega sinal que o limiar descarta.

`C_completo` viola o invariante estrito deliberadamente, e tem teste que **falha se a
redundância desaparecer**.

**Redação no TCC II:** apresentar os dois níveis de ortogonalidade explicitamente. A
afirmação de ortogonalidade estrita vale para A e B; a plena, só para A.

---

### D-013 · Imputação como estágio anterior ao `ColumnTransformer`
**Status:** CONFORME (ordenação imposta pela §4.4)
**Origem:** §6.3 fase 3 (R2/R3) · §4.4 e §4.5 da spec

A §4.5 descreve o `KNNImputer` dentro de cada bloco do `ColumnTransformer`. A §4.4 exige
que as compostas sejam construídas **após** a imputação, já que dependem de
`has_autorrelato` e `colesterol_alto` imputados. As duas não são simultaneamente
satisfazíveis: uma composta não pode ser coluna de entrada de um `ColumnTransformer` que
ainda vai imputar seus próprios componentes.

**Decisão:** a imputação vira o primeiro estágio do `Pipeline` (`ImputadorPNS`), seguida
de `ConstrutorCompostas` e, por último, do `ColumnTransformer`, que seleciona o conjunto
e aplica escala, passthrough e one-hot. O método permanece o KNN ajustado só no treino —
R2 e R3 são cumpridos integralmente, porque tudo roda dentro do mesmo `Pipeline` e o
`fit` só vê o treino.

**Efeito colateral deliberado:** o imputador roda sobre **todas** as preditoras cruas, não
só as do conjunto escolhido. É necessário: o conjunto B não tem `colesterol_alto` como
coluna própria, mas precisa dela imputada para calcular `multimorbidade`.

**Redação no TCC II:** nenhuma justificativa adicional necessária — é a forma concreta de
executar o que a §6.3 fase 3 estabelece.

---

### D-014 · Espaço métrico controlado na imputação por KNN
**Status:** CONFORME (correção de implementação)
**Origem:** §6.3 fase 3 (R3)

O `KNNImputer` seleciona vizinhos por distância euclidiana. Aplicado às escalas
brutas, como no baseline e na primeira versão de `features.py`, o resultado é
dominado pelas variáveis de maior amplitude numérica. Medido no conjunto de treino:

| Variável | Participação na distância (escala bruta) | Após padronização |
|---|---|---|
| `pas` | 38,2% | 16,7% |
| `idade` | 27,8% | 16,7% |
| `circ_cintura` | 18,0% | 16,7% |
| `imc` | 2,7% | 16,7% |
| `rcest` | < 0,001% | 16,7% |

`rcest` não alterava a ordenação dos vizinhos, e `imc` quase não. A seleção era
artefato da unidade de medida, não de similaridade clínica. As nominais agravavam o
problema: a distância entre Norte (1) e Sul (4) não é três vezes a distância entre
Norte e Nordeste (2), e nenhuma dessas diferenças significa coisa alguma.

**Correção.** O espaço métrico passa a ser as contínuas **padronizadas** por um
`StandardScaler` interno ao `ImputadorPNS`, ajustado só no treino (R2), mais
`sexo_fem` — a única categórica admitida, por ser binária, a única codificação em que
a distância tem significado direto. `raca_cor`, `regiao`, `has_autorrelato`,
`escolaridade` e `saude_autoaval` saem da métrica e **permanecem como alvos** de
imputação: não votam na escolha dos vizinhos, mas continuam sendo imputadas por KNN,
em conformidade com R3.

Após imputar, `inverse_transform` devolve as contínuas à escala original — as
compostas da §4.4 comparam `imc` a 30 e `circ_cintura` a 80/94 cm, cortes clínicos em
unidades reais. A padronização é interna à imputação e não vaza: o `StandardScaler` do
`ColumnTransformer` permanece inalterado.

**Implementação.** Uma imputação por coluna categórica alvo, cada uma sobre o mesmo
espaço métrico acrescido da própria coluna alvo. Onde a alvo está ausente — exatamente
as linhas a imputar — a dimensão é descartada pelo `nan_euclidean`, de modo que a
distância efetiva é só a do espaço declarado. O `KNNImputer` do scikit-learn é usado
sem reimplementação.

**Efeito medido.** A imputação muda: 6,1% dos 1.048 valores de `colesterol_alto` e
32,3% dos 223 de `has_autorrelato` no treino recebem código diferente do que a versão
anterior atribuía.

**Redação no TCC II:** nenhuma justificativa adicional necessária. A §6.3 fase 3
estabelece o `KNNImputer`; padronizar antes de medir distância é pré-requisito do
método, não escolha discricionária. Vale uma nota de rodapé na descrição da imputação
explicitando que as nominais ficam fora da métrica e por quê.

---

### D-015 · Escopo do `pipeline[:-1]` no cálculo do VIF
**Status:** CONFORME (precisão de implementação)
**Origem:** §6.5 · §5 da spec v2 · D-002

A §5 da spec prescreve `pipeline[:-1].transform(X_train)` para obter a matriz de design.
A expressão pressupõe um pipeline de modelagem, cujo último estágio é o estimador. Na
etapa 3 ainda não há estimador: o objeto disponível é o preprocessador de três estágios,
e nele `[:-1]` descartaria justamente o `ColumnTransformer` — devolvendo a saída de
`ConstrutorCompostas`, isto é, as preditoras **antes** da codificação, sem nenhuma dummy.

O VIF calculado ali seria o diagnóstico incompleto do baseline: exatamente a falha que
D-002 corrige.

**Decisão:** `matriz_de_design` descarta o último estágio **somente** quando ele é um
estimador (detectado por `hasattr(estagio, "predict")`). Com um preprocessador, aplica
todos os estágios. As duas formas produzem matrizes idênticas, verificado por teste.

**Redação no TCC II:** nenhuma justificativa adicional necessária. A §6.5 pede VIF sobre
a matriz de design; isto garante que seja ela, e não uma matriz intermediária.

---

### D-016 · Resultados de VIF divergem da expectativa da §5
**Status:** DIVERGÊNCIA DE EXPECTATIVA — resultado a reportar
**Origem:** §6.5 · §5 da spec v2

A §5 da spec registra a expectativa "VIF elevado em C por redundância deliberada;
residual em B por coarsening antropométrico; baixo em A", explicitamente marcada como
"a verificar, não a presumir". A verificação, no treino (n = 6.760):

| Conjunto | Colunas | VIF máximo | VIF mediano | Acima de 10 |
|---|---|---|---|---|
| `A_atomico` | 20 | 3,51 | 1,40 | 0 |
| `B_composto` | 16 | 3,92 | 1,40 | 0 |
| `C_completo` | 23 | 14,03 | 1,62 | 2 |

**Duas divergências, nenhuma corrigida no código:**

1. **Em C, o VIF acima do limiar não vem da redundância que D-001 aponta.** As duas
   colunas sinalizadas são `rcest` (14,03) e `circ_cintura` (13,73) — o par
   antropométrico. As colunas que carregam a duplicação de diagnóstico ficaram abaixo do
   limiar: `multimorbidade` 8,60, `has_combinada` 6,03, `has_autorrelato_3` 2,85,
   `colesterol_sim` 2,52. A redundância de canal existe e é visível na elevação relativa
   (`multimorbidade` 8,60 em C contra 2,01 em B), mas não cruza VIF > 10.

2. **Em B, a redundância por coarsening não se distingue de A em magnitude.** VIF máximo
   3,92 contra 3,51 de A, medianas idênticas. O efeito do coarsening antropométrico
   sobre `circ_cintura` é de 0,41 no VIF — presente, porém marginal.

**A redundância do escore é total, e o VIF mede menos da metade dela.** Duas regressões
auxiliares sobre `multimorbidade`, no mesmo conjunto C e no mesmo treino:

| Regressão auxiliar de `multimorbidade` | Regressores | R² |
|---|---|---|
| Identidade algébrica da §4.4 | `has_combinada`, `colesterol_sim`, indicadora de obesidade | **1,000000** |
| Auxiliar do VIF | demais colunas da matriz de design | **0,883778** (VIF 8,60) |

O primeiro R² não é estimativa: é a identidade da §4.4 escrita como regressão. A soma das
três parcelas reproduz o escore linha a linha nas 6.760 observações de treino, verificado
por teste. O escore não carrega **nenhuma** informação ausente de suas parcelas.

O segundo R² é menor porque a terceira parcela — a indicadora de obesidade — não é coluna
do modelo. A regressão auxiliar só dispõe de `imc` e `circ_cintura` **contínuas**, e o
limiar que as binariza (30 kg/m², 80/94 cm) não é recuperável por combinação linear.

O VIF, portanto, subestima a redundância por construção: reporta 8,60 onde a dependência é
determinística e o VIF verdadeiro seria infinito. Não é falha de cálculo — é o limite do
diagnóstico linear diante de uma composição por ponto de corte. Implementado em
`colinearidade.decomposicao_do_escore`.

**Implicação para a argumentação:** o VIF sozinho **não** sustenta a escolha de A ou B
sobre C pelo critério da §6.5, já que só sinaliza o par `rcest`/`circ_cintura`. O
argumento contra a redundância de canal permanece o Princípio 4 da §6.3.2 —
interpretabilidade do SHAP, não instabilidade numérica detectável por VIF. A remoção de
`rcest` em A e B, por outro lado, tem lastro empírico direto: é a única violação de
limiar observada.

**Redação no TCC II:** reportar a tabela como está. O resultado negativo é legítimo e
deve constar: a ablação A/B/C se justifica por parcimônia e interpretabilidade, com o
VIF confirmando apenas a colinearidade antropométrica. Não ajustar o limiar nem o
código para produzir o esperado.

O par R² = 1,000 / R² = 0,884 é o argumento a levar à banca caso se questione por que C
foi descartado com VIF abaixo do limiar: a redundância é total e demonstrável por
identidade algébrica; o VIF apenas não tem como vê-la inteira.

---

### D-017 · R² da regressão auxiliar reportado ao lado do VIF
**Status:** CONFORME (leitura do diagnóstico)
**Origem:** §6.5

O VIF isolado subestima a percepção da redundância. `multimorbidade` em C tem VIF 8,60 —
abaixo do limiar de 10, portanto não sinalizado — o que corresponde a **R² = 0,884**: 88,4%
da variância do escore já é explicada pelas demais colunas da matriz de design.

A tabela comparativa passa a trazer a coluna `r2_auxiliar = 1 - 1/VIF`. Mesma informação,
escala mais legível para o texto: "88% de variância redundante" comunica o que "VIF 8,60"
esconde.

**Redação no TCC II:** apresentar as duas colunas juntas. O limiar de 10 permanece o
critério formal da §6.5; o R² é a leitura que sustenta a discussão.

---

### D-018 · Coeficientes de Pearson reportados em tabela
**Status:** CONFORME (lacuna de implementação corrigida)
**Origem:** §6.5 · D-003

A §6.5 exige Pearson **e** VIF como resultado. Entregar apenas o heatmap deixa o
coeficiente inacessível ao texto — não se cita um pixel. `pares_pearson` converte a matriz
na lista de pares ordenada por |r|, exportada em `reports/pearson_coeficientes.tex` e
`.csv` junto das figuras.

Coeficientes no treino, conjunto C (n = 6.760):

| Par | Pearson |
|---|---|
| `rcest` × `circ_cintura` | **0,916** |
| `imc` × `rcest` | 0,799 |
| `imc` × `circ_cintura` | 0,793 |
| `pas` × `pad` | 0,743 |
| `rcest` × `multimorbidade` | 0,636 |
| `idade` × `imc` | 0,137 |

O bloco antropométrico é onde a colinearidade se concentra — coerente com o VIF, que só
sinalizou `rcest` e `circ_cintura`.

---

### D-019 · Ablação de `rcest` — execução do protocolo da §6.5
**Status:** EXECUÇÃO DE PROTOCOLO
**Origem:** §6.5

A §6.5 autoriza remover experimentalmente a variável de maior VIF dentro de um bloco
temático e recalcular o diagnóstico. Executado sobre `rcest` em `C_completo`
(`colinearidade.ablacao_vif`), no treino:

| Coluna | VIF antes | VIF depois | Δ | R² antes | R² depois |
|---|---|---|---|---|---|
| `circ_cintura` | 13,73 | **4,70** | −9,03 | 0,927 | 0,787 |
| `multimorbidade` | 8,60 | 8,60 | −0,0002 | 0,884 | 0,884 |
| `has_combinada` | 6,03 | 6,03 | −0,0003 | 0,834 | 0,834 |
| `imc` | 3,39 | 3,12 | −0,26 | 0,705 | 0,680 |

**Resultado:** sem `rcest`, **nenhuma coluna de C cruza o limiar**. A remoção resolve
integralmente a única violação de VIF observada, e o efeito é cirúrgico — restrito ao
bloco antropométrico. `multimorbidade`, `has_combinada` e as nominais não se movem: a
redundância de canal que D-001 aponta é independente da colinearidade antropométrica.

Confirma, por dois caminhos distintos, o que D-016 registrou: o VIF diagnostica o par
`rcest`/`circ_cintura`, e nada além disso. A justificativa para A e B continua sendo o
Princípio 4, não o VIF.

**Redação no TCC II:** apresentar a tabela como execução do protocolo da §6.5. A exclusão
de `rcest` de A e B deixa de ser escolha de projeto e passa a ter lastro empírico direto.

---

### D-020 · Ramo defensivo da seleção de limiar
**Status:** CONFORME (precisão de implementação)
**Origem:** §6 da spec · D-004

A spec determina: se nenhum limiar atingir o recall-alvo, devolver o de maior recall e
sinalizar. Implementado — com uma constatação que vale registrar: em
`precision_recall_curve` o menor limiar classifica todas as observações como positivas,
logo o recall máximo da curva é sempre 1,0. **Nenhum alvo válido cai nesse ramo.**

O fallback permanece como defesa contra curvas degeneradas, coberto por teste com a curva
substituída. Já `recall_alvo` fora de (0, 1] passou a levantar `ValueError`: é erro de
chamada, e transformá-lo em aviso deixaria passar um ponto de operação sem sentido.

**Redação no TCC II:** nenhuma. Detalhe de implementação, registrado para que a ausência
do aviso nos resultados não seja lida como omissão.

---

### D-021 · Dispersão entre folds reportada ao lado das métricas de teste
**Status:** CONFORME (leitura do resultado)
**Origem:** §6.3 fase 5 · §6 da spec

A tabela comparativa passa a trazer, para cada métrica assinada, a média e o desvio
padrão entre os cinco folds da validação cruzada, ao lado do valor de teste. Sem a
dispersão não há escala para julgar se a diferença entre conjuntos é sinal ou variação
de partição.

Recall no teste (LogReg, limiar calibrado por CV no treino):

| Conjunto | Recall (teste) | Recall CV (média ± dp) |
|---|---|---|
| `A_atomico` | 0,748 | 0,802 ± 0,027 |
| `B_composto` | 0,765 | 0,802 ± 0,008 |
| `C_completo` | 0,756 | 0,802 ± 0,026 |

**Consequência direta para a interpretação:** a amplitude entre os três conjuntos no
teste é de 0,017 em recall, contra desvio padrão de até 0,027 entre folds do mesmo
conjunto. **A diferença entre A, B e C está dentro da variação de partição.** O mesmo
vale para especificidade (amplitude 0,012; dp até 0,023), F1 (0,009; 0,011) e AUC-ROC
(0,011; 0,015).

**Redação no TCC II:** não afirmar superioridade de nenhum conjunto com base nesses
números. A escolha entre A, B e C se decide por parcimônia e interpretabilidade
(Princípio 4), não por desempenho — que é estatisticamente indistinguível. Reportar
sempre com a dispersão; a tabela sem ela convidaria à leitura errada.

---

### D-022 · Probabilidades não interpretáveis como risco individual
**Status:** EXTRAPOLAÇÃO — limitação a declarar no texto
**Origem:** §5 (Cost-Sensitive Learning) · §6.3 fase 5

**Hipótese verificada e confirmada.** Com `class_weight='balanced'` (R4), o modelo é
ajustado como se as classes fossem equilibradas. O resultado é um deslocamento
sistemático de nível nas probabilidades:

| Conjunto | Brier | Brier trivial | Brier skill | Prevalência | Prob. média | Inflação | ECE | Slope | Intercepto |
|---|---|---|---|---|---|---|---|---|---|
| `A_atomico` | 0,1922 | 0,0654 | −1,94 | 0,0704 | 0,3935 | 5,59× | 0,323 | 0,873 | −2,571 |
| `B_composto` | 0,1959 | 0,0654 | −2,00 | 0,0704 | 0,3987 | 5,67× | 0,328 | 0,920 | −2,589 |
| `C_completo` | 0,1927 | 0,0654 | −1,95 | 0,0704 | 0,3926 | 5,58× | 0,322 | 0,836 | −2,561 |

O Brier de ~0,19 contra 0,065 de um previsor constante na prevalência **não indica
modelo ruim**: indica probabilidades na escala errada. Duas evidências separam as duas
leituras:

1. **Slope de calibração entre 0,84 e 0,92** — próximo de 1. A *ordenação* das
   probabilidades é boa; a discriminação (AUC-ROC ≈ 0,78) está preservada.
2. **Intercepto entre −2,561 e −2,589**, contra logito da prevalência de treino
   `log(0,0701 / 0,9299) = −2,585`. O deslocamento reproduz, com três casas, exatamente
   o reponderamento de classes. Não é ruído nem má especificação: é o efeito algébrico
   previsível do `class_weight='balanced'`.

A curva de calibração por decis confirma: até o oitavo decil a frequência observada fica
entre 0,00 e 0,07 enquanto a probabilidade média sobe de 0,08 a 0,58; só no décimo decil
o observado chega a 0,22, contra 0,82 predito.

**Consequência — e é uma restrição de uso, não uma nota técnica:**

> As probabilidades produzidas pelo modelo **não são interpretáveis como risco
> individual**. Um valor de 0,60 não significa 60% de chance de HbA1c ≥ 6,5%: a
> frequência observada nessa faixa é de aproximadamente 7%. As saídas são **escores de
> ordenação**, válidos para priorizar quem encaminhar a exame, e o ponto de operação é o
> limiar calibrado — não a probabilidade.

Qualquer interface que venha a exibir a saída ao profissional de saúde (§6.3 fase 6,
FastAPI/Streamlit) precisa apresentar a decisão de triagem ou o escore relativo, jamais a
probabilidade bruta rotulada como risco.

**Caminho de correção, fora do escopo da Sprint 1:** calibração posterior por Platt
scaling ou isotônica, ajustada em partição de validação separada do treino, nunca no
teste. Entra no backlog de calibração probabilística já previsto na §6 da spec.

**Redação no TCC II:** declarar explicitamente na seção de resultados e na de limitações.
O par slope ≈ 1 / intercepto ≈ logito da prevalência é a evidência a apresentar: mostra
que o modelo discrimina bem e que o problema é de escala, corrigível por calibração.

---
---

### D-023 · Comparadores não-ML avaliados com limiar fixo
**Status:** CONFORME (precisão de implementação)
**Origem:** §7 da spec · D-006

`RegraClinicaADA` e `DummyClassifier` devolvem indicadoras em {0, 1}, não escores.
Calibrar limiar sobre elas é operação sem significado — uma curva precision-recall de
dois pontos não define ponto de operação, e a regra **é** o ponto de operação.

No Dummy o efeito seria pior que inócuo: a calibração encontraria o limiar 0,0 e
converteria "prevê sempre a maioria" em "prevê sempre positivo", com recall 1,0 —
destruindo exatamente o piso que ele existe para marcar. Verificado por teste.

**Decisão:** `evaluate.avaliar` recebe `limiar` opcional. Quando informado, dispensa a
calibração e registra `limiar_calibrado=False` na tabela, distinguindo os dois regimes.
Os comparadores usam `LIMIAR_INDICADORA = 0,5`.

**Limitação a declarar:** para a regra ADA, **AUC-ROC (0,656) e PR-AUC ficam degradadas
por construção**, não por desempenho: com um único corte, a curva é uma quebra. Brier e
calibração não têm sentido sobre 0/1 categórico. As métricas comparáveis são Recall,
Especificidade e NNS.

---

### D-024 · Critério de sucesso da §7 depende do ponto de operação
**Status:** EXTRAPOLAÇÃO — resultado a reportar
**Origem:** §7 da spec · D-006

A §7 define: superar a regra clínica em Recall **com NNS menor**. Aplicado diretamente,
o critério dá **negativo** — e a razão não é o desempenho dos classificadores.

**Comparação no ponto de operação padrão** (`RECALL_ALVO = 0,80`), teste n = 1.691:

| Estratégia | Encaminhados | Taxa | Detectados | Perdidos | Recall | NNS |
|---|---|---|---|---|---|---|
| LogReg · B_composto | 661 | 39,1% | 91 | 28 | 0,765 | **7,26** |
| LogReg · C_completo | 679 | 40,2% | 90 | 29 | 0,756 | 7,54 |
| LogReg · A_atomico | 675 | 39,9% | 89 | 30 | 0,748 | 7,58 |
| **Regra ADA** | 974 | 57,6% | 103 | 16 | **0,866** | 9,46 |
| Dummy (maioria) | 0 | 0% | 0 | 119 | 0,000 | ∞ |

Nenhum modelo domina: a regra tem recall maior (0,866), o modelo tem NNS menor (7,26).
**Mas os dois operam em pontos diferentes da curva recall × precisão**, e comparar NNS
entre recalls diferentes não responde nada — qualquer classificador com escore contínuo
compra recall pagando em precisão.

**Comparação em pé de igualdade.** O recall da regra **no treino** é 0,8945
(`benchmark.recall_da_regra_no_treino` — medido no treino, nunca no teste, sob pena de
vazamento). Recalibrando o modelo para esse alvo:

| Estratégia | Recall-alvo (treino) | Recall obtido (teste) | Recall da regra | Encaminhados | Detectados | NNS | Δ NNS | Supera |
|---|---|---|---|---|---|---|---|---|
| LogReg · B_composto | 0,8945 | 0,9160 | 0,8655 | 1.000 | 109 | 9,17 | −0,28 | **sim** |
| LogReg · A_atomico | 0,8945 | 0,8992 | 0,8655 | 986 | 107 | 9,22 | −0,24 | **sim** |
| LogReg · C_completo | 0,8945 | 0,8739 | 0,8655 | 946 | 104 | 9,10 | −0,36 | **sim** |
| Regra ADA | — | 0,8655 | — | 974 | 103 | 9,46 | — | — |

> **Nota obrigatória sob esta tabela.** A equiparação de recall ocorre **no treino**, por
> construção: o limiar do modelo é calibrado para o recall que a regra obtém no treino
> (0,8945). O valor realizado no teste desvia desse alvo, e o desvio é **esperado** — o
> limiar é fixado fora da amostra de avaliação, sob pena de vazamento. Os recalls no teste
> **não são iguais** entre modelo e regra, e não deveriam ser: o que está equiparado é o
> regime de operação, não o resultado. O NNS permanece a métrica de comparação.

No mesmo regime de recall, os três conjuntos cumprem o critério: mais casos detectados
com menos encaminhamentos por caso.

**A margem é pequena, e isso precisa ser dito.** A redução de NNS fica entre 0,24 e 0,36
— cerca de 3%. Em termos operacionais: para detectar 6 casos a mais que a regra
(109 contra 103), B_composto encaminha 26 pessoas a mais (1.000 contra 974). O ganho
existe e é consistente nos três conjuntos, mas não é a diferença entre triar e não triar.

**Redação no TCC II:** apresentar as duas tabelas. A primeira mostra por que a comparação
ingênua não decide; a segunda, o resultado sob comparação válida. Declarar a magnitude do
ganho honestamente — o argumento a favor do modelo é a **flexibilidade do ponto de
operação**, que a regra não tem: a regra entrega um único par (recall, NNS), enquanto o
modelo percorre a curva conforme a capacidade instalada da UBS.

**Pendência:** a comparação depende de `RECALL_ALVO`, ainda provisório (D-004). Confirmar
com o orientador qual ponto de operação reportar como principal.

---

### D-025 · Apresentação da comparação e das métricas não interpretáveis
**Status:** CONFORME (correção de apresentação)
**Origem:** §7 da spec · D-023 · D-024

Três correções de reporte. Nenhum cálculo, limiar ou partição foi alterado.

**1. Recall-alvo e recall obtido em colunas separadas.** A tabela da comparação
equiparada exibia o modelo em 0,916 contra 0,866 da regra sob o rótulo "recall
equiparado". Um leitor atento nota que os valores não são iguais — e a objeção que ele
levanta é exatamente a que motivou a recalibração, o que desmonta o argumento em vez de
sustentá-lo. `supera_a_regra_clinica` passa a devolver `recall_alvo` (0,8945, medido no
treino, idêntico para os três conjuntos), `recall_obtido` (teste) e `recall_regra`, com a
nota de equiparação anexada em `tabela.attrs["nota"]`. Comparadores de limiar fixo têm
`recall_alvo` vazio, não o `RECALL_ALVO` herdado por default.

**2. Nota de rodapé sobre métricas dependentes de curva.** A advertência de D-023 existia
apenas na docstring de `predict_proba` — invisível para quem lê a tabela consolidada, que
é onde o erro de leitura acontece: AUC-ROC 0,656 da regra ao lado de 0,781 do modelo
parecem a mesma métrica. `evaluate.comparar` ganha a coluna `corte_unico`, **detectada
pela saída** (no máximo dois valores distintos), não pelo tipo do estimador; e
`exportar_latex` gera a nota automaticamente quando a tabela tem tais linhas. Suprimir
exige `nota=""` explícito.

**3. A taxa de encaminhamento enunciada.** `benchmark.leitura_operacional` escreve o
argumento que a coluna `taxa_encaminhamento` continha sem enunciar — ver D-026.

**Redação no TCC II:** a nota de equiparação e a de corte único acompanham as respectivas
tabelas no texto. Não são detalhes de implementação: sem elas, ambas as tabelas induzem à
leitura errada.

---

### D-026 · Taxa de encaminhamento como argumento de aplicabilidade
**Status:** EXTRAPOLAÇÃO — resultado a reportar
**Origem:** §7 da spec · D-024

O número estava na tabela de carga operacional desde a etapa 5, mas não enunciado em
lugar nenhum — e é ele que decide a aplicabilidade na Atenção Primária:

> **A regra ADA encaminha 974 de 1.691 pessoas — 57,6% da amostra — para detectar 103
> casos. O modelo, no ponto de operação padrão, encaminha 661 (39,1%) para detectar 91.**

Uma triagem que manda mais da metade da população adulta ao laboratório **não seleciona
quem examinar**: apenas desloca a fila para a etapa seguinte. O propósito declarado do
trabalho — priorizar quem encaminhar sob capacidade instalada limitada — é justamente o
que a regra atual não entrega.

Isto reposiciona a conclusão de D-024. Naquele registro, o ganho do modelo aparece como
redução marginal de NNS (≈3%) sob recall equiparado, o que é honesto mas subestima o
argumento. A leitura completa é outra:

| Regime | Encaminhados | Detectados | Leitura |
|---|---|---|---|
| Regra ADA | 974 (57,6%) | 103 | ponto único, não ajustável |
| Modelo, recall equiparado | 1.000 (59,1%) | 109 | mesmo regime, +6 casos |
| Modelo, ponto padrão | 661 (39,1%) | 91 | **−32% de encaminhamentos**, −12 casos |

A vantagem do modelo não é fazer o mesmo um pouco melhor: é **poder escolher o ponto de
operação** conforme a capacidade da rede, o que a regra por definição não permite. A
troca é explícita e quantificada — 313 encaminhamentos a menos ao custo de 12 casos não
detectados, ou o inverso.

**Cuidado de leitura entre as duas tabelas.** No regime equiparado o modelo encaminha
**mais** gente que a regra em números absolutos: 1.000 contra 974. Lida isoladamente, a
tabela de carga operacional sugere desvantagem. A resposta está no recall realizado —
0,916 contra 0,866 — e é o que o NNS já incorpora, por normalizar pelos casos detectados:
26 encaminhamentos a mais para 6 casos a mais é uma troca favorável, e é por isso que o
NNS cai de 9,46 para 9,17. Volume absoluto só é comparável entre estratégias que operam
no mesmo recall.

`benchmark.leitura_operacional(padrao, equiparados=...)` emite essa linha junto da leitura
de aplicabilidade, e `tabela_nns` carrega a advertência em `attrs["nota"]`, de modo que
nenhuma das duas tabelas circule sem ela.

**Redação no TCC II:** enunciar a taxa de encaminhamento junto do NNS sempre que a
comparação com a regra for apresentada. A frase "encaminha 57,6% da população adulta" faz
o trabalho argumentativo que "NNS 9,46" não faz, para leitor clínico ou de banca.

**Pendência:** a escolha do ponto de operação a reportar como principal (D-004, D-024)
deixa de ser detalhe técnico e passa a ser a decisão central do trabalho. Levar ao
orientador com estas três linhas na mão.


## Números da base — referência para as decisões acima

Medidos em 16/09/2026, `random_state=42`. Reprodução: `pytest tests/ -v`.

| Item | Valor |
|---|---|
| Linhas no CSV bruto | 8.952 |
| Excluídas: HbA1c ausente (R3) | −411 |
| Excluídas: bloco de medição 100% ausente | −90 |
| **Amostra analítica** | **8.451** · 593 positivos · prevalência 0,070169 |
| Treino / teste (80/20 estratificado) | 6.760 (474 pos., 0,070118) / 1.691 (119 pos., 0,070373) |
| Ausentes na base / no treino | `colesterol_alto` 1.297/1.048 · `has_autorrelato` 261/223 · `regiao` 4/3 · `raca_cor` 2/1 |
| Contínuas com ausentes | nenhuma |

---

## Três leituras que os números NÃO autorizam

São as inferências mais naturais a partir das tabelas deste documento, e as três estão
erradas. Registradas juntas porque cada uma é refutada por uma decisão diferente.

**1. "O VIF justifica descartar C."** As colunas acima do limiar em C são `rcest` e
`circ_cintura`, ambas antropométricas. As que carregam a redundância de canal —
`multimorbidade` 8,60, `has_combinada` 6,03, `colesterol_sim` 2,52 — ficam abaixo de 10.
O argumento é o Princípio 4, e a evidência é o par R² 1,000 / 0,884. (D-016, D-019)

**2. "B_composto é o melhor conjunto."** A vantagem de B em recall (0,765 contra 0,748 de
A) é menor que o desvio padrão entre folds do próprio A (0,027). A escolha se decide por
parcimônia e interpretabilidade. (D-021)

**3. "A probabilidade do modelo é o risco do paciente."** Um valor de 0,60 corresponde a
frequência observada de ~7%. São escores de ordenação; a decisão é o limiar. (D-022)

---

## Índice e pendências

| ID | Assunto | Status |
|---|---|---|
| D-001 | Ablação de conjuntos A/B/C | Execução de protocolo |
| D-002 | VIF sobre a matriz de design pós-transformação | Conforme |
| D-003 | Correlação de Pearson entre contínuas | Conforme |
| D-004 | Seleção de limiar por Recall-alvo | Extrapolação |
| D-005 | Métricas complementares (PR-AUC, Brier, NNS) | Extrapolação |
| D-006 | Benchmark por regra clínica ADA | Extrapolação |
| D-007 | Agrupamento de `raca_cor` | Extrapolação — não implementada |
| D-008 | Categoria "não sabe" para `colesterol_alto` | Divergência — variante comparada |
| D-009 | `peso_lab` fora do treinamento | Pendência de redação |
| D-010 | `peso_kg` e `altura_cm` fora das preditoras | Conforme |
| D-011 | Contagem de ausentes de `colesterol_alto` | Conforme — corrigir `CLAUDE.md` |
| D-012 | `has_combinada` fora de B; invariantes em dois níveis | Execução de protocolo |
| D-013 | Imputação antes do `ColumnTransformer` | Conforme |
| D-014 | Espaço métrico controlado no KNN | Conforme |
| D-015 | Escopo do `pipeline[:-1]` no VIF | Conforme |
| D-016 | VIF diverge da expectativa da §5 | Divergência de expectativa |
| D-017 | R² auxiliar ao lado do VIF | Conforme |
| D-018 | Coeficientes de Pearson em tabela | Conforme |
| D-019 | Ablação de `rcest` | Execução de protocolo |
| D-020 | Ramo defensivo da seleção de limiar | Conforme |
| D-021 | Dispersão entre folds | Conforme |
| D-022 | Probabilidades não interpretáveis como risco | Extrapolação — limitação a declarar |
| D-023 | Comparadores não-ML com limiar fixo | Conforme |
| D-024 | Critério da §7 depende do ponto de operação | Extrapolação — resultado a reportar |
| D-025 | Apresentação da comparação e das métricas de curva | Conforme |
| D-026 | Taxa de encaminhamento como argumento de aplicabilidade | Extrapolação — resultado a reportar |

**Antes da pré-banca (05/10/2026):**

1. Confirmar `RECALL_ALVO = 0,80` com o orientador — alteração obriga a regerar figuras e tabelas (D-004)
2. Redigir por que a ponderação amostral não se aplica ao treinamento (D-009)
3. Corrigir 1.319 → 1.297 no `CLAUDE.md` (D-011)
4. Corrigir a §4.2 da spec: ortogonalidade plena vale para A; estrita, para A e B (D-012)
5. Corrigir a §5 da spec: `pipeline[:-1]` só se aplica a pipeline com estimador (D-015)
6. Declarar a limitação de calibração nos resultados e nas limitações (D-022)
7. Reavaliar agrupamento de `raca_cor` com IC bootstrap na etapa 6 (D-007)
8. Definir com o orientador qual ponto de operação reportar como principal na comparação com a regra ADA (D-024)

**Artefatos:** `reports/` — `vif_comparativo.{tex,csv}` · `vif_resumo.tex` ·
`pearson_coeficientes.{tex,csv}` · `vif_ablacao_rcest.tex` · `comparativo_logreg.tex`.
`reports/figures/` — `pearson_{A,B,C}.png` · `confusao_*` · `roc_*` · `pr_*` ·
`calibracao_*`.


## Sprint 2 — Random Forest

*A preencher.*

## Sprint 3 — SHAP

*A preencher.*

Nota antecipada: a §6.5 remete explicitamente à fase de implementação do TCC II a
definição das estratégias de mitigação do impacto da multicolinearidade sobre a
explicabilidade via SHAP. Essa decisão será registrada aqui quando tomada.

---

## Itens fechados

*Nenhum até o momento.* Mover para cá as decisões já redigidas no LaTeX, com referência
à seção correspondente do TCC II.