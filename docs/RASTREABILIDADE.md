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