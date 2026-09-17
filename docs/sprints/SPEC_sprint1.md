# SPEC — Sprint 1 · hba1c-screening-pns2013

> **Documento encerrado — 16/09/2026.** Sprint cumprida. O corpo abaixo é preservado
> como está: ele registra o que foi **efetivamente especificado**, e reescrevê-lo para
> refletir o repositório de hoje apagaria esse registro.
>
> Alguns caminhos citados no corpo mudaram na reorganização posterior (D-027 a D-031).
> A correspondência é esta:
>
> | No corpo | Hoje |
> |---|---|
> | `reports/figures/` | `reports/figuras/` |
> | `reports/*.tex`, `reports/*.csv` | `reports/tabelas/` |
> | `docs/SPEC_sprint1.md` | `docs/sprints/SPEC_sprint1.md` (este arquivo) |
>
> Mudou também **onde os resultados são produzidos**: a spec não previa ponto único de
> reprodução, e os números da Sprint 1 acabaram gerados por código de sessão que não
> sobreviveu à sessão. `src/experimento.py` e `notebooks/03_resultados.ipynb` fecham essa
> lacuna e são posteriores a este documento (D-027).
>
> Estado atual do projeto: `docs/ESTADO.md`. Decisões e sua origem: `docs/RASTREABILIDADE.md`.

**Versão 2** — incorpora D-012, D-013, D-014 e a correção de contagem de ausentes.

**Objetivo:** refatorar o notebook baseline em módulos testáveis, corrigir violações do
Princípio 4 de seleção de variáveis e construir o harness de avaliação reutilizado por
todos os modelos.

**Prazo:** 15–21/09/2026 · **Marco:** pré-banca 05/10/2026
**Metodologia de referência:** Projeto de TCC I assinado, seções 6.3 a 6.5.

Toda decisão tem rastreabilidade para a seção de origem. Decisões sem lastro no TCC I
estão isoladas na §10 e exigem redação no TCC II.

### Registro de alterações da v2

| ID | Alteração | Seção |
|---|---|---|
| D-012 | `has_combinada` removida de `B_composto`; invariante de ortogonalidade reformulado | §4.2 |
| D-013 | Imputação migra do `ColumnTransformer` para o primeiro estágio do `Pipeline` | §4.5 |
| D-014 | Espaço métrico do KNN restrito a contínuas padronizadas | §4.5 |
| — | `colesterol_alto`: 1.297 ausentes (≈15,4%), não 1.319 | §3, §10 |

---

## 0. Restrições não negociáveis

| # | Regra | Origem |
|---|---|---|
| R1 | Nenhuma estatística entre linhas antes do split. Divisão 80/20 estratificada pelo alvo | §6.3 fase 3 |
| R2 | Imputação, escalonamento e codificação: `fit` só no treino, `transform` no teste | §6.3 fase 3 |
| R3 | Ausentes das variáveis classificatórias tratados por **`KNNImputer`**, ajustado só no treino. Registros sem HbA1c válida são **excluídos**, jamais imputados | §6.3 fase 3 |
| R4 | **SMOTE não é utilizado.** Cost-Sensitive Learning via `class_weight='balanced'` | §5, §6.3 fase 3 |
| R5 | Variáveis comportamentais permanecem excluídas — causalidade reversa em desenho transversal | §6.3.2, princípio 3 |
| R6 | Apenas variáveis **não invasivas**, rotineiramente capturadas pelo PEC/SISAB | §6.3.2, princípios 1 e 2 |
| R7 | Variáveis conceitualmente **ortogonais**. Redundância que dilua a interpretabilidade via SHAP deve ser eliminada | §6.3.2, princípio 4 |
| R8 | Validação por `StratifiedKFold(k=5)`. **Recall é a métrica prioritária** | §6.3 fase 5 |
| R9 | Diagnóstico de multicolinearidade: **correlação de Pearson entre contínuas + VIF**, limiar VIF > 10 | §6.5 |
| R10 | `peso_lab` pondera a **EDA descritiva**. **Não entra no treino.** Ver §9 | §6.3 fase 2 |
| R11 | `random_state=42` em toda operação estocástica | reprodutibilidade |

---

## 1. Estrutura do repositório

```
hba1c-screening-pns2013/
├── data/raw/                    # GITIGNORED
├── src/
│   ├── config.py                # constantes, DE_PARA, codificações, cortes clínicos
│   ├── data.py                  # carga, limpeza, exclusões (row-wise apenas)
│   ├── features.py              # imputação, compostas, conjuntos, preprocessador
│   ├── colinearidade.py         # Pearson + VIF pós-transformação
│   ├── evaluate.py              # harness de métricas, limiar, figuras
│   └── benchmark.py             # comparadores não-ML
├── notebooks/
│   ├── 01_eda.ipynb             # EDA ponderada por peso_lab
│   └── 02_modelagem.ipynb       # orquestra src/, não implementa
├── docs/
│   ├── SPEC_sprint1.md
│   ├── RASTREABILIDADE.md
│   └── baseline/                # notebook congelado, não editar
├── reports/figures/
├── tests/
├── CLAUDE.md
└── requirements.txt
```

---

## 2. `src/config.py` — IMPLEMENTADO

```python
RANDOM_STATE = 42
TEST_SIZE = 0.20
LIMIAR_HBA1C = 6.5          # critério ADA 2010 (§6.3.1)
CV_FOLDS = 5                # §6.3 fase 5
VIF_LIMIAR = 10             # §6.5
RECALL_ALVO = 0.80          # §10 item 3 — confirmar com orientador

PATH_RAW  # resolvido a partir de Path(__file__).parents[1]

DE_PARA = {...}             # 16 variáveis, códigos PNS -> nomes legíveis
COD_SEXO, COD_RACA, COD_REGIAO, COD_HAS, COD_COLEST

# Cortes clínicos (Sociedade Brasileira de Diabetes, 2019) — §6.4.1
CORTE_PAS, CORTE_PAD = 140, 90
CORTE_IMC_OBESIDADE = 30
CORTE_CINTURA_F, CORTE_CINTURA_M = 80, 94
```

---

## 3. `src/data.py` — IMPLEMENTADO

Somente operações **row-wise** (R1). Nenhuma imputação.

`carregar_bruto` · `limpar` · `separar` · `carregar_analitico`

**Contrato de saída, verificado na base real:**
```
len(df) == 8451
df.target.mean() == 0.07017
contínuas sem ausentes
ausentes: raca_cor=2, regiao=4, has_autorrelato=261, colesterol_alto=1297
idade ∈ [18, 101]
```

`peso_kg` e `altura_cm` permanecem no DataFrame limpo (a EDA precisa deles) mas ficam
fora de `X` — entram no modelo apenas via `imc` e `rcest`, conforme §6.4.1 (D-010).

---

## 4. `src/features.py`

### 4.1 O problema a corrigir

Na matriz do baseline, `has_autorrelato` entrava **três vezes** (one-hot +
`has_combinada` + `multimorbidade`) e `colesterol_alto` **duas vezes** (binária isolada +
`multimorbidade`).

Viola o **Princípio 4 da §6.3.2** — parcimônia estatística, ortogonalidade conceitual e
não-diluição da interpretabilidade via SHAP. Não é preferência de implementação: é não
conformidade com a metodologia assinada.

### 4.2 Conjuntos de atributos (ablação)

A §6.5 autoriza avaliar por experimentação o impacto da remoção de variáveis
correlacionadas do mesmo bloco temático. A ablação é a execução desse protocolo.

```python
FEATURE_SETS = {
    # A — decomposto: componentes atômicos, sem escores agregados.
    #     Ortogonalidade estrita. Máxima legibilidade do SHAP.
    "A_atomico": {
        "num": ["idade", "imc", "circ_cintura", "pas", "pad",
                "escolaridade", "saude_autoaval"],
        "bin": ["sexo_fem", "plano_sim"],
        "nom": ["raca_cor", "regiao", "has_autorrelato", "colesterol_alto"],
    },
    # B — agregado: o escore ordinal SUBSTITUI seus componentes de diagnóstico
    #     e pressão. Só assim a hipótese de sinergia da §6.4.1 é testável.
    "B_composto": {
        "num": ["idade", "imc", "circ_cintura",
                "escolaridade", "saude_autoaval", "multimorbidade"],
        "bin": ["sexo_fem", "plano_sim"],
        "nom": ["raca_cor", "regiao"],
    },
    # C — completo: as 16 variáveis conceituais do TCC I, com RCEst.
    #     Reproduz o baseline. Quantifica o custo da redundância.
    "C_completo": {
        "num": ["idade", "imc", "circ_cintura", "rcest", "pas", "pad",
                "escolaridade", "saude_autoaval", "multimorbidade"],
        "bin": ["sexo_fem", "plano_sim", "colesterol_sim", "has_combinada"],
        "nom": ["raca_cor", "regiao", "has_autorrelato"],
    },
}
```

#### Correção D-012

`has_combinada` foi removida de `B_composto`. Ela é a **primeira parcela literal** de
`multimorbidade`; mantê-la ao lado do escore reproduz em B exatamente a violação que a
§4.1 acusa no baseline, e impede o teste da hipótese da §6.4.1 — o modelo recuperaria a
parcela linearmente, esvaziando o propósito do agregado.

#### Invariantes — dois níveis distintos

A formulação original ("nenhuma variável fonte alimenta o modelo por dois canais em A e
B") era forte demais e B não tinha como cumpri-la. Substituída por dois predicados
implementados separadamente:

**Invariante estrito — `fontes_isoladas_e_agregadas`**
Nenhuma variável entra simultaneamente como coluna própria e como **parcela idêntica**
de um agregado.
Cumprido por **A e B**. Violado por **C**, deliberadamente.

**Invariante amplo — `fontes_em_canal_duplo`**
Nenhuma variável fonte contribui ao modelo por mais de um caminho, em qualquer forma.
Cumprido apenas por **A** (conjunto de sobreposição vazio).

**Sobreposição residual admitida em B — por *coarsening*:**
`imc`, `circ_cintura` e `sexo_fem` entram como colunas próprias e alimentam o componente
de obesidade de `multimorbidade` em forma **binarizada por ponto de corte clínico**. Não
é o mesmo canal: o gradiente contínuo carrega sinal que o limiar descarta. A magnitude
dessa redundância é quantificada pelo VIF na etapa 3, não presumida.

**Verificação por teste:** cada conjunto afirma explicitamente qual invariante cumpre.
`C_completo` tem teste que **falha se a redundância desaparecer** — ela é proposital.

### 4.3 Recodificações binárias

```python
sexo_fem       = (sexo == 2).astype(int)
plano_sim      = (plano_saude == 1).astype(int)
colesterol_sim = (colesterol_alto == 1).astype(int)
```

### 4.4 Variáveis compostas (§6.4.1)

```python
# HAS Combinada: pressão aferida >= 140/90 (SBD 2019) OU autorrelato afirmativo.
# Código 2 de Q002 ("apenas na gravidez") NÃO é HAS crônica.
has_combinada = (pas >= CORTE_PAS) | (pad >= CORTE_PAD) | (has_autorrelato == 1)

# Obesidade: disjuntiva e sensível ao sexo, contornando a limitação reconhecida
# do IMC isolado para adiposidade central.
obesidade = (imc >= 30) | (sexo_fem & (circ_cintura >= 80)) | (~sexo_fem & (circ_cintura >= 94))

# Multimorbidade Metabólica: escore ordinal 0–3.
multimorbidade = has_combinada + (colesterol_alto == 1) + obesidade
```

Construídas **após a imputação** e em **escala clínica original** — os cortes são
expressos em mmHg, kg/m² e cm.

Implementadas como transformador de `Pipeline` (`ConstrutorCompostas`), nunca como função
aplicada ao DataFrame completo.

### 4.5 Arquitetura do pipeline

#### Correção D-013 — imputação fora do `ColumnTransformer`

A v1 posicionava o `KNNImputer` dentro de cada bloco do `ColumnTransformer`. Isso é
incompatível com a §4.4: `multimorbidade` não pode ser coluna de entrada de um
transformador que ainda vai imputar `colesterol_alto`. Os dois requisitos não são
simultaneamente satisfazíveis.

A imputação passa a ser o **primeiro estágio do `Pipeline`**:

```
Pipeline([
    ("imputacao",  ImputadorPNS()),          # todas as 14 preditoras
    ("compostas",  ConstrutorCompostas()),   # escala clínica original
    ("preparo",    ColumnTransformer(...)),  # seleção + escala + one-hot
])
```

R2 e R3 permanecem íntegros: mesmo `Pipeline`, `fit` só vê o treino.

O imputador opera sobre **todas as 14 preditoras**, independentemente do conjunto ativo.
Necessário porque B precisa de `colesterol_alto` imputada sem tê-la como coluna — e
garante que A, B e C recebam base imputada idêntica, sem o que a comparação não seria
válida.

#### Correção D-014 — espaço métrico do KNN

Dois defeitos na versão anterior:

1. **Escalas brutas.** O `StandardScaler` estava no `ColumnTransformer`, portanto a
   jusante da imputação. A distância euclidiana era dominada por `pas` (~70–220) e
   `circ_cintura` (50–150), enquanto `rcest` (0,33–0,95) era praticamente invisível e
   `imc` pesava uma fração de `pas`. Os 1.297 ausentes de `colesterol_alto` estavam sendo
   imputados por vizinhos escolhidos por pressão arterial, não por perfil metabólico.
2. **Códigos nominais como dimensão.** `raca_cor`, `regiao` e `has_autorrelato` entravam
   na distância como inteiros. Região 3 não está entre 2 e 4 — a distância não tem
   significado nessas dimensões.

**Desenho do `ImputadorPNS`:**

```
fit   : StandardScaler ajustado no treino, sobre o espaço métrico
        KNNImputer(n_neighbors=5, weights="distance") no espaço padronizado
transform: imputa -> inverse_transform -> devolve unidades clínicas originais
```

| Papel | Variáveis |
|---|---|
| **Espaço métrico** (dimensões de distância) | `idade`, `imc`, `circ_cintura`, `rcest`, `pas`, `pad` (padronizadas) + `sexo_fem` (distância binária interpretável) |
| **Alvos de imputação, fora do espaço métrico** | `raca_cor`, `regiao`, `has_autorrelato`, `colesterol_alto`, `escolaridade`, `saude_autoaval`, `plano_saude` |

`escolaridade` e `saude_autoaval` ficam fora do espaço métrico apesar de ordinais: a
ordinalidade delas já é premissa questionável no modelo (§10 item 6); acumular duas
suposições sobre a mesma variável degrada a auditabilidade.

O `inverse_transform` é obrigatório — as compostas da §4.4 usam cortes clínicos em
unidades reais. O `StandardScaler` do `ColumnTransformer` permanece inalterado e
independente deste.

**Propriedade favorável:** após a limpeza da §3 as contínuas não têm ausentes, logo o
espaço métrico é **completo para todas as linhas**. Nenhum vizinho é selecionado com
vetor de distância parcial.

#### `KNNImputerCategorico`

O KNN opera sobre códigos numéricos e produz valores fracionários. Exige arredondamento
ao inteiro mais próximo **e** contenção ao conjunto de códigos observados no treino, para
que extrapolação nas bordas não fabrique código inexistente. Encapsulado em transformador
próprio, com docstring registrando a decisão — não pode ficar implícito.

#### `construir_preprocessador(feature_set) -> Pipeline`

`ColumnTransformer` final:
```
num -> StandardScaler
bin -> passthrough
nom -> OneHotEncoder(drop="first", handle_unknown="ignore")
```

---

## 5. `src/colinearidade.py`

A §6.5 exige **duas** ferramentas. Ambas obrigatórias.

### `matriz_pearson(X_train, cols_continuas) -> pd.DataFrame`
Correlação de Pearson entre as contínuas. Tabela + heatmap em
`reports/figures/pearson_{feature_set}.png`.

### `vif_matriz_design(pipeline, X_train, feature_set) -> pd.DataFrame`

**Correção necessária:** o baseline calculava VIF apenas sobre o bloco numérico cru.
`has_combinada` (binária) e `has_autorrelato` (nominal) ficaram fora do diagnóstico —
razão pela qual a redundância de D-001 não foi detectada.

O VIF roda sobre a **matriz de design pós-transformação**, incluindo dummies:
- `pipeline[:-1].transform(X_train)` + `get_feature_names_out()`
- Regressão auxiliar por coluna: `VIF_j = 1 / (1 - R²_j)`
- Sinalizar `VIF > VIF_LIMIAR`

Rodar para A, B e C. Tabela comparativa exportável via `.to_latex()` — é **resultado do
TCC**, materializando o protocolo da §6.5.

**Expectativa (a verificar, não a presumir):** VIF elevado em C por redundância
deliberada; residual em B por coarsening antropométrico; baixo em A.

---

## 6. `src/evaluate.py` — harness

### `avaliar(estimador, X_tr, y_tr, X_te, y_te, nome, recall_alvo) -> dict`

**Etapa 1 — CV no treino:** `StratifiedKFold(5, shuffle=True, random_state=42)` (§6.3 fase 5).

**Etapa 2 — Seleção de limiar.** Não usar `predict()` (limiar fixo 0,5). Com 7% de
positivos o limiar é decisão de projeto. Calibrar por CV **no treino**, nunca no teste:

```python
prec, rec, thr = precision_recall_curve(y_tr_cv, proba_tr_cv)
limiar = max(t for p, r, t in zip(prec, rec, thr) if r >= recall_alvo)
```

**Etapa 3 — Métricas no teste, com o limiar fixado.**

Assinadas na §6.3 fase 5, obrigatórias em toda tabela:

| Métrica | Papel |
|---|---|
| **Recall (Sensibilidade)** | prioritária — contexto de triagem |
| **Especificidade** | quantifica a carga de falsos positivos |
| **F1-Score** | equilíbrio sob desbalanceamento |
| **AUC-ROC** | capacidade discriminativa global |
| Matriz de Confusão | VP, VN, FP, FN |

Complementares (§10 item 4): PR-AUC, Brier, NNS.

**Etapa 4 — NNS.** `nns = 1 / precisao_no_limiar`. Tradução operacional: quantas pessoas
a UBS encaminha a exame para identificar um caso. Precisão 0,151 → NNS ≈ 6,6.

**Etapa 5 — Figuras** (`reports/figures/`, paleta Clinical Teal):
matriz de confusão · ROC · Precision-Recall com ponto de operação · curva de calibração.

### `comparar(resultados) -> pd.DataFrame`
Uma linha por (modelo × conjunto). Exportável via `.to_latex()`.

---

## 7. `src/benchmark.py`

A §2 e a §5 do TCC I já posicionam FINDRISC e BrDMrisc como o estado da arte do qual o
trabalho se diferencia — pesos fixos e calibração em população europeia, no caso do
FINDRISC. O benchmark operacionaliza esse argumento.

### `RegraClinicaADA` — `BaseEstimator, ClassifierMixin`
```python
predicao = (idade >= 45) | (imc >= 30)
```
O que a APS executa hoje **sem** machine learning.

### `DummyClassifier(strategy="most_frequent")`
Piso absoluto. Acurácia ≈0,93 com Recall zero — a armadilha da acurácia.

### Sobre FINDRISC e BrDMrisc
**Não reproduzíveis nesta base.** Exigem histórico familiar e variáveis comportamentais,
excluídas por R5. O BrDMrisc é escore de **incidência** derivado de coorte longitudinal,
incompatível com o desenho transversal da PNS 2013. Documentar a impossibilidade é
resultado, não limitação omitida.

**Critério de sucesso:** superar a regra clínica em Recall **com NNS menor**. Resultado
negativo é legítimo e deve ser reportado.

---

## 8. Ordem de execução

| # | Entrega | Status | Validação |
|---|---|---|---|
| 1 | `config.py` + `data.py` + `tests/test_data.py` | **CONCLUÍDA** | 14 testes · contrato da §3 conferido |
| 2 | `features.py` + `tests/test_features.py` | **CONCLUÍDA — requer patch D-012/D-014** | invariantes da §4.2 |
| 3 | `colinearidade.py`: Pearson + VIF pós-transformação | pendente | tabela A/B/C gerada |
| 4 | `evaluate.py` completo | pendente | end-to-end com LogReg, cinco métricas assinadas |
| 5 | `benchmark.py`: regra ADA + Dummy | pendente | avaliados pelo mesmo harness |
| 6 | LogReg × {A, B, C} + IC bootstrap nos coeficientes | pendente | tabela consolidada exportada |

**Só então** Random Forest (Sprint 2) e SHAP (Sprint 3).
Implemente **uma etapa por vez**, validando antes de avançar.

---

## 9. `peso_lab` — decisão de código e pendência de redação

**No código:** pondera exclusivamente a EDA descritiva de `01_eda.ipynb` (§6.3 fase 2).
**Não é passado a nenhum estimador.** Nenhuma função de `features.py`, `colinearidade.py`,
`evaluate.py` ou `benchmark.py` o recebe como parâmetro.

**Pendência de redação (não implementar):** a §5 rejeita o SMOTE argumentando que
observações sintéticas quebram a estrutura de ponderação amostral, mas não explicita por
que essa ponderação não se aplica ao treinamento. Lacuna a fechar no LaTeX antes da
defesa. Item marcado pelo autor.

---

## 10. Decisões que EXTRAPOLAM o TCC I

Registro completo e atualizado em `docs/RASTREABILIDADE.md`.

1. **Agrupamento de `raca_cor`** (D-007) — **não implementada.** Rodar primeiro com as
   cinco categorias e IC bootstrap; agrupar só se os intervalos confirmarem a
   instabilidade. No baseline, `raca_cor_3` (n=81, 6 eventos) produziu o maior
   coeficiente do modelo enquanto a prevalência marginal entre categorias é plana
   (6,8%–9,2%). Reavaliar na etapa 6.
2. **Categoria "não sabe/não mediu"** para `colesterol_alto` (D-008) — **1.297 ausentes
   (≈15,4%)**, provável MNAR. Divergente de R3. Variante comparada ao KNN, nunca
   substituta.
3. **Seleção de limiar por Recall-alvo** (D-004). `RECALL_ALVO = 0.80` provisório —
   confirmar com o orientador antes da etapa 4. Mudança posterior obriga a regerar todas
   as figuras.
4. **Métricas complementares** (D-005): PR-AUC, Brier, NNS. Adicionais, jamais
   substitutas das cinco assinadas.
5. **Benchmark por regra clínica ADA** (D-006).
6. **`escolaridade` e `saude_autoaval` como numéricas ordinais.** Premissa de linearidade
   herdada do baseline, não justificada no TCC I. Variante one-hot fica para depois da
   pré-banca se houver tempo.

## 11. Fora do escopo desta sprint

FastAPI e Streamlit pertencem à fase de Implantação (§6.3 fase 6), após a pré-banca.
SHAP entra na Sprint 3. Não antecipe nenhum dos três.