# ESTADO DO PROJETO

**Atualizado:** 16/09/2026 · fim da Sprint 1, após a reorganização de reprodutibilidade
e o fechamento do documento de resultados
**Marco:** pré-banca 05/10/2026 — 19 dias

Documento regenerado ao fim de cada sprint. Descreve **onde o projeto está**, não como
chegou aqui — isso é `RASTREABILIDADE.md`.

---

## 1. Identificação

Classificação de perfis associados à hiperglicemia crônica clinicamente significativa
(HbA1c ≥ 6,5%) a partir de variáveis não invasivas dos microdados da Subamostra
Laboratorial da PNS 2013, com vistas à aplicabilidade na APS via PEC/SISAB.

IFFar São Borja · Sistemas de Informação · Orientador: Prof. Icaro Lins Iglesias
Repositório: `hba1c-screening-pns2013` (privado até a defesa)
Framework: CRISP-DM

**Metodologia vinculante:** Projeto de TCC I assinado. Qualquer decisão ausente dele
exige registro em `RASTREABILIDADE.md`.

---

## 2. Dados

Amostra analítica: **8.451 indivíduos** · prevalência do desfecho **7,02%**
Split 80/20 estratificado · treino 6.760, teste 1.691 · `random_state=42`

Prevalência ponderada por `peso_lab`: **6,36%** — estimativa populacional, reportada
apenas na EDA descritiva (R9, D-031). Não é a prevalência que contextualiza as métricas:
essa é a amostral, e é ela que o `class_weight='balanced'` reequilibra.

Exclusões: 411 sem HbA1c válida, 90 com bloco de medição integralmente ausente.
Ausentes remanescentes ao fim da limpeza: `colesterol_alto` 1.297, `has_autorrelato` 261,
`regiao` 4, `raca_cor` 2.

---

## 3. Código em produção

```
src/config.py           constantes, DE_PARA, codificações PNS, cortes clínicos SBD 2019
src/data.py             carga, limpeza, exclusões — row-wise, sem imputação
src/features.py         ImputadorPNS, ConstrutorCompostas, FEATURE_SETS, preprocessador
src/colinearidade.py    Pearson, VIF pós-transformação, decomposicao_do_escore
src/evaluate.py         avaliar(), comparar(), calibração de limiar, figuras
src/benchmark.py        RegraClinicaADA, Dummy, recall_da_regra_no_treino, leitura_operacional
src/models.py           factory Estimador, logistica(), random_forest(), montar()
src/experimento.py      rodar_grid(), carga_operacional(), conferir_reproducao() — CLI

notebooks/03_resultados.ipynb   documento de arguição — orquestra src/, commitado com outputs
```

Testes: `test_data.py`, `test_features.py`, `test_colinearidade.py`, `test_evaluate.py`,
`test_benchmark.py`, `test_experimento.py`, `test_notebook.py`. 170 testes, suíte
integralmente verde. O marcado `lento` reexecuta o notebook e compara a tabela emitida
com a versionada; `make test-rapido` o pula no ciclo curto.

**Reprodução:** `python -m src.experimento` (alvo `make resultados`) regenera as duas
tabelas da seção 4 a partir do CSV bruto e confere o resultado contra
`reports/tabelas/comparativo_logreg.tex`, falhando com status 1 se algum número divergir
na terceira casa. É o ponto único de reprodução — ver D-027.

Saídas em `reports/tabelas/` (`.tex` e `.csv`) e `reports/figuras/` (`.png`).

`notebooks/03_resultados.ipynb` apresenta o mesmo grid na ordem da arguição — amostra,
colinearidade, escolha do conjunto, desempenho, calibração, comparadores, carga
operacional. **Orquestra e não implementa:** toda chamada vem de `src/`. É a fonte das
figuras do LaTeX e é commitado com outputs, pelo mesmo motivo que `docs/baseline/`.
Reexecução: `make notebook`.

Por ser commitado com outputs, o notebook não falha sozinho quando `src/` muda —
`tests/test_notebook.py` é a guarda: reexecuta e compara a tabela emitida com
`comparativo_consolidado.tex` **como está versionada**, lida antes da execução (D-032).

**Pipeline de três estágios** (`construir_preprocessador`):
`ImputadorPNS` → `ConstrutorCompostas` → `ColumnTransformer`

`ImputadorPNS` define espaço métrico explícito: seis contínuas padronizadas por scaler
interno + `sexo_fem`. Nominais são alvos de imputação sem votar na vizinhança.
`inverse_transform` devolve escala clínica antes das compostas.

**Conjuntos de atributos:** `A_atomico` (20 colunas), `B_composto` (16), `C_completo` (23).

---

## 4. Resultados consolidados — LogReg, teste n=1.691

| Conjunto | Limiar | Recall | Especif. | F1 | AUC-ROC | NNS |
|---|---|---|---|---|---|---|
| A_atomico | 0,424 | 0,748 | 0,627 | 0,224 | 0,775 | 7,58 |
| B_composto | 0,445 | 0,765 | 0,637 | 0,233 | 0,781 | 7,26 |
| C_completo | 0,419 | 0,756 | 0,625 | 0,226 | 0,770 | 7,54 |
| Regra ADA | fixo | 0,866 | 0,446 | 0,188² | 0,656¹ | 9,46 |
| Dummy | fixo | 0,000 | 1,000 | 0,000 | 0,500¹ | ∞ |

¹ Classificadores de corte único. Métricas dependentes de curva não são comparáveis.
² Valor regenerado por `src/experimento.py` (0,188472). A versão anterior deste documento
registrava 0,189, arredondamento de transcrição sem consequência interpretativa. Vale o
valor emitido (D-027).

Artefatos: `reports/tabelas/comparativo_consolidado.{tex,csv}` e
`carga_operacional.{tex,csv}`.

**Carga operacional — três regimes**

| Regime | Encaminhados | Detectados |
|---|---|---|
| Regra ADA | 974 (57,6%) | 103 |
| Modelo, recall equiparado (alvo 0,8945 no treino) | 1.000 (59,1%) | 109 |
| Modelo, ponto padrão (recall-alvo 0,80) | 661 (39,1%) | 91 |

---

## 5. Cinco achados que sustentam a defesa

1. **`C_completo` descartado.** Redundância total por identidade algébrica: R² = 1,000 ao
   reconstruir a indicadora de obesidade, contra R² = 0,884 da regressão auxiliar do VIF.
   O diagnóstico linear é estruturalmente cego a composições por ponto de corte.
2. **A e B indistinguíveis em desempenho.** Amplitude de 0,017 em recall entre conjuntos,
   contra desvio de até 0,027 entre folds do mesmo conjunto. A escolha se decide por
   parcimônia e interpretabilidade, não por métrica.
3. **`rcest` removida por protocolo da §6.5.** VIF 14,03; Pearson 0,916 com
   `circ_cintura`. Após remoção, `circ_cintura` cai de 13,73 para 4,70.
4. **Probabilidades não são risco.** Intercepto de calibração −2,58 contra logito da
   prevalência de treino −2,585. Deslocamento algébrico previsível do
   `class_weight='balanced'`. Slope 0,84–0,92: a ordenação é válida, a escala não.
5. **A regra clínica não é triagem em escala.** Encaminha 57,6% da população adulta e
   opera em ponto único. A contribuição do modelo é a curva de operação ajustável à
   capacidade instalada da rede.

---

## 6. Decisões em aberto

| # | Questão | Recomendação | Quem decide |
|---|---|---|---|
| 1 | Ponto de operação principal a reportar | `B_composto` em recall 0,80, com os três regimes como análise de suporte | Orientador |
| 2 | Agrupamento de `raca_cor` (D-007) | Decidir após IC bootstrap dos coeficientes, não antes | Dados |
| 3 | Conjunto mínimo (4–5 variáveis) | Se houver tempo após SHAP | Cronograma |
| 4 | Redação sobre `peso_lab` no LaTeX (D-009) | Pendência marcada pelo autor | Autor |
| 5 | `escolaridade`/`saude_autoaval` one-hot | Pós pré-banca | Cronograma |
| 6 | Hiperparâmetros da Random Forest | Placeholder declarado; fora de `MODELOS_PADRAO` até a Sprint 2 | Sprint 2 |

---

## 7. Restrições permanentes

SMOTE não é utilizado — Cost-Sensitive Learning via `class_weight='balanced'` (§5).
`peso_lab` não entra no treino, apenas na EDA descritiva (§6.3 fase 2).
Variáveis comportamentais excluídas por causalidade reversa (§6.3.2, princípio 3).
Apenas variáveis não invasivas capturáveis pelo PEC/SISAB (§6.3.2, princípios 1 e 2).
XGBoost fora de escopo. FastAPI e Streamlit após a pré-banca (§6.3 fase 6).
Linguagem: "associação com", nunca "risco de desenvolver"; "modelo de classificação",
nunca "preditivo".
`random_state=42` em toda operação estocástica.

---

## 8. Próxima sprint

**Sprint 2 — Random Forest.** Especificação em `docs/sprints/SPEC_sprint2.md`.

O grid já aceita a RF: `python -m src.experimento --modelos LogReg "Random Forest"`. Os
hiperparâmetros em `models.random_forest` são placeholder declarado e o modelo está fora
de `MODELOS_PADRAO` — nenhum número de RF entra em tabela publicada antes que a busca
defina os valores (D-029).

Em seguida, Sprint 3 — SHAP global e local, `LinearExplainer` para LogReg e
`TreeExplainer` para RF.

Condição de contorno do cronograma: se a Sprint 2 exceder 23/09, reavaliar a ordem — um
trabalho com SHAP sobre a LogReg é completo; um com RF sem SHAP não é.