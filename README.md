# HbA1c ≥ 6,5% · PNS 2013

Classificação de perfis associados à hiperglicemia crônica clinicamente significativa
(HbA1c ≥ 6,5%) a partir de variáveis não invasivas da Subamostra Laboratorial da
PNS 2013. Dois modelos: regressão logística e floresta aleatória. O desbalanceamento
entra por `class_weight`. O painel lê as tabelas já geradas e, numa página só,
reproduz o experimento a partir do CSV bruto.

O notebook orquestra. `src/` implementa. `painel/` apresenta.

| Documento | Para quê |
|---|---|
| [docs/arquitetura.md](docs/arquitetura.md) | Fluxos, módulos, contratos e o que cada comando emite |
| [docs/manutencao.md](docs/manutencao.md) | Onde mexer, o que o teste cobra e o que ainda está em aberto |
| [data/README.md](data/README.md) | Onde colocar o CSV bruto |
| [docs/baseline/readme.md](docs/baseline/readme.md) | Notebook congelado da primeira matriz de atributos |

`docs/ESTADO.md` e `docs/RASTREABILIDADE.md`, quando existem nesta cópia, ficam fora
do Git. O primeiro transcreve números. O segundo registra decisões. A fonte dos
números publicados é `reports/tabelas/`.

## Ambiente

O interpretador tem de ser o da `.venv`. Um Python solto no PATH pode ter Streamlit
e não ter scikit-learn: o painel sobe e a reprodução ao vivo quebra no import.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
.venv/bin/python -m pip install -r requirements.txt        # Linux e macOS
```

Baixe o CSV da PNS e coloque em `data/raw/`, como descreve `data/README.md`.
Sem esse arquivo, carga, notebook e reprodução ao vivo são pulados.

`make` é atalho. Em máquina sem `make`, rode a linha que o alvo comenta.
O `Makefile` escolhe o Python da `.venv` quando ela existe.

## Comandos

| Alvo | O que faz |
|---|---|
| `make test` | Suíte inteira, inclusive os testes longos |
| `make test-rapido` | A mesma suíte sem o marcador `lento` |
| `make resultados` | `python -m src.experimento`: regera as tabelas do grid e confere as âncoras |
| `make figuras` | O mesmo grid, gravando as figuras do teste |
| `make shap` | Atribuições SHAP da floresta publicada. Fora do caminho de `resultados` |
| `make notebook` | Reexecuta `notebooks/03_resultados.ipynb` no lugar, com os outputs |
| `make painel` | `python -m streamlit run painel/app.py`, só em localhost |

A busca de hiperparâmetros é outro comando, e não entra em `make resultados`:

```bash
python -m src.tuning
```

## O que não mudar por acidente

- Nenhuma estatística entre linhas antes do `train_test_split`. Split 80/20, estratificado pelo alvo, `random_state=42`.
- Imputação, escala e codificação: `fit` no treino, `transform` no teste.
- Registro sem HbA1c válida sai da base. O alvo não é imputado.
- `peso_lab` pondera a descrição da amostra e fica fora do treino.
- HbA1c é desfecho. Variáveis comportamentais ficam de fora.
- A floresta publicada está em literais em `src/models.py`. Ela não lê o CSV da busca na hora de treinar.
- `comparativo_logreg.tex` e `comparativo_rf.tex` são âncoras. Número que diverge na terceira casa faz `src.experimento` sair com status 1.
- Notebook e tabelas versionadas mudam juntos. O teste longo compara a tabela que o notebook emite com o `.tex` lido antes da reexecução.
