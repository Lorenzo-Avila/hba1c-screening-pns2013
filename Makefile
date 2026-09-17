# Contrato de reprodução do projeto.
#
# Os alvos são invólucros finos sobre comandos Python. `make` não é dependência de
# execução — nenhum alvo faz mais do que a linha que ele mostra, e o comando
# equivalente está anotado ao lado. Em máquina sem `make` (o caso do ambiente de
# desenvolvimento atual, Windows), rode a linha direto.

PYTHON ?= python

.PHONY: tudo test test-rapido resultados figuras notebook limpar ajuda

## tudo — suíte de testes e reprodução completa, na ordem em que importam
tudo: test resultados notebook

## test — contrato de dados, features, colinearidade, harness e notebook.
## Inclui a reexecução do notebook (marcador `lento`, alguns minutos): é a única
## guarda do artefato que é commitado com outputs.
test:
	$(PYTHON) -m pytest tests/ -v

## test-rapido — a mesma suíte sem a reexecução do notebook, para o ciclo curto
##   equivale a: pytest tests/ -v -m "not lento"
test-rapido:
	$(PYTHON) -m pytest tests/ -v -m "not lento"

## resultados — emite reports/tabelas/ e confere contra o artefato da Sprint 1.
## Falha (status 1) se algum número divergir na terceira casa.
##   equivale a: python -m src.experimento
resultados:
	$(PYTHON) -m src.experimento

## figuras — o mesmo grid, salvando ROC, PR, confusão e calibração do teste
##   equivale a: python -m src.experimento --figuras
figuras:
	$(PYTHON) -m src.experimento --figuras

## notebook — reexecuta o documento de resultados in-place, preservando os outputs.
## É o mesmo grid de `figuras`, apresentado na ordem da arguição. Commitado com
## outputs de propósito: é a fonte das figuras do LaTeX, como `docs/baseline/`.
##   equivale a: jupyter nbconvert --execute --to notebook --inplace notebooks/03_resultados.ipynb
notebook:
	$(PYTHON) -m jupyter nbconvert --execute --to notebook --inplace notebooks/03_resultados.ipynb

## limpar — remove os artefatos gerados, preservando os .tex versionados
limpar:
	$(PYTHON) -c "import shutil, pathlib; [p.unlink() for p in pathlib.Path('reports/figuras').glob('*.png')]"

## ajuda — lista os alvos
ajuda:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/^## //'
