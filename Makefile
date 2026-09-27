# Contrato de reprodução do projeto.
#
# Os alvos são invólucros finos sobre comandos Python. `make` não é dependência de
# execução — nenhum alvo faz mais do que a linha que ele mostra, e o comando
# equivalente está anotado ao lado. Em máquina sem `make` (o caso do ambiente de
# desenvolvimento atual, Windows), rode a linha direto.

# Interpretador: o da `.venv` do projeto quando ela existe, porque é ela que tem as
# versões de `requirements.txt`. Um `python` solto no PATH pode ter streamlit e não
# ter sklearn: o painel sobe e a reprodução ao vivo falha no import. Sobrescreva com
# `make PYTHON=...` se o ambiente estiver em outro lugar.
ifneq ($(wildcard .venv/Scripts/python.exe),)
PYTHON ?= .venv/Scripts/python
else ifneq ($(wildcard .venv/bin/python),)
PYTHON ?= .venv/bin/python
else
PYTHON ?= python
endif

.PHONY: tudo test test-rapido resultados figuras shap notebook painel limpar ajuda

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

## shap — atribuições SHAP da floresta publicada, fora do caminho crítico.
## O regime de contraste é caro e nenhum número publicado depende dele, por isso
## ele não entra em `resultados`.
##   equivale a: python -m src.explicabilidade
shap:
	$(PYTHON) -m src.explicabilidade

## notebook — reexecuta o documento de resultados in-place, preservando os outputs.
## É o mesmo grid de `figuras`, apresentado na ordem da arguição. Commitado com
## outputs de propósito: é a fonte das figuras do LaTeX, como `docs/baseline/`.
##   equivale a: jupyter nbconvert --execute --to notebook --inplace notebooks/03_resultados.ipynb
notebook:
	$(PYTHON) -m jupyter nbconvert --execute --to notebook --inplace notebooks/03_resultados.ipynb

## painel: painel de resultados sobre os CSV de reports/tabelas/, em localhost.
## As telas leem artefato versionado. A reprodução ao vivo dispara o experimento
## em outro processo e não regrava reports/tabelas/.
##   equivale a: streamlit run painel/app.py
painel:
	$(PYTHON) -m streamlit run painel/app.py

## limpar — remove os artefatos gerados, preservando os .tex versionados
limpar:
	$(PYTHON) -c "import shutil, pathlib; [p.unlink() for p in pathlib.Path('reports/figuras').glob('*.png')]"

## ajuda — lista os alvos
ajuda:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/^## //'
