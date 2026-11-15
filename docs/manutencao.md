# Manutenção

Leia [arquitetura.md](arquitetura.md) antes de alterar fluxo. Este arquivo diz
onde a mudança mora e qual teste quebra se ela vazar.

## Quando um número publicado muda

`python -m src.experimento` compara o grid com duas âncoras em
`reports/tabelas/`: `comparativo_logreg.tex` e `comparativo_rf.tex`. Status 1
é divergência na terceira casa ou composição errada. Primeiro descubra se o
código regrediu. Se o resultado novo for de propósito:

1. Rode o experimento e leia a célula que divergiu.
2. Congele a âncora da floresta só com `--congelar-ancora`, na mesma execução
   que grava em `reports/tabelas/`. A âncora da logística existe para acusar
   movimento indevido nas linhas antigas. Regravá-la sem motivo apaga essa
   conferência.
3. Reexecute o notebook (`make notebook`) no mesmo conjunto de mudanças. O
   teste longo lê o `.tex` versionado antes de rodar o notebook e compara com
   o que a execução emite. Notebook novo com tabela velha, ou o contrário,
   falha.
4. Se a tela precisar da coluna nova, declare-a em `painel/dados.py`. Coluna
   que o painel não usa pode existir no CSV. Coluna que a tela lê e não está
   declarada quebra `dados.ler`.

Guarda que fixa resultado medido e falha pede investigação. Não edite a
asserção, nem troque a métrica, para a suíte voltar ao verde. O `xfail` estrito
do regime equiparado em `B_composto` é o caso em que a falha é o achado: a
floresta, nesse regime, não supera a regra clínica em NNS. Remover o `xfail`
sem o critério voltar a ser cumprido esconde o achado.

## Onde mexer

| Intenção | Arquivos | O que mais acompanha |
|---|---|---|
| Nova coluna da PNS | `config.DE_PARA`, grupos em `config`, `data.limpar` se o tipo ou a ausência mudarem, `features.FONTES` e `FEATURE_SETS` | `tests/test_data.py`, `tests/test_features.py`, rótulo em `painel/textos.py` |
| Outro conjunto de atributos | `FEATURE_SETS` e, se a floresta for rodar nele, `experimento.CONJUNTOS_FLORESTA` e `tuning.CONJUNTOS` | Diagnóstico em `colinearidade`, composição esperada do grid, notebook |
| Ablação de uma variável | `features.conjunto_sem` e `colinearidade.ablacao_vif` | Não crie uma quarta chave em `FEATURE_SETS` para um recorte de análise |
| Outro corte clínico | `config` (HbA1c, PAS, PAD, IMC de obesidade, cintura). A regra SBD lê `CORTE_IDADE_SBD` e `CORTE_IMC_SBD` em `benchmark.py` | A regra não lê `CORTE_IMC_OBESIDADE`. Corte duplicado em outro arquivo diverge em silêncio |
| Outro limiar de operação | `config.RECALL_ALVO` | Todo número calibrado do grid muda. As âncoras e o notebook vão junto |
| Outra configuração da floresta | Busca em `src/tuning.py`, depois o literal em `models.random_forest` | `tests/test_tuning.py` confere os dois. Não leia `busca_rf.csv` dentro de `random_forest` |
| Outro modelo | Fábrica em `models.py`, marca de conjunto e de limiar no `Estimador`, entrada em `MODELOS_PADRAO` | Preprocessador só para quem depende do conjunto. XGBoost está fora do escopo |
| Métrica ou figura nova | `evaluate.py`, exportação em `experimento.py` | `.to_latex()` na tabela que for ao texto. Figura salva em `reports/figuras/` e devolve o caminho |
| Página nova no painel | `painel/paginas/`, registro em `app.SECOES` | Frase em `textos.py`, gráfico em `graficos.py`, cor em `estilo.py`. Sem import de `src/`, sem widget de entrada |
| Coluna nova na tela | Emissão em `src/`, declaração em `dados.ARTEFATOS` | Casas decimais iguais às do `.tex` irmão |

O notebook não ganha lógica. Função nova nasce em `src/`. O notebook só chama.

`peso_kg` e `altura_cm` chegam ao modelo por `imc` e `rcest`. Colocá-los também
como coluna própria duplica canal. A proveniência em `FONTES` e os testes de
canal duplo existem para isso aparecer.

## Painel

`tests/test_painel.py` cobre o contrato da interface:

- import proibido (`src`, sklearn e afins) em qualquer arquivo de `painel/`
- widget de entrada proibido, com subprocesso e botão só na reprodução
- todo link de `app.SECOES` aponta para arquivo que existe, e toda página está
  numa seção
- leitura de arquivo só em `dados.py`
- artefato declarado tem CSV versionado e as colunas esperadas
- nenhuma página traz nome de célula como literal
- cor hexadecimal só em `estilo.py`
- vocabulário e ajuda dos cartões, inclusive no HTML
- a reprodução pelo botão, teste longo, confere e não altera `reports/tabelas/`

Número exibido é coluna de artefato. Filtro, ordem, formato, rótulo e cor não
calculam métrica. Se a tela precisar de um índice que hoje só existe no log,
emita a coluna em `src/` e leia o CSV. Não reimplemente a conta em `painel/`.

A reprodução ao vivo chama `sys.executable`. Suba o painel com o Python da
`.venv`. O `Makefile` já escolhe esse interpretador no alvo `painel`.

## Testes longos

O marcador `lento` pula no `make test-rapido`. Hoje ele marca:

- reexecução do notebook contra a tabela versionada
- reprodução do regime principal de SHAP contra os CSV
- mesma semente da busca devolvendo o mesmo vencedor, com poucas iterações
- `src.experimento --saida` gravando fora de `reports/tabelas/`
- o botão de reprodução do painel

A suíte que chama `experimento.executar` ou reexecuta o notebook pede a fixture
`tabelas_preservadas`. Sem ela, o teste deixa em `reports/tabelas/` o recorte
que usou, e o teste seguinte lê o rastro. O backend do matplotlib na suíte é
`Agg`, para figura ir a arquivo e não abrir janela.

Testes que dependem do CSV bruto pulam com motivo se `data/raw/` está vazio.
`tests/test_rastreabilidade.py` pula se `docs/RASTREABILIDADE.md` não existe.
Quando o arquivo existe, confere se o índice e o corpo de cada decisão declaram
a mesma classificação. O teste depende do formato que o arquivo já tem: o
índice, o identificador repetido no título de cada decisão e a linha de Status
logo abaixo do título. Mudar esse formato faz a guarda falhar. A guarda não
congela resultado experimental.

## O que o código deixa em aberto

Estas pendências estão conscientes. Implementar uma delas muda resultado ou
texto, e não é limpeza.

- O recall-alvo 0,80 está provisório em `config.py`.
- Nenhuma célula está eleita como modelo principal. A referência da barra
  lateral é o menor NNS no ponto padrão. O critério frente à regra só se avalia
  no regime de recall equiparado, e nele a célula de referência em `B_composto`
  não cumpre. Quatro das cinco células calibráveis cumprem. Mostrar os dois
  regimes lado a lado é o comportamento atual.
- Agrupar `raca_cor` não está implementado. O SHAP da floresta e o coeficiente
  da logística do baseline não contam a mesma história para essa variável. A
  pendência do coeficiente pede intervalo, não um agrupamento decidido antes.
- Ausência de `colesterol_alto` continua no KNN. Categoria explícita de não
  resposta, para comparar com o KNN, não tem código.
- `LinearExplainer` da logística não foi executado. A explicabilidade publicada
  é `TreeExplainer` da floresta em A e em B.
- `peso_lab` fica fora do treino. A justificativa para o texto é esta: o peso
  amostral corrige estimativa de parâmetro populacional. Aqui a tarefa é
  classificação de indivíduo. Ponderar a perda não traz validade externa e
  briga com `class_weight`, que já reequilibra as classes. O texto do trabalho
  ainda precisa dizer isso com essas palavras.
- Escolaridade e saúde autoavaliada seguem como numéricas no bloco `num`, não
  como one-hot.
- API e formulário que classifiquem um indivíduo não existem. O painel não é
  esse formulário.

## Fora do desenho

Não entram, mesmo como experimento lateral no grid publicado:

- SMOTE ou outra reamostragem sintética
- variável de dieta, atividade física ou tabagismo
- HbA1c, ou qualquer medida laboratorial, como preditor
- `peso_lab` no `fit`
- XGBoost
- estatística entre linhas antes do split
- `fit` de imputador, scaler ou encoder enxergando o teste
