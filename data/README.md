# Dados

**Fonte:** IBGE — PNS 2013, Subamostra Laboratorial (microdados públicos).
Download: https://ftp.ibge.gov.br/PNS/2013/Divulgacoes/Outros/Exames/

**Arquivo:** `EXAMES_PNS_2013_BRUTA.csv` · 8.952 linhas × 509 colunas
**MD5:** f2d80eea58229bc29b8d5e482d7f778c

Não versionado. Baixe manualmente e coloque em `data/raw/`.
Decimal em padrão brasileiro (vírgula), convertido em `src/data.py`.

`python -m src.experimento`, o notebook de resultados e a reprodução ao vivo
leem esse caminho. Sem o arquivo, esses comandos e os testes que carregam a
base são pulados. O contrato da amostra depois da limpeza está em
`tests/test_data.py`.
