"""Gravação de artefato tabular em disco, sem depender de mais nada de `src/`.

Existe por causa de uma seta apontando para o lado errado. `salvar_csv` nasceu dentro de
`evaluate`, e `colinearidade` passou a importar o harness de avaliação inteiro só para
gravar um CSV de diagnóstico. Diagnóstico de multicolinearidade não depende de como o
projeto avalia modelo, e acoplamento assim ninguém consegue justificar depois (D-050).

O módulo não importa nenhum outro de `src/`, nem `config`. Quem chama diz onde gravar, e
é por isso que ele pode ser importado de qualquer lugar sem arrastar nada junto.

**Ponto único de gravação de CSV do projeto.** Nenhum outro módulo chama `to_csv`
direto: valor gravado por caminho próprio escapa do arredondamento e devolve o diff
permanente que D-048 fechou.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Casas decimais na gravação de CSV. O `.tex` já sai em três casas, mas o CSV gravava o
# float inteiro, e a floresta roda com `n_jobs=-1`: a ordem em que os processos reduzem
# varia entre execuções e move o último bit. Nenhum resultado muda, e o `.tex` nem
# mostra. O dano é outro: `reports/tabelas/` passava a aparecer modificado a cada
# execução, e esse é o único diretório do projeto onde ninguém pode se acostumar a
# passar o olho sem ler. Nove casas é folgado para um projeto que confere em três
# (D-048).
CASAS_CSV = 9


def salvar_csv(tabela: pd.DataFrame, destino: Path | str,
               casas: int = CASAS_CSV) -> Path:
    """Grava a tabela como CSV no caminho pedido, com os floats arredondados.

    `DataFrame.round` mexe só nas colunas numéricas, então rótulo, booleano e `inf`
    atravessam intactos.

    Args:
        destino: caminho completo do arquivo. Os diretórios que faltarem são criados.
            O módulo não conhece `config`, e é de propósito: quem sabe onde ficam os
            artefatos é quem chama.
        casas: precisão gravada. Passe mais se alguma conferência precisar, mas note
            que o ruído de paralelismo volta junto.

    Returns:
        O caminho gravado.
    """
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    tabela.round(casas).to_csv(destino, index=False, encoding="utf-8")
    return destino
