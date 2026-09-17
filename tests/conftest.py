"""Configuração compartilhada da suíte.

Duas responsabilidades, as duas nascidas do mesmo problema: partes da suíte escrevem
em `reports/tabelas/`, que é diretório **versionado**.

1. **`tabelas_preservadas`** restaura `reports/tabelas/` ao estado original ao fim do
   teste, dê no que der. Todo teste que chame `experimento.executar` ou reexecute o
   notebook precisa dela: sem isso o teste deixa a tabela consolidada com o recorte
   que usou (dois modelos, um conjunto) no lugar do grid inteiro, e o teste seguinte
   lê o rastro do anterior. Foi assim que a guarda do notebook falhou na primeira
   execução da suíte completa.

2. **O marcador `lento`** identifica a reexecução do notebook, para o ciclo curto
   poder pulá-la:

       pytest tests/ -v                 suíte inteira, com a reexecução
       pytest tests/ -m "not lento"     só as guardas baratas

   O padrão é rodar tudo. O teste lento existe porque o notebook é commitado com
   outputs e não falha sozinho quando `src/` muda; desativá-lo por padrão devolveria
   o projeto ao estado que ele existe para impedir.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from src import config as cfg


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "lento: reexecuta o notebook de resultados de ponta a ponta (minutos).",
    )


@pytest.fixture
def tabelas_preservadas():
    """Isola `reports/tabelas/` das escritas do teste.

    Teste que suja diretório versionado é teste que na prática se desativa sozinho:
    ninguém convive com `git status` poluído depois de rodar a suíte.
    """
    if not cfg.PATH_TABELAS.exists():
        pytest.skip(f"Diretório de tabelas ausente em {cfg.PATH_TABELAS}.")
    with tempfile.TemporaryDirectory() as tmp:
        espelho = Path(tmp) / "tabelas"
        shutil.copytree(cfg.PATH_TABELAS, espelho)
        try:
            yield
        finally:
            shutil.rmtree(cfg.PATH_TABELAS, ignore_errors=True)
            shutil.copytree(espelho, cfg.PATH_TABELAS)
