"""Guardas do painel de resultados (D-054).

O painel lê `reports/tabelas/` e não calcula, não importa `src/`, não recebe
entrada e não usa vocabulário de risco. Cada guarda abaixo verifica uma dessas
fronteiras por AST ou pela renderização com `AppTest`.
"""

import ast
import functools
import json
import re
import subprocess
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest
from google.protobuf.json_format import MessageToDict
from streamlit.testing.v1 import AppTest

from painel import dados, textos

RAIZ = Path(__file__).resolve().parent.parent
PAINEL = RAIZ / "painel"
ARQUIVOS = sorted(PAINEL.rglob("*.py"))
PAGINAS = ["amostra", "colinearidade", "floresta_logistica", "explicabilidade",
           "casos_locais"]

IMPORTS_PROIBIDOS = {"sklearn", "shap", "joblib", "pickle", "src", "subprocess"}
WIDGETS_PROIBIDOS = {
    "text_input", "number_input", "text_area", "date_input", "time_input",
    "file_uploader", "camera_input", "form", "slider", "select_slider",
    "data_editor", "chat_input", "color_picker", "button",
}
LEITURAS = {"read_csv", "open", "read_text"}
VOCABULARIO_PROIBIDO = ["risco", "probabilidade de", "preditiv", "protetor",
                        "referência", "referencia", "modelo principal"]
CELULA_LITERAL = re.compile(
    r"(Random Forest|LogReg)\s*·|\b(A_atomico|B_composto|C_completo)\b"
)


def _arvore(arquivo: Path) -> ast.AST:
    return ast.parse(arquivo.read_text(encoding="utf-8"), filename=str(arquivo))


def _rel(arquivo: Path) -> str:
    return arquivo.relative_to(RAIZ).as_posix()


# ---------------------------------------------------------------------------
# Guardas estáticas, por AST
# ---------------------------------------------------------------------------

def test_painel_nao_importa_modulo_proibido():
    violacoes = []
    for arquivo in ARQUIVOS:
        for no in ast.walk(_arvore(arquivo)):
            if isinstance(no, ast.Import):
                modulos = [alias.name for alias in no.names]
            elif isinstance(no, ast.ImportFrom) and no.module:
                modulos = [no.module]
            else:
                continue
            for modulo in modulos:
                if modulo.split(".")[0] in IMPORTS_PROIBIDOS:
                    violacoes.append(f"{_rel(arquivo)}:{no.lineno} importa {modulo}")
    assert not violacoes, "imports proibidos no painel:\n" + "\n".join(violacoes)


def test_painel_nao_usa_widget_de_entrada():
    violacoes = []
    for arquivo in ARQUIVOS:
        for no in ast.walk(_arvore(arquivo)):
            if isinstance(no, ast.Attribute) and no.attr in WIDGETS_PROIBIDOS:
                violacoes.append(f"{_rel(arquivo)}:{no.lineno} usa {no.attr}")
            elif isinstance(no, ast.ImportFrom):
                for alias in no.names:
                    if alias.name in WIDGETS_PROIBIDOS:
                        violacoes.append(
                            f"{_rel(arquivo)}:{no.lineno} importa {alias.name}")
    assert not violacoes, "widgets de entrada no painel:\n" + "\n".join(violacoes)


def test_leitura_de_arquivo_so_em_dados():
    violacoes = []
    for arquivo in ARQUIVOS:
        if arquivo == PAINEL / "dados.py":
            continue
        for no in ast.walk(_arvore(arquivo)):
            if not isinstance(no, ast.Call):
                continue
            funcao = no.func
            nome = (funcao.id if isinstance(funcao, ast.Name)
                    else funcao.attr if isinstance(funcao, ast.Attribute) else None)
            if nome in LEITURAS:
                violacoes.append(f"{_rel(arquivo)}:{no.lineno} chama {nome}")
    assert not violacoes, "leitura fora de painel/dados.py:\n" + "\n".join(violacoes)


def test_artefatos_declarados_estao_versionados():
    versionados = set(subprocess.run(
        ["git", "ls-files", "reports/tabelas"], cwd=RAIZ, capture_output=True,
        text=True, check=True,
    ).stdout.split())
    fora = [nome for nome in dados.ARTEFATOS
            if f"reports/tabelas/{nome}.csv" not in versionados]
    assert not fora, f"artefatos declarados e não versionados: {fora}"


def test_nenhum_nome_de_celula_como_literal():
    violacoes = []
    for arquivo in ARQUIVOS:
        for no in ast.walk(_arvore(arquivo)):
            if (isinstance(no, ast.Constant) and isinstance(no.value, str)
                    and CELULA_LITERAL.search(no.value)):
                violacoes.append(f"{_rel(arquivo)}:{no.lineno} {no.value!r}")
    assert not violacoes, "nome de célula como literal:\n" + "\n".join(violacoes)


# ---------------------------------------------------------------------------
# Leitura
# ---------------------------------------------------------------------------

def test_todo_artefato_declarado_carrega_com_as_colunas_esperadas():
    for nome, spec in dados.ARTEFATOS.items():
        tabela = dados.ler(nome)
        assert set(spec["colunas"]) <= set(tabela.columns), nome
        assert set(spec["casas"]) <= set(spec["colunas"]), nome


def test_dados_falha_nomeando_artefato_e_coluna_ausente(tmp_path):
    original = pd.read_csv(dados.RAIZ / "mcnemar_pareado.csv")
    original.drop(columns=["ambos_erram"]).to_csv(
        tmp_path / "mcnemar_pareado.csv", index=False
    )
    with pytest.raises(dados.ArtefatoInvalido,
                       match=r"'mcnemar_pareado'.*ambos_erram"):
        dados.ler("mcnemar_pareado", raiz=tmp_path)


# ---------------------------------------------------------------------------
# Renderização
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _renderizar(pagina: str) -> AppTest:
    app = AppTest.from_file(str(PAINEL / "app.py"), default_timeout=60)
    app.run()
    app.switch_page(f"paginas/{pagina}.py")
    app.run()
    return app


def _nos(no):
    yield no
    for filho in getattr(no, "children", {}).values():
        yield from _nos(filho)


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for chave, valor in obj.items():
            yield str(chave)
            yield from _strings(valor)
    elif isinstance(obj, list):
        for item in obj:
            yield from _strings(item)


def _texto_da_tabela(tabela: pd.DataFrame) -> list:
    return ([str(c) for c in tabela.columns] + [str(i) for i in tabela.index]
            + [str(v) for v in tabela.to_numpy().ravel()])


def _texto_renderizado(app: AppTest) -> str:
    """Todo texto que chega à tela: elementos, rótulos, tabelas e gráficos."""
    partes = []
    for no in _nos(app._tree):
        tipo = getattr(no, "type", "")
        if tipo in ("dataframe", "table", "arrow_data_frame"):
            partes += _texto_da_tabela(no.value)
        elif tipo == "vega_lite_chart":
            partes += list(_strings(json.loads(no.proto.spec)))
            for conjunto in no.proto.datasets:
                tabela = pa.ipc.open_stream(conjunto.data.data).read_pandas()
                partes += _texto_da_tabela(tabela)
        elif getattr(no, "proto", None) is not None:
            partes += list(_strings(MessageToDict(no.proto)))
        if tipo == "selectbox":
            partes += list(no.options)
    return "\n".join(partes)


def _literais_de_textos() -> list:
    """Frases de `textos.py`, inteiras e quebradas em frases, maiores primeiro."""
    frases = []
    for nome, valor in vars(textos).items():
        if not nome.isupper():
            continue
        candidatos = (valor.values() if isinstance(valor, dict)
                      else [valor] if isinstance(valor, str) else [])
        for texto in candidatos:
            frases.append(texto)
            frases += re.split(r"(?<=\.) ", texto)
    return sorted(set(frases), key=len, reverse=True)


@pytest.mark.parametrize("pagina", PAGINAS)
def test_pagina_renderiza_sem_excecao(pagina):
    app = _renderizar(pagina)
    assert not app.exception, [e.value for e in app.exception]
    assert app.title[0].value == textos.TITULO
    assert app.caption[-1].value == textos.RODAPE


@pytest.mark.parametrize("pagina", PAGINAS)
def test_vocabulario_proibido_fora_de_textos(pagina):
    texto = _texto_renderizado(_renderizar(pagina))
    for literal in _literais_de_textos():
        texto = texto.replace(literal, "")
    achados = [termo for termo in VOCABULARIO_PROIBIDO
               if termo in texto.lower()]
    trechos = [texto[max(0, texto.lower().find(t) - 40):
                     texto.lower().find(t) + 40] for t in achados]
    assert not achados, f"{pagina}: vocabulário proibido {achados} em {trechos}"


def test_casos_locais_exibe_ressalva_e_valor_observado():
    # Instância nova: o teste troca a seleção e não pode sujar o cache.
    app = _renderizar.__wrapped__("casos_locais")
    grafico = app.get("vega_lite_chart")[0]
    subtitulo = json.loads(grafico.proto.spec)["title"]["subtitle"]
    assert " ".join(subtitulo) == textos.RESSALVA_LOCAL, "ressalva fora do gráfico"
    assert any(textos.RESSALVA_LOCAL in w.value for w in app.warning)

    tabela = app.dataframe[0].value
    assert "valor observado" in tabela.columns
    assert (tabela["valor observado"] != "").all()

    # Índice presente em mais de uma célula: a tela diz que é o mesmo indivíduo.
    casos = dados.ler("shap_casos_locais")
    por_indice = casos.groupby("indice")["celula"].nunique()
    repetidos = list(por_indice[por_indice > 1].index)
    assert repetidos, "o artefato deixou de ter índice repetido entre células"
    caixa = app.selectbox[0]
    opcao = next(o for o in caixa.options if o.endswith(f"índice {repetidos[0]}"))
    caixa.select_index(caixa.options.index(opcao)).run()
    assert not app.exception
    assert any("mesmo indivíduo" in i.value for i in app.info)
