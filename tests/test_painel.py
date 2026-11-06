"""Guardas do painel de resultados.

O painel lê `reports/tabelas/` e não calcula, não importa `src/`, não usa
vocabulário de risco e concentra as cores num módulo. Só a página de classificação
individual recebe entrada, e ela fala com a API por HTTP em vez de ler artefato. Cada
guarda abaixo verifica uma dessas fronteiras por AST ou pela renderização com
`AppTest`.
"""

import ast
import functools
import hashlib
import html
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

# Página e seção. Projeto e clínica não mostram identificador de decisão.
PAGINAS = {
    "visao_geral": "projeto",
    "desempenho": "clinica",
    "triagem": "clinica",
    "fatores": "clinica",
    "casos_exemplares": "clinica",
    "classificacao": "clinica",
    "amostra": "tecnica",
    "colinearidade": "tecnica",
    "modelagem": "tecnica",
    "floresta_logistica": "tecnica",
    "calibracao": "tecnica",
    "explicabilidade_tecnica": "tecnica",
    "reproducao": "projeto",
}

IMPORTS_PROIBIDOS = {"sklearn", "shap", "joblib", "pickle", "src", "subprocess"}
WIDGETS_PROIBIDOS = {
    "text_input", "number_input", "text_area", "date_input", "time_input",
    "file_uploader", "camera_input", "form", "slider", "select_slider",
    "data_editor", "chat_input", "color_picker", "button",
}
# A reprodução ao vivo é a única página que dispara processo e tem botão.
REPRODUCAO = PAINEL / "paginas" / "reproducao.py"
# A classificação individual é a única página com formulário e a única que fala
# com a API.
CLASSIFICACAO = PAINEL / "paginas" / "classificacao.py"
LIBERADOS = {"subprocess": {REPRODUCAO}, "button": {REPRODUCAO},
             "form": {CLASSIFICACAO}, "number_input": {CLASSIFICACAO}}
LEITURAS = {"read_csv", "open", "read_text"}
VOCABULARIO_PROIBIDO = ["risco", "probabilidade de", "preditiv", "protetor",
                        "acurácia", "acuracia", "accuracy",
                        "referência", "referencia", "modelo principal"]
IDENTIFICADOR_DE_DECISAO = re.compile(r"\bD-\d{3}\b|\bR\d{1,2}\b")
CELULA_LITERAL = re.compile(
    r"(Random Forest|LogReg)\s*·|\b(A_atomico|B_composto|C_completo)\b"
)
COR_HEX = re.compile(r"#[0-9A-Fa-f]{3}(?:[0-9A-Fa-f]{3})?\b")


def _arvore(arquivo: Path) -> ast.AST:
    return ast.parse(arquivo.read_text(encoding="utf-8"), filename=str(arquivo))


def _rel(arquivo: Path) -> str:
    return arquivo.relative_to(RAIZ).as_posix()


def _constantes(arquivo: Path):
    for no in ast.walk(_arvore(arquivo)):
        if isinstance(no, ast.Constant) and isinstance(no.value, str):
            yield no


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
                raiz = modulo.split(".")[0]
                if raiz in IMPORTS_PROIBIDOS and arquivo not in LIBERADOS.get(raiz, ()):
                    violacoes.append(f"{_rel(arquivo)}:{no.lineno} importa {modulo}")
    assert not violacoes, "imports proibidos no painel:\n" + "\n".join(violacoes)


def test_painel_nao_usa_widget_de_entrada():
    violacoes = []
    for arquivo in ARQUIVOS:
        for no in ast.walk(_arvore(arquivo)):
            if (isinstance(no, ast.Attribute) and no.attr in WIDGETS_PROIBIDOS
                    and arquivo not in LIBERADOS.get(no.attr, ())):
                violacoes.append(f"{_rel(arquivo)}:{no.lineno} usa {no.attr}")
            elif isinstance(no, ast.ImportFrom):
                for alias in no.names:
                    if alias.name in WIDGETS_PROIBIDOS:
                        violacoes.append(
                            f"{_rel(arquivo)}:{no.lineno} importa {alias.name}")
    assert not violacoes, "widgets de entrada no painel:\n" + "\n".join(violacoes)


def test_subprocess_e_button_so_na_reproducao():
    """A liberação é por arquivo: a página de reprodução usa os dois, e só ela."""
    usos = {"subprocess": set(), "button": set()}
    for arquivo in ARQUIVOS:
        for no in ast.walk(_arvore(arquivo)):
            if isinstance(no, ast.Import) and any(
                    a.name.split(".")[0] == "subprocess" for a in no.names):
                usos["subprocess"].add(arquivo)
            elif isinstance(no, ast.Attribute) and no.attr == "button":
                usos["button"].add(arquivo)
    assert usos == {"subprocess": {REPRODUCAO}, "button": {REPRODUCAO}}, usos


def test_formulario_e_api_so_na_classificacao():
    """Formulário, `cliente` e `httpx` ficam na página de classificação e no cliente."""
    usos = {"formulario": set(), "cliente": set(), "httpx": set()}
    for arquivo in ARQUIVOS:
        for no in ast.walk(_arvore(arquivo)):
            if isinstance(no, ast.Attribute) and no.attr in (
                    "form", "form_submit_button", "number_input"):
                usos["formulario"].add(arquivo)
            elif isinstance(no, ast.Import):
                for alias in no.names:
                    if alias.name in ("cliente", "httpx"):
                        usos[alias.name].add(arquivo)
    assert usos == {"formulario": {CLASSIFICACAO}, "cliente": {CLASSIFICACAO},
                    "httpx": {CLASSIFICACAO, PAINEL / "cliente.py"}}, usos


def _chamadas_de_page(no: ast.AST) -> list:
    """Caminhos passados a `st.Page` dentro do nó."""
    return [chamada.args[0].value for chamada in ast.walk(no)
            if isinstance(chamada, ast.Call)
            and isinstance(chamada.func, ast.Attribute) and chamada.func.attr == "Page"]


def _paginas_em_secoes() -> list:
    """Caminhos passados a `st.Page` dentro do dicionário `SECOES` de `app.py`."""
    secoes = next(no.value for no in ast.walk(_arvore(PAINEL / "app.py"))
                  if isinstance(no, ast.Assign)
                  and any(isinstance(a, ast.Name) and a.id == "SECOES"
                          for a in no.targets))
    return _chamadas_de_page(secoes)


def test_todo_link_de_navegacao_aponta_para_arquivo_existente():
    todas = _chamadas_de_page(_arvore(PAINEL / "app.py"))
    assert todas and sorted(todas) == sorted(_paginas_em_secoes()), (
        "st.Page fora de SECOES")
    quebrados = [caminho for caminho in todas if not (PAINEL / caminho).is_file()]
    assert not quebrados, f"st.Page aponta para arquivo inexistente: {quebrados}"


def test_todo_arquivo_de_pagina_esta_em_secoes():
    arquivos = {f"paginas/{p.name}" for p in (PAINEL / "paginas").glob("*.py")
                if p.name != "__init__.py"}
    ligados = set(_paginas_em_secoes())
    assert not arquivos - ligados, f"página fora da navegação: {arquivos - ligados}"
    assert {Path(c).stem for c in ligados} == set(PAGINAS), "PAGINAS desatualizado"


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
    violacoes = [f"{_rel(arquivo)}:{no.lineno} {no.value!r}"
                 for arquivo in ARQUIVOS for no in _constantes(arquivo)
                 if CELULA_LITERAL.search(no.value)]
    assert not violacoes, "nome de célula como literal:\n" + "\n".join(violacoes)


def test_cor_hexadecimal_so_em_estilo():
    violacoes = [f"{_rel(arquivo)}:{no.lineno} {COR_HEX.search(no.value).group()}"
                 for arquivo in ARQUIVOS if arquivo != PAINEL / "estilo.py"
                 for no in _constantes(arquivo) if COR_HEX.search(no.value)]
    assert not violacoes, "cor fora de painel/estilo.py:\n" + "\n".join(violacoes)


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

def _novo_app(pagina: str) -> AppTest:
    app = AppTest.from_file(str(PAINEL / "app.py"), default_timeout=60)
    app.run()
    app.switch_page(f"paginas/{pagina}.py")
    app.run()
    return app


@functools.lru_cache(maxsize=None)
def _renderizar(pagina: str) -> AppTest:
    return _novo_app(pagina)


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
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _strings(item)


def _texto_da_tabela(tabela: pd.DataFrame) -> list:
    return ([str(c) for c in tabela.columns] + [str(i) for i in tabela.index]
            + [str(v) for v in tabela.to_numpy().ravel()])


def _html_renderizado(app: AppTest) -> str:
    """HTML passado a `st.markdown`, como chega ao navegador."""
    return "\n".join(m.value for m in app.markdown)


def _texto_renderizado(app: AppTest) -> str:
    """Todo texto que chega à tela: elementos, HTML, tabelas e gráficos.

    O HTML entra sem as tags e com as entidades decodificadas, que é o que o
    leitor vê.
    """
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
    texto = "\n".join(partes)
    return html.unescape(re.sub(r"<[^>]+>", "\n", texto))


def _literais_de_textos() -> list:
    """Frases de `textos.py`, inteiras e quebradas em frases, maiores primeiro."""
    frases = []
    for nome, valor in vars(textos).items():
        if not nome.isupper():
            continue
        for texto in _strings(valor):
            frases.append(texto)
            frases += re.split(r"(?<=\.) ", texto)
    return sorted(set(frases), key=len, reverse=True)


@pytest.mark.parametrize("pagina", PAGINAS)
def test_pagina_renderiza_sem_excecao(pagina):
    app = _renderizar(pagina)
    assert not app.exception, [e.value for e in app.exception]
    corpo = _html_renderizado(app)
    assert html.escape(textos.PAGINAS[pagina][0]) in corpo, "título da página ausente"
    assert "Sprints 1 a 3" not in corpo, "rodapé removido voltou à tela"


@pytest.mark.parametrize("pagina", PAGINAS)
def test_vocabulario_proibido_fora_de_textos(pagina):
    texto = _texto_renderizado(_renderizar(pagina))
    if PAGINAS[pagina] != "tecnica":
        ids = sorted(set(IDENTIFICADOR_DE_DECISAO.findall(texto)))
        assert not ids, f"{pagina}: identificador de decisão na tela: {ids}"
    for literal in _literais_de_textos():
        texto = texto.replace(literal, "")
    minusculo = texto.lower()
    achados = [termo for termo in VOCABULARIO_PROIBIDO if termo in minusculo]
    trechos = [texto[max(0, minusculo.find(t) - 40):minusculo.find(t) + 40]
               for t in achados]
    assert not achados, f"{pagina}: vocabulário proibido {achados} em {trechos}"


def test_cartoes_tem_ajuda_do_glossario():
    # Estático: toda chamada a `metrica` passa `ajuda` indexando o glossário.
    violacoes = []
    for arquivo in ARQUIVOS:
        for no in ast.walk(_arvore(arquivo)):
            if not (isinstance(no, ast.Call) and isinstance(no.func, ast.Attribute)
                    and no.func.attr == "metrica"):
                continue
            ajuda = next((k.value for k in no.keywords if k.arg == "ajuda"), None)
            base = ajuda.value if isinstance(ajuda, ast.Subscript) else None
            do_glossario = (
                (isinstance(base, ast.Attribute) and base.attr == "GLOSSARIO")
                or (isinstance(base, ast.Name) and base.id == "g")
            )
            if not do_glossario:
                violacoes.append(f"{_rel(arquivo)}:{no.lineno}")
    assert not violacoes, "cartão sem ajuda do glossário:\n" + "\n".join(violacoes)

    # Renderizado: todo cartão das onze páginas traz uma ajuda que está no glossário.
    glossario = set(textos.GLOSSARIO.values())
    total = 0
    for pagina in PAGINAS:
        corpo = _html_renderizado(_renderizar(pagina))
        cartoes = re.findall(r'<div class="cartao"[^>]*>', corpo)
        total += len(cartoes)
        for cartao in cartoes:
            ajuda = re.search(r'data-ajuda="([^"]*)"', cartao)
            assert ajuda and html.unescape(ajuda.group(1)) in glossario, (
                f"{pagina}: cartão sem ajuda do glossário: {cartao[:120]}")
    assert total > 0, "nenhum cartão renderizado"


def test_desempenho_exibe_faixa_ada_e_cartao_f1():
    corpo = _html_renderizado(_renderizar("desempenho"))
    assert html.escape(textos.COMPARADOR_ADA) in corpo, "faixa da regra ADA ausente"
    cartao_f1 = re.search(
        r'<div class="cartao"[^>]*data-ajuda="([^"]*)"[^>]*><div class="rotulo">F1<',
        corpo,
    )
    assert cartao_f1, "cartão de F1 ausente"
    assert html.unescape(cartao_f1.group(1)) == textos.GLOSSARIO["f1"]


def _caixa_de_casos(app: AppTest):
    return next(c for c in app.selectbox if str(c.key).startswith("caso_exemplar"))


def _selecionar_celula(app: AppTest, modelo: str, conjunto: str) -> None:
    app.selectbox(key="selecao_modelo").select(modelo).run()
    app.selectbox(key="selecao_conjunto").select(conjunto).run()
    assert not app.exception, [e.value for e in app.exception]


def test_casos_exemplares_exibe_faixa_ressalva_e_valor_observado():
    # Instância nova: o teste troca a seleção e não pode sujar o cache.
    app = _novo_app("casos_exemplares")
    corpo = _html_renderizado(app)
    assert html.escape(textos.FAIXA_CASOS) in corpo, "faixa de não classificação ausente"
    assert html.escape(textos.RESSALVA_LOCAL) in corpo, "ressalva fora do quadro"

    grafico = app.get("vega_lite_chart")[0]
    subtitulo = json.loads(grafico.proto.spec)["title"]["subtitle"]
    assert " ".join(subtitulo) == textos.RESSALVA_LOCAL, "ressalva fora do gráfico"
    assert '<div class="campo">' in corpo, "valores observados ausentes"

    # Índice presente nas duas células: a tela diz que é o mesmo indivíduo.
    casos = dados.ler("shap_casos_locais")
    repetidos = list(casos[casos["indice_nas_duas_celulas"]]["indice"].unique())
    assert repetidos, "o artefato deixou de ter índice repetido entre células"
    caixa = _caixa_de_casos(app)
    opcao = next(o for o in caixa.options if o.endswith(f"índice {repetidos[0]}"))
    caixa.select_index(caixa.options.index(opcao)).run()
    assert not app.exception
    assert "mesmo indivíduo" in _html_renderizado(app)


def test_waterfall_ajusta_so_a_largura():
    """Com `fit`, padrão do Streamlit, título e legenda comem a altura das barras e
    os rótulos de linhas vizinhas se sobrepõem. O waterfall declara `fit-x`."""
    app = _renderizar("casos_exemplares")
    specs = [json.loads(no.proto.spec) for no in _nos(app._tree)
             if getattr(no, "type", "") == "vega_lite_chart"]
    assert specs, "waterfall não renderizado"
    assert [s.get("autosize", {}).get("type") for s in specs] == ["fit-x"] * len(specs)


def test_casos_exemplares_seguem_a_celula_da_barra_lateral():
    """Cada célula da floresta mostra os seus quatro casos, e o selo diz qual é."""
    casos = dados.ler("shap_casos_locais")
    app = _novo_app("casos_exemplares")
    for (modelo, conjunto), grupo in casos.groupby(["modelo", "celula"]):
        _selecionar_celula(app, modelo, conjunto)
        esperados = grupo[["caso", "indice"]].drop_duplicates()
        opcoes = _caixa_de_casos(app).options
        assert len(opcoes) == len(esperados) == 4, (modelo, conjunto, opcoes)
        for caso, indice in esperados.itertuples(index=False):
            assert f"{textos.CASOS[caso]} · índice {indice}" in opcoes
        selo = textos.SELO_CELULA.format(celula=f"{modelo} · {conjunto}")
        assert html.escape(selo) in _html_renderizado(app), "selo de célula errado"


def test_casos_exemplares_avisam_quando_a_celula_nao_tem_shap():
    """Célula calibrável sem SHAP, como a logística, mostra o aviso e nenhum caso."""
    casos = dados.ler("shap_casos_locais")
    consolidado = dados.ler("comparativo_consolidado")
    com_shap = set(zip(casos["modelo"], casos["celula"]))
    sem_shap = [(m, c) for m, c, unico in consolidado[
                    ["modelo", "conjunto", "corte_unico"]].itertuples(index=False)
                if not unico and (m, c) not in com_shap]
    assert sem_shap, "toda célula calibrável passou a ter SHAP"
    app = _novo_app("casos_exemplares")
    _selecionar_celula(app, *sem_shap[0])
    corpo = _html_renderizado(app)
    assert html.escape(textos.SHAP_SO_FLORESTA) in corpo, "aviso de SHAP ausente"
    assert not [c for c in app.selectbox if str(c.key).startswith("caso_exemplar")]
    assert not app.get("vega_lite_chart"), "waterfall sem caso da célula"


def test_triagem_exibe_rotulo_do_regime_e_regime_do_custo_em_texto():
    corpo = _html_renderizado(_renderizar("triagem"))
    carga = dados.ler("carga_operacional")
    for _, linha in carga.iterrows():
        assert html.escape(linha["rotulo"]) in corpo, linha["rotulo"]
        if linha["rotulo"] != linha["regime"]:
            assert html.escape(linha["regime"]) not in corpo, linha["regime"]
    custo = dados.ler("custo_marginal_pareado")
    assert set(custo["regime"]) == set(textos.REGIMES_CUSTO)
    for chave, rotulo in textos.REGIMES_CUSTO.items():
        assert f"<td>{html.escape(rotulo)}</td>" in corpo, rotulo
        assert f"<td>{chave}</td>" not in corpo, chave


def test_reproducao_antes_do_clique():
    """Sem clicar: comando, tempo esperado, o que não é regenerado e o botão."""
    app = _renderizar("reproducao")
    corpo = _html_renderizado(app)
    assert textos.COMANDO_REPRODUCAO in [c.value for c in app.code], "comando ausente"
    assert html.escape(textos.REPRODUCAO_TEMPO) in corpo
    assert html.escape(textos.REPRODUCAO_FORA) in corpo
    assert app.button(key="reproduzir").label == textos.BOTAO_REPRODUZIR
    for rotulo, _ in textos.REPRODUCAO_STATUS.values():
        assert html.escape(rotulo) not in corpo, "status antes da execução"


def test_reproducao_avisa_quando_o_interpretador_nao_importa_o_experimento(monkeypatch):
    """Python sem sklearn: aviso com interpretador e correção, e nenhum experimento."""
    erro = "ModuleNotFoundError: No module named 'sklearn'"

    def import_falha(comando, **kwargs):
        assert comando[1:] == ["-c", "import src.experimento"], comando
        return subprocess.CompletedProcess(comando, 1, "", f"Traceback\n{erro}\n")

    def popen_proibido(*args, **kwargs):
        raise AssertionError("experimento disparado com ambiente quebrado")

    monkeypatch.setattr(subprocess, "run", import_falha)
    monkeypatch.setattr(subprocess, "Popen", popen_proibido)
    app = _novo_app("reproducao")
    app.button(key="reproduzir").click().run()
    assert not app.exception, [e.value for e in app.exception]

    corpo = html.unescape(_html_renderizado(app))
    assert textos.AMBIENTE_TITULO in corpo
    assert erro in corpo, "erro do import ausente"
    assert textos.AMBIENTE_CORRECAO in corpo
    assert textos.COMANDO_PAINEL_VENV in [c.value for c in app.code]
    for rotulo, _ in textos.REPRODUCAO_STATUS.values():
        assert rotulo not in corpo, "status de reprodução sem reprodução"


def _hashes(diretorio: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(diretorio.iterdir()) if p.is_file()}


@pytest.mark.lento
def test_reproducao_pelo_botao_confere_e_nao_toca_nas_tabelas():
    """Clique no botão: status 0, conferência na tela e `reports/tabelas/` intacto."""
    if not (RAIZ / "data" / "raw" / "EXAMES_PNS_2013_BRUTA.csv").exists():
        pytest.skip("CSV bruto ausente (não versionado).")
    antes = _hashes(dados.RAIZ)
    app = AppTest.from_file(str(PAINEL / "app.py"), default_timeout=420)
    app.run()
    app.switch_page("paginas/reproducao.py")
    app.run()
    app.button(key="reproduzir").click().run()
    assert not app.exception, [e.value for e in app.exception]

    corpo = _html_renderizado(app)
    log = "\n".join(c.value for c in app.code)
    ok, _ = textos.REPRODUCAO_STATUS[True]
    assert html.escape(ok) in corpo, log[-3000:]
    assert '<div class="linha-log"' in corpo, "linhas de conferência sem destaque"
    assert "Composição conferida" in corpo and "Reprodução conferida" in corpo
    assert "comparativo_consolidado" in corpo, "lista de artefatos ausente"
    assert _hashes(dados.RAIZ) == antes, "reports/tabelas/ mudou durante a reprodução"


# ---------------------------------------------------------------------------
# Classificação individual, com a API do cliente de teste
# ---------------------------------------------------------------------------

def _requer_artefato():
    from src import config as cfg
    from src import servir as sv
    if not cfg.PATH_RAW.exists() or not (sv.PATH_MODELO / sv.ARQUIVO_METADADOS).exists():
        pytest.skip("CSV bruto ou artefato ausente; rode python -m src.servir.")


@pytest.fixture(scope="module")
def api():
    _requer_artefato()
    from fastapi.testclient import TestClient
    from api.app import criar_app
    return TestClient(criar_app())


@pytest.fixture(scope="module")
def casos_b():
    """Os quatro casos da célula servida, com o registro clínico de cada um."""
    _requer_artefato()
    from src import data as dt
    from src import servir as sv
    from tests.test_api import _payload
    analitico = dt.carregar_analitico()
    tabela = dados.carregar("shap_casos_locais")
    recorte = tabela[tabela["celula"] == sv.CONJUNTO]
    unicos = recorte[["caso", "indice", "escore"]].drop_duplicates().set_index("caso")
    return {caso: (_payload(analitico.loc[int(linha.indice)]), linha.escore)
            for caso, linha in unicos.iterrows()}


@pytest.fixture
def cliente_api(monkeypatch):
    """O `cliente` que a página importa, o mesmo objeto que o `AppTest` usa."""
    monkeypatch.syspath_prepend(str(PAINEL))
    import cliente
    return cliente


@pytest.fixture
def classificacao(cliente_api, api, monkeypatch):
    monkeypatch.setattr(cliente_api, "ler_metadados",
                        lambda: api.get("/metadados").json())
    monkeypatch.setattr(cliente_api, "classificar",
                        lambda registro: api.post("/classificacao", json=registro))
    app = AppTest.from_file(str(PAINEL / "app.py"), default_timeout=60)
    app.run()
    app.switch_page("paginas/classificacao.py")
    return app.run()


def _enviar(app: AppTest, corpo: dict) -> AppTest:
    for campo, valor in corpo.items():
        if campo in CATEGORICAS_ENTRADA:
            app.selectbox(key=campo).set_value(valor)
        else:
            app.number_input(key=campo).set_value(valor)
    app.button[0].click().run()
    assert not app.exception, [e.value for e in app.exception]
    return app


CATEGORICAS_ENTRADA = {"sexo", "raca_cor", "regiao", "escolaridade", "saude_autoaval",
                       "plano_saude", "has_autorrelato", "colesterol_alto"}
ENCAMINHA = {"VP": True, "FP": True, "FN": False, "VN": False}


@pytest.mark.parametrize("caso", ["VP", "FP", "FN", "VN"])
def test_classificacao_exibe_decisao_e_escore_dos_casos(classificacao, casos_b, caso):
    corpo, escore = casos_b[caso]
    app = _enviar(classificacao, corpo)
    texto = _html_renderizado(app)
    decisao = "Encaminhar a exame" if ENCAMINHA[caso] else "Não encaminhar"
    assert html.escape(decisao) in texto, caso
    assert html.escape(f"{escore:.3f}".replace(".", ",")) in texto, caso


def test_classificacao_sem_vocabulario_vetado_fora_da_restricao(classificacao, casos_b):
    from api import textos as textos_api
    app = _enviar(classificacao, casos_b["VP"][0])
    texto = _texto_renderizado(app)
    restricao = textos_api.RESTRICAO_DE_LEITURA
    assert restricao in texto, "restrição de leitura ausente da tela"
    # A restrição vem da API e é a única constante liberada; o waterfall a quebra em
    # frases no subtítulo. Sai antes dos literais do painel, que poderiam cortá-la.
    for frase in sorted({restricao, *re.split(r"(?<=\.) ", restricao)},
                        key=len, reverse=True):
        texto = texto.replace(frase, "")
    for literal in _literais_de_textos():
        texto = texto.replace(literal, "")
    minusculo = texto.lower()
    achados = [t for t in VOCABULARIO_PROIBIDO if t in minusculo]
    assert not achados, achados


def test_classificacao_marca_ausente_como_imputado(classificacao, casos_b):
    corpo = {**casos_b["VP"][0], "colesterol_alto": None}
    app = _enviar(classificacao, corpo)
    texto = _texto_renderizado(app)
    assert "Colesterol alto autorreferido" in _html_renderizado(app)
    assert "[imputada]" in texto


def test_classificacao_mostra_recusa_da_api(classificacao, casos_b):
    app = _enviar(classificacao, {**casos_b["VP"][0], "pas": None})
    assert "recusou" in app.error[0].value and "pas" in app.error[0].value


def test_classificacao_avisa_api_fora_do_ar(cliente_api, monkeypatch):
    import httpx

    def recusa():
        raise httpx.ConnectError("sem conexão")
    monkeypatch.setattr(cliente_api, "ler_metadados", recusa)
    app = AppTest.from_file(str(PAINEL / "app.py"), default_timeout=60)
    app.run()
    app.switch_page("paginas/classificacao.py")
    app.run()
    assert not app.exception
    assert "make api" in _html_renderizado(app)
