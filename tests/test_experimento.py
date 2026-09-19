"""Contratos da fábrica de estimadores e do grid.

Dois invariantes sustentam a reprodutibilidade e nenhum dos dois é óbvio pela
leitura do código:

1. `models.estimadores` **delega** os comparadores ao `benchmark`. Se algum dia
   alguém os reinstanciar aqui, passam a existir duas configurações para o mesmo
   objeto e elas divergem em silêncio.
2. O produto cartesiano **não é cheio**. Estimador invariante ao conjunto rende
   uma linha, não três. Cartesiano cheio faria a tabela deixar de bater com o
   `ESTADO.md` por artefato do grid: divergência falsa em cima da verificação que
   existe justamente para dar confiança.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src import benchmark as bm
from src import config as cfg
from src import experimento as xp
from src import models as md


# ---------------------------------------------------------------------------
# Fábrica
# ---------------------------------------------------------------------------

def test_estimadores_traz_os_quatro():
    """Logística, Random Forest e os dois pisos não-ML."""
    grid = md.estimadores()
    assert set(grid) == {"LogReg", "Random Forest", *bm.construir_comparadores()}


def test_comparadores_sao_delegados_e_nao_reconstruidos():
    """A configuração dos pisos continua sendo do `benchmark`, não desta fábrica."""
    do_benchmark = bm.construir_comparadores()
    grid = md.estimadores()
    for nome, referencia in do_benchmark.items():
        montado = md.montar(grid[nome], conjunto=None)
        assert type(montado) is type(referencia)
        assert montado.get_params() == referencia.get_params()


def test_pisos_marcados_como_invariantes_e_com_limiar_fixo():
    """A regra lê idade e IMC crus; o Dummy ignora X. Nenhum depende do conjunto."""
    grid = md.estimadores()
    for nome in bm.construir_comparadores():
        assert grid[nome].invariante_ao_conjunto
        assert grid[nome].limiar_fixo == bm.LIMIAR_INDICADORA


def test_modelos_dependem_do_conjunto():
    """LogReg e RF saem embrulhados no preprocessador."""
    grid = md.estimadores()
    for nome in ("LogReg", "Random Forest"):
        assert not grid[nome].invariante_ao_conjunto
        assert grid[nome].limiar_fixo is None
        montado = md.montar(grid[nome], "B_composto")
        assert isinstance(montado, Pipeline)
        assert list(dict(montado.steps)) == ["pre", "clf"]


def test_montar_sem_conjunto_falha_para_quem_depende_dele():
    """Erro explícito em vez de conjunto silenciosamente errado."""
    with pytest.raises(ValueError, match="depende do conjunto"):
        md.montar(md.estimadores()["LogReg"], conjunto=None)


def test_piso_nao_recebe_preprocessador():
    """O corte de IMC 30 da regra ADA não sobrevive a uma padronização."""
    montado = md.montar(md.estimadores()["Regra ADA"], conjunto="B_composto")
    assert not isinstance(montado, Pipeline)


def test_parametros_metodologicos_dos_modelos():
    """Cost-sensitive e semente fixa valem para os dois algoritmos, cada um no seu grau.

    A exigência é por estimador, e não uma lista branca comum, para o teste marcar onde
    a variação mora. A logística tem que trazer o literal `"balanced"`, que é o que o
    texto assinado nomeia e onde não existe variação a admitir; afrouxar ali perderia a
    conformidade literal sem ganho nenhum. A floresta aceita os dois por D-034, porque
    com bootstrap o peso precisa ser recalculado dentro de cada subamostra e `balanced`
    não faz isso.

    O que está sob guarda nos dois casos é R4: reponderamento de classes no lugar de
    reamostragem sintética.
    """
    admitidos = {md.logistica: ("balanced",),
                 md.random_forest: ("balanced", "balanced_subsample")}
    for construir, esperados in admitidos.items():
        params = construir().get_params()
        assert params["class_weight"] in esperados, (
            f"{construir.__name__}: class_weight {params['class_weight']!r} fora de "
            f"{esperados}."
        )
        assert params["random_state"] == cfg.RANDOM_STATE


def test_dummy_nao_e_compartilhado_entre_celulas():
    """Cada célula recebe objeto novo; nada carrega ajuste de outra."""
    grid = md.estimadores()
    a, b = md.montar(grid["Dummy (maioria)"]), md.montar(grid["Dummy (maioria)"])
    assert a is not b
    assert isinstance(a, DummyClassifier)


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def particao():
    if not cfg.PATH_RAW.exists():
        pytest.skip(f"CSV bruto ausente em {cfg.PATH_RAW} (não versionado).")
    return xp.preparar_particao()


@pytest.fixture(scope="module")
def grid_dois_conjuntos(particao):
    """LogReg sobre dois conjuntos mais os dois pisos: 4 linhas, não 6."""
    return xp.rodar_grid(conjuntos=["A_atomico", "B_composto"],
                         particao=particao, salvar_figuras=False)


def test_produto_cartesiano_nao_e_cheio(grid_dois_conjuntos):
    """Dois modelos calibráveis × 2 conjuntos + 2 pisos × 1 = 6 linhas, não 8."""
    linhas = [(r["modelo"], r["conjunto"]) for r in grid_dois_conjuntos]
    assert linhas == [
        ("LogReg", "A_atomico"),
        ("LogReg", "B_composto"),
        (xp.NOME_FLORESTA, "A_atomico"),
        (xp.NOME_FLORESTA, "B_composto"),
        ("Regra ADA", xp.SEM_CONJUNTO),
        ("Dummy (maioria)", xp.SEM_CONJUNTO),
    ]


def test_pisos_aparecem_uma_vez_so(grid_dois_conjuntos):
    """O sintoma do cartesiano cheio seria linhas idênticas repetidas."""
    for nome in bm.construir_comparadores():
        assert sum(r["modelo"] == nome for r in grid_dois_conjuntos) == 1


def test_pisos_nao_passam_por_calibracao(grid_dois_conjuntos):
    """Calibrar sobre indicadora converteria o Dummy em "sempre positivo"."""
    for r in grid_dois_conjuntos:
        if r["modelo"] in bm.construir_comparadores():
            assert r["limiar_calibrado"] is False
            assert r["limiar"] == bm.LIMIAR_INDICADORA
    dummy = next(r for r in grid_dois_conjuntos if r["modelo"] == "Dummy (maioria)")
    assert dummy["recall"] == 0.0  # o piso continua sendo piso


def test_floresta_no_padrao_mas_fora_do_conjunto_descartado():
    """A floresta publica em A e B, e não em `C_completo`.

    `C_completo` saiu pelo diagnóstico de multicolinearidade (D-016), e publicar
    desempenho de conjunto descartado sugere que ele ainda está em disputa. A logística
    continua nos três porque a linha de C é o lastro empírico do descarte (D-042).
    """
    assert xp.NOME_FLORESTA in xp.MODELOS_PADRAO
    assert "C_completo" not in xp.CONJUNTOS_FLORESTA

    celulas = xp.composicao_esperada()
    assert (xp.NOME_FLORESTA, "C_completo") not in celulas
    assert ("LogReg", "C_completo") in celulas


def test_composicao_publicada_tem_sete_celulas():
    """Logística em três conjuntos, floresta em dois, mais os dois pisos."""
    assert xp.composicao_esperada() == [
        ("LogReg", "A_atomico"),
        ("LogReg", "B_composto"),
        ("LogReg", "C_completo"),
        (xp.NOME_FLORESTA, "A_atomico"),
        (xp.NOME_FLORESTA, "B_composto"),
        ("Regra ADA", xp.SEM_CONJUNTO),
        ("Dummy (maioria)", xp.SEM_CONJUNTO),
    ]


def test_conferencia_de_composicao_acusa_linha_removida():
    """Linha que some tem que ser acusada, e a conferência de valores não acusa.

    A conferência de reprodução intersecta índices: ela compara o que os dois lados têm
    em comum e uma tabela truncada passa com os números todos certos. Foi assim que a
    consolidada sobreviveu com duas linhas e a suíte verde (D-044).
    """
    completa = pd.DataFrame(xp.composicao_esperada(), columns=["modelo", "conjunto"])
    assert xp.conferir_composicao(completa).empty

    truncada = completa.drop(index=4).reset_index(drop=True)
    faltando = xp.conferir_composicao(truncada)
    assert len(faltando) == 1
    assert faltando.loc[0, "situacao"] == "ausente do emitido"
    assert "B_composto" in faltando.loc[0, "linha"]

    sobrando = xp.conferir_composicao(
        pd.concat([completa,
                   pd.DataFrame([{"modelo": xp.NOME_FLORESTA,
                                  "conjunto": "C_completo"}])],
                  ignore_index=True)
    )
    assert list(sobrando["situacao"]) == ["inesperada no emitido"]


def test_modelo_desconhecido_falha_cedo(particao):
    with pytest.raises(ValueError, match="desconhecido"):
        xp.rodar_grid(modelos=["XGBoost"], particao=particao)


def test_modelo_de_referencia_ignora_os_pisos(grid_dois_conjuntos):
    """O Dummy tem NNS infinito e a regra opera em limiar fixo: nenhum é candidato.

    O critério continua sendo o menor NNS entre os calibráveis, sem mudança (D-030). O
    que mudou foi quem ganha: a floresta entrou no grid e ficou com o menor NNS, então
    ela assume a carga operacional. Isso é consequência do critério, não escolha nova, e
    o desempate por parcimônia de D-038 não chega a ser acionado porque não houve empate
    em recall.
    """
    escolhido = xp.modelo_de_referencia(grid_dois_conjuntos)
    calibraveis = [r for r in grid_dois_conjuntos if r.get("limiar_calibrado", True)]

    assert escolhido["modelo"] not in bm.construir_comparadores()
    assert escolhido["limiar_calibrado"] is True
    assert escolhido["nns"] == min(r["nns"] for r in calibraveis)


# ---------------------------------------------------------------------------
# Conferência de reprodução
# ---------------------------------------------------------------------------

def test_referencia_da_sprint1_esta_versionada():
    """Sem ela não há âncora: a conferência passaria por vacuidade."""
    assert (cfg.PATH_TABELAS / xp.REFERENCIA).exists()


def test_as_duas_ancoras_estao_versionadas():
    """A da logística cobre as três linhas dela; a da floresta cobre as duas novas.

    A antiga ganhou função a mais: com a floresta no grid, ela prova que acrescentá-la
    não moveu nenhum número da logística. Se mover, é defeito no grid (D-039).
    """
    for nome in xp.REFERENCIAS:
        assert (cfg.PATH_TABELAS / nome).exists(), f"âncora ausente: {nome}"


def test_tabelas_versionadas_trazem_as_sete_linhas():
    """Contagem, e não só valor.

    Esta é a guarda que faltava: as conferências comparam a interseção dos índices e
    deixam passar tabela truncada. Aqui o artefato versionado é medido pelo número de
    linhas e pela composição (D-044).
    """
    esperada = xp.composicao_esperada()

    assert len(esperada) == 7

    for nome in (xp.NOME_CONSOLIDADO, xp.NOME_CALIBRACAO):
        emitida = pd.read_csv(cfg.PATH_TABELAS / f"{nome}.csv")
        # A composição vem antes da contagem de propósito: ela nomeia a célula que
        # falta, e `6 != 7` sozinho não diz qual linha sumiu.
        faltando = xp.conferir_composicao(emitida, esperada)
        assert faltando.empty, (
            f"{nome}.csv com composição errada:\n{faltando.to_string(index=False)}"
        )
        assert len(emitida) == len(esperada), (
            f"{nome}.csv tem {len(emitida)} linhas, esperadas {len(esperada)}."
        )


def test_conferencia_acusa_divergencia_injetada(grid_dois_conjuntos):
    """Uma conferência que nunca reprova não verifica nada."""
    from src import evaluate as ev

    tabela = ev.comparar(grid_dois_conjuntos).copy()
    tabela.loc[tabela["conjunto"] == "A_atomico", "recall"] += 0.01
    divergencias = xp.conferir_reproducao(tabela, exigir_cobertura=False)
    assert not divergencias.empty
    assert (divergencias["coluna"] == "recall").any()


def test_conferencia_tolera_divergencia_abaixo_da_terceira_casa(grid_dois_conjuntos):
    """O contrato é três casas decimais, não igualdade de ponto flutuante."""
    from src import evaluate as ev

    tabela = ev.comparar(grid_dois_conjuntos).copy()
    tabela["recall"] += 1e-6
    assert xp.conferir_reproducao(tabela, exigir_cobertura=False).empty


def test_conferencia_compara_so_as_colunas_que_a_referencia_cobre(grid_dois_conjuntos):
    """A âncora é anterior às colunas de dispersão e de calibração.

    Coluna que a âncora não tem fica de fora da comparação. Linha que ela tem, não: a
    tabela de dois conjuntos não traz `C_completo`, e isso passa a ser acusado.
    """
    from src import evaluate as ev

    tabela = ev.comparar(grid_dois_conjuntos)
    assert xp.conferir_reproducao(tabela, exigir_cobertura=False).empty

    faltando = xp.conferir_reproducao(tabela)
    assert list(faltando["coluna"].unique()) == ["(linha inteira)"]
    assert faltando["linha"].str.contains("C_completo").all()


# ---------------------------------------------------------------------------
# Carga operacional
# ---------------------------------------------------------------------------

def test_carga_operacional_tem_tres_regimes(particao, grid_dois_conjuntos):
    tabela = xp.carga_operacional(grid_dois_conjuntos, particao)
    assert len(tabela) == 3
    assert tabela["regime"].iloc[0] == "Regra ADA"
    assert "recall equiparado" in tabela["regime"].iloc[1]
    assert "ponto padrão" in tabela["regime"].iloc[2]


@pytest.mark.xfail(
    strict=True,
    reason="O critério de sucesso do projeto não é cumprido no regime equiparado pelo "
           "modelo de referência atual: a floresta detecta mais casos que a regra e "
           "gasta mais encaminhamento por caso, NNS 9,53 contra 9,46 (D-046). A falha "
           "fica visível de propósito, e não é ajustada. `strict=True` porque se o "
           "critério voltar a ser cumprido a suíte tem que avisar, em vez de passar "
           "calada.",
)
def test_regime_equiparado_supera_a_regra_em_nns(particao, grid_dois_conjuntos):
    """O argumento é esse: mais casos por encaminhamento, não menos volume."""
    tabela = xp.carga_operacional(grid_dois_conjuntos, particao)
    regra, equiparado = tabela.iloc[0], tabela.iloc[1]
    assert equiparado["nns"] < regra["nns"]
    assert equiparado["casos_detectados"] > regra["casos_detectados"]


def test_regime_equiparado_calibra_pelo_recall_da_regra_no_treino(particao,
                                                                 grid_dois_conjuntos):
    """O mecanismo do regime equiparado, sem congelar quem ganha.

    A guarda que compara NNS contra a regra está marcada `xfail`, porque o critério de
    sucesso deixou de ser cumprido (D-046), e sob `xfail` nada mais do corpo dela roda.
    O que se perdeu ali não era resultado, era mecanismo, e mecanismo é o que teste
    verifica: de onde sai o alvo e se a calibração mirou nele.

    Este teste **não afirma** que o modelo detecta mais casos que a regra. Afirma que o
    alvo veio do recall da regra medido no treino, e não no teste, e que o recall
    realizado caiu na vizinhança dele.
    """
    alvo = bm.recall_da_regra_no_treino(particao.X_tr, particao.y_tr)
    tabela = xp.carga_operacional(grid_dois_conjuntos, particao)
    regra, equiparado = tabela.iloc[0], tabela.iloc[1]

    # O alvo publicado no rótulo é o do treino. Se alguém trocasse a medição para o
    # teste, o vazamento passaria despercebido: os dois números são próximos e o
    # resultado continuaria plausível. Por isso o teste exige um e recusa o outro.
    assert f"alvo {alvo:.4f} no treino" in equiparado["regime"]
    assert f"alvo {regra['recall']:.4f}" not in equiparado["regime"]
    assert alvo != pytest.approx(regra["recall"])

    # O limiar é fixado fora da amostra de avaliação, então o recall realizado desvia
    # do alvo. O desvio é esperado; o que não pode é o realizado ignorar o alvo.
    assert equiparado["recall"] == pytest.approx(alvo, abs=0.10)
    # E o regime equiparado tem que operar acima do ponto padrão, que é o que o
    # distingue dele: alvo maior, mais gente encaminhada.
    assert equiparado["encaminhados"] > tabela.iloc[2]["encaminhados"]


def test_leitura_do_equiparado_segue_o_sinal_do_nns(particao, grid_dois_conjuntos):
    """A frase gerada lê o número, em vez de afirmar vitória por escrito.

    A versão anterior trazia "mantendo NNS inferior" fixo no texto. Com a floresta na
    referência isso passou a contradizer o número ao lado, e texto gerado que contradiz
    o próprio dado é pior que texto ausente: ele vai para o documento sem ninguém
    conferir.
    """
    tabela = xp.carga_operacional(grid_dois_conjuntos, particao)
    regra, equiparado = tabela.iloc[0], tabela.iloc[1]
    leitura = tabela.attrs["leitura"]

    if equiparado["nns"] > regra["nns"]:
        assert "NNS superior" in leitura and "mantendo NNS inferior" not in leitura
    elif equiparado["nns"] < regra["nns"]:
        assert "mantendo NNS inferior" in leitura


def test_carga_operacional_carrega_a_advertencia(particao, grid_dois_conjuntos):
    """Nenhuma das duas tabelas deve circular sem a nota de volume absoluto."""
    tabela = xp.carga_operacional(grid_dois_conjuntos, particao)
    assert tabela.attrs["nota"] == bm.NOTA_VOLUME_ABSOLUTO
    assert "recall equiparado" in tabela.attrs["leitura"]


# ---------------------------------------------------------------------------
# Regime equiparado, célula a célula
# ---------------------------------------------------------------------------

def test_equiparado_cobre_toda_celula_calibravel(particao, grid_dois_conjuntos):
    """Uma linha por célula calibrável do grid, mais a da regra.

    `carga_operacional` equipara só o modelo de referência, e o número que falta é
    metade do contraste que sustenta o achado da sprint: no regime equiparado a
    floresta gasta mais encaminhamento por caso que a logística. Publicar a comparação
    emitindo um lado só repete a lacuna que D-027 fechou (D-047).
    """
    tabela = xp.equiparado_por_modelo(grid_dois_conjuntos, particao)
    calibraveis = [r for r in grid_dois_conjuntos if r.get("limiar_calibrado", True)]

    assert len(tabela) == len(calibraveis) + 1
    assert tabela["estrategia"].iloc[0] == "Regra ADA"
    assert set(zip(tabela["modelo"].iloc[1:], tabela["conjunto"].iloc[1:])) == {
        (r["modelo"], r["conjunto"]) for r in calibraveis
    }
    assert {"recall_alvo", "recall_obtido", "recall_regra",
            "encaminhados", "casos_detectados", "nns"} <= set(tabela.columns)


def test_equiparado_calibra_pelo_recall_da_regra_no_treino(particao,
                                                           grid_dois_conjuntos):
    """O alvo vem do treino, e é o mesmo para todas as células.

    Calibrar pelo recall que a regra obtém no **teste** vazaria o teste para dentro da
    decisão de limiar, e o vazamento passaria despercebido: os dois números são
    próximos e o resultado continuaria plausível. Por isso o teste exige um e recusa o
    outro.
    """
    alvo = bm.recall_da_regra_no_treino(particao.X_tr, particao.y_tr)
    tabela = xp.equiparado_por_modelo(grid_dois_conjuntos, particao)
    regra = next(r for r in grid_dois_conjuntos if r["modelo"] == "Regra ADA")

    assert pd.isna(tabela["recall_alvo"].iloc[0])   # a regra opera em limiar fixo
    assert (tabela["recall_alvo"].iloc[1:] == alvo).all()
    assert alvo != pytest.approx(regra["recall"])
    assert (tabela["recall_alvo"] != tabela["recall_regra"]).iloc[1:].all()

    # O limiar é fixado fora da amostra de avaliação, então o realizado desvia do
    # alvo. O desvio é esperado; o que não pode é o realizado ignorar o alvo.
    assert tabela["recall_obtido"].iloc[1:].to_numpy() == pytest.approx(alvo, abs=0.10)


def test_equiparado_bate_com_a_carga_operacional(particao, grid_dois_conjuntos):
    """As duas tabelas medem a mesma célula por caminhos independentes.

    A linha do modelo de referência é reavaliada aqui em vez de aproveitada da carga
    operacional, e é isso que faz uma conferir a outra. Divergência entre elas significa
    que uma das duas mudou de alvo, de partição ou de estimador sem que nada acusasse.
    """
    carga = xp.carga_operacional(grid_dois_conjuntos, particao)
    referencia = xp.modelo_de_referencia(grid_dois_conjuntos)
    tabela = xp.equiparado_por_modelo(grid_dois_conjuntos, particao)

    linha = tabela[(tabela["modelo"] == referencia["modelo"])
                   & (tabela["conjunto"] == referencia["conjunto"])].iloc[0]
    da_carga = carga.iloc[1]

    assert linha["encaminhados"] == da_carga["encaminhados"]
    assert linha["casos_detectados"] == da_carga["casos_detectados"]
    assert round(linha["nns"], 3) == round(da_carga["nns"], 3)
    assert round(linha["recall_obtido"], 3) == round(da_carga["recall"], 3)


def test_equiparado_exige_a_regra(particao, grid_dois_conjuntos):
    sem_regra = [r for r in grid_dois_conjuntos if r["modelo"] != "Regra ADA"]
    with pytest.raises(ValueError, match="ausente"):
        xp.equiparado_por_modelo(sem_regra, particao)


def test_artefato_equiparado_traz_as_seis_linhas_publicadas():
    """O artefato versionado, com os dois lados do contraste dentro dele.

    Cinco células calibráveis mais a regra. A linha da logística em `B_composto` é o
    número que o `ESTADO.md` e D-046 citavam sem ter `.csv` para apontar, e agora tem.
    Está fixado aqui de propósito: quando uma guarda que fixa resultado medido falha,
    investiga-se, não se ajusta a asserção (D-016, D-046).
    """
    caminho = cfg.PATH_TABELAS / f"{xp.NOME_EQUIPARADO}.csv"
    assert caminho.exists(), f"artefato ausente: {caminho}"

    emitido = pd.read_csv(caminho)
    calibraveis = [c for c in xp.composicao_esperada()
                   if c[1] != xp.SEM_CONJUNTO]
    assert len(emitido) == len(calibraveis) + 1 == 6

    logreg = emitido[(emitido["modelo"] == "LogReg")
                     & (emitido["conjunto"] == "B_composto")].iloc[0]
    assert logreg["encaminhados"] == 1000
    assert logreg["casos_detectados"] == 109
    assert round(logreg["recall_obtido"], 3) == 0.916
    assert round(logreg["nns"], 6) == 9.174312

    floresta = emitido[(emitido["modelo"] == xp.NOME_FLORESTA)
                       & (emitido["conjunto"] == "B_composto")].iloc[0]
    da_carga = pd.read_csv(cfg.PATH_TABELAS / f"{xp.NOME_CARGA}.csv").iloc[1]
    assert floresta["encaminhados"] == da_carga["encaminhados"]
    assert round(floresta["nns"], 3) == round(da_carga["nns"], 3)


# ---------------------------------------------------------------------------
# Gravação dos CSV
# ---------------------------------------------------------------------------

def test_csv_sai_arredondado(tabelas_preservadas):
    """Ponto flutuante cru no CSV fazia `reports/tabelas/` diferir a cada execução.

    O `n_jobs=-1` da floresta muda a ordem em que os processos reduzem, e o último bit
    acompanha. Nenhum resultado muda, o `.tex` nem mostra, mas o diretório versionado
    passa a ter diff permanente, e é o único do projeto onde ninguém pode se acostumar
    a passar o olho sem ler (D-048).
    """
    from src import artefatos as art

    tabela = pd.DataFrame([{"modelo": "X", "conjunto": "—",
                            "valor": 0.1234567891234567, "n": 7, "flag": True}])
    caminhos = xp.exportar(tabela, "_teste_arredondamento", nota="")
    lido = pd.read_csv(caminhos["csv"])

    assert lido.loc[0, "valor"] == round(0.1234567891234567, art.CASAS_CSV)
    assert lido.loc[0, "n"] == 7          # inteiro atravessa intacto
    assert bool(lido.loc[0, "flag"]) is True
    assert lido.loc[0, "conjunto"] == "—"


def test_duas_gravacoes_do_mesmo_numero_dao_o_mesmo_byte(tabelas_preservadas):
    """Ruído abaixo da nona casa não pode chegar ao arquivo.

    É a condição que fecha o diff permanente: dois valores que diferem no último bit
    têm de sair idênticos, senão o `git status` volta a sujar a cada execução.
    """
    valor = 0.30550177603514955
    bytes_gravados = set()
    for deslocado in (valor, valor - 5e-17):
        caminho = xp.exportar(pd.DataFrame([{"v": deslocado}]),
                              "_teste_arredondamento", nota="")["csv"]
        bytes_gravados.add(caminho.read_bytes())
    assert len(bytes_gravados) == 1


def test_carga_operacional_exige_a_regra(particao, grid_dois_conjuntos):
    sem_regra = [r for r in grid_dois_conjuntos if r["modelo"] != "Regra ADA"]
    with pytest.raises(ValueError, match="ausente"):
        xp.carga_operacional(sem_regra, particao)


def test_executar_aceita_e_devolve_a_particao(particao, tabelas_preservadas):
    """O notebook diagnostica colinearidade sobre `X_tr` antes de rodar o grid.

    Sem este parâmetro ele releria o CSV para refazer exatamente o mesmo split, ou,
    pior, reimplementaria a orquestração de `executar` célula a célula.

    `tabelas_preservadas` não é zelo: `executar` **exporta**, e este teste roda um
    recorte do grid (dois modelos, um conjunto). Sem a fixture, ele deixaria
    `comparativo_consolidado.tex` com duas linhas no lugar das cinco, e a guarda do
    notebook falharia por rastro de teste anterior, não por defeito real. Foi
    exatamente o que aconteceu na primeira execução da suíte completa.
    """
    saida = xp.executar(modelos=["LogReg", "Regra ADA"], conjuntos=["B_composto"],
                        particao=particao, conferir=False)
    assert saida["particao"] is particao


def test_executar_emite_a_calibracao(particao, tabelas_preservadas):
    """Slope e intercepto ganham arquivo, porque `comparar` não tem coluna para eles.

    São o par que sustenta a leitura de D-022 e de D-040: slope perto de 1 diz que a
    ordenação está preservada, e o intercepto diz o tamanho do deslocamento de nível.
    Sem esta exportação eles só existiriam dentro de um dicionário em memória, e quem
    transcreve para o texto não teria contra o que conferir.
    """
    saida = xp.executar(modelos=["LogReg", "Regra ADA"], conjuntos=["B_composto"],
                        particao=particao, conferir=False)
    assert xp.NOME_CALIBRACAO in saida["artefatos"]

    emitido = pd.read_csv(cfg.PATH_TABELAS / f"{xp.NOME_CALIBRACAO}.csv")
    assert {"slope", "intercepto"} <= set(emitido.columns)
    assert len(emitido) == len(saida["resultados"])


def test_descritivas_ganham_artefato(particao, tabelas_preservadas):
    """O quadro da amostra e as contagens de ausentes saem em arquivo.

    A prevalência ponderada por `peso_lab` (D-031) e a contagem de ausentes de
    `colesterol_alto` (D-011) vão ao texto, e até agora saíam de `data` direto para o
    documento, sem passar por artefato nenhum.
    """
    artefatos = xp.exportar_descritivas(particao)
    assert set(artefatos) == {xp.NOME_RESUMO, xp.NOME_AUSENTES}

    resumo = pd.read_csv(cfg.PATH_TABELAS / f"{xp.NOME_RESUMO}.csv")
    assert resumo["item"].str.contains("ponderada").any()

    ausentes = pd.read_csv(cfg.PATH_TABELAS / f"{xp.NOME_AUSENTES}.csv")
    assert "no_treino" in ausentes.columns
    assert ausentes.loc[0, "variavel"] == "colesterol_alto"
