"""Classificação individual: formulário que consulta a API e mostra a resposta.

Única página que recebe dado de entrada e única que não lê `reports/tabelas/`. O
cálculo fica todo na API: a página monta o registro, envia por `cliente` e exibe
decisão, escore contra o limiar, decomposição em cascata e a restrição de leitura que
vêm na resposta. A API serve só a célula de referência, então a seleção da barra
lateral não vale aqui.
"""

import httpx
import pandas as pd
import streamlit as st

import cliente
import componentes
import graficos
import textos

CASAS = 3

titulo, subtitulo, abertura = textos.PAGINAS["classificacao"]
componentes.cabecalho(titulo, subtitulo, [])
componentes.nota(abertura)

try:
    metadados = cliente.ler_metadados()
except httpx.HTTPError:
    componentes.alerta(textos.API_FORA.format(url=cliente.URL_API))
    st.stop()

componentes.selos([(textos.SELO_SERVIDO.format(modelo=metadados["modelo"],
                                               conjunto=metadados["conjunto"]), "azul")])
dominios = {campo: {int(codigo): rotulo for codigo, rotulo in opcoes.items()}
            for campo, opcoes in metadados["dominios"].items()}
g = textos.GLOSSARIO


def _rotulo_opcao(campo: str):
    return lambda codigo: f"{codigo} · {dominios[campo][codigo]}"


st.subheader(textos.FORMULARIO)
with st.form("registro"):
    componentes.nota(textos.AUSENCIA_ACEITA.format(campos=", ".join(
        textos.ROTULOS_CAMPOS[c] for c in metadados["campos_que_aceitam_ausencia"])))
    colunas = st.columns(2)
    registro = {}
    for posicao, campo in enumerate(metadados["campos_entrada"]):
        with colunas[posicao % 2]:
            rotulo = textos.ROTULOS_CAMPOS[campo]
            if campo in dominios:
                registro[campo] = st.selectbox(
                    rotulo, list(dominios[campo]), index=None, key=campo,
                    format_func=_rotulo_opcao(campo), placeholder=textos.NAO_INFORMADO)
            else:
                registro[campo] = st.number_input(rotulo, value=None, key=campo,
                                                  placeholder=textos.NAO_INFORMADO)
    enviado = st.form_submit_button(textos.CLASSIFICAR, type="primary")

if not enviado:
    st.stop()

resposta = cliente.classificar(registro)
if resposta.status_code == 422:
    detalhe = resposta.json()["detail"]
    if isinstance(detalhe, list):   # erro de esquema do pydantic, um por campo
        detalhe = "; ".join(erro["msg"] for erro in detalhe)
    st.error(f"{textos.RECUSADO}: {detalhe}")
    st.stop()
resposta.raise_for_status()
resultado = resposta.json()

st.subheader(textos.DECISAO_TITULO)
componentes.selos([(resultado["decisao"].capitalize(),
                    "azul" if resultado["encaminhar"] else "cinza")])
componentes.grade([
    componentes.metrica("Escore de ordenação",
                        graficos.numero(resultado["escore_de_ordenacao"], CASAS),
                        ajuda=g["escore"]),
    componentes.metrica("Limiar calibrado", graficos.numero(resultado["limiar"], CASAS),
                        ajuda=g["limiar"]),
])
componentes.nota(resultado["aviso"])

if resultado["campos_imputados"]:
    componentes.alerta(textos.IMPUTADOS.format(campos=", ".join(
        textos.ROTULOS_CAMPOS[c] for c in resultado["campos_imputados"])))

contribuicoes = pd.DataFrame(resultado["contribuicoes"])
caso = contribuicoes.rename(columns={"valor": "valor_observado"}).assign(
    valor_base=resultado["valor_de_base"],
    escore=resultado["escore_de_ordenacao"],
    limiar=resultado["limiar"],
)
rotulos = {
    linha.atributo: textos.ROTULOS_CLINICOS.get(linha.atributo, linha.atributo)
    + (textos.MARCA_IMPUTADA if linha.imputada else "")
    for linha in contribuicoes.itertuples()
}
st.altair_chart(
    graficos.waterfall(caso, resultado["restricao_de_leitura"], CASAS, rotulos),
    width="stretch",
)
componentes.nota(textos.WATERFALL)
componentes.quadro(textos.LEITURA_TITULO, resultado["restricao_de_leitura"])
