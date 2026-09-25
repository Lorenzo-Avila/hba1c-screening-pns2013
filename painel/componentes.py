"""Componentes visuais em HTML e CSS: cartões, selos, faixas, matriz e tabelas.

Cada função escapa o texto que recebe e desenha com as classes de `estilo.css`.
Nenhuma cor é escrita aqui: as que variam por tom vêm de `estilo`.
"""

from html import escape

import pandas as pd
import streamlit as st

import estilo


def _html(conteudo: str) -> None:
    st.markdown(conteudo, unsafe_allow_html=True)


def aplicar_estilo() -> None:
    """Injeta a folha de estilo do painel."""
    _html(estilo.css())


def selo(texto: str, tom: str = "azul") -> str:
    """HTML de um selo arredondado. `tom` é uma chave de `estilo.SELOS`."""
    fundo, cor = estilo.SELOS[tom]
    return (f'<span class="selo" style="background:{fundo};color:{cor}">'
            f"{escape(texto)}</span>")


def selos(itens: list) -> None:
    """Linha de selos. Cada item é um par (texto, tom)."""
    _html('<div class="selos">' + "".join(selo(t, tom) for t, tom in itens) + "</div>")


def cabecalho(titulo: str, subtitulo: str, itens_selo: list) -> None:
    """Título, subtítulo e selos no topo da página."""
    _html(
        f'<div class="cabecalho"><h1>{escape(titulo)}</h1>'
        f'<div class="sub">{escape(subtitulo)}</div></div>'
    )
    selos(itens_selo)


def metrica(rotulo: str, valor: str, ajuda: str, nota: str = "",
            tom: str = "neutro") -> str:
    """HTML de um cartão de métrica.

    `ajuda` é obrigatória e sai de `textos.GLOSSARIO`: aparece ao passar o mouse
    sobre o rótulo. `tom` é uma chave de `estilo.TONS`.
    """
    cor, fundo = estilo.TONS[tom]
    nota_html = f'<div class="nota">{escape(nota)}</div>' if nota else ""
    return (
        f'<div class="cartao" style="background:{fundo}" title="{escape(ajuda)}" '
        f'data-ajuda="{escape(ajuda)}">'
        f'<div class="rotulo">{escape(rotulo)}<span class="ajuda">ⓘ</span></div>'
        f'<div class="valor" style="color:{cor}">{escape(valor)}</div>{nota_html}</div>'
    )


def grade(cartoes: list) -> None:
    """Cartões de métrica lado a lado, quebrando linha conforme a largura."""
    _html('<div class="grade">' + "".join(cartoes) + "</div>")


def faixa(titulo: str, cartoes: list) -> None:
    """Faixa horizontal com título e cartões compactos."""
    _html(f'<div class="faixa"><div class="titulo">{escape(titulo)}</div>'
          f'<div class="grade">{"".join(cartoes)}</div></div>')


def bloco(titulo: str, cartoes: list) -> None:
    """Bloco com título e cartões em duas colunas, para comparar regimes."""
    _html(f'<div class="painel-bloco"><div class="titulo">{escape(titulo)}</div>'
          f'<div class="grade">{"".join(cartoes)}</div></div>')


def alerta(texto: str) -> None:
    """Faixa de aviso em azul, para restrição de uso da tela."""
    _html(f'<div class="faixa-alerta">{escape(texto)}</div>')


def quadro(titulo: str, texto: str) -> None:
    """Quadro azul de leitura."""
    _html(f'<div class="quadro"><div class="titulo">{escape(titulo)}</div>'
          f"{escape(texto)}</div>")


def linha_de_log(texto: str, tom: str) -> None:
    """Linha de log em fonte monoespaçada, destacada com as cores do selo `tom`."""
    fundo, cor = estilo.SELOS[tom]
    _html(f'<div class="linha-log" style="background:{fundo};border-left-color:{cor};'
          f'color:{cor}">{escape(texto)}</div>')


def nota(texto: str) -> None:
    """Nota curta em cinza."""
    _html(f'<div class="nota-fixa">{escape(texto)}</div>')


def matriz_confusao(celulas: list) -> None:
    """Matriz 2×2 em cartões. `celulas` traz quatro cartões na ordem da grade."""
    _html('<div class="matriz">' + "".join(celulas) + "</div>")


def funil(etapas: list) -> None:
    """Funil em etapas. Cada etapa é (rótulo, número formatado, classe)."""
    partes = []
    for i, (rotulo, numero, classe) in enumerate(etapas):
        if i:
            partes.append('<div class="seta">›</div>')
        partes.append(f'<div class="etapa {classe}"><div class="n">{escape(numero)}</div>'
                      f'<div class="r">{escape(rotulo)}</div></div>')
    _html('<div class="funil">' + "".join(partes) + "</div>")


def duas_colunas(esquerda: tuple, direita: tuple) -> None:
    """Dois blocos lado a lado. Cada lado é (título, lista de linhas)."""
    def lado(titulo, linhas):
        itens = "".join(f"<li>{escape(l)}</li>" for l in linhas)
        return (f'<div class="painel-bloco"><div class="titulo">{escape(titulo)}</div>'
                f"<ul>{itens}</ul></div>")
    _html(f'<div class="duas-colunas">{lado(*esquerda)}{lado(*direita)}</div>')


def linha_tempo(fases: list) -> None:
    """Fases com estado. Cada fase é (nome, estado, concluida)."""
    partes = "".join(
        f'<div class="fase{"" if concluida else " pendente"}">'
        f'<div class="nome">{escape(nome)}</div><div class="estado">{escape(estado)}</div></div>'
        for nome, estado, concluida in fases
    )
    _html(f'<div class="linha-tempo">{partes}</div>')


def campos(pares: list) -> None:
    """Campos só de leitura. Cada par é (rótulo, valor formatado)."""
    partes = "".join(
        f'<div class="campo"><div class="r">{escape(r)}</div>'
        f'<div class="v">{escape(v)}</div></div>'
        for r, v in pares
    )
    _html(f'<div class="campos">{partes}</div>')


def tabela(df: pd.DataFrame, colunas_html: tuple = ()) -> None:
    """Tabela em HTML. Colunas em `colunas_html` já chegam como HTML (selos)."""
    cabeca = "".join(f"<th>{escape(str(c))}</th>" for c in df.columns)
    linhas = []
    for _, linha in df.iterrows():
        celulas = "".join(
            f"<td>{linha[c] if c in colunas_html else escape(str(linha[c]))}</td>"
            for c in df.columns
        )
        linhas.append(f"<tr>{celulas}</tr>")
    _html(f'<div class="rolagem"><table class="tabela"><thead><tr>{cabeca}</tr></thead>'
          f'<tbody>{"".join(linhas)}</tbody></table></div>')
