"""Paleta e folha de estilo do painel.

Único módulo com código de cor. Componentes, gráficos e diagramas importam as
constantes daqui; `.streamlit/config.toml` repete a paleta para o tema base.
"""

FONTE = "Inter, 'Segoe UI', system-ui, sans-serif"

FUNDO = "#F8FAFC"
CARTAO = "#FFFFFF"
BORDA = "#E2E8F0"
TEXTO = "#0F172A"
TEXTO_SECUNDARIO = "#64748B"

AZUL = "#2563EB"
AZUL_ESCURO = "#1D4ED8"
AZUL_CLARO = "#93C5FD"
AZUL_SUAVE = "#BFDBFE"
AZUL_FUNDO = "#E0F2FE"

VERDE = "#059669"
VERDE_FUNDO = "#ECFDF5"
VERMELHO = "#DC2626"
VERMELHO_FUNDO = "#FEF2F2"
ROXO = "#9333EA"
CINZA = "#475569"
CINZA_FUNDO = "#F1F5F9"

SELO_VERDE = ("#DCFCE7", "#15803D")
SELO_AZUL = ("#DBEAFE", "#1D4ED8")
SELO_CINZA = ("#F1F5F9", "#475569")
SELO_VERMELHO = ("#FEE2E2", "#B91C1C")

SELOS = {
    "verde": SELO_VERDE,
    "azul": SELO_AZUL,
    "cinza": SELO_CINZA,
    "vermelho": SELO_VERMELHO,
}

# Tom de cartão: cor do valor e fundo. Acerto em verde, erro em vermelho,
# encaminhamento em azul. O rótulo em texto acompanha sempre.
TONS = {
    "neutro": (TEXTO, CARTAO),
    "acerto": (VERDE, VERDE_FUNDO),
    "erro": (VERMELHO, VERMELHO_FUNDO),
    "azul": (AZUL_ESCURO, AZUL_FUNDO),
}


def css() -> str:
    """Folha de estilo injetada uma vez por execução do app."""
    return f"""
<style>
html, body, .stApp, .stMarkdown, p, li, label, h1, h2, h3, h4, td, th {{
  font-family: {FONTE};
}}
.stApp {{ background: {FUNDO}; color: {TEXTO}; }}
.block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
[data-testid="stSidebar"] {{ background: {CARTAO}; border-right: 1px solid {BORDA}; }}
[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] {{
  border-left: 3px solid transparent; border-radius: 0 8px 8px 0; padding-left: .6rem;
}}
.st-key-nav-ativo a[data-testid="stPageLink-NavLink"] {{
  background: {AZUL_FUNDO}; border-left-color: {AZUL};
}}
.st-key-nav-ativo a[data-testid="stPageLink-NavLink"] p {{
  color: {AZUL_ESCURO}; font-weight: 600;
}}
[data-testid="stSidebar"] [data-testid="stPageLink"] {{ margin: 0; }}
.marca-titulo {{ font-size: 1.35rem; font-weight: 700; color: {TEXTO}; margin: 0; }}
.marca-sub {{ font-size: .85rem; color: {TEXTO_SECUNDARIO}; margin: .1rem 0 .6rem 0; }}
.secao-nav {{
  font-size: .72rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;
  color: {TEXTO_SECUNDARIO}; margin: 1rem 0 .2rem .2rem;
}}
.cabecalho h1 {{ font-size: 1.9rem; font-weight: 700; margin: 0; color: {TEXTO}; }}
.cabecalho .sub {{ color: {TEXTO_SECUNDARIO}; margin: .25rem 0 .6rem 0; font-size: 1rem; }}
.selos {{ display: flex; flex-wrap: wrap; gap: .4rem; margin: .2rem 0 1rem 0; }}
.selo {{
  display: inline-block; padding: .18rem .65rem; border-radius: 999px;
  font-size: .8rem; font-weight: 600;
}}
.grade {{
  display: grid; gap: .8rem; margin: .4rem 0 1rem 0;
  grid-template-columns: repeat(auto-fit, minmax(122px, 1fr));
}}
.cartao {{
  background: {CARTAO}; border: 1px solid {BORDA}; border-radius: 12px;
  padding: .85rem 1rem; box-shadow: 0 1px 2px rgba(15, 23, 42, .05);
}}
.cartao .rotulo {{ font-size: .82rem; color: {TEXTO_SECUNDARIO}; display: flex; gap: .3rem; }}
.cartao .ajuda {{ cursor: help; color: {TEXTO_SECUNDARIO}; }}
.cartao .valor {{ font-size: 1.65rem; font-weight: 700; line-height: 1.25; margin-top: .15rem; }}
.cartao .nota {{ font-size: .78rem; color: {TEXTO_SECUNDARIO}; margin-top: .15rem; }}
.faixa {{
  background: {CINZA_FUNDO}; border: 1px solid {BORDA}; border-radius: 12px;
  padding: .75rem 1rem; margin: .3rem 0 1rem 0;
}}
.faixa .titulo {{ font-weight: 600; font-size: .9rem; margin-bottom: .4rem; }}
.faixa .grade {{ margin: 0; }}
.faixa .cartao {{ background: transparent; border: none; box-shadow: none; padding: .2rem .4rem; }}
.faixa .cartao .valor {{ font-size: 1.25rem; }}
.faixa-alerta {{
  background: {AZUL_FUNDO}; border-left: 4px solid {AZUL}; border-radius: 8px;
  padding: .7rem 1rem; margin: .3rem 0 1rem 0; color: {AZUL_ESCURO}; font-weight: 500;
}}
.quadro {{
  background: {AZUL_FUNDO}; border: 1px solid {AZUL_SUAVE}; border-radius: 12px;
  padding: .8rem 1rem; margin: .6rem 0 1rem 0; color: {TEXTO};
}}
.quadro .titulo {{ font-weight: 700; color: {AZUL_ESCURO}; margin-bottom: .25rem; }}
.nota-fixa {{ font-size: .85rem; color: {TEXTO_SECUNDARIO}; margin: .2rem 0 .8rem 0; }}
.painel-bloco {{
  background: {CARTAO}; border: 1px solid {BORDA}; border-radius: 12px;
  padding: .9rem 1rem; box-shadow: 0 1px 2px rgba(15, 23, 42, .05); height: 100%;
}}
.painel-bloco .titulo {{ font-weight: 700; font-size: .95rem; margin-bottom: .5rem; }}
.painel-bloco .grade {{ grid-template-columns: repeat(2, 1fr); margin: 0; }}
.matriz {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: .6rem; margin: .4rem 0 1rem 0; }}
.matriz .cartao .valor {{ font-size: 1.9rem; }}
.funil {{ display: flex; flex-wrap: wrap; align-items: stretch; gap: .5rem; margin: .4rem 0 1rem 0; }}
.funil .etapa {{
  flex: 1 1 130px; background: {CARTAO}; border: 1px solid {BORDA}; border-radius: 12px;
  padding: .7rem .8rem; text-align: center;
}}
.funil .etapa .n {{ font-size: 1.35rem; font-weight: 700; color: {TEXTO}; }}
.funil .etapa .r {{ font-size: .78rem; color: {TEXTO_SECUNDARIO}; }}
.funil .etapa.exclusao .n {{ color: {TEXTO_SECUNDARIO}; }}
.funil .etapa.destaque {{ border-color: {AZUL}; background: {AZUL_FUNDO}; }}
.funil .seta {{ align-self: center; color: {TEXTO_SECUNDARIO}; font-size: 1.2rem; }}
.duas-colunas {{ display: grid; grid-template-columns: 1fr 1fr; gap: .8rem; margin: .4rem 0 1rem 0; }}
.duas-colunas ul {{ margin: .3rem 0 0 1rem; padding: 0; }}
.duas-colunas li {{ margin: .2rem 0; }}
.linha-tempo {{ display: flex; flex-wrap: wrap; gap: .5rem; margin: .4rem 0 1rem 0; }}
.linha-tempo .fase {{
  flex: 1 1 140px; border: 1px solid {BORDA}; border-radius: 12px; padding: .6rem .7rem;
  background: {CARTAO}; border-top: 4px solid {VERDE};
}}
.linha-tempo .fase.pendente {{ border-top-color: {TEXTO_SECUNDARIO}; background: {CINZA_FUNDO}; }}
.linha-tempo .fase .nome {{ font-weight: 600; font-size: .88rem; }}
.linha-tempo .fase .estado {{ font-size: .78rem; color: {TEXTO_SECUNDARIO}; }}
.campos {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: .5rem; }}
.campo .r {{ font-size: .78rem; color: {TEXTO_SECUNDARIO}; margin-bottom: .15rem; }}
.campo .v {{
  background: {CINZA_FUNDO}; border: 1px solid {BORDA}; border-radius: 8px;
  padding: .4rem .6rem; color: {TEXTO}; font-size: .92rem;
}}
div.rolagem {{ overflow-x: auto; margin: .3rem 0 1rem 0; }}
table.tabela {{
  width: 100%; border-collapse: collapse; background: {CARTAO}; border: 1px solid {BORDA};
  border-radius: 12px; overflow: hidden; font-size: .85rem; margin: 0;
}}
table.tabela th {{
  text-align: left; color: {TEXTO_SECUNDARIO}; font-weight: 600; background: {FUNDO};
  padding: .5rem .7rem; border-bottom: 1px solid {BORDA};
}}
table.tabela td {{ padding: .45rem .7rem; border-bottom: 1px solid {BORDA}; color: {TEXTO}; }}
</style>
"""
