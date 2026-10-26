"""Constantes, caminhos, mapeamentos e semente do projeto.

Fonte única dos parâmetros metodológicos. Todo valor usado por mais de um módulo
de `src/` mora aqui e só aqui, nunca copiado para dentro de um notebook: valor
replicado é divergência silenciosa esperando acontecer, e cada uma delas custa a
rastreabilidade das decisões que sustentam a metodologia.
"""

from pathlib import Path

# --- Reprodutibilidade e protocolo de validação ---
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# --- Critérios clínicos e diagnósticos ---
LIMIAR_HBA1C = 6.5   # critério ADA 2010
VIF_LIMIAR = 10      # acima disso, multicolinearidade a tratar
RECALL_ALVO = 0.80   # provisório: confirmar com o orientador antes da pré-banca

# --- Caminhos ---
RAIZ = Path(__file__).resolve().parents[1]
PATH_RAW = RAIZ / "data" / "raw" / "EXAMES_PNS_2013_BRUTA.csv"
PATH_REPORTS = RAIZ / "reports"
PATH_TABELAS = PATH_REPORTS / "tabelas"   # .tex e .csv
PATH_FIGURAS = PATH_REPORTS / "figuras"  # .png

# --- De-para: código oficial PNS 2013 -> nome legível usado no código ---
DE_PARA = {
    "C008": "idade",           "Z001": "sexo",           "Z003": "raca_cor",
    "z004": "peso_kg",         "z005": "altura_cm",      "W00303": "circ_cintura",
    "W00407": "pas",           "W00408": "pad",          "Q002": "has_autorrelato",
    "Q060": "colesterol_alto", "N001": "saude_autoaval", "VDD004": "escolaridade",
    "I001": "plano_saude",     "regiao": "regiao",       "Z034": "hba1c",
    "peso_lab": "peso_lab",
}
COLS_ORIGEM = list(DE_PARA.keys())

# --- Grupos conceituais (guiam conversão de tipo e tratamento) ---
CONTINUAS = ["idade", "peso_kg", "altura_cm", "circ_cintura", "pas", "pad"]
CATEGORICAS = ["sexo", "raca_cor", "regiao", "escolaridade", "saude_autoaval",
               "plano_saude", "has_autorrelato", "colesterol_alto"]
MEDICAO = ["peso_kg", "altura_cm", "circ_cintura", "pas", "pad"]
ALVO_CONTINUO = "hba1c"
ALVO = "target"
PESO = "peso_lab"

# Preditoras conceituais devolvidas por `separar`. `peso_kg` e `altura_cm` ficam
# fora de propósito: chegam ao modelo só pelas derivadas `imc` e `rcest`.
PREDITORAS = ["idade", "sexo", "imc", "circ_cintura", "rcest", "pas", "pad",
              "has_autorrelato", "colesterol_alto", "raca_cor", "escolaridade",
              "plano_saude", "regiao", "saude_autoaval"]

# --- Codificação oficial PNS 2013 ---
COD_SEXO = {1: "Masculino", 2: "Feminino"}
COD_RACA = {1: "Branca", 2: "Preta", 3: "Amarela", 4: "Parda", 5: "Indígena", 9: "Ignorado"}
COD_REGIAO = {1: "Norte", 2: "Nordeste", 3: "Sudeste", 4: "Sul", 5: "Centro-Oeste"}
COD_HAS = {1: "Sim", 2: "Apenas na gravidez", 3: "Não"}
COD_COLEST = {1: "Sim", 2: "Não"}
# VDD004 no dicionário de variáveis da PNS 2013, sistema de 8 anos.
COD_ESCOLARIDADE = {
    1: "Sem instrução",
    2: "Fundamental incompleto ou equivalente",
    3: "Fundamental completo ou equivalente",
    4: "Médio incompleto ou equivalente",
    5: "Médio completo ou equivalente",
    6: "Superior incompleto ou equivalente",
    7: "Superior completo",
}
# N001 no dicionário de variáveis da PNS 2013.
COD_SAUDE_AUTOAVAL = {
    1: "Muito boa",
    2: "Boa",
    3: "Regular",
    4: "Ruim",
    5: "Muito ruim",
}

# Código de não resposta: vira NaN na limpeza.
COD_IGNORADO_RACA = 9

# --- Pontos de corte clínicos (Sociedade Brasileira de Diabetes, 2019) ---
CORTE_PAS, CORTE_PAD = 140, 90
CORTE_IMC_OBESIDADE = 30
CORTE_CINTURA_F, CORTE_CINTURA_M = 80, 94

# --- Paleta (identidade visual do projeto) ---
TEAL = "#0F766E"
TEAL_SOFT = "#5EEAD4"
CORAL = "#E76F51"
INK = "#1F2937"
GRID = "#E5E7EB"
