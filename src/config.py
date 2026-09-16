"""Constantes, caminhos, mapeamentos e semente do projeto.

Modulo criado para consolidar, em fonte única, os parâmetros metodológicos definidos ao longo de todo o projeto.
Todo valor utilizado por mais de um módulo de `src/` deve estar
exclusivamente aqui, nunca replicado em notebooks, de modo que nenhuma
divergência silenciosa possa comprometer a rastreabilidade das decisões que
sustentam a metodologia.
"""

from pathlib import Path

# --- Reprodutibilidade e protocolo de validação (R10/R11, §6.3 fases 3 e 5) ---
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# --- Critérios clínicos e diagnósticos ---
LIMIAR_HBA1C = 6.5   # critério ADA 2010
VIF_LIMIAR = 10      # §6.5
RECALL_ALVO = 0.80   # provisório — confirmar com orientador antes da pré-banca (D-004)

# --- Caminhos ---
RAIZ = Path(__file__).resolve().parents[1]
PATH_RAW = RAIZ / "data" / "raw" / "EXAMES_PNS_2013_BRUTA.csv"
PATH_REPORTS = RAIZ / "reports"
PATH_FIGURAS = PATH_REPORTS / "figures"

# --- De-para: código oficial PNS 2013 -> nome legível (rastreabilidade com o TCC) ---
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

# Preditoras conceituais devolvidas por `separar`. `peso_kg` e `altura_cm` foram
# deliberadamente excluídas desta lista, de forma que passam a integrar o modelo
# tão somente por meio das derivadas `imc` e `rcest`, conforme estabelecido na
# §6.4.1.
PREDITORAS = ["idade", "sexo", "imc", "circ_cintura", "rcest", "pas", "pad",
              "has_autorrelato", "colesterol_alto", "raca_cor", "escolaridade",
              "plano_saude", "regiao", "saude_autoaval"]

# --- Codificação oficial PNS 2013 ---
COD_SEXO = {1: "Masculino", 2: "Feminino"}
COD_RACA = {1: "Branca", 2: "Preta", 3: "Amarela", 4: "Parda", 5: "Indígena", 9: "Ignorado"}
COD_REGIAO = {1: "Norte", 2: "Nordeste", 3: "Sudeste", 4: "Sul", 5: "Centro-Oeste"}
COD_HAS = {1: "Sim", 2: "Apenas na gravidez", 3: "Não"}
COD_COLEST = {1: "Sim", 2: "Não"}

# Código de ausência a decodificar como NaN na limpeza (§3 da spec).
COD_IGNORADO_RACA = 9

# --- Pontos de corte clínicos (Sociedade Brasileira de Diabetes, 2019) — §6.4.1 ---
CORTE_PAS, CORTE_PAD = 140, 90
CORTE_IMC_OBESIDADE = 30
CORTE_CINTURA_F, CORTE_CINTURA_M = 80, 94

# --- Paleta (identidade visual do projeto) ---
TEAL = "#0F766E"
TEAL_SOFT = "#5EEAD4"
CORAL = "#E76F51"
INK = "#1F2937"
GRID = "#E5E7EB"
