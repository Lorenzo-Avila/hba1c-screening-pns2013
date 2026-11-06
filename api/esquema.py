"""Contrato da API: entrada em nível clínico e formato das respostas.

A entrada traz peso e altura, não IMC, porque é o que um prontuário ou formulário
tem. Todo campo é obrigatório na requisição e aceita `null`: ausência é declarada,
nunca deduzida de um campo esquecido. Quais ausências são aceitas, `api.app` decide.

Nenhum nome de campo usa termo vetado pela guarda de vocabulário: a saída é decisão
e escore de ordenação, não risco.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src import config as cfg

# Dicionários da PNS 2013 que validam cada categórica. `raca_cor = 9` ("Ignorado")
# é código válido e vira ausente em `data.derivar`, como no treino.
DOMINIOS = {
    "sexo": cfg.COD_SEXO,
    "raca_cor": cfg.COD_RACA,
    "regiao": cfg.COD_REGIAO,
    "escolaridade": cfg.COD_ESCOLARIDADE,
    "saude_autoaval": cfg.COD_SAUDE_AUTOAVAL,
    "plano_saude": cfg.COD_PLANO,
    "has_autorrelato": cfg.COD_HAS,
    "colesterol_alto": cfg.COD_COLEST,
}


class Registro(BaseModel):
    """Um indivíduo, nos 14 campos que `data.derivar` e o `ImputadorPNS` consomem.

    As faixas das contínuas são de plausibilidade fisiológica, não as do treino:
    barram erro de digitação e de unidade (altura em metros, peso em gramas). A
    idade começa em 18 porque a amostra é adulta.
    """

    model_config = ConfigDict(extra="forbid")

    idade: float | None = Field(ge=18, le=120)
    peso_kg: float | None = Field(ge=20, le=300)
    altura_cm: float | None = Field(ge=100, le=250)
    circ_cintura: float | None = Field(ge=30, le=250)
    pas: float | None = Field(ge=50, le=300)
    pad: float | None = Field(ge=20, le=200)
    sexo: int | None
    raca_cor: int | None
    regiao: int | None
    escolaridade: int | None
    saude_autoaval: int | None
    plano_saude: int | None
    has_autorrelato: int | None
    colesterol_alto: int | None

    @model_validator(mode="after")
    def _codigo_no_dicionario(self):
        for campo, dicionario in DOMINIOS.items():
            valor = getattr(self, campo)
            if valor is not None and valor not in dicionario:
                raise ValueError(
                    f"{campo}={valor} fora da codificação da PNS 2013: "
                    f"{sorted(dicionario)}"
                )
        return self


class Contribuicao(BaseModel):
    """Contribuição de uma variável do modelo ao escore, somadas as indicadoras.

    `valor` é o que o modelo recebeu, na escala clínica: o informado ou, quando
    `imputada`, o que a imputação atribuiu. `categoria_omitida` é a categoria que o
    one-hot deixou de fora, nas nominais; nas demais, `None`.
    """

    atributo: str
    contribuicao: float
    valor: float
    imputada: bool
    categoria_omitida: str | None


class Classificacao(BaseModel):
    decisao: str
    encaminhar: bool
    escore_de_ordenacao: float
    limiar: float
    aviso: str
    valor_de_base: float
    contribuicoes: list[Contribuicao]
    campos_imputados: list[str]
    restricao_de_leitura: str


class Metadados(BaseModel):
    modelo: str
    conjunto: str
    limiar: float
    recall_alvo: float
    versoes: dict[str, str]
    campos_entrada: list[str]
    campos_que_aceitam_ausencia: list[str]
    # Codificação da PNS de cada categórica: o formulário monta as opções daqui e
    # não precisa importar `src/`.
    dominios: dict[str, dict[int, str]]
    aviso: str
