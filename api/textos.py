"""Textos fixos da resposta da API.

Ficam num lugar só para a guarda de vocabulário saber o que conferir. A restrição de
leitura é a única constante liberada: ela nega os termos vetados ("não é risco",
"nunca efeito protetor"), e retirá-los dela apagaria justamente a negação. Ela
acompanha toda decomposição porque, com valor de base perto de 0,5, quase todo
indivíduo recebe contribuições negativas, e sem o aviso isso se lê como proteção.
"""

AVISO_ESCORE = (
    "Escore de ordenação, não estimativa individual. Só a comparação com o limiar "
    "tem leitura operacional: escore igual ou acima do limiar encaminha a exame de "
    "HbA1c."
)

RESTRICAO_DE_LEITURA = (
    "O escore é de ordenação: não é risco individual nem probabilidade de HbA1c "
    "≥ 6,5%. O valor de base fica perto de 0,5 por causa do reponderamento de "
    "classes. Contribuição negativa indica valor abaixo do indivíduo médio sob "
    "reponderamento, nunca efeito protetor. O valor de base somado às contribuições "
    "reconstrói o escore, não o risco."
)

ENCAMINHAR = "encaminhar a exame"
NAO_ENCAMINHAR = "não encaminhar"
