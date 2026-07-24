# Archivo: rag-api/acronyms.py
# Descripción: Expansión de acrónimos de dominio en la consulta (PLAN.md §3.1).
#
# Muchas normas se citan por su acrónimo en las preguntas ("¿qué es el REBT?")
# pero el cuerpo del PDF apenas contiene el acrónimo: usa el nombre completo o
# referencias tipo ITC-BT-NN. Resultado: la consulta no puede casar con su
# propia norma ni por léxico (BM25) ni acercarse por densa. Expandir el acrónimo
# a su término completo añade el vocabulario que sí está indexado —y, de paso,
# tokens acentuados que la rama BM25 encuentra (el índice Whoosh no pliega
# acentos). No requiere re-indexar.
#
# Sólo acrónimos NO ambiguos en este corpus (electricidad / telecom / edificación).
# Extender esta tabla es barato y reversible; añadir ambiguos (p.ej. "RD") no.

import re

ACRONYM_EXPANSIONS = {
    "REBT": "Reglamento Electrotécnico para Baja Tensión",
    "ITC-BT": "Instrucción Técnica Complementaria para Baja Tensión",
    "RITE": "Reglamento de Instalaciones Térmicas en los Edificios",
    "CTE": "Código Técnico de la Edificación",
    "MBTS": "Muy Baja Tensión de Seguridad",
    "MBTP": "Muy Baja Tensión de Protección",
    "ICT": "Infraestructura Común de Telecomunicaciones",
    "RETIE": "Reglamento Técnico de Instalaciones Eléctricas",
}


def expand_acronyms(query: str):
    """Expande acrónimos conocidos presentes en la consulta.

    Cada acrónimo detectado como palabra completa (sin distinguir mayúsculas y
    sin cortar dentro de otra palabra o de un código tipo ITC-BT-44) se conserva
    y se le añade su término completo una sola vez, al final de la consulta.

    Args:
        query: Consulta original del usuario.

    Returns:
        Tupla (consulta_expandida, lista_de_acrónimos_encontrados). Si no hay
        coincidencias, devuelve la consulta sin cambios y una lista vacía.
    """
    if not query:
        return query, []

    found = []
    additions = []
    for acronym, expansion in ACRONYM_EXPANSIONS.items():
        # Fronteras propias: ni letra/dígito ni guion a los lados, para no casar
        # dentro de una palabra ni dentro de un código más largo (ITC-BT-44).
        pattern = r"(?<![\w-])" + re.escape(acronym) + r"(?![\w-])"
        if re.search(pattern, query, flags=re.IGNORECASE):
            found.append(acronym)
            additions.append(expansion)

    if not additions:
        return query, found

    return query + " " + " ".join(additions), found
