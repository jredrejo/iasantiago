"""
Reparación de texto mal decodificado por los extractores de PDF.

Medido el 2026-07-21 sobre los índices en vivo: el 27,2 % de los fragmentos de
Programming, el 14,9 % de Chemistry y el 13,7 % de Dibujo contenían texto
corrupto. El texto corrupto no casa con ninguna consulta, así que esos
fragmentos ocupan sitio en el índice sin ser recuperables nunca.

Tres defectos distintos comparten síntoma; sólo dos son reparables aquí:

A) ``GLYPH<NNN>``  -- NNN es el punto de código en decimal.
   ``MaGLYPH<237>z`` -> ``Maíz``. Reversible exactamente.

B) ``Æ Ø æ œ Œ``   -- bytes Latin-1 leídos con Adobe StandardEncoding.
   ``mantendrÆ`` -> ``mantendrá``. Reversible exactamente.

C) ``GLYPH<c=N,font=/ABCDEF+Nombre>`` -- fuente subconjunto sin tabla
   ToUnicode. Sólo sobrevive el índice de glifo: la identidad del carácter NO
   está en el PDF. **Irreparable aquí**; hace falta OCR o un PDF limpio. Se
   detecta y se avisa para que no pase inadvertido.

TODAS las correspondencias de este módulo están verificadas contra el corpus
real, no deducidas de la tabla de Adobe. Esa distinción importa: al derivarlas
sobre el papel salían `œ`->`Ñ` y `Œ`->`Á`, y el corpus demuestra que son
`œ`->`ú` (2050 casos: "Nœmero" = "Número") y `Œ`->`Í` (51 casos:
"RODRŒGUEZ" = "RODRÍGUEZ"). Aplicar la tabla deducida habría corrompido texto
que hoy está bien. **No añadir correspondencias sin comprobarlas contra los
índices.**
"""

import logging
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# --- (A) GLYPH<NNN>: punto de código en decimal ---------------------------
_GLYPH_DECIMAL = re.compile(r"GLYPH<(\d{1,7})>")

# --- (C) fuente subconjunto sin ToUnicode: irreparable --------------------
_GLYPH_FONT = re.compile(r"GLYPH<c=\d+,font=[^>]*>")

# --- (B) StandardEncoding leído como Latin-1 ------------------------------
# Del carácter que VEMOS al que DEBERÍA ser. Frecuencias observadas al
# verificar (Chemistry + Dibujo + Programming).
_STANDARD_ENCODING_FIXES: Dict[str, str] = {
    "Æ": "á",  # 7218  "MÆs" -> "más"
    "Ø": "é",  # 5522  "gØrmenes" -> "gérmenes"
    "œ": "ú",  # 2050  "Nœmero" -> "Número"
    "æ": "ñ",  # 1403  "Espaæol" -> "Español"
    "Œ": "Í",  #   51  "RODRŒGUEZ" -> "RODRÍGUEZ"
}
# Deliberadamente FUERA de la tabla, con motivo:
#   ø  -> ambiguo: "cømún" es corrupción pero "Brønsted-Lowry" es un apellido
#         danés legítimo, y sólo hay 100 casos. Reparar rompería el nombre.
#   Ł  -> 2 casos, ambos franceses legítimos ("BibliothŁque" = "Bibliothèque").
#   Ð  -> 2 casos, teclas de flecha, no es un acento.
#   ł þ-> 0 casos en el corpus.

# (B) sólo se aplica cuando el carácter toca una letra. Protege los usos
# legítimos: el diámetro "Ø 90 mm" de Dibujo/Chemistry (comprobado: de 3090
# coincidencias en rag_dibujo, 3020 eran mojibake y exactamente 1 diámetro).
_LETTER = r"[A-Za-zÀ-ÿ]"
_CLASS = "".join(re.escape(c) for c in _STANDARD_ENCODING_FIXES)
_ADJACENT = re.compile(
    rf"(?<={_LETTER})([{_CLASS}])|([{_CLASS}])(?={_LETTER})"
)


def _decode_codepoint(cp: int) -> str:
    """
    Convierte el número de un GLYPH<NNN> en su carácter.

    El rango 128-159 no son caracteres imprimibles en Unicode sino controles
    C1: ahí el productor del PDF usaba cp1252, donde 151 es la raya (—) y 149
    el bolo (•). Verificado: 1562 casos de GLYPH<151> y 76 de GLYPH<149> en
    Chemistry, que con chr() darían caracteres de control invisibles.
    """
    if 128 <= cp <= 159:
        try:
            return bytes([cp]).decode("cp1252")
        except UnicodeDecodeError:
            return ""
    if cp == 127 or cp < 32:  # DEL y controles C0: no aportan nada
        return ""
    if cp > 0x10FFFF or 0xD800 <= cp <= 0xDFFF:  # fuera de rango o sustituto
        return ""
    return chr(cp)


def _fix_glyph_decimal(text: str) -> Tuple[str, int]:
    """Sustituye GLYPH<NNN> por su carácter. Devuelve (texto, nº arreglos)."""
    n = 0

    def repl(m: "re.Match") -> str:
        nonlocal n
        ch = _decode_codepoint(int(m.group(1)))
        if ch == "" and m.group(1) not in ("127",) and int(m.group(1)) > 159:
            return m.group(0)  # no supimos decodificarlo: mejor no tocarlo
        n += 1
        return ch

    return _GLYPH_DECIMAL.sub(repl, text), n


def _fix_standard_encoding(text: str) -> Tuple[str, int]:
    """Deshace StandardEncoding→Latin-1, sólo junto a letras."""
    n = 0

    def repl(m: "re.Match") -> str:
        nonlocal n
        n += 1
        return _STANDARD_ENCODING_FIXES[m.group(1) or m.group(2)]

    return _ADJACENT.sub(repl, text), n


def repair_text(text: str) -> Tuple[str, Dict[str, int]]:
    """
    Repara un texto. Devuelve (texto_reparado, contadores).

    Contadores: `glyph_decimal`, `standard_encoding` y `unrecoverable`
    (marcas de tipo C encontradas, que no se tocan).
    """
    if not text:
        return text, {"glyph_decimal": 0, "standard_encoding": 0, "unrecoverable": 0}

    unrecoverable = len(_GLYPH_FONT.findall(text))
    text, a = _fix_glyph_decimal(text)
    text, b = _fix_standard_encoding(text)
    return text, {
        "glyph_decimal": a,
        "standard_encoding": b,
        "unrecoverable": unrecoverable,
    }


def repair_chunks(chunks: List) -> List:
    """
    Repara el texto de los Chunk in situ y deja un resumen en el log.

    Se aplica ANTES de deduplicar: al reparar, dos fragmentos que sólo
    diferían en la corrupción pasan a ser idénticos y deben colapsar en uno.
    """
    totals = {"glyph_decimal": 0, "standard_encoding": 0, "unrecoverable": 0}
    touched = 0
    lost = 0

    for c in chunks:
        fixed, counts = repair_text(c.text)
        for k in totals:
            totals[k] += counts[k]
        if fixed != c.text:
            c.text = fixed
            touched += 1
        if counts["unrecoverable"]:
            lost += 1

    if touched:
        logger.info(
            f"[REPAIR] {touched} fragmentos reparados "
            f"(GLYPH<n>: {totals['glyph_decimal']}, "
            f"StandardEncoding: {totals['standard_encoding']})"
        )
    if lost:
        logger.warning(
            f"[REPAIR] {lost} fragmentos con fuente subconjunto sin ToUnicode: "
            f"texto NO recuperable sin OCR. Revisar el PDF de origen."
        )
    return chunks
