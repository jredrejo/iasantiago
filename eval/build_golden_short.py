#!/usr/bin/env python3
"""Genera `eval/golden_short.json` acortando los golden sets existentes.

**Para qué sirve y para qué NO.** El 41 % del tráfico real por el Filter tiene
menos de `BM25_FALLBACK_TOKEN_THRESHOLD` tokens y se sirve por la rama
BM25-sola, sin vector denso ni RRF. Ninguna de las 157 consultas golden baja de
6 tokens, así que esa rama **no la mide nada**. Esto la hace medible.

- **Sirve como comparación pareada** (BM25-solo ↔ híbrido): las dos ramas ven la
  misma consulta y el mismo ground truth, así que la ambigüedad extra de una
  consulta corta las penaliza igual.
- **NO sirve como línea base absoluta.** Acortar una pregunta la vuelve más
  ambigua: "declinaciones" apunta legítimamente a muchas páginas y el
  `relevant_pages` heredado sólo reconoce una. Los R@k de este fichero son un
  suelo, no la calidad del sistema. No lo metas en la tabla de líneas base.

**El sesgo va a favor del BM25, a propósito.** Cada consulta corta conserva los
términos distintivos de la larga, que es justo lo que necesita la coincidencia
léxica; un usuario real escribe otras palabras. Si aun así gana el híbrido, la
conclusión es fuerte.

Regenera, no edites a mano: `python3 eval/build_golden_short.py`.
"""
from __future__ import annotations

import json
import os
import sys

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_TOKENS = 3  # < BM25_FALLBACK_TOKEN_THRESHOLD (4), o no se ejerce la rama

# Índice dentro de cada golden set -> consulta corta. El índice es el orden del
# fichero: si alguien reordena un golden set, este mapa deja de valer y la
# comprobación de longitud no lo detectaría. Por eso se verifica también que el
# número de casos coincida.
SHORT = {
    "golden_afd.json": [
        "luxación anteroinferior hombro", "series repeticiones entrenamiento",
        "anfetaminas cocaína rendimiento", "grasas monoinsaturadas poliinsaturadas",
        "agua peso corporal", "vitamina E cocinar",
        "inmovilización columna accidentado", "convulsiones parciales simples",
        "terremoto dentro coche", "hidratos termogénesis",
        "BCAA daño muscular", "aminoácidos esenciales",
        "magnesio metabolismo deportista", "electrolitos",
        "recuperar hidratos entrenamiento",
    ],
    "golden_chemistry.json": [
        "HPLC cromatografía líquidos", "cromatografía capa fina",
        "enlace frecuencia infrarrojo", "electrodos amperométricos",
        "diámetro interno columna", "atomizadores continuos discretos",
        "tiocianato potasio hierro", "ácido fuerte débil",
        "porcentaje nitrógeno cloroplatinato", "lámpara deuterio fondo",
        "aerosoles análisis microbiológico", "filtros de profundidad",
        "esencia agua colonia", "cifra significativa incertidumbre",
        "Pseudomonas gram negativo", "ondas oxígeno polarografía",
        "moler muestras pastillas", "voltametría cationes metálicos",
    ],
    "golden_chemistry_docling.json": [
        "coeficiente actividad medio", "disociación ácido barbitúrico",
        "AEDT calmagita estandarización", "dióxido azufre cerio",
        "membrana líquida calcio", "línea solidus",
        "abrasivos desgastar", "refractómetro Abbé",
        "anodizado electrodeposición", "paño lana pulir",
        "diámetro bolas molienda", "frasco Schöniger",
        "analizador halógeno humedad", "filtro retención partículas",
        "diagrama heptano octano", "esencia limón arrastre",
        "muestra tanque cerrado", "colonias Campylobacter",
        "secar placa Proteus", "incubación levaduras mohos",
        "número más probable", "eficacia medio cultivo",
        "densidad gases decímetro", "composición jabón bases",
        "comburente combustible", "equilibrio evaporación condensación",
        "presión vapor agua",
    ],
    "golden_dibujo.json": [
        "Ronda noche Rembrandt", "Santa Cristina Lena",
        "arte bizantino macedónica", "iglesia San Carlino",
        "intersección recta cono", "plantas independientes perspectiva",
        "arquitectura cobijo estética", "estofado policromía escultura",
        "esquemas compositivos pintura", "Arts and Crafts",
        "marca Durex",
    ],
    "golden_electricidad.json": [
        "r2114 V20", "tiempo diferencial Vdc",
        "ciclo inversión bomba", "grado protección condensación",
        "protección know-how", "invertir consigna binaria",
        "fallo F07936", "sustituir Control Unit",
        "interfaz X127 S210", "PROFIdrive 9003 sobretensión",
        "escribir parámetro p0010", "resistencia freno S210",
        "protección datos CPU", "parametrizar módulos STEP7",
        "PS 60W alimentación", "estados operativos S7-1200",
        "Memory Card protegida", "herramientas online TIA",
        "sección conductores protección", "atmósferas polvo explosivo",
        "empresa instaladora habilitación", "registro principal óptico",
        "planos proyecto ICT", "criterio amplitud mínima",
        "placa D3076-K", "seguridad robot KUKA",
    ],
    "golden_fol.json": [
        "Fondo Garantía Salarial", "asamblea trabajadores horario",
        "directivas maternidad jóvenes", "formación trabajadores temporales",
        "obligaciones fabricantes maquinaria", "servicio prevención ajeno",
        "incapacidad temporal prórroga", "riesgo durante lactancia",
        "jubilación anticipada voluntaria",
    ],
    "golden_latin.json": [
        "Eutropio origen griego", "sobrenombre Torcuato",
        "emperador filósofo estoico", "frase Julio César",
        "Fedro fábulas", "tablillas enceradas",
        "brevis gravis fortis", "erga dignus",
    ],
    "golden_mecanica.json": [
        "mecanizado ultrasonidos sonotrodo", "mango troquel cortador",
        "aptitud rectificado vanadio", "ensamblaje componente fijo",
        "anotaciones fabricación SolidWorks", "campo imprecisión intercambiables",
        "eje único agujero", "máximo material MMC",
        "tolerancia proyectada", "velocidad corte constante",
        "punto cambio herramienta", "triscado hoja sierra",
        "corte plasma piloto", "operación de trazado",
    ],
    "golden_programming.json": [
        "tabla HTML tr", "flexbox justify-content",
        "redundancia inconsistencia archivos", "modelo datos lógico",
        "removeprefix cadenas Python", "if precio entrada",
        "argumentos arbitrarios tupla", "if else JavaScript",
        "expresiones regulares replace", "shiftOut Arduino",
        "biblioteca Wire Arduino", "operadores asignación Python",
        "secuencia Fibonacci",
    ],
    "golden_sostenibilidad.json": [
        "dimensiones de sostenibilidad", "siglas ASG",
        "criterios ASG inversores", "impacto Revolución Industrial",
        "Pacto Verde Europeo", "ESRS empresas",
        "contribuciones NDC París", "emisión inmisión calidad",
        "país redujo emisiones", "economía circular residuos",
        "objetivo renovables 2020", "muertes trabajo OIT",
        "Integrated Reporting IR", "Heineken materia prima",
        "inversión LEGO sostenibles", "Nueva Visión Agricultura",
    ],
}


def main() -> int:
    out = []
    problems = []
    for fname, shorts in SHORT.items():
        path = os.path.join(EVAL_DIR, fname)
        cases = json.load(open(path, encoding="utf-8"))
        if len(cases) != len(shorts):
            problems.append(
                f"{fname}: {len(cases)} casos en el golden, {len(shorts)} consultas cortas"
            )
            continue
        for case, short in zip(cases, shorts):
            n = len(short.split())
            if n > MAX_TOKENS:
                problems.append(f"{fname}: {short!r} tiene {n} tokens (máx {MAX_TOKENS})")
            out.append(
                {
                    "query": short,
                    "topic": case["topic"],
                    "relevant_pages": case["relevant_pages"],
                    "long_query": case["query"],
                    "source_set": fname,
                }
            )

    if problems:
        for p in problems:
            print(f"ERROR: {p}", file=sys.stderr)
        return 1

    dest = os.path.join(EVAL_DIR, "golden_short.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    print(f"{len(out)} casos escritos en {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
