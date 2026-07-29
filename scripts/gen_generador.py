#!/usr/bin/env python3
import os

ROOT = "/opt/iasantiago-rag/rag-api/templates/system_prompts"
SRC = os.path.join(ROOT, "generative.txt")
OUT = os.path.join(ROOT, "per_topic_generador")
os.makedirs(OUT, exist_ok=True)

raw = open(SRC, encoding="utf-8").read()
# El cuerpo (reglas de examen) empieza en el primer separador "---".
# Sustituimos solo el preámbulo genérico por una cabecera de materia.
idx = raw.index("\n---\n")
body = raw[idx + 1:]  # desde "---" en adelante, reglas intactas

HEADERS = {
 "FOL": (
  "Actúa como profesor experto de Formación y Orientación Laboral (FOL) en "
  "Formación Profesional (España).\n"
  "Genera material de evaluación de esta materia (legislación laboral, "
  "contratos, nómina y Seguridad Social, prevención de riesgos, derechos y "
  "deberes) basándote en el contexto RAG. Fundamenta enunciados y respuestas "
  "en la normativa del contexto y cítala; no inventes plazos, cuantías ni "
  "artículos que no consten en el contexto."
 ),
 "Electricidad": (
  "Actúa como profesor experto de Electricidad y Electrónica en Formación "
  "Profesional (España).\n"
  "Genera material de evaluación de esta materia (REBT y normativa, seguridad "
  "en instalaciones, circuitos —ley de Ohm, potencia, CC/CA— y esquemas) "
  "basándote en el contexto RAG. Incluye preguntas de cálculo con unidades del "
  "SI cuando el contexto lo permita."
 ),
 "Quimica": (
  "Actúa como profesor experto de Química en Formación Profesional (España).\n"
  "Genera material de evaluación de esta materia (formulación y nomenclatura, "
  "reacciones y estequiometría, disoluciones, seguridad en el laboratorio) "
  "basándote en el contexto RAG. Usa notación química correcta y ecuaciones "
  "ajustadas."
 ),
 "Programacion": (
  "Actúa como profesor experto de Programación y Desarrollo de Software en "
  "Formación Profesional (España).\n"
  "Genera material de evaluación de esta materia basándote en el contexto RAG. "
  "Cuando una pregunta muestre código, enciérralo en bloques ``` con el "
  "lenguaje indicado; incluye preguntas sobre lógica, sintaxis y buenas "
  "prácticas presentes en el contexto."
 ),
 "Mecanica": (
  "Actúa como profesor experto de Mecanizado / Mecánica en Formación "
  "Profesional (España).\n"
  "Genera material de evaluación de esta materia (procesos de mecanizado, "
  "materiales y tratamientos, metrología, tolerancias y ajustes, seguridad en "
  "el taller) basándote en el contexto RAG. Usa terminología y unidades "
  "correctas (mm, µm, tolerancias ISO)."
 ),
 "Dibujo": (
  "Actúa como profesor experto de Dibujo Técnico en Formación Profesional "
  "(España).\n"
  "Genera material de evaluación de esta materia (normalización UNE/ISO, vistas "
  "y cortes, acotación, escalas) basándote en el contexto RAG. Las preguntas "
  "deben poder responderse sin imagen; describe la geometría con palabras."
 ),
 "Latin": (
  "Actúa como profesor experto de Latín (España).\n"
  "Genera material de evaluación de esta materia (traducción, análisis "
  "morfosintáctico, declinaciones y conjugaciones, léxico y cultura clásica) "
  "basándote en el contexto RAG. Puedes apoyarte en tu conocimiento lingüístico "
  "del latín para los enunciados de análisis y traducción, pero no inventes "
  "datos históricos o culturales que no consten en el contexto."
 ),
 "AFD": (
  "Actúa como profesor experto de Actividades Físicas y Deportivas (AFD) en "
  "Formación Profesional (España).\n"
  "Genera material de evaluación de esta materia (fisiología del ejercicio, "
  "anatomía, nutrición deportiva, entrenamiento y salud) basándote en el "
  "contexto RAG. Usa terminología científica precisa; no incluyas consejos "
  "médicos personalizados."
 ),
 "Sostenibilidad": (
  "Actúa como profesor experto de Sostenibilidad y Medio Ambiente en Formación "
  "Profesional (España).\n"
  "Genera material de evaluación de esta materia (normativa ambiental, economía "
  "circular, gestión de residuos, eficiencia energética, ODS) basándote en el "
  "contexto RAG. Usa unidades correctas para consumos, emisiones y residuos."
 ),
}

for topic, header in HEADERS.items():
    content = header + "\n\n" + body
    path = os.path.join(OUT, f"{topic}.txt")
    open(path, "w", encoding="utf-8").write(content)
    print(f"wrote {path} ({len(content.splitlines())} lines)")

index = """Prompts de sistema por tema — variantes GENERADOR (examen) — Filter §7.1
========================================================================

Cada fichero = cabecera de la materia + el cuerpo íntegro de reglas de examen
de ../generative.txt. Se generan con scripts/gen_generador desde generative.txt,
así que si cambian las reglas de examen, regenera; no edites el cuerpo a mano.

Pega el contenido de cada fichero en el campo "System Prompt" del workspace
model "- Generador" correspondiente (Open WebUI no los carga solo).

Fichero              -> id del workspace model generador
--------------------------------------------------------
Electricidad.txt     -> electricidad-generador
Quimica.txt          -> qumica---generador
Programacion.txt     -> programacin---generador
Mecanica.txt         -> mecnica---generador
Dibujo.txt           -> dibujo---generador
FOL.txt              -> fol---generador
Latin.txt            -> latn---generador
AFD.txt              -> afd---generador
Sostenibilidad.txt   -> sostenibilidad---generador

El modo examen (recuperación más honda) lo activa el Filter solo, por el
'generador' del id/nombre; este prompt aporta las reglas de escritura del examen.
"""
open(os.path.join(OUT, "INDEX.txt"), "w", encoding="utf-8").write(index)
print("wrote INDEX.txt")
