# Archivo: rag-api/tests/test_intent.py
# Descripción: Tests de detect_generative_intent con frases reales de alumnos
#
# Ejecutar con pytest (`pytest tests/`) o directamente:
#   python tests/test_intent.py

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chat.intent import detect_generative_intent  # noqa: E402

# (frase, esperado_generativo)
CASES = [
    # --- Generativas: verbo creativo + objeto ---
    ("Crea un examen de 10 preguntas sobre enlaces químicos", True),
    ("Hazme 5 ejercicios de estequiometría", True),
    ("Genera una prueba de evaluación del tema 3", True),
    ("Prepárame un cuestionario sobre la célula", True),
    ("¿Puedes elaborar un test tipo opción múltiple del tema 2?", True),
    ("Quiero que me hagas un examen de FOL", True),
    ("Diseña una actividad práctica sobre circuitos en paralelo", True),
    ("Redacta preguntas cortas acerca de la revolución industrial", True),
    ("propón ejercicios de repaso para el examen final", True),
    ("Escribe un resumen del tema 4", True),
    # --- Generativas: cantidad explícita de ítems ---
    ("10 preguntas tipo test del tema 2", True),
    ("dame 3 problemas de dinámica con solución", True),
    # --- Generativas: reorganizar en un formato ---
    ("Resume el tema 4 en un esquema", True),
    ("Organiza los conceptos del tema en un mapa conceptual", True),
    ("sintetiza la unidad 2 como lista de puntos clave", True),
    # --- NO generativas: dudas que solo mencionan las palabras clave ---
    ("¿En qué página está el ejercicio 3?", False),
    ("Tengo una pregunta sobre la fotosíntesis", False),
    ("¿Cómo se hace la prueba del ácido?", False),
    ("No sé hacer el ejercicio 2 de la página 34", False),
    ("¿Qué actividades económicas aparecen en el tema 2?", False),
    ("¿Me explicas la solución del problema del enunciado?", False),
    ("¿Cuándo es el examen del tema 5?", False),
    ("¿Qué entra en la evaluación de este trimestre?", False),
    ("Explícame qué es un mapa de Karnaugh", False),
    ("¿Dónde está la lista de materiales de la práctica?", False),
    ("¿Qué significa esta pregunta del libro?", False),
    ("cual es la respuesta a la cuestión 4", False),
]


def test_generative_intent_cases():
    failures = []
    for phrase, expected in CASES:
        got = detect_generative_intent(phrase)
        if got != expected:
            failures.append(
                f"  '{phrase}' → {'GENERATIVA' if got else 'respuesta'} "
                f"(esperado: {'GENERATIVA' if expected else 'respuesta'})"
            )
    assert not failures, "Casos fallidos:\n" + "\n".join(failures)


if __name__ == "__main__":
    import logging

    logging.disable(logging.CRITICAL)
    test_generative_intent_cases()
    print(f"OK: {len(CASES)} casos de intención pasan")
