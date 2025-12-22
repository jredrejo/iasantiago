#!/usr/bin/env python3
"""
Script de prueba para verificar que la traducción de queries funciona correctamente.
Ejecuta esto para probar el pipeline de traducción antes del despliegue.
"""

from translation import translate_query, detect_language, should_translate

test_queries = [
    ("dime que es Quality of Service en MQTT", "es"),  # Español
    ("Qu'est-ce que la qualité de service en MQTT?", "fr"),  # Francés
    ("What is Quality of Service in MQTT?", "en"),  # Inglés (no debe traducirse)
    ("Was ist Quality of Service in MQTT?", "de"),  # Alemán
    ("Qual è la qualità del servizio in MQTT?", "it"),  # Italiano
]

print("=" * 80)
print("SUITE DE PRUEBAS DE TRADUCCIÓN")
print("=" * 80)

for query, expected_lang in test_queries:
    print(f"\n📝 Query original: {query}")
    print(f"   Idioma esperado: {expected_lang}")

    # Detectar idioma
    detected = detect_language(query)
    print(f"   Idioma detectado: {detected}")

    # Probar traducción si no es inglés
    if detected != "en":
        translated, source_lang = translate_query(query, detected, "en")
        print(f"   Query traducido: {translated}")
        print(f"   Idioma origen: {source_lang}")
    else:
        print(f"   ✓ Ya está en inglés, no se necesita traducción")

    # Probar should_translate
    needs_translation = should_translate(query)
    print(f"   Debe traducirse: {needs_translation}")

print("\n" + "=" * 80)
print("✓ Prueba de traducción completada")
print("=" * 80)
