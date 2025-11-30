#!/usr/bin/env python3
"""
Test script to verify query translation works correctly.
Run this to test the translation pipeline before deploying.
"""

from translation import translate_query, detect_language, should_translate

test_queries = [
    ("dime que es Quality of Service en MQTT", "es"),  # Spanish
    ("Qu'est-ce que la qualité de service en MQTT?", "fr"),  # French
    ("What is Quality of Service in MQTT?", "en"),  # English (should not translate)
    ("Was ist Quality of Service in MQTT?", "de"),  # German
    ("Qual è la qualità del servizio in MQTT?", "it"),  # Italian
]

print("=" * 80)
print("TRANSLATION TEST SUITE")
print("=" * 80)

for query, expected_lang in test_queries:
    print(f"\n📝 Original query: {query}")
    print(f"   Expected language: {expected_lang}")
    
    # Detect language
    detected = detect_language(query)
    print(f"   Detected language: {detected}")
    
    # Test translation if not English
    if detected != "en":
        translated, source_lang = translate_query(query, detected, "en")
        print(f"   Translated query: {translated}")
        print(f"   Source language: {source_lang}")
    else:
        print(f"   ✓ Already in English, no translation needed")
    
    # Test should_translate
    needs_translation = should_translate(query)
    print(f"   Should translate: {needs_translation}")

print("\n" + "=" * 80)
print("✓ Translation test complete")
print("=" * 80)
