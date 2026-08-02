# Archivo: rag-api/tests/test_retrieve_topic_validation.py
# Descripción: contrato de /retrieve ante un tema que el servicio no conoce.
#
# El punto 8 de PLAN.md: un tema fuera de TOPIC_LABELS devolvía 200 con lista
# vacía, indistinguible de "el corpus no tiene la respuesta". Eso disfrazó de
# problema de calidad los cuatro workspace models "- Generador" rotos del
# 2026-08-02. Estos tests fijan la distinción que se perdía:
#
#   tema desconocido        -> 400 (culpa del llamador)
#   tema válido sin colección -> 200 vacío (culpa de operación; §7.1 dice no tumbar)
#
# Se usa TestClient sin `with`, así que NO se ejecuta el lifespan y no se carga
# ningún modelo: las dos ramas cortan antes de tocar embeddings o Qdrant.

import pytest
from fastapi.testclient import TestClient

import app as app_module
from config.settings import OPENAI_API_KEY

AUTH = {"Authorization": f"Bearer {OPENAI_API_KEY}"}


@pytest.fixture
def client():
    return TestClient(app_module.app)


@pytest.fixture
def topics(monkeypatch):
    """TOPIC_LABELS conocido y fijo, para no depender del .env de la máquina."""
    labels = ["Latin", "Mecanica"]
    monkeypatch.setattr(app_module, "TOPIC_LABELS", labels)
    return labels


def _body(topic: str) -> dict:
    return {"query": "quid est", "topic": topic}


# ---------------------------------------------------------------- 400


def test_tema_desconocido_devuelve_400(client, topics):
    r = client.post("/retrieve", json=_body("Latín"), headers=AUTH)
    assert r.status_code == 400


def test_el_400_nombra_el_tema_recibido_y_los_validos(client, topics):
    """El mensaje tiene que bastar sin abrir el log: es el punto del cambio."""
    r = client.post("/retrieve", json=_body("Latín"), headers=AUTH)
    detail = r.json()["detail"]
    assert detail["error"] == "unknown_topic"
    assert detail["topic"] == "Latín"
    assert detail["valid_topics"] == topics


def test_tema_desconocido_no_llega_a_consultar_qdrant(client, topics, monkeypatch):
    """La validación va ANTES de la red: un tema mal escrito no cuesta un viaje."""
    llamadas = []
    monkeypatch.setattr(
        app_module, "collection_exists", lambda t: llamadas.append(t) or True
    )
    client.post("/retrieve", json=_body("Mecánica"), headers=AUTH)
    assert llamadas == []


def test_la_distincion_es_exacta_no_por_mayusculas_ni_acentos(client, topics):
    """'Latin' es válido y 'latin'/'Latín' no.

    Plegar acentos aquí habría tapado el bug del 2026-08-02 en vez de mostrarlo:
    el objetivo es que una etiqueta que no coincide se vea, no que cuele.
    """
    for malo in ("latin", "LATIN", "Latín", "Mecánica", ""):
        r = client.post("/retrieve", json=_body(malo), headers=AUTH)
        assert r.status_code == 400, malo


# ---------------------------------------------------------------- 200 vacío


def test_tema_valido_sin_coleccion_sigue_degradando_con_elegancia(
    client, topics, monkeypatch
):
    """§7.1: que falte la colección no puede tumbar el chat del alumno."""
    monkeypatch.setattr(app_module, "collection_exists", lambda t: False)
    r = client.post("/retrieve", json=_body("Latin"), headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["context"] == ""
    assert body["citations"] == []
    assert body["meta"]["num_chunks"] == 0


def test_falta_de_coleccion_se_marca_distinto_de_tema_desconocido(
    client, topics, monkeypatch
):
    """Los dos fallos eran el mismo `meta.error`; ahora se pueden separar."""
    monkeypatch.setattr(app_module, "collection_exists", lambda t: False)
    r = client.post("/retrieve", json=_body("Latin"), headers=AUTH)
    assert r.json()["meta"]["error"] == "missing_collection"


# ---------------------------------------------------------------- camino feliz


def test_el_camino_feliz_no_cambia(client, topics, monkeypatch):
    """Un tema válido con colección atraviesa la validación sin tocarse."""
    monkeypatch.setattr(app_module, "collection_exists", lambda t: True)
    monkeypatch.setattr(
        app_module,
        "choose_retrieval",
        lambda *a, **k: ([], {"mode": "bm25", "original_language": "es"}),
    )
    monkeypatch.setattr(app_module, "attach_citations", lambda *a, **k: ("", []))
    monkeypatch.setattr(app_module, "telemetry_log", lambda *a, **k: None)

    r = client.post("/retrieve", json=_body("Latin"), headers=AUTH)
    assert r.status_code == 200
    assert r.json()["meta"]["topic"] == "Latin"
    assert "error" not in r.json()["meta"]


def test_sin_credenciales_sigue_siendo_401_no_400(client, topics):
    """El orden importa: la autenticación va antes que la validación de tema."""
    r = client.post("/retrieve", json=_body("Latín"))
    assert r.status_code in (401, 403)
