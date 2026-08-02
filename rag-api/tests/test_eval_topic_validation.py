# Archivo: rag-api/tests/test_eval_topic_validation.py
# Descripción: contrato de /v1/eval/offline ante temas que el servicio no conoce.
#
# El cabo que dejó abierto el punto 8 de PLAN.md: se validó `/retrieve` y no el
# endpoint de evaluación, donde un tema mal escrito medía **0.000 en silencio**.
# Eso es peor que en `/retrieve`, porque 0.000 no se lee como "te has equivocado
# de etiqueta" sino como "el retrieval es malísimo", y es la cifra con la que se
# deciden las fases. Misma separación por dueño que `/retrieve`:
#
#   tema desconocido          -> 400 (culpa del llamador)
#   tema válido sin colección -> 200 con aviso (culpa de operación)
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


def _case(topic: str, query: str = "quid est") -> dict:
    return {"query": query, "topic": topic, "relevant_pages": ["x.pdf#1"]}


# ---------------------------------------------------------------- 400


def test_tema_desconocido_devuelve_400(client, topics):
    r = client.post("/v1/eval/offline", json=[_case("Latín")], headers=AUTH)
    assert r.status_code == 400


def test_el_400_nombra_los_temas_malos_y_los_validos(client, topics):
    """El mensaje tiene que bastar sin abrir el log."""
    r = client.post("/v1/eval/offline", json=[_case("Latín")], headers=AUTH)
    detail = r.json()["detail"]
    assert detail["error"] == "unknown_topic"
    assert detail["topics"] == ["Latín"]
    assert detail["valid_topics"] == topics


def test_reune_todos_los_temas_malos_no_solo_el_primero(client, topics):
    """Con 157 casos, arreglarlos de uno en uno es una tirada por errata."""
    cases = [_case("Latín"), _case("Mecanica"), _case("Química"), _case("Latín")]
    r = client.post("/v1/eval/offline", json=cases, headers=AUTH)
    # Ordenados y sin repetir; `Mecanica` es válido y no debe aparecer.
    assert r.json()["detail"]["topics"] == ["Latín", "Química"]


def test_un_solo_caso_malo_tumba_la_tirada_entera(client, topics):
    """Medir 156 bien y 1 en silencio da un ponderado que nadie puede usar."""
    cases = [_case("Latin") for _ in range(10)] + [_case("latin")]
    r = client.post("/v1/eval/offline", json=cases, headers=AUTH)
    assert r.status_code == 400


def test_la_validacion_va_antes_de_recuperar_nada(client, topics, monkeypatch):
    """Una tirada son minutos: fallar al final no le sirve a nadie."""
    llamadas = []
    monkeypatch.setattr(
        app_module, "choose_retrieval", lambda *a, **k: llamadas.append(a) or ([], {})
    )
    monkeypatch.setattr(app_module, "collection_exists", lambda t: True)
    client.post("/v1/eval/offline", json=[_case("Mecánica")], headers=AUTH)
    assert llamadas == []


def test_la_distincion_es_exacta_no_por_mayusculas_ni_acentos(client, topics):
    """Igual que en /retrieve: plegar acentos taparía la errata en vez de mostrarla."""
    for malo in ("latin", "LATIN", "Latín", "Mecánica", ""):
        r = client.post("/v1/eval/offline", json=[_case(malo)], headers=AUTH)
        assert r.status_code == 400, malo


# ------------------------------------------------- tema válido sin colección


def test_falta_de_coleccion_no_es_400_sino_aviso(client, topics, monkeypatch):
    """No es culpa de quien llama, así que no se le devuelve como error suyo."""
    monkeypatch.setattr(app_module, "collection_exists", lambda t: False)
    monkeypatch.setattr(app_module, "choose_retrieval", lambda *a, **k: ([], {}))
    r = client.post("/v1/eval/offline", json=[_case("Latin")], headers=AUTH)
    assert r.status_code == 200
    assert any("sin colección" in w for w in r.json()["warnings"])


def test_el_aviso_dice_que_el_resultado_no_describe_al_retrieval(
    client, topics, monkeypatch
):
    """Es el punto: que quien lea la tabla sepa que ese 0.000 no es calidad."""
    monkeypatch.setattr(app_module, "collection_exists", lambda t: False)
    monkeypatch.setattr(app_module, "choose_retrieval", lambda *a, **k: ([], {}))
    r = client.post("/v1/eval/offline", json=[_case("Latin")], headers=AUTH)
    aviso = next(w for w in r.json()["warnings"] if "sin colección" in w)
    assert "Latin" in aviso and "0.000" in aviso


def test_el_aviso_de_tema_va_antes_que_los_de_caso(client, topics, monkeypatch):
    """Si falta la colección, los avisos por caso son consecuencia, no causa."""
    monkeypatch.setattr(app_module, "collection_exists", lambda t: False)
    monkeypatch.setattr(app_module, "choose_retrieval", lambda *a, **k: ([], {}))
    caso_sin_gt = {"query": "sin ground truth", "topic": "Latin", "relevant_pages": []}
    r = client.post("/v1/eval/offline", json=[caso_sin_gt], headers=AUTH)
    warnings = r.json()["warnings"]
    assert "sin colección" in warnings[0]


def test_no_se_consulta_qdrant_una_vez_por_caso(client, topics, monkeypatch):
    """157 casos de 10 temas son 10 comprobaciones, no 157."""
    llamadas = []
    monkeypatch.setattr(
        app_module, "collection_exists", lambda t: llamadas.append(t) or True
    )
    monkeypatch.setattr(app_module, "choose_retrieval", lambda *a, **k: ([], {}))
    cases = [_case("Latin") for _ in range(5)] + [_case("Mecanica") for _ in range(5)]
    client.post("/v1/eval/offline", json=cases, headers=AUTH)
    assert sorted(llamadas) == ["Latin", "Mecanica"]


# ---------------------------------------------------------------- camino feliz


def test_el_camino_feliz_no_cambia(client, topics, monkeypatch):
    """La guarda corre antes de los diez flujos y no altera ninguno."""
    monkeypatch.setattr(app_module, "collection_exists", lambda t: True)
    monkeypatch.setattr(app_module, "choose_retrieval", lambda *a, **k: ([], {}))
    r = client.post("/v1/eval/offline", json=[_case("Latin")], headers=AUTH)
    assert r.status_code == 200
    assert "aggregate" in r.json()
    assert not any("sin colección" in w for w in r.json()["warnings"])
