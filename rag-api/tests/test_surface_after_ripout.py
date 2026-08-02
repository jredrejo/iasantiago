# Archivo: rag-api/tests/test_surface_after_ripout.py
# Descripción: la superficie HTTP que rag-api expone tras el rip-out del §7.1.
#
# El punto 6 de PLAN.md. rag-api dejó de ser un proxy de chat compatible con la
# API de OpenAI para ser un servicio de retrieval puro: `GET /v1/models` y
# `POST /v1/chat/completions` ya no existen y la orquestación es de Open WebUI.
#
# Estos tests existen por un modo de fallo concreto, no por ceremonia: un
# `import` olvidado o un decorador que sobrevive al borrado dejan la ruta viva y
# medio rota, y nadie lo nota hasta que alguien la llama. Aquí se afirma que la
# ruta **no está**, que las tres que se quedan **sí**, y que rag-api ya no tiene
# manera de hablar con vLLM.
#
# Se usa TestClient sin `with`, así que NO se ejecuta el lifespan y no se carga
# ningún modelo.

import importlib

import pytest
from fastapi.testclient import TestClient

import app as app_module
from config.settings import OPENAI_API_KEY

AUTH = {"Authorization": f"Bearer {OPENAI_API_KEY}"}


@pytest.fixture
def client():
    return TestClient(app_module.app)


def _rutas() -> set:
    return {r.path for r in app_module.app.routes}


# ------------------------------------------------- lo que se fue

RETIRADAS = ["/v1/models", "/v1/chat/completions"]


@pytest.mark.parametrize("ruta", RETIRADAS)
def test_la_ruta_no_esta_registrada(ruta):
    """En la tabla de rutas, que es donde se ve un decorador superviviente."""
    assert ruta not in _rutas()


def test_v1_models_responde_404(client):
    """404 y no 401: la autenticación no llega a mirar una ruta que no existe."""
    r = client.get("/v1/models", headers=AUTH)
    assert r.status_code == 404


def test_chat_completions_responde_404(client):
    r = client.post(
        "/v1/chat/completions",
        json={"model": "topic:Latin", "messages": [{"role": "user", "content": "hola"}]},
        headers=AUTH,
    )
    assert r.status_code == 404


def test_no_queda_ninguna_ruta_que_publique_modelos_topic(client):
    """La forma `topic:X` era el contrato con Open WebUI; no debe quedar rastro.

    Se comprueba sobre el cuerpo real de las respuestas, no sobre el código: un
    endpoint nuevo que volviera a publicarlas sería el mismo error con otro
    nombre.
    """
    r = client.get("/healthz")
    assert "topic:" not in r.text


# ------------------------------------------------- lo que se queda


@pytest.mark.parametrize("ruta", ["/healthz", "/retrieve", "/v1/eval/offline"])
def test_las_rutas_que_sirven_siguen_registradas(ruta):
    """`/v1/eval/offline` comparte prefijo con lo retirado y NO se va con ello."""
    assert ruta in _rutas()


def test_healthz_sigue_sin_autenticacion_y_lista_los_temas(client):
    """Es la señal de readiness del healthcheck de compose."""
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert isinstance(body["topics"], list)


def test_retrieve_sigue_exigiendo_credenciales(client):
    r = client.post("/retrieve", json={"query": "x", "topic": "Latin"})
    assert r.status_code in (401, 403)


# ------------------------------------------------- lo que ya no puede pasar


def test_rag_api_ya_no_puede_hablar_con_vllm():
    """Ni cliente, ni URL upstream, ni nombre de modelo servido.

    Es la mitad del rip-out que no se ve en la tabla de rutas: mientras el
    cliente siguiera importable, cualquier endpoint futuro podría volver a
    generar desde aquí sin que nadie lo decidiera.
    """
    for modulo in ("core.vllm_client", "chat", "chat.intent", "token_utils"):
        with pytest.raises(ImportError):
            importlib.import_module(modulo)


def test_settings_no_expone_configuracion_de_generacion():
    """Los parámetros de muestreo los decide ahora Open WebUI, por modelo."""
    settings = importlib.import_module("config.settings")
    for nombre in (
        "UPSTREAM_OPENAI_URL",
        "VLLM_SERVED_MODEL",
        "RESPONSE_TEMPERATURE",
        "GENERATIVE_TEMPERATURE",
        "MIN_RESPONSE_TOKENS",
        "GENERATIVE_MAX_TOKENS_PERCENT",
        "RESPONSE_MAX_TOKENS_PERCENT",
    ):
        assert not hasattr(settings, nombre), nombre


def test_lo_que_sigue_haciendo_falta_del_modelo_no_se_borro():
    """El tokenizador y la ventana no son configuración de chat: acotan el
    contexto que `/retrieve` devuelve, y borrarlos de paso habría roto el
    recorte por tokens sin que ningún test de rutas lo viera."""
    settings = importlib.import_module("config.settings")
    for nombre in ("VLLM_MODEL", "VLLM_MAX_MODEL_LEN", "VLLM_MAX_TOKENS", "OPENAI_API_KEY"):
        assert hasattr(settings, nombre), nombre
