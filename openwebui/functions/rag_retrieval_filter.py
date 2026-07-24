"""
title: IASantiago RAG Retrieval
author: iasantiago
version: 0.1.0
required_open_webui_version: 0.5.0
description: >
    Inlet Filter que consulta el servicio de retrieval puro de rag-api
    (POST /retrieve, PLAN.md §7.1) e inyecta el contexto con citaciones en el
    ultimo mensaje del usuario. Open WebUI conserva el historial y hace
    streaming directo contra vLLM; rag-api queda como servicio de retrieval.

    Se instala como *Function* de tipo Filter y se activa por cada workspace
    model de tema. Es un prototipo que corre EN PARALELO a la ruta topic:X
    actual para poder hacer A/B (PLAN.md §5, paso 8): no borra nada de rag-api.
"""

import json
from typing import Optional

import aiohttp
from pydantic import BaseModel, Field


class Filter:
    class Valves(BaseModel):
        rag_api_url: str = Field(
            default="http://rag-api:8001",
            description="Base URL de rag-api (red interna de docker).",
        )
        api_key: str = Field(
            default="",
            description="Bearer para rag-api = OPENAI_API_KEY. Obligatorio.",
        )
        topic_map: str = Field(
            default="{}",
            description=(
                'JSON {"<model_id>": "<TopicLabel>"}. Si el model id no esta '
                "aqui, se usa el nombre del workspace model tal cual como tema. "
                'Ej: {"quimica": "Chemistry", "electricidad": "Electricidad"}'
            ),
        )
        strip_suffixes: str = Field(
            default="- Generador,-Generador,— Generador, Generador,- Examen,-Examen",
            description=(
                "Sufijos (separados por coma) que se quitan del nombre del modelo "
                "antes de resolver el tema, para que las variantes 'Electricidad - "
                "Generador' apunten al mismo tema 'Electricidad'. topic_map tiene "
                "prioridad sobre esto."
            ),
        )
        default_topic: str = Field(
            default="",
            description="Tema de reserva si no se resuelve ninguno (vacio = error visible).",
        )
        generative: bool = Field(
            default=False,
            description="Modo examen: recupera mas hondo. Actívalo en la variante '— generador'.",
        )
        top_k: int = Field(
            default=0,
            description="Override de profundidad. 0 = usar el valor por defecto de rag-api.",
        )
        timeout: int = Field(default=60, description="Timeout HTTP en segundos.")
        show_status: bool = Field(
            default=True, description="Emitir estado 'Consultando documentos…' en la UI."
        )

    def __init__(self):
        self.valves = self.Valves()

    # ------------------------------------------------------------------
    def _resolve_topic(self, model: Optional[dict]) -> Optional[str]:
        """Deriva el tema del workspace model.

        Prioridad: topic_map[id] -> topic_map[name] -> name del modelo ->
        default_topic. El caso de cero-config es nombrar el workspace model
        exactamente igual que el tema (p. ej. "Electricidad").
        """
        if not model:
            return self.valves.default_topic or None
        try:
            mapping = json.loads(self.valves.topic_map or "{}")
        except json.JSONDecodeError:
            mapping = {}

        model_id = model.get("id") or ""
        # El nombre legible puede venir en model["name"] o model["info"]["name"].
        name = model.get("name") or (model.get("info") or {}).get("name") or ""

        for key in (model_id, name):
            if key and key in mapping:
                return mapping[key]

        # Quitar sufijos de variante ("Electricidad - Generador" -> "Electricidad")
        # para que las variantes compartan el tema base sin tener que mapearlas.
        resolved = name
        for suffix in (s.strip() for s in (self.valves.strip_suffixes or "").split(",")):
            if suffix and resolved.lower().endswith(suffix.lower()):
                resolved = resolved[: -len(suffix)]
                break
        resolved = resolved.rstrip(" \t-–—")

        return resolved or self.valves.default_topic or None

    async def _emit(self, emitter, description: str, done: bool):
        if emitter and self.valves.show_status:
            await emitter(
                {
                    "type": "status",
                    "data": {"description": description, "done": done},
                }
            )

    # ------------------------------------------------------------------
    async def inlet(
        self,
        body: dict,
        __event_emitter__=None,
        __user__: Optional[dict] = None,
        __model__: Optional[dict] = None,
    ) -> dict:
        messages = body.get("messages") or []
        if not messages:
            return body

        # Ultimo mensaje de usuario.
        last_user_idx = next(
            (i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"),
            None,
        )
        if last_user_idx is None:
            return body

        query = (messages[last_user_idx].get("content") or "").strip()
        if not query:
            return body

        topic = self._resolve_topic(__model__)
        if not topic:
            await self._emit(__event_emitter__, "RAG: no se pudo resolver el tema", True)
            return body

        await self._emit(__event_emitter__, "Consultando documentos…", False)

        payload = {
            "query": query,
            "topic": topic,
            "generative": self.valves.generative,
        }
        if self.valves.top_k > 0:
            payload["top_k"] = self.valves.top_k

        headers = {"Content-Type": "application/json"}
        if self.valves.api_key:
            headers["Authorization"] = f"Bearer {self.valves.api_key}"
        # Propagar el email del usuario para la telemetria hasheada de rag-api.
        email = (__user__ or {}).get("email")
        if email:
            headers["X-Email"] = email

        try:
            timeout = aiohttp.ClientTimeout(total=self.valves.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.valves.rag_api_url}/retrieve",
                    json=payload,
                    headers=headers,
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
        except Exception as e:
            await self._emit(__event_emitter__, f"RAG no disponible: {e}", True)
            # Degradar con elegancia: sin contexto, el modelo responde igual.
            return body

        context = data.get("context") or ""
        num = (data.get("meta") or {}).get("num_chunks", 0)

        if context and num:
            # Contexto al final del mensaje del usuario, para conservar el prefijo
            # cacheable del historial anterior (misma ubicacion que rag-api).
            messages[last_user_idx]["content"] = f"{query}\n\n{context}"
            body["messages"] = messages

        await self._emit(
            __event_emitter__,
            f"{num} fragmentos recuperados" if num else "Sin fragmentos relevantes",
            True,
        )
        return body
