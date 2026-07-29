"""
title: IASantiago RAG Retrieval
author: iasantiago
version: 0.2.0
required_open_webui_version: 0.5.0
description: >
    Inlet Filter que consulta el servicio de retrieval puro de rag-api
    (POST /retrieve, PLAN.md §7.1) e inyecta el contexto con citaciones en el
    ultimo mensaje del usuario. Open WebUI conserva el historial y hace
    streaming directo contra vLLM; rag-api queda como servicio de retrieval.

    UN SOLO Filter sirve a los modelos normales y a los '- Generador': el modo
    examen se detecta por el nombre/id del modelo (ver `generative_markers`), no
    por una valve global, así que ya no hacen falta dos copias del Filter con
    distinta valve `generative` (v0.2.0). Se activa por cada workspace model.
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
        generative_markers: str = Field(
            default="generador,generator,examen",
            description=(
                "Tokens (separados por coma) que, si aparecen en el id o el nombre "
                "del modelo, activan el modo examen (recupera mas hondo). Así un "
                "solo Filter distingue 'qumica' de 'qumica---generador' sin valve "
                "por modelo. Busqueda por substring, sin mayusculas/acentos."
            ),
        )
        generative: bool = Field(
            default=False,
            description=(
                "Fuerza el modo examen para TODOS los modelos de este Filter, "
                "ignorando generative_markers. Normalmente False: deja que el "
                "nombre/id del modelo decida."
            ),
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

    def _is_generative(self, model: Optional[dict]) -> bool:
        """Decide el modo examen a partir del modelo, no de una valve global.

        La valve `generative` (si está a True) fuerza el modo para todos. Si no,
        se mira si el id o el nombre del modelo contienen alguno de los
        `generative_markers` (p. ej. 'generador'), lo que permite que un único
        Filter sirva tanto a 'qumica' como a 'qumica---generador'.
        """
        if self.valves.generative:
            return True
        if not model:
            return False
        haystack = " ".join(
            (
                model.get("id") or "",
                model.get("name") or "",
                (model.get("info") or {}).get("name") or "",
            )
        ).lower()
        markers = (
            m.strip().lower()
            for m in (self.valves.generative_markers or "").split(",")
        )
        return any(m and m in haystack for m in markers)

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
            "generative": self._is_generative(__model__),
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
