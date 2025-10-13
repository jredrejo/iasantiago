def extract_topic_from_model_name(model_name: str, fallback: str) -> str:
    # model_name esperado: "topic:Chemistry / Electronics / Programming"
    if model_name.startswith("topic:"):
        return model_name.split("topic:", 1)[1].strip()
    return fallback
