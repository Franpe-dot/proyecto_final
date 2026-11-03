"""Configuración común de los tests.

Los tests unitarios (`tests/unit`) no usan red: LLM con respuestas guionadas, Redis en
memoria (fakeredis) y checkpointer en memoria. Los e2e (`tests/e2e`) pegan contra el
sistema levantado con `docker compose up`.
"""

import os

# Antes de importar `app`: get_settings() se cachea en la primera llamada.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_MODEL", "fake-model")
os.environ.setdefault("CHROMA_COLLECTION", "test")
os.environ.setdefault("EMBEDDING_MODEL", "fake-embeddings")
os.environ["TRACING_ENABLED"] = "false"
