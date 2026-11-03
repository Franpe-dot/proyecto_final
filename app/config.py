"""Configuración del sistema, leída desde variables de entorno / `.env`.

Nada que dependa del despliegue (claves, modelos, URLs, nombres de colección o de cola)
está escrito en el código: todo sale de acá.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_anthropic import ChatAnthropic
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


class Settings(BaseSettings):
    # LLM
    anthropic_api_key: SecretStr
    anthropic_model: str
    max_tokens: int = Field(default=4096, gt=0)
    # None = no se envía. Los modelos actuales rechazan valores no default.
    temperature: float | None = None

    # Orquestador
    max_steps: int = Field(default=6, ge=1, description="Delegaciones por consulta.")
    hitl_monto_umbral: float = Field(
        default=1_000_000,
        ge=0,
        description="Desde este monto, registrar una solicitud requiere aprobación humana.",
    )

    # Redis: cola de jobs, estado de jobs, checkpointer y solicitudes registradas
    redis_url: str = "redis://localhost:6379"
    queue_name: str = "intelligence:jobs"
    job_ttl_seconds: int = Field(default=7 * 24 * 3600, gt=0)
    worker_concurrency: int = Field(default=2, ge=1)

    # RAG: ChromaDB en modo servidor + embeddings e5
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str
    embedding_model: str

    # Observabilidad: Arize Phoenix (OpenTelemetry / OpenInference)
    tracing_enabled: bool = True
    phoenix_collector_endpoint: str = "http://localhost:6006/v1/traces"
    phoenix_project_name: str = "intelligence-system"

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # los valores vienen del entorno


def build_llm(settings: Settings) -> ChatAnthropic:
    """Modelo de Anthropic que comparten el supervisor, los especialistas y la síntesis."""
    kwargs: dict[str, Any] = {
        "model": settings.anthropic_model,
        "api_key": settings.anthropic_api_key,
        "max_tokens": settings.max_tokens,
        "max_retries": 3,
    }
    if settings.temperature is not None:
        kwargs["temperature"] = settings.temperature
    return ChatAnthropic(**kwargs)
