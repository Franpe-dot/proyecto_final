"""Observabilidad con Arize Phoenix (OpenInference sobre OpenTelemetry).

`register(auto_instrument=True)` detecta los instrumentadores instalados
(`openinference-instrumentation-langchain`) y envía cada nodo del grafo, cada llamada al
LLM y cada tool como spans al colector de Phoenix.
"""

import logging
import os

from opentelemetry.sdk.trace import TracerProvider

from app.config import Settings

logger = logging.getLogger(__name__)


def setup_tracing(settings: Settings, service: str) -> TracerProvider | None:
    if not settings.tracing_enabled:
        logger.info("Tracing deshabilitado (TRACING_ENABLED=false)")
        return None
    from phoenix.otel import register

    # El SDK de Anthropic emite sus propios spans cuando hay un tracer provider global.
    # Duplican los del instrumentador de LangChain (que ya registra tokens y costo) y
    # quedan fuera del árbol del grafo, así que se apagan. Tiene que hacerse antes de
    # crear el cliente (build_llm).
    os.environ.setdefault("ANTHROPIC_OPEN_TELEMETRY", "false")

    provider = register(
        project_name=settings.phoenix_project_name,
        endpoint=settings.phoenix_collector_endpoint,
        auto_instrument=True,
        batch=True,
        set_global_tracer_provider=True,
        verbose=False,
    )
    logger.info(
        "Tracing [%s] → %s (proyecto %s)",
        service,
        settings.phoenix_collector_endpoint,
        settings.phoenix_project_name,
    )
    return provider


def shutdown_tracing(provider: TracerProvider | None) -> None:
    """Envía los spans pendientes antes de salir."""
    if provider is not None:
        provider.force_flush()
        provider.shutdown()
