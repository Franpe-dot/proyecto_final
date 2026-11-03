"""Dobles de prueba: un chat model con salidas estructuradas guionadas."""

from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from app.actions import SolicitudStore
from app.graph.builder import SynthesisOutput, build_graph
from app.graph.supervisor import SupervisorDecision
from app.rag.retriever import PolicyRetriever
from app.config import get_settings

Script = Callable[[type, list[BaseMessage]], Any]


class ScriptedLLM(BaseChatModel):
    """Responde `with_structured_output` con lo que devuelve `script(schema, mensajes)`."""

    script: Script

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage("ok"))])

    def bind_tools(self, tools: Any, **kwargs: Any) -> Runnable[Any, Any]:  # type: ignore[override]
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable[Any, Any]:  # type: ignore[override]
        return RunnableLambda(lambda messages: self.script(schema, messages))


SOLICITUD_OK: dict[str, Any] = {
    "cliente": "Juan Pérez",
    "calificacion": "A",
    "monto": 3_000_000,
    "plazo_meses": 36,
    "tna_pct": 52,
    "cuota_mensual": 166_070.12,
    "ingreso_neto_mensual": 900_000,
    "justificacion": "Cuota de $166.070 = 18,45% del ingreso, bajo el tope del 30%.",
}


def finishing_script(solicitud: dict[str, Any] | None) -> Script:
    """El supervisor finaliza de entrada y la síntesis propone (o no) una solicitud."""

    def script(schema: type, messages: list[BaseMessage]) -> Any:
        if schema is SupervisorDecision:
            return SupervisorDecision(evaluacion="Alcanza.", next="FINISH", instruccion="")
        if schema is SynthesisOutput:
            return {"respuesta": "Respuesta de prueba.", "solicitud": solicitud}
        raise AssertionError(f"schema inesperado: {schema}")

    return script


def make_graph(store: SolicitudStore, script: Script, threshold: float = 1_000_000) -> Any:
    return build_graph(
        ScriptedLLM(script=script),
        PolicyRetriever(get_settings()),  # perezoso: nunca se conecta en estos tests
        store,
        max_steps=6,
        hitl_threshold=threshold,
        checkpointer=InMemorySaver(),
    )
