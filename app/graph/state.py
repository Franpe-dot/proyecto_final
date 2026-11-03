"""Estado compartido del orquestador.

Además de `messages` (heredado de `MessagesState`), el estado lleva un libro de aportes
(`contributions`) que registra qué agente produjo qué información, con qué instrucción y
con qué herramientas. El supervisor decide a partir de ese libro, no del historial crudo.

Como el checkpointer conserva el estado entre consultas de un mismo `thread_id`, cada
consulta es un `turn`: los aportes y decisiones se etiquetan con su turno y el supervisor
solo mira los del turno actual. Los mensajes sí se acumulan: son la memoria conversacional.
"""

import operator
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import MessagesState

RESEARCH_AGENT = "research_agent"
ANALYST_AGENT = "analyst_agent"
FINISH = "FINISH"

AgentName = Literal["research_agent", "analyst_agent"]
NextAgent = Literal["research_agent", "analyst_agent", "FINISH"]
StopReason = Literal["suficiencia", "limite_de_pasos"]


class ToolCallRecord(TypedDict):
    name: str
    args: dict[str, Any]


class Contribution(TypedDict):
    """Aporte de un especialista: quién, para qué, con qué herramientas y qué devolvió."""

    turn: int
    step: int
    agent: AgentName
    task: str
    output: str
    tool_calls: list[ToolCallRecord]


class Decision(TypedDict):
    """Decisión del supervisor: evaluación del último aporte y a quién le toca."""

    turn: int
    step: int
    evaluation: str
    next_agent: NextAgent
    instruction: str


class OrchestratorState(MessagesState):
    turn: int
    # Los especialistas y el supervisor solo agregan: operator.add evita que un nodo pise
    # lo que aportó otro.
    contributions: Annotated[list[Contribution], operator.add]
    decisions: Annotated[list[Decision], operator.add]
    # Ruteo: lo escribe el supervisor y lo leen la arista condicional y el especialista.
    next_agent: NextAgent
    instruction: str
    # Condición de parada (se reinicia en cada turno).
    step_count: int
    task_completed: bool
    stop_reason: StopReason | None
    # HITL: dicts validados con los modelos de app.schemas.
    proposed_action: dict[str, Any] | None  # SolicitudCredito
    approval: dict[str, Any] | None  # ApprovalRecord
    action_result: dict[str, Any] | None  # ActionOutcome


def current_contributions(state: OrchestratorState) -> list[Contribution]:
    turn = state.get("turn", 0)
    return [c for c in state.get("contributions", []) if c["turn"] == turn]
