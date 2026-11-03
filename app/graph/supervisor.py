"""Nodo Supervisor: valida los aportes con una rúbrica y decide quién interviene."""

import json
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.graph.agents import text_of
from app.graph.state import (
    ANALYST_AGENT,
    FINISH,
    RESEARCH_AGENT,
    Contribution,
    Decision,
    NextAgent,
    OrchestratorState,
    current_contributions,
)

SYSTEM_PROMPT = """\
Sos el supervisor de un equipo de análisis de FinCrece S.A., una financiera de créditos \
personales. No respondés la consulta ni hacés cálculos: decidís quién interviene ahora o si \
es momento de finalizar.

Especialistas:
- research_agent: busca datos en la base de políticas (otorgamiento, tasas y costos, mora y \
cobranzas, refinanciación y cancelación). No calcula.
- analyst_agent: hace cálculos (cuota por sistema francés, ratio cuota/ingreso, recargos por \
mora) con los datos que ya consiguió research_agent. No busca información.

Dada la consulta del usuario, el historial de la conversación y el libro de aportes, elegí:
- "research_agent" si falta algún dato de política.
- "analyst_agent" si los datos de política están y falta algún cálculo.
- "FINISH" si los aportes alcanzan para responder toda la consulta, o si la consulta no \
necesita a ningún especialista.

Si el usuario pide registrar o cargar una solicitud de crédito, antes de FINISH tienen que \
estar: la TNA de su calificación, los límites de monto y plazo, el tope del ratio \
cuota/ingreso, la cuota calculada y la evaluación del ratio. El registro en sí lo hace otra \
etapa del sistema, con aprobación humana: vos no lo delegás.

Rúbrica de validación. Antes de avanzar o finalizar, revisá el último aporte:
- Aporte de research_agent:
  R1. Cada dato cita su archivo fuente.
  R2. Cubre todos los datos de política que la consulta necesita, con su valor exacto.
  R3. No hay datos sin respaldo en una búsqueda (`buscar_politicas` figura en sus herramientas).
- Aporte de analyst_agent:
  A1. Cada resultado sale de una herramienta de cálculo registrada en el libro.
  A2. Los argumentos de esas herramientas coinciden con los datos de research_agent y con los \
datos del caso que dio el usuario.
  A3. Cubre todas las partes cuantitativas de la consulta.

Si el último aporte no cumple algún criterio, devolvele la tarea al mismo especialista \
indicando qué criterio falló y qué tiene que corregir. Si analyst_agent informa que le falta \
un dato de política, pedíselo a research_agent y después volvé a analyst_agent.

Conflictos entre especialistas: en datos de política manda research_agent, porque cita la \
fuente; en resultados numéricos manda analyst_agent, porque usa herramientas. Si un cálculo \
usó un valor distinto al que informó research_agent, el cálculo se rehace con el valor de \
research_agent.

Criterio de suficiencia: no pidas refinamientos por estilo, redacción o datos que la consulta \
no necesita. Si research_agent ya buscó un dato y no está en la base, no lo vuelvas a pedir: \
finalizá y la respuesta final lo aclarará.

Seguimientos: si la consulta se apoya en la conversación anterior ("¿y a 36 meses?"), tomá \
del historial los datos del caso que falten.

La instrucción para el especialista tiene que ser autocontenida: él no ve la consulta \
original ni esta conversación, solo tu instrucción y los aportes relevantes. Incluí los datos \
del caso que necesite (montos, plazos, ingresos, días).
"""

# Turnos anteriores que ve el supervisor para resolver seguimientos.
_HISTORY_TURNS = 3


class SupervisorDecision(BaseModel):
    """Decisión de ruteo del supervisor (salida estructurada del LLM)."""

    evaluacion: str = Field(
        description=(
            "Resultado de aplicar la rúbrica al último aporte, criterio por criterio, en una o "
            "dos oraciones. Si todavía no hay aportes, qué necesita la consulta."
        )
    )
    next: NextAgent = Field(description="Quién interviene ahora, o FINISH para finalizar.")
    instruccion: str = Field(
        description="Tarea autocontenida para el especialista elegido. Vacío si next es FINISH."
    )


SupervisorNode = Callable[[OrchestratorState], Awaitable[dict[str, Any]]]


def last_user_query(state: OrchestratorState) -> str:
    for message in reversed(state["messages"]):
        if message.type == "human":
            return text_of(message.content)
    raise ValueError("El estado no tiene ninguna consulta del usuario")


def format_history(state: OrchestratorState, turns: int = _HISTORY_TURNS) -> str:
    """Pares consulta/respuesta final de los turnos anteriores (sin la consulta actual)."""
    messages = state["messages"]
    current = max(i for i, m in enumerate(messages) if m.type == "human")
    lines: list[str] = []
    for message in messages[:current]:
        if message.type == "human":
            lines.append(f"Usuario: {text_of(message.content)}")
        elif message.type == "ai" and message.name in ("synthesizer", "operations"):
            lines.append(f"Sistema: {text_of(message.content)}")
    # Cada turno aporta dos líneas (tres si registró una solicitud).
    recent = lines[-2 * turns :]
    return "\n".join(recent) if recent else "(primera consulta de la conversación)"


def format_contributions(contributions: list[Contribution]) -> str:
    """Libro de aportes tal como lo leen el supervisor y el sintetizador."""
    if not contributions:
        return "(todavía no hay aportes)"
    blocks: list[str] = []
    for c in contributions:
        tools = (
            "\n".join(
                f"  - {call['name']}({json.dumps(call['args'], ensure_ascii=False)})"
                for call in c["tool_calls"]
            )
            or "  (ninguna)"
        )
        blocks.append(
            f"[Paso {c['step']}] {c['agent']}\n"
            f"Tarea asignada: {c['task']}\n"
            f"Herramientas usadas:\n{tools}\n"
            f"Entrega:\n{c['output']}"
        )
    return "\n\n".join(blocks)


def make_supervisor_node(llm: BaseChatModel, max_steps: int) -> SupervisorNode:
    # json_schema y no tool calling forzado: los modelos con thinking no admiten forzar una tool.
    router = llm.with_structured_output(SupervisorDecision, method="json_schema")

    async def supervisor(state: OrchestratorState) -> dict[str, Any]:
        turn = state["turn"]
        step_count = state.get("step_count", 0)

        # Freno del "supervisor infinito": al agotar los pasos se finaliza sin consultar al LLM.
        if step_count >= max_steps:
            forced: Decision = {
                "turn": turn,
                "step": step_count,
                "evaluation": f"Se alcanzó el límite de delegaciones ({max_steps}).",
                "next_agent": FINISH,
                "instruction": "",
            }
            return {
                "decisions": [forced],
                "next_agent": FINISH,
                "instruction": "",
                "task_completed": False,
                "stop_reason": "limite_de_pasos",
            }

        # El supervisor decide sobre el libro de aportes, no sobre el historial de mensajes.
        prompt = (
            f"HISTORIAL DE LA CONVERSACIÓN:\n{format_history(state)}\n\n"
            f"CONSULTA DEL USUARIO:\n{last_user_query(state)}\n\n"
            f"LIBRO DE APORTES:\n{format_contributions(current_contributions(state))}\n\n"
            f"Delegaciones usadas: {step_count} de {max_steps}."
        )
        result = await router.ainvoke([SystemMessage(SYSTEM_PROMPT), HumanMessage(prompt)])
        decision_out = SupervisorDecision.model_validate(result)

        finished = decision_out.next == FINISH
        next_step = step_count if finished else step_count + 1
        decision: Decision = {
            "turn": turn,
            "step": next_step,
            "evaluation": decision_out.evaluacion,
            "next_agent": decision_out.next,
            "instruction": "" if finished else decision_out.instruccion,
        }
        update: dict[str, Any] = {
            "decisions": [decision],
            "next_agent": decision_out.next,
            "instruction": decision["instruction"],
            "step_count": next_step,
            "task_completed": finished,
        }
        if finished:
            update["stop_reason"] = "suficiencia"
        return update

    return supervisor


def route_from_supervisor(
    state: OrchestratorState,
) -> Literal["research_agent", "analyst_agent", "synthesizer"]:
    """Arista condicional: mapea la decisión del supervisor al nombre del nodo siguiente."""
    next_agent = state["next_agent"]
    if next_agent == RESEARCH_AGENT:
        return "research_agent"
    if next_agent == ANALYST_AGENT:
        return "analyst_agent"
    return "synthesizer"
