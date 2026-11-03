"""Adaptador entre un especialista ReAct y el estado compartido del orquestador."""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph

from app.graph.state import (
    AgentName,
    Contribution,
    OrchestratorState,
    ToolCallRecord,
    current_contributions,
)

# Techo del ciclo interno modelo↔herramientas de cada especialista.
_AGENT_RECURSION_LIMIT = 16

SpecialistNode = Callable[[OrchestratorState], Awaitable[dict[str, Any]]]


def text_of(content: str | list[Any]) -> str:
    """Extrae el texto de un mensaje, ignorando bloques que no sean de texto (thinking, tool_use)."""
    if isinstance(content, str):
        return content.strip()
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts).strip()


def _build_brief(
    state: OrchestratorState, name: AgentName, context_from: tuple[AgentName, ...]
) -> str:
    """Arma lo único que ve el especialista: la instrucción y el contexto que necesita."""
    contributions = current_contributions(state)
    context = [c for c in contributions if c["agent"] in context_from and c["agent"] != name]
    # De sus propios aportes solo importa el último: es el que el supervisor pidió refinar.
    own = [c for c in contributions if c["agent"] == name][-1:]

    parts = [f"TAREA:\n{state['instruction']}"]
    if context:
        parts.append(
            "DATOS DISPONIBLES (aportados por otros especialistas):\n"
            + "\n\n".join(f"[{c['agent']}, paso {c['step']}]\n{c['output']}" for c in context)
        )
    if own:
        parts.append(
            f"TU ENTREGA ANTERIOR (a corregir o completar según la tarea):\n{own[0]['output']}"
        )
    return "\n\n".join(parts)


def make_specialist_node(
    name: AgentName,
    agent: CompiledStateGraph[Any, Any, Any, Any],
    context_from: tuple[AgentName, ...] = (),
) -> SpecialistNode:
    """Envuelve un agente ReAct como nodo del grafo.

    El especialista no recibe el historial ni la metadata del sistema: solo la instrucción
    del supervisor y los aportes de los agentes listados en `context_from`. De vuelta, al
    estado compartido sube únicamente su conclusión y el registro de las herramientas que
    usó; sus mensajes internos (tool calls y resultados) no salen del nodo, aunque sí quedan
    en la traza de Phoenix como spans hijos del nodo.
    """

    async def node(state: OrchestratorState) -> dict[str, Any]:
        brief = _build_brief(state, name, context_from)
        result = await agent.ainvoke(
            {"messages": [HumanMessage(brief)]},
            {"recursion_limit": _AGENT_RECURSION_LIMIT},
        )
        messages = result["messages"]
        tool_calls: list[ToolCallRecord] = [
            {"name": call["name"], "args": call["args"]}
            for message in messages
            if isinstance(message, AIMessage)
            for call in message.tool_calls
        ]
        output = text_of(messages[-1].content)
        contribution: Contribution = {
            "turn": state["turn"],
            "step": state["step_count"],
            "agent": name,
            "task": state["instruction"],
            "output": output,
            "tool_calls": tool_calls,
        }
        return {
            "contributions": [contribution],
            "messages": [AIMessage(content=output, name=name)],
        }

    return node
