"""Grafo: condición de parada del supervisor, HITL y persistencia por thread."""

from typing import Any

import fakeredis
import pytest
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app.actions import SolicitudStore
from app.graph.supervisor import make_supervisor_node

from .fakes import SOLICITUD_OK, ScriptedLLM, finishing_script, make_graph


@pytest.fixture
def store() -> SolicitudStore:
    return SolicitudStore(fakeredis.FakeAsyncRedis(decode_responses=True))


def _config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


async def test_supervisor_corta_al_limite_sin_llamar_al_llm() -> None:
    def script(schema: type, messages: Any) -> Any:
        raise AssertionError("no debería consultar al LLM")

    node = make_supervisor_node(ScriptedLLM(script=script), max_steps=2)
    update = await node({"messages": [HumanMessage("hola")], "turn": 1, "step_count": 2})  # type: ignore[typeddict-item]
    assert update["next_agent"] == "FINISH"
    assert update["stop_reason"] == "limite_de_pasos"


async def test_sin_solicitud_termina_sin_aprobacion(store: SolicitudStore) -> None:
    graph = make_graph(store, finishing_script(None))
    await graph.ainvoke({"messages": [HumanMessage("¿Qué tasa tiene la calificación B?")]}, _config("t1"))
    snapshot = await graph.aget_state(_config("t1"))
    assert not snapshot.interrupts
    assert snapshot.values["action_result"] is None
    assert await store.list() == []


async def test_solicitud_sobre_umbral_pausa_y_se_aprueba(store: SolicitudStore) -> None:
    graph = make_graph(store, finishing_script(SOLICITUD_OK))
    await graph.ainvoke({"messages": [HumanMessage("Registrá la solicitud")]}, _config("t2"))

    snapshot = await graph.aget_state(_config("t2"))
    assert snapshot.next == ("human_approval",)
    assert snapshot.interrupts[0].value["solicitud"]["cliente"] == "Juan Pérez"
    # Pausado: todavía no se registró nada.
    assert await store.list() == []

    await graph.ainvoke(Command(resume={"aprobado": True, "revisor": "ana.risk"}), _config("t2"))
    snapshot = await graph.aget_state(_config("t2"))
    result = snapshot.values["action_result"]
    assert result["estado"] == "registrada"
    assert result["aprobacion"]["modo"] == "humana"
    [registrada] = await store.list()
    assert registrada.solicitud_id == "SOL-t2-1"
    assert registrada.aprobacion.revisor == "ana.risk"


async def test_solicitud_rechazada_no_registra(store: SolicitudStore) -> None:
    graph = make_graph(store, finishing_script(SOLICITUD_OK))
    await graph.ainvoke({"messages": [HumanMessage("Registrá la solicitud")]}, _config("t3"))
    await graph.ainvoke(
        Command(resume={"aprobado": False, "revisor": "ana.risk", "comentario": "Falta legajo"}),
        _config("t3"),
    )
    snapshot = await graph.aget_state(_config("t3"))
    assert snapshot.values["action_result"]["estado"] == "rechazada"
    assert "Falta legajo" in snapshot.values["messages"][-1].content
    assert await store.list() == []


async def test_solicitud_bajo_umbral_se_aprueba_sola(store: SolicitudStore) -> None:
    graph = make_graph(store, finishing_script(SOLICITUD_OK), threshold=5_000_000)
    await graph.ainvoke({"messages": [HumanMessage("Registrá la solicitud")]}, _config("t4"))
    snapshot = await graph.aget_state(_config("t4"))
    assert not snapshot.interrupts
    assert snapshot.values["action_result"]["aprobacion"]["modo"] == "automatica"
    assert len(await store.list()) == 1


async def test_solicitud_fuera_de_politica_se_descarta(store: SolicitudStore) -> None:
    graph = make_graph(store, finishing_script({**SOLICITUD_OK, "monto": 10_000_000}))
    await graph.ainvoke({"messages": [HumanMessage("Registrá la solicitud")]}, _config("t5"))
    snapshot = await graph.aget_state(_config("t5"))
    assert not snapshot.interrupts
    assert snapshot.values["proposed_action"] is None
    assert "fuera de política (monto)" in snapshot.values["messages"][-1].content


async def test_turnos_acumulan_memoria_en_el_thread(store: SolicitudStore) -> None:
    graph = make_graph(store, finishing_script(None))
    await graph.ainvoke({"messages": [HumanMessage("Primera consulta")]}, _config("t6"))
    await graph.ainvoke({"messages": [HumanMessage("Segunda consulta")]}, _config("t6"))
    values = (await graph.aget_state(_config("t6"))).values
    assert values["turn"] == 2
    humans = [m.content for m in values["messages"] if m.type == "human"]
    assert humans == ["Primera consulta", "Segunda consulta"]
