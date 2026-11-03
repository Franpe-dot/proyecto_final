"""API + worker con Redis en memoria: validación, códigos HTTP y el ciclo completo del HITL."""

from collections.abc import AsyncIterator
from typing import Any

import fakeredis
import httpx
import pytest

from app.actions import SolicitudStore
from app.config import get_settings
from app.jobs import JobStore
from app.main import app
from app.worker import handle

from .fakes import SOLICITUD_OK, finishing_script, make_graph


@pytest.fixture
async def env() -> AsyncIterator[dict[str, Any]]:
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    store = SolicitudStore(redis)
    # Sin lifespan: se inyecta el estado con dobles en memoria.
    app.state.jobs = JobStore(redis, get_settings())
    app.state.solicitudes = store
    app.state.graph = make_graph(store, finishing_script(SOLICITUD_OK))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield {"client": client, "jobs": app.state.jobs, "graph": app.state.graph}


async def _drain(env: dict[str, Any]) -> None:
    """Hace de worker: procesa lo que haya en la cola."""
    jobs: JobStore = env["jobs"]
    while (job_id := await jobs.dequeue(timeout=1)) is not None:
        await handle(job_id, env["graph"], jobs, get_settings())


async def test_process_guarda_pending_y_encola(env: dict[str, Any]) -> None:
    r = await env["client"].post("/process", json={"query": "¿Qué tasa tiene la B?"})
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "pending"
    status = await env["client"].get(body["status_url"])
    assert status.status_code == 200
    assert status.json()["status"] == "pending"


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "  "},
        {"query": "x" * 2001},
        {"query": "hola que tal", "extra": 1},
        {"query": "hola que tal", "thread_id": "con espacios"},
    ],
)
async def test_process_valida_entrada(env: dict[str, Any], payload: dict[str, Any]) -> None:
    r = await env["client"].post("/process", json=payload)
    assert r.status_code == 422


async def test_status_404_y_422(env: dict[str, Any]) -> None:
    assert (await env["client"].get(f"/status/{'0' * 32}")).status_code == 404
    assert (await env["client"].get("/status/no-es-un-id")).status_code == 422


async def test_thread_ocupado_devuelve_409(env: dict[str, Any]) -> None:
    payload = {"query": "Primera consulta", "thread_id": "cliente-42"}
    assert (await env["client"].post("/process", json=payload)).status_code == 202
    assert (await env["client"].post("/process", json=payload)).status_code == 409


async def test_ciclo_completo_con_aprobacion(env: dict[str, Any]) -> None:
    client: httpx.AsyncClient = env["client"]
    r = await client.post("/process", json={"query": "Registrá la solicitud", "thread_id": "c1"})
    job_id = r.json()["job_id"]

    # Aprobar antes de que el worker lo procese: todavía no hay nada que aprobar.
    early = await client.post(f"/approve/{job_id}", json={"aprobado": True, "revisor": "ana"})
    assert early.status_code == 409

    await _drain(env)
    job = (await client.get(f"/status/{job_id}")).json()
    assert job["status"] == "awaiting_approval"
    assert job["approval_request"]["solicitud"]["monto"] == 3_000_000
    # Mientras espera aprobación, la conversación sigue tomada.
    busy = await client.post("/process", json={"query": "Otra consulta", "thread_id": "c1"})
    assert busy.status_code == 409

    assert (await client.post(f"/approve/{job_id}", json={"aprobado": True})).status_code == 422
    ok = await client.post(f"/approve/{job_id}", json={"aprobado": True, "revisor": "ana.risk"})
    assert ok.status_code == 202
    again = await client.post(f"/approve/{job_id}", json={"aprobado": True, "revisor": "ana.risk"})
    assert again.status_code == 409

    await _drain(env)
    job = (await client.get(f"/status/{job_id}")).json()
    assert job["status"] == "completed"
    assert job["decision"]["revisor"] == "ana.risk"
    assert job["result"]["accion"]["estado"] == "registrada"
    assert job["result"]["recorrido"][-2:] == ["human_approval", "execute_action"]

    solicitudes = (await client.get("/solicitudes")).json()
    assert [s["solicitud_id"] for s in solicitudes] == ["SOL-c1-1"]

    thread = (await client.get("/threads/c1")).json()
    assert thread["turnos"] == 1
    assert thread["pendiente_aprobacion"] is False
    assert [m["role"] for m in thread["messages"]] == ["user", "assistant", "assistant"]

    # Terminado el turno, la conversación acepta otra consulta.
    follow = await client.post("/process", json={"query": "Otra consulta", "thread_id": "c1"})
    assert follow.status_code == 202


async def test_thread_inexistente_404(env: dict[str, Any]) -> None:
    assert (await env["client"].get("/threads/no-existe")).status_code == 404
