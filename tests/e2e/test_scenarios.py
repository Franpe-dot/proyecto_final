"""Pruebas del sistema completo. Cada escenario deja su traza en Phoenix.

    docker compose up -d         # o ./run.sh
    pytest tests/e2e -m e2e -v -s

Usan la API real (cola + worker + Claude + Chroma + Redis). Se saltean si la API no
responde en API_URL (default http://localhost:8000). El escenario de reinicio corre
`docker compose restart api worker` y solo se ejecuta con E2E_RESTART=1.
"""

import asyncio
import os
import subprocess
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

API_URL = os.environ.get("API_URL", "http://localhost:8000")
TIMEOUT_S = float(os.environ.get("E2E_TIMEOUT", "300"))

pytestmark = pytest.mark.e2e


def _api_up() -> bool:
    try:
        return httpx.get(f"{API_URL}/health", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


if not _api_up():
    pytest.skip(f"API no disponible en {API_URL}", allow_module_level=True)


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(base_url=API_URL, timeout=30) as c:
        yield c


def _thread(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


async def submit(client: httpx.AsyncClient, query: str, thread_id: str | None = None) -> str:
    payload: dict[str, Any] = {"query": query}
    if thread_id:
        payload["thread_id"] = thread_id
    r = await client.post("/process", json=payload)
    assert r.status_code == 202, r.text
    return str(r.json()["job_id"])


async def wait(client: httpx.AsyncClient, job_id: str, until: set[str]) -> dict[str, Any]:
    """Polling a /status hasta que el job llegue a uno de los estados `until`."""
    deadline = time.monotonic() + TIMEOUT_S
    while time.monotonic() < deadline:
        job = (await client.get(f"/status/{job_id}")).json()
        if job["status"] in until:
            return job  # type: ignore[no-any-return]
        assert job["status"] != "failed", job["error"]
        await asyncio.sleep(2)
    raise TimeoutError(f"El job {job_id} no llegó a {until} en {TIMEOUT_S}s")


def show(title: str, job: dict[str, Any]) -> None:
    result = job.get("result") or {}
    print(f"\n=== {title} [{job['status']}] thread={job['thread_id']}")
    print("recorrido:", " → ".join(result.get("recorrido", [])))
    if result.get("respuesta"):
        print(result["respuesta"])


# 1 ---------------------------------------------------------------------------------------
async def test_01_rag_sin_calculos(client: httpx.AsyncClient) -> None:
    """Solo conocimiento: el supervisor no necesita al analista."""
    job_id = await submit(
        client, "¿Desde qué día de mora se puede pedir una refinanciación y a qué tasa?"
    )
    job = await wait(client, job_id, {"completed"})
    show("RAG sin cálculos", job)
    assert "research_agent" in job["result"]["recorrido"]
    assert "analyst_agent" not in job["result"]["recorrido"]
    assert "31" in job["result"]["respuesta"]


# 2 ---------------------------------------------------------------------------------------
async def test_02_multipaso_rag_y_calculo(client: httpx.AsyncClient) -> None:
    """Razonamiento multi-paso: datos de política → cálculos → síntesis."""
    job_id = await submit(
        client,
        "Un cliente con calificación B pide un crédito de $2.000.000 a 24 meses y tiene "
        "ingresos netos de $450.000 por mes. ¿Cuánto pagaría de cuota? ¿Cumple con el ratio "
        "cuota/ingreso? Y si se atrasa 20 días en pagar una cuota, ¿cuánto paga por esa cuota?",
    )
    job = await wait(client, job_id, {"completed"})
    show("Multi-paso", job)
    path = job["result"]["recorrido"]
    assert path.index("research_agent") < path.index("analyst_agent")
    tools = {t for a in job["result"]["aportes"] for t in a["tools"]}
    assert {"buscar_politicas", "calcular_cuota_frances", "evaluar_ratio_cuota_ingreso"} <= tools
    assert job["result"]["accion"] is None


# 3 ---------------------------------------------------------------------------------------
async def test_03_memoria_conversacional(client: httpx.AsyncClient) -> None:
    """Un seguimiento en el mismo thread usa los datos del turno anterior."""
    thread_id = _thread("memoria")
    first = await submit(
        client, "¿Cuál es la cuota de $1.500.000 a 12 meses para un cliente AA?", thread_id
    )
    await wait(client, first, {"completed"})
    second = await submit(client, "¿Y si lo pidiera a 36 meses?", thread_id)
    job = await wait(client, second, {"completed"})
    show("Memoria", job)
    thread = (await client.get(f"/threads/{thread_id}")).json()
    assert thread["turnos"] == 2
    cuota_tools = [a for a in job["result"]["aportes"] if "calcular_cuota_frances" in a["tools"]]
    assert cuota_tools, "el seguimiento debería recalcular la cuota"
    assert "1.500.000" in cuota_tools[-1]["task"] or "1500000" in cuota_tools[-1]["task"]


# 4 ---------------------------------------------------------------------------------------
SOLICITUD_GRANDE = (
    "{cliente}, calificación A, pide $3.000.000 a 36 meses y tiene ingresos netos de "
    "$900.000 por mes. Calculá la cuota, verificá el ratio cuota/ingreso y, si cumple, "
    "registrá la solicitud."
)


async def test_04_hitl_aprobada(client: httpx.AsyncClient) -> None:
    """Monto sobre el umbral: el grafo se pausa en interrupt() hasta que un humano aprueba."""
    job_id = await submit(client, SOLICITUD_GRANDE.format(cliente="Juan Pérez"))
    job = await wait(client, job_id, {"awaiting_approval", "completed"})
    show("HITL (pausado)", job)
    assert job["status"] == "awaiting_approval"
    assert job["approval_request"]["solicitud"]["monto"] == 3_000_000

    r = await client.post(
        f"/approve/{job_id}", json={"aprobado": True, "revisor": "ana.risk", "comentario": "OK"}
    )
    assert r.status_code == 202
    job = await wait(client, job_id, {"completed"})
    show("HITL (aprobado)", job)
    accion = job["result"]["accion"]
    assert accion["estado"] == "registrada"
    assert accion["aprobacion"]["revisor"] == "ana.risk"
    registradas = (await client.get("/solicitudes")).json()
    assert accion["solicitud_id"] in {s["solicitud_id"] for s in registradas}


# 5 ---------------------------------------------------------------------------------------
async def test_05_hitl_rechazada(client: httpx.AsyncClient) -> None:
    job_id = await submit(client, SOLICITUD_GRANDE.format(cliente="María Gómez"))
    job = await wait(client, job_id, {"awaiting_approval"})
    r = await client.post(
        f"/approve/{job_id}",
        json={"aprobado": False, "revisor": "ana.risk", "comentario": "Falta recibo de sueldo"},
    )
    assert r.status_code == 202
    job = await wait(client, job_id, {"completed"})
    show("HITL (rechazado)", job)
    assert job["result"]["accion"]["estado"] == "rechazada"
    assert job["result"]["accion"]["solicitud_id"] is None


# 6 ---------------------------------------------------------------------------------------
async def test_06_bajo_umbral_sin_humano(client: httpx.AsyncClient) -> None:
    """Acción no crítica: se aprueba sola y queda registrada como automática."""
    job_id = await submit(
        client,
        "Carlos Ruiz, calificación AA, pide $500.000 a 12 meses con ingresos netos de $800.000. "
        "Calculá la cuota, verificá el ratio y registrá la solicitud si cumple.",
    )
    job = await wait(client, job_id, {"completed", "awaiting_approval"})
    show("Bajo umbral", job)
    assert job["status"] == "completed"
    assert job["result"]["accion"]["aprobacion"]["modo"] == "automatica"


# 7 ---------------------------------------------------------------------------------------
async def test_07_validacion_y_errores(client: httpx.AsyncClient) -> None:
    """Pydantic en la entrada de la API y códigos HTTP de la máquina de estados."""
    assert (await client.post("/process", json={"query": ""})).status_code == 422
    assert (await client.post("/process", json={"pregunta": "hola"})).status_code == 422
    assert (await client.get(f"/status/{'0' * 32}")).status_code == 404
    job_id = await submit(client, "¿Qué productos no ofrece FinCrece?")
    r = await client.post(f"/approve/{job_id}", json={"aprobado": True, "revisor": "ana.risk"})
    assert r.status_code == 409
    await wait(client, job_id, {"completed"})


# 8 ---------------------------------------------------------------------------------------
@pytest.mark.skipif(os.environ.get("E2E_RESTART") != "1", reason="requiere E2E_RESTART=1")
async def test_08_persistencia_tras_reinicio(client: httpx.AsyncClient) -> None:
    """Un hilo pausado sobrevive al reinicio de la API y del worker (AsyncRedisSaver)."""
    thread_id = _thread("reinicio")
    job_id = await submit(client, SOLICITUD_GRANDE.format(cliente="Lucía Díaz"), thread_id)
    await wait(client, job_id, {"awaiting_approval"})

    subprocess.run(["docker", "compose", "restart", "api", "worker"], check=True)
    deadline = time.monotonic() + 120
    while not _api_up():
        assert time.monotonic() < deadline, "la API no volvió después del reinicio"
        await asyncio.sleep(2)

    thread = (await client.get(f"/threads/{thread_id}")).json()
    assert thread["pendiente_aprobacion"] is True
    r = await client.post(f"/approve/{job_id}", json={"aprobado": True, "revisor": "ana.risk"})
    assert r.status_code == 202
    job = await wait(client, job_id, {"completed"})
    show("Tras reinicio", job)
    assert job["result"]["accion"]["estado"] == "registrada"
