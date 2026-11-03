"""API de The Intelligence System (FastAPI). Documentación interactiva en `/docs`.

La API no ejecuta el grafo: valida, encola y responde en milisegundos. El trabajo lo hace
`app.worker`, y el cliente consulta el avance por polling a `/status/{job_id}`.
"""

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request, status
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.graph.state import CompiledStateGraph
from redis.asyncio import Redis

from app.actions import SolicitudStore
from app.config import build_llm, get_settings
from app.graph.agents import text_of
from app.graph.builder import OPERATIONS, SYNTHESIZER, build_graph
from app.jobs import JobStore, ThreadBusyError
from app.rag.retriever import PolicyRetriever
from app.schemas import (
    JOB_ID_PATTERN,
    THREAD_ID_PATTERN,
    ApprovalDecision,
    HealthResponse,
    JobAccepted,
    JobRecord,
    MessageView,
    ProcessRequest,
    SolicitudRegistrada,
    ThreadView,
)
from app.tracing import setup_tracing, shutdown_tracing

logger = logging.getLogger("app.api")

JobId = Annotated[str, Path(pattern=JOB_ID_PATTERN, description="Id devuelto por /process.")]
ThreadId = Annotated[str, Path(pattern=THREAD_ID_PATTERN)]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = get_settings()
    provider = setup_tracing(settings, service="api")
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    async with AsyncRedisSaver.from_conn_string(settings.redis_url) as checkpointer:
        await checkpointer.asetup()
        solicitudes = SolicitudStore(redis)
        # La API solo lee el estado de las conversaciones: arma el mismo grafo que el worker
        # (sin conectarse a Chroma ni cargar embeddings, que son perezosos) para usar aget_state.
        app.state.graph = build_graph(
            build_llm(settings),
            PolicyRetriever(settings),
            solicitudes,
            max_steps=settings.max_steps,
            hitl_threshold=settings.hitl_monto_umbral,
            checkpointer=checkpointer,
        )
        app.state.jobs = JobStore(redis, settings)
        app.state.solicitudes = solicitudes
        yield
    await redis.aclose()
    shutdown_tracing(provider)


app = FastAPI(
    title="The Intelligence System — FinCrece S.A.",
    version="1.0.0",
    description=(
        "Sistema multi-agente (Supervisor + Investigador RAG + Analista) sobre las políticas de "
        "crédito de FinCrece S.A. Procesamiento asíncrono con cola Redis, memoria persistente "
        "por conversación (checkpointer de LangGraph en Redis), aprobación humana para "
        "registrar solicitudes de crédito y trazas en Arize Phoenix."
    ),
    lifespan=lifespan,
    # Sin spans HTTP nativos: en Phoenix interesan las trazas del grafo (en el worker), y
    # el polling a /status llenaría el proyecto de spans de 1 ms.
    telemetry={"tracing": False, "metrics": False, "logs": False},
)


def get_jobs(request: Request) -> JobStore:
    return request.app.state.jobs  # type: ignore[no-any-return]


def get_graph(request: Request) -> CompiledStateGraph[Any, Any, Any, Any]:
    return request.app.state.graph  # type: ignore[no-any-return]


def get_solicitudes(request: Request) -> SolicitudStore:
    return request.app.state.solicitudes  # type: ignore[no-any-return]


Jobs = Annotated[JobStore, Depends(get_jobs)]
Graph = Annotated[CompiledStateGraph[Any, Any, Any, Any], Depends(get_graph)]
Solicitudes = Annotated[SolicitudStore, Depends(get_solicitudes)]


def _accepted(job: JobRecord) -> JobAccepted:
    return JobAccepted(
        job_id=job.job_id,
        thread_id=job.thread_id,
        status=job.status,
        status_url=f"/status/{job.job_id}",
    )


@app.post(
    "/process",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobAccepted,
    tags=["jobs"],
    summary="Encola una consulta",
    responses={409: {"description": "La conversación ya tiene un job en curso"}},
)
async def process(body: ProcessRequest, jobs: Jobs) -> JobAccepted:
    """Crea un job para la consulta y lo encola. Con `thread_id` continúa una conversación."""
    job_id = uuid.uuid4().hex
    thread_id = body.thread_id or job_id
    try:
        await jobs.acquire_thread(thread_id, job_id)
    except ThreadBusyError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    job = JobRecord(job_id=job_id, thread_id=thread_id, query=body.query)
    await jobs.enqueue(job)
    return _accepted(job)


@app.get(
    "/status/{job_id}",
    response_model=JobRecord,
    tags=["jobs"],
    summary="Estado y resultado de un job",
    responses={404: {"description": "Job inexistente o expirado"}},
)
async def get_status(job_id: JobId, jobs: Jobs) -> JobRecord:
    job = await jobs.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job no encontrado")
    return job


@app.post(
    "/approve/{job_id}",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobAccepted,
    tags=["hitl"],
    summary="Aprueba o rechaza la acción pendiente de un job",
    responses={
        404: {"description": "Job inexistente"},
        409: {"description": "El job no está esperando aprobación"},
    },
)
async def approve(job_id: JobId, decision: ApprovalDecision, jobs: Jobs) -> JobAccepted:
    """Registra la decisión del revisor y encola la reanudación del mismo `thread_id`."""
    job = await jobs.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job no encontrado")
    if job.status != "awaiting_approval" or not await jobs.claim_resume(job_id):
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"El job no está esperando aprobación (estado: {job.status})"
        )
    job.kind = "resume"
    job.decision = decision
    job.status = "pending"
    await jobs.enqueue(job)
    return _accepted(job)


@app.get(
    "/threads/{thread_id}",
    response_model=ThreadView,
    tags=["conversaciones"],
    summary="Historial de una conversación (leído del checkpointer)",
    responses={404: {"description": "Conversación inexistente"}},
)
async def get_thread(thread_id: ThreadId, graph: Graph) -> ThreadView:
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    snapshot = await graph.aget_state(config)
    if not snapshot.values:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversación no encontrada")
    messages: list[MessageView] = []
    for m in snapshot.values.get("messages", []):
        # Solo la transcripción para el usuario: consultas, respuestas finales y acciones.
        if m.type == "human":
            messages.append(MessageView(role="user", name=None, content=text_of(m.content)))
        elif m.type == "ai" and m.name in (SYNTHESIZER, OPERATIONS):
            messages.append(MessageView(role="assistant", name=m.name, content=text_of(m.content)))
    return ThreadView(
        thread_id=thread_id,
        turnos=snapshot.values.get("turn", 0),
        pendiente_aprobacion=bool(snapshot.interrupts),
        messages=messages,
    )


@app.get(
    "/solicitudes",
    response_model=list[SolicitudRegistrada],
    tags=["hitl"],
    summary="Solicitudes de crédito registradas (con su aprobación)",
)
async def list_solicitudes(
    solicitudes: Solicitudes, limit: Annotated[int, Query(ge=1, le=200)] = 50
) -> list[SolicitudRegistrada]:
    return await solicitudes.list(limit)


@app.get("/health", response_model=HealthResponse, tags=["sistema"])
async def health(jobs: Jobs) -> HealthResponse:
    try:
        redis_ok = bool(await jobs.redis.ping())
    except Exception:
        redis_ok = False
    return HealthResponse(status="ok" if redis_ok else "degraded", redis=redis_ok)
