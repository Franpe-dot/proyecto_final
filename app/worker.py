"""Worker: consume la cola de Redis y ejecuta el grafo.

    python -m app.worker

Proceso independiente de la API (se escala levantando más réplicas). Por cada job:
`pending → processing → completed | awaiting_approval | failed`. Si el grafo queda en un
`interrupt()`, el job pasa a `awaiting_approval` con el payload de la pregunta; cuando llega
la decisión, la API reencola el mismo job como `resume` y el worker reanuda el hilo con
`Command(resume=...)`.
"""

import asyncio
import logging
import signal
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from openinference.instrumentation import using_attributes
from redis.asyncio import Redis

from app.actions import SolicitudStore
from app.config import Settings, build_llm, get_settings
from app.graph.agents import text_of
from app.graph.builder import PROMPT_TEMPLATE_VERSION, SYNTHESIZER, build_graph, recursion_limit
from app.graph.state import current_contributions
from app.jobs import JobStore
from app.rag.ingest import ensure_index
from app.rag.retriever import PolicyRetriever
from app.schemas import (
    ActionOutcome,
    ApprovalRequest,
    ContributionView,
    JobRecord,
    JobResult,
)
from app.tracing import setup_tracing, shutdown_tracing

logger = logging.getLogger("app.worker")

Graph = CompiledStateGraph[Any, Any, Any, Any]


def build_result(values: dict[str, Any], path: list[str]) -> JobResult:
    """Arma la respuesta del job a partir del estado final del turno."""
    answer = next(
        (
            text_of(m.content)
            for m in reversed(values["messages"])
            if m.type == "ai" and m.name == SYNTHESIZER
        ),
        "",
    )
    action = values.get("action_result")
    return JobResult(
        respuesta=answer,
        recorrido=path,
        stop_reason=values.get("stop_reason"),
        aportes=[
            ContributionView(
                step=c["step"],
                agent=c["agent"],
                task=c["task"],
                tools=[call["name"] for call in c["tool_calls"]],
            )
            for c in current_contributions(values)  # type: ignore[arg-type]
        ],
        accion=ActionOutcome.model_validate(action) if action else None,
    )


async def run_job(graph: Graph, jobs: JobStore, job: JobRecord, settings: Settings) -> JobRecord:
    """Ejecuta (o reanuda) el turno de un job y deja el resultado en Redis."""
    job.status = "processing"
    await jobs.save(job)

    tags = [f"kind:{job.kind}", f"prompt:{PROMPT_TEMPLATE_VERSION}"]
    metadata = {"job_id": job.job_id, "prompt_template_version": PROMPT_TEMPLATE_VERSION}
    config: RunnableConfig = {
        "configurable": {"thread_id": job.thread_id},
        "recursion_limit": recursion_limit(settings.max_steps),
        "run_name": f"job:{job.kind}",
        "tags": tags,
        "metadata": metadata,
    }
    graph_input: Any
    if job.kind == "resume":
        assert job.decision is not None
        graph_input = Command(resume=job.decision.model_dump())
    else:
        graph_input = {"messages": [HumanMessage(job.query)]}

    # Recorrido real del grafo, nodo por nodo (incluye la reanudación si la hubo).
    path = list(job.result.recorrido) if job.result else []
    # session_id agrupa en Phoenix todas las trazas de una conversación.
    with using_attributes(session_id=job.thread_id, metadata=metadata, tags=tags):
        async for update in graph.astream(graph_input, config, stream_mode="updates"):
            path.extend(node for node in update if node != "__interrupt__")

    snapshot = await graph.aget_state(config)
    if snapshot.interrupts:
        job.status = "awaiting_approval"
        job.approval_request = ApprovalRequest.model_validate(snapshot.interrupts[0].value)
        # Se guarda el recorrido parcial para continuarlo al reanudar.
        job.result = JobResult(respuesta="", recorrido=path, stop_reason=None, aportes=[])
    else:
        job.status = "completed"
        job.result = build_result(snapshot.values, path)
        await jobs.release_thread(job.thread_id, job.job_id)
    await jobs.save(job)
    return job


async def handle(job_id: str, graph: Graph, jobs: JobStore, settings: Settings) -> None:
    job = await jobs.get(job_id)
    if job is None:
        logger.warning("Job %s no encontrado (¿expiró?)", job_id)
        return
    try:
        job = await run_job(graph, jobs, job, settings)
        logger.info("Job %s [%s] → %s", job_id, job.kind, job.status)
    except Exception as exc:
        # Un job roto no debe matar al worker.
        logger.exception("Job %s falló", job_id)
        job.status = "failed"
        job.error = f"{type(exc).__name__}: {exc}"
        await jobs.save(job)
        await jobs.release_thread(job.thread_id, job.job_id)


async def consume(
    name: str, graph: Graph, jobs: JobStore, settings: Settings, stop: asyncio.Event
) -> None:
    logger.info("Consumidor %s escuchando %s", name, settings.queue_name)
    while not stop.is_set():
        # timeout corto para poder revisar `stop` y cerrar ordenadamente.
        job_id = await jobs.dequeue(timeout=2)
        if job_id is not None:
            await handle(job_id, graph, jobs, settings)


async def _ensure_index_with_retry(retriever: PolicyRetriever, attempts: int = 10) -> int:
    """Al levantar con Docker, Chroma puede tardar unos segundos más que el worker."""
    for attempt in range(1, attempts + 1):
        try:
            return await ensure_index(retriever)
        except Exception as exc:
            if attempt == attempts:
                raise
            logger.warning("Chroma no disponible (%s); reintento %d/%d", exc, attempt, attempts)
            await asyncio.sleep(3)
    raise AssertionError("inalcanzable")


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = get_settings()
    provider = setup_tracing(settings, service="worker")

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # Windows: Ctrl+C llega como KeyboardInterrupt
            pass

    retriever = PolicyRetriever(settings)
    total = await _ensure_index_with_retry(retriever)
    logger.info("Base de políticas lista (%d fragmentos)", total)

    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    try:
        async with AsyncRedisSaver.from_conn_string(settings.redis_url) as checkpointer:
            await checkpointer.asetup()
            graph = build_graph(
                build_llm(settings),
                retriever,
                SolicitudStore(redis),
                max_steps=settings.max_steps,
                hitl_threshold=settings.hitl_monto_umbral,
                checkpointer=checkpointer,
            )
            jobs = JobStore(redis, settings)
            await asyncio.gather(
                *(
                    consume(f"c{i}", graph, jobs, settings, stop)
                    for i in range(settings.worker_concurrency)
                )
            )
    finally:
        await redis.aclose()
        shutdown_tracing(provider)
        logger.info("Worker detenido")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
