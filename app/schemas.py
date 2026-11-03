"""Modelos Pydantic de la API y del dominio que cruza la frontera grafo ↔ API.

En el estado del grafo se guardan dicts (`model_dump()`) para que el checkpointer los
serialice sin registrar tipos propios; cada vez que salen o entran del grafo se validan
contra estos modelos.
"""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

THREAD_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
JOB_ID_PATTERN = r"^[0-9a-f]{32}$"

Calificacion = Literal["AA", "A", "B", "C"]
JobStatus = Literal["pending", "processing", "awaiting_approval", "completed", "failed"]
JobKind = Literal["run", "resume"]


def utcnow() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------------- dominio


class SolicitudCredito(BaseModel):
    """Solicitud de crédito que el sistema propone registrar (acción con efecto real)."""

    cliente: str = Field(min_length=1, max_length=120)
    calificacion: Calificacion
    monto: float = Field(ge=300_000, le=8_000_000, description="Límites de la política.")
    plazo_meses: Literal[6, 12, 18, 24, 36, 48]
    tna_pct: float = Field(gt=0, le=500)
    cuota_mensual: float = Field(gt=0)
    ingreso_neto_mensual: float = Field(gt=0)
    justificacion: str = Field(min_length=10, max_length=1500)


class ApprovalDecision(BaseModel):
    """Lo que manda el revisor humano para reanudar un grafo pausado."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    aprobado: bool
    revisor: str = Field(min_length=2, max_length=80, examples=["ana.risk"])
    comentario: str | None = Field(default=None, max_length=500)


class ApprovalRecord(BaseModel):
    """Registro de auditoría de una aprobación: quién, cuándo, cómo y qué decidió."""

    aprobado: bool
    modo: Literal["automatica", "humana"]
    revisor: str | None = None
    comentario: str | None = None
    decidido_en: datetime = Field(default_factory=utcnow)


class ActionOutcome(BaseModel):
    estado: Literal["registrada", "rechazada"]
    solicitud_id: str | None = None
    aprobacion: ApprovalRecord


class SolicitudRegistrada(BaseModel):
    solicitud_id: str
    thread_id: str
    solicitud: SolicitudCredito
    aprobacion: ApprovalRecord
    registrada_en: datetime = Field(default_factory=utcnow)


class ApprovalRequest(BaseModel):
    """Payload del `interrupt()`: lo que ve el revisor antes de decidir."""

    pregunta: str
    solicitud: SolicitudCredito
    umbral: float


# --------------------------------------------------------------------------- API


class ProcessRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(
        min_length=3,
        max_length=2000,
        examples=["¿Cuánto pagaría de cuota un cliente B por $2.000.000 a 24 meses?"],
    )
    thread_id: str | None = Field(
        default=None,
        pattern=THREAD_ID_PATTERN,
        description="Conversación a continuar. Si se omite, se crea una nueva.",
    )


class ContributionView(BaseModel):
    step: int
    agent: str
    task: str
    tools: list[str]


class JobResult(BaseModel):
    respuesta: str
    recorrido: list[str] = Field(description="Nodos del grafo ejecutados, en orden.")
    stop_reason: str | None
    aportes: list[ContributionView]
    accion: ActionOutcome | None = None


class JobRecord(BaseModel):
    """Estado de un job en Redis. Es también la respuesta de `GET /status/{job_id}`."""

    job_id: str
    thread_id: str
    kind: JobKind = "run"
    status: JobStatus = "pending"
    query: str
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    approval_request: ApprovalRequest | None = None
    decision: ApprovalDecision | None = None
    result: JobResult | None = None
    error: str | None = None


class JobAccepted(BaseModel):
    job_id: str
    thread_id: str
    status: JobStatus
    status_url: str


class MessageView(BaseModel):
    role: Literal["user", "assistant"]
    name: str | None
    content: str


class ThreadView(BaseModel):
    thread_id: str
    turnos: int
    pendiente_aprobacion: bool
    messages: list[MessageView]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    redis: bool
