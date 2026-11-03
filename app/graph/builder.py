"""Grafo del orquestador: supervisor → especialistas → síntesis → (HITL) → acción.

    START → intake → supervisor ─┬→ research_agent ─┐
                        ↑        ├→ analyst_agent ──┤
                        └────────┼──────────────────┘
                                 └→ synthesizer ─┬→ END
                                                 └→ human_approval → execute_action → END

`human_approval` llama a `interrupt()` cuando la solicitud supera el umbral: el grafo se
detiene, el checkpointer guarda el estado del `thread_id` y se reanuda con
`Command(resume=...)`. Los efectos (registrar la solicitud) están en el nodo siguiente,
porque al reanudar el nodo interrumpido se re-ejecuta desde el principio.
"""

from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field, ValidationError

from app.actions import SolicitudStore
from app.graph.agents import (
    build_analyst_agent,
    build_research_agent,
    make_specialist_node,
)
from app.graph.state import (
    ANALYST_AGENT,
    RESEARCH_AGENT,
    OrchestratorState,
    current_contributions,
)
from app.graph.supervisor import (
    format_contributions,
    format_history,
    last_user_query,
    make_supervisor_node,
    route_from_supervisor,
)
from app.rag.retriever import PolicyRetriever
from app.schemas import (
    ActionOutcome,
    ApprovalDecision,
    ApprovalRecord,
    ApprovalRequest,
    Calificacion,
    SolicitudCredito,
    SolicitudRegistrada,
)

# Se loguea como metadata en cada traza: si cambia un prompt, se sube la versión.
PROMPT_TEMPLATE_VERSION = "m7-v1"

INTAKE = "intake"
SUPERVISOR = "supervisor"
SYNTHESIZER = "synthesizer"
HUMAN_APPROVAL = "human_approval"
EXECUTE_ACTION = "execute_action"
OPERATIONS = "operations"  # `name` de los mensajes que deja execute_action

SYNTHESIS_PROMPT = """\
Sos quien redacta la respuesta final de un equipo de análisis de FinCrece S.A. Recibís la \
consulta del usuario, el historial de la conversación y el libro de aportes de los \
especialistas.

Reglas para `respuesta`:
- Usá solo lo que está en el libro de aportes. No agregues datos ni recalcules nada.
- Si un dato de política y un cálculo se contradicen, vale el dato de research_agent y \
aclaralo. Si hay varias entregas del mismo especialista, vale la más reciente.
- Respondé primero lo que el usuario preguntó, con los números finales, y después el detalle.
- Citá entre paréntesis el archivo fuente de cada dato de política.
- Si algo de la consulta quedó sin responder, decilo explícitamente al final.
- Respondé en español, de forma breve, con los montos en formato $14.500.

Reglas para `solicitud`:
- Completala solo si el usuario pidió explícitamente registrar, cargar o dar de alta la \
solicitud de crédito Y el libro de aportes muestra que el monto y el plazo están dentro de la \
política y que la cuota cumple el ratio cuota/ingreso.
- Si el usuario lo pidió pero el caso no cumple la política, dejala en null y explicá en \
`respuesta` por qué no se puede registrar.
- Si el usuario no lo pidió, dejala en null.
- `cliente`: el nombre que dio el usuario, o "sin identificar". `justificacion`: una o dos \
oraciones con los números que respaldan la solicitud.
- En `respuesta` no digas que la solicitud ya quedó registrada: la registra otra etapa, \
que puede requerir aprobación humana.
"""

_LIMIT_NOTE = (
    "\n\nAVISO: el análisis se cortó por límite de pasos antes de que el supervisor lo diera "
    "por completo. Indicá qué partes pueden estar incompletas o sin validar, y no propongas "
    "registrar ninguna solicitud."
)


class SolicitudPropuesta(BaseModel):
    """Versión para el LLM de `SolicitudCredito`, sin restricciones numéricas en el schema.

    Las restricciones se aplican después, al validar contra `SolicitudCredito`: así una
    propuesta fuera de política se descarta en código en vez de depender del modelo.
    """

    cliente: str
    calificacion: Calificacion
    monto: float = Field(description="Monto solicitado, en pesos.")
    plazo_meses: int
    tna_pct: float = Field(description="TNA aplicada, en porcentaje.")
    cuota_mensual: float = Field(description="Cuota calculada por analyst_agent, en pesos.")
    ingreso_neto_mensual: float
    justificacion: str


class SynthesisOutput(BaseModel):
    respuesta: str = Field(description="Respuesta final para el usuario.")
    solicitud: SolicitudPropuesta | None = Field(
        description="Solicitud a registrar, o null si no corresponde."
    )


def _pesos(amount: float, decimals: int = 0) -> str:
    """Formato argentino: $1.234.567,89."""
    text = f"{amount:,.{decimals}f}".replace(",", "_").replace(".", ",").replace("_", ".")
    return f"${text}"


def _thread_id(config: RunnableConfig) -> str:
    return str(config.get("configurable", {}).get("thread_id", "local"))


def build_graph(
    llm: BaseChatModel,
    retriever: PolicyRetriever,
    solicitudes: SolicitudStore,
    *,
    max_steps: int,
    hitl_threshold: float,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Arma y compila el orquestador con el checkpointer recibido."""
    synthesis_llm = llm.with_structured_output(SynthesisOutput, method="json_schema")

    async def intake(state: OrchestratorState) -> dict[str, Any]:
        """Abre un turno nuevo: reinicia el ruteo y la condición de parada."""
        return {
            "turn": state.get("turn", 0) + 1,
            "step_count": 0,
            "instruction": "",
            "task_completed": False,
            "stop_reason": None,
            "proposed_action": None,
            "approval": None,
            "action_result": None,
        }

    async def synthesizer(state: OrchestratorState) -> dict[str, Any]:
        prompt = (
            f"HISTORIAL DE LA CONVERSACIÓN:\n{format_history(state)}\n\n"
            f"CONSULTA DEL USUARIO:\n{last_user_query(state)}\n\n"
            f"LIBRO DE APORTES:\n{format_contributions(current_contributions(state))}"
        )
        limit_reached = state.get("stop_reason") == "limite_de_pasos"
        if limit_reached:
            prompt += _LIMIT_NOTE
        raw = await synthesis_llm.ainvoke([SystemMessage(SYNTHESIS_PROMPT), HumanMessage(prompt)])
        output = SynthesisOutput.model_validate(raw)

        answer = output.respuesta.strip()
        proposed: dict[str, Any] | None = None
        if output.solicitud is not None and not limit_reached:
            try:
                proposed = SolicitudCredito.model_validate(output.solicitud.model_dump()).model_dump()
            except ValidationError as exc:
                fields = ", ".join(str(e["loc"][0]) for e in exc.errors())
                answer += (
                    f"\n\nNo se propone registrar la solicitud: hay datos fuera de política ({fields})."
                )
        return {
            "messages": [AIMessage(content=answer, name=SYNTHESIZER)],
            "proposed_action": proposed,
        }

    async def human_approval(state: OrchestratorState) -> dict[str, Any]:
        # Sin efectos antes del interrupt(): al reanudar, este nodo corre de nuevo desde acá.
        solicitud = SolicitudCredito.model_validate(state["proposed_action"])
        if solicitud.monto < hitl_threshold:
            record = ApprovalRecord(
                aprobado=True,
                modo="automatica",
                comentario=f"Monto menor al umbral de aprobación humana ({_pesos(hitl_threshold)}).",
            )
            return {"approval": record.model_dump(mode="json")}

        request = ApprovalRequest(
            pregunta=(
                f"¿Aprobás registrar la solicitud de {solicitud.cliente} por "
                f"{_pesos(solicitud.monto)} a {solicitud.plazo_meses} meses "
                f"(cuota {_pesos(solicitud.cuota_mensual, 2)})?"
            ),
            solicitud=solicitud,
            umbral=hitl_threshold,
        )
        resume_value = interrupt(request.model_dump(mode="json"))
        decision = ApprovalDecision.model_validate(resume_value)
        record = ApprovalRecord(
            aprobado=decision.aprobado,
            modo="humana",
            revisor=decision.revisor,
            comentario=decision.comentario,
        )
        return {"approval": record.model_dump(mode="json")}

    async def execute_action(state: OrchestratorState, config: RunnableConfig) -> dict[str, Any]:
        solicitud = SolicitudCredito.model_validate(state["proposed_action"])
        approval = ApprovalRecord.model_validate(state["approval"])
        thread_id = _thread_id(config)

        if not approval.aprobado:
            outcome = ActionOutcome(estado="rechazada", aprobacion=approval)
            text = f"La solicitud no se registró: la rechazó {approval.revisor}."
            if approval.comentario:
                text += f" Motivo: {approval.comentario}"
        else:
            # Id determinista: si el nodo se re-ejecuta, el registro no se duplica.
            solicitud_id = f"SOL-{thread_id}-{state['turn']}"
            await solicitudes.register(
                SolicitudRegistrada(
                    solicitud_id=solicitud_id,
                    thread_id=thread_id,
                    solicitud=solicitud,
                    aprobacion=approval,
                )
            )
            outcome = ActionOutcome(
                estado="registrada", solicitud_id=solicitud_id, aprobacion=approval
            )
            quien = approval.revisor if approval.modo == "humana" else "aprobación automática"
            text = f"Solicitud {solicitud_id} registrada ({quien})."
        return {
            "action_result": outcome.model_dump(mode="json"),
            "messages": [AIMessage(content=text, name=OPERATIONS)],
        }

    def route_after_synthesis(state: OrchestratorState) -> Literal["human_approval", "__end__"]:
        return HUMAN_APPROVAL if state.get("proposed_action") else END

    workflow = StateGraph(OrchestratorState)
    workflow.add_node(INTAKE, intake)
    workflow.add_node(SUPERVISOR, make_supervisor_node(llm, max_steps))
    workflow.add_node(
        RESEARCH_AGENT,
        make_specialist_node(RESEARCH_AGENT, build_research_agent(llm, retriever)),
    )
    # El analista necesita los datos del investigador; el investigador no necesita nada del analista.
    workflow.add_node(
        ANALYST_AGENT,
        make_specialist_node(ANALYST_AGENT, build_analyst_agent(llm), context_from=(RESEARCH_AGENT,)),
    )
    workflow.add_node(SYNTHESIZER, synthesizer)
    workflow.add_node(HUMAN_APPROVAL, human_approval)
    workflow.add_node(EXECUTE_ACTION, execute_action)

    workflow.add_edge(START, INTAKE)
    workflow.add_edge(INTAKE, SUPERVISOR)
    workflow.add_conditional_edges(SUPERVISOR, route_from_supervisor)
    workflow.add_edge(RESEARCH_AGENT, SUPERVISOR)
    workflow.add_edge(ANALYST_AGENT, SUPERVISOR)
    workflow.add_conditional_edges(SYNTHESIZER, route_after_synthesis)
    workflow.add_edge(HUMAN_APPROVAL, EXECUTE_ACTION)
    workflow.add_edge(EXECUTE_ACTION, END)
    return workflow.compile(checkpointer=checkpointer)


def recursion_limit(max_steps: int) -> int:
    """Red de seguridad del grafo; el corte real lo hace el contador del supervisor.

    Cada delegación son 2 super-pasos (supervisor + especialista); el margen cubre intake,
    síntesis, aprobación y acción.
    """
    return 2 * max_steps + 8
