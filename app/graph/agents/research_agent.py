"""Agente de investigación: consulta la base de políticas (RAG sobre ChromaDB)."""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from app.graph.state import RESEARCH_AGENT
from app.rag.retriever import PolicyRetriever, SearchQuery, Tema

SYSTEM_PROMPT = """\
Sos el especialista en investigación de un equipo de análisis de FinCrece S.A., una \
financiera de créditos personales. Tu único trabajo es encontrar en la base de políticas \
los datos que pide la tarea.

Reglas:
- Usá `buscar_politicas` para todo dato. No uses conocimiento propio ni supongas valores.
- Hacé una búsqueda por cada tema distinto de la tarea (tasas, requisitos, mora, etc.). Si \
sabés en qué documento está el dato, filtrá con `tema`.
- No hagas cálculos ni saques conclusiones: de eso se ocupa otro especialista.
- Si un dato no aparece después de buscarlo, decilo explícitamente en "No encontrado".

Formato de la entrega:
DATOS ENCONTRADOS
- <dato con su valor exacto> (fuente: <archivo>, sección "<sección>")
NO ENCONTRADO
- <dato pedido que no está en la base> (o "nada")
"""


def make_search_tool(retriever: PolicyRetriever) -> BaseTool:
    async def buscar_politicas(consulta: str, k: int = 3, tema: Tema | None = None) -> dict[str, Any]:
        # La entrada ya llega validada por `args_schema`; la salida se valida al construir
        # el `SearchResult`.
        result = await retriever.search(SearchQuery(consulta=consulta, k=k, tema=tema))
        return result.model_dump()

    return StructuredTool.from_function(
        coroutine=buscar_politicas,
        name="buscar_politicas",
        description=(
            "Busca por similitud semántica en las políticas de crédito de FinCrece S.A. "
            "Temas: otorgamiento (requisitos, montos, plazos, ratio cuota/ingreso), "
            "tasas_y_costos, mora_y_cobranzas, refinanciacion_cancelacion. Devuelve los "
            "fragmentos más relevantes con archivo fuente, sección y relevancia (0 a 1)."
        ),
        args_schema=SearchQuery,
    )


def build_research_agent(
    llm: BaseChatModel, retriever: PolicyRetriever
) -> CompiledStateGraph[Any, Any, Any, Any]:
    return create_react_agent(
        llm, [make_search_tool(retriever)], prompt=SYSTEM_PROMPT, name=RESEARCH_AGENT
    )
