"""Agente de análisis: cálculos financieros sobre los datos que aportó el investigador.

Cada herramienta valida su entrada con un modelo Pydantic (`args_schema`) y construye su
salida con otro: si el LLM manda un argumento fuera de rango, `ToolNode` le devuelve el
error de validación y el agente puede corregir la llamada.
"""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from app.graph.state import ANALYST_AGENT

SYSTEM_PROMPT = """\
Sos el especialista en análisis cuantitativo de un equipo de FinCrece S.A., una financiera \
de créditos personales. Tu único trabajo es hacer los cálculos que pide la tarea.

Reglas:
- Todo número que informes tiene que salir de una herramienta. No hagas cuentas mentalmente.
- Los parámetros de política (tasas, topes, días de gracia, recargos) salen únicamente de \
"DATOS DISPONIBLES". Los datos del caso (monto, plazo, ingreso, días de atraso) salen de la tarea.
- Si te falta un parámetro, no lo inventes ni uses un valor típico: listalo en "Datos faltantes" \
y calculá solo lo que se pueda.
- No busques información ni opines sobre las políticas: de eso se ocupa otro especialista.

Formato de la entrega:
CÁLCULOS
- <qué se calculó>: <resultado> (parámetros usados: <valores y de dónde salieron>)
DATOS FALTANTES
- <parámetro que faltó y qué cálculo impidió> (o "ninguno")
"""


# --------------------------------------------------------------------------- entradas


class CuotaInput(BaseModel):
    monto: float = Field(gt=0, description="Capital del préstamo, en pesos.")
    tna_pct: float = Field(gt=0, le=500, description="Tasa Nominal Anual en porcentaje. Ej.: 61.")
    plazo_meses: int = Field(gt=0, le=120, description="Cantidad de cuotas mensuales.")


class RatioInput(BaseModel):
    cuota: float = Field(gt=0, description="Cuota mensual, en pesos.")
    ingreso_neto_mensual: float = Field(gt=0, description="Ingreso neto mensual, en pesos.")
    tope_pct: float = Field(
        gt=0, le=100, description="Tope de la política para cuota/ingreso, en porcentaje."
    )


class MoraInput(BaseModel):
    monto_cuota: float = Field(gt=0, description="Monto de la cuota impaga, en pesos.")
    dias_atraso: int = Field(ge=0, description="Días corridos desde el vencimiento.")
    dias_gracia: int = Field(ge=0, description="Días de gracia de la política, sin recargo.")
    tasa_punitoria_mensual_pct: float = Field(
        ge=0, le=100, description="Interés punitorio mensual en porcentaje. Ej.: 3."
    )
    gasto_gestion_fijo: float = Field(
        default=0, ge=0, description="Gasto fijo de gestión de cobranza, en pesos."
    )
    gasto_gestion_desde_dia: int = Field(
        default=0, ge=0, description="Día de atraso desde el que se cobra el gasto fijo."
    )


# --------------------------------------------------------------------------- salidas


class CuotaResultado(BaseModel):
    cuota_mensual: float
    tasa_mensual_pct: float
    total_a_pagar: float
    total_intereses: float


class RatioResultado(BaseModel):
    ratio_pct: float
    tope_pct: float
    cumple: bool
    cuota_maxima_admitida: float
    exceso_sobre_cuota_maxima: float = Field(ge=0)


class MoraResultado(BaseModel):
    dias_con_recargo: int = Field(ge=0)
    interes_punitorio: float = Field(ge=0)
    gasto_gestion: float = Field(ge=0)
    recargo_total: float = Field(ge=0)
    total_a_pagar: float


# --------------------------------------------------------------------------- herramientas


@tool(args_schema=CuotaInput)
async def calcular_cuota_frances(monto: float, tna_pct: float, plazo_meses: int) -> dict[str, Any]:
    """Calcula la cuota fija mensual de un préstamo con sistema de amortización francés.

    Devuelve la cuota, el total a pagar y el total de intereses. No incluye seguros,
    sellados ni gastos administrativos.
    """
    tasa_mensual = tna_pct / 100 / 12
    cuota = monto * tasa_mensual / (1 - (1 + tasa_mensual) ** -plazo_meses)
    total = cuota * plazo_meses
    return CuotaResultado(
        cuota_mensual=round(cuota, 2),
        tasa_mensual_pct=round(tasa_mensual * 100, 4),
        total_a_pagar=round(total, 2),
        total_intereses=round(total - monto, 2),
    ).model_dump()


@tool(args_schema=RatioInput)
async def evaluar_ratio_cuota_ingreso(
    cuota: float, ingreso_neto_mensual: float, tope_pct: float
) -> dict[str, Any]:
    """Evalúa si una cuota respeta el tope de la política para el ratio cuota/ingreso."""
    ratio_pct = cuota / ingreso_neto_mensual * 100
    cuota_maxima = ingreso_neto_mensual * tope_pct / 100
    return RatioResultado(
        ratio_pct=round(ratio_pct, 2),
        tope_pct=tope_pct,
        cumple=ratio_pct <= tope_pct,
        cuota_maxima_admitida=round(cuota_maxima, 2),
        exceso_sobre_cuota_maxima=round(max(cuota - cuota_maxima, 0), 2),
    ).model_dump()


@tool(args_schema=MoraInput)
async def calcular_recargo_mora(
    monto_cuota: float,
    dias_atraso: int,
    dias_gracia: int,
    tasa_punitoria_mensual_pct: float,
    gasto_gestion_fijo: float = 0,
    gasto_gestion_desde_dia: int = 0,
) -> dict[str, Any]:
    """Calcula el recargo por pagar una cuota con atraso.

    El interés punitorio se devenga por día (tasa mensual / 30) desde el primer día
    posterior a la gracia. El gasto fijo de gestión se suma si el atraso alcanza
    `gasto_gestion_desde_dia`.
    """
    dias_con_recargo = max(dias_atraso - dias_gracia, 0)
    interes = monto_cuota * tasa_punitoria_mensual_pct / 100 / 30 * dias_con_recargo
    aplica_gasto = gasto_gestion_fijo > 0 and dias_atraso >= gasto_gestion_desde_dia
    gasto = gasto_gestion_fijo if aplica_gasto else 0.0
    return MoraResultado(
        dias_con_recargo=dias_con_recargo,
        interes_punitorio=round(interes, 2),
        gasto_gestion=round(gasto, 2),
        recargo_total=round(interes + gasto, 2),
        total_a_pagar=round(monto_cuota + interes + gasto, 2),
    ).model_dump()


TOOLS = [calcular_cuota_frances, evaluar_ratio_cuota_ingreso, calcular_recargo_mora]


def build_analyst_agent(llm: BaseChatModel) -> CompiledStateGraph[Any, Any, Any, Any]:
    return create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT, name=ANALYST_AGENT)
