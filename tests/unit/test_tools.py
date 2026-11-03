"""Herramientas de los agentes: cálculos y validación Pydantic de entradas."""

import pytest
from pydantic import ValidationError

from app.graph.agents.analyst_agent import (
    calcular_cuota_frances,
    calcular_recargo_mora,
    evaluar_ratio_cuota_ingreso,
)
from app.rag.ingest import load_chunks
from app.rag.retriever import SearchQuery


async def test_cuota_frances() -> None:
    out = await calcular_cuota_frances.ainvoke({"monto": 2_000_000, "tna_pct": 61, "plazo_meses": 24})
    assert out["cuota_mensual"] == pytest.approx(146_118.99, abs=0.01)
    assert out["total_intereses"] == pytest.approx(out["total_a_pagar"] - 2_000_000, abs=0.01)


async def test_ratio_no_cumple() -> None:
    out = await evaluar_ratio_cuota_ingreso.ainvoke(
        {"cuota": 150_000, "ingreso_neto_mensual": 450_000, "tope_pct": 30}
    )
    assert out["cumple"] is False
    assert out["cuota_maxima_admitida"] == 135_000
    assert out["exceso_sobre_cuota_maxima"] == 15_000


async def test_recargo_mora_con_gracia_y_gasto() -> None:
    out = await calcular_recargo_mora.ainvoke(
        {
            "monto_cuota": 100_000,
            "dias_atraso": 35,
            "dias_gracia": 5,
            "tasa_punitoria_mensual_pct": 3,
            "gasto_gestion_fijo": 2_500,
            "gasto_gestion_desde_dia": 31,
        }
    )
    assert out["dias_con_recargo"] == 30
    assert out["interes_punitorio"] == 3_000
    assert out["recargo_total"] == 5_500


@pytest.mark.parametrize(
    "args",
    [
        {"monto": -1, "tna_pct": 61, "plazo_meses": 24},
        {"monto": 1_000, "tna_pct": 0, "plazo_meses": 24},
        {"monto": 1_000, "tna_pct": 61, "plazo_meses": 500},
    ],
)
async def test_cuota_rechaza_argumentos_invalidos(args: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        await calcular_cuota_frances.ainvoke(args)


def test_search_query_valida_tema_y_k() -> None:
    with pytest.raises(ValidationError):
        SearchQuery(consulta="tasas", tema="hipotecas")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        SearchQuery(consulta="tasas", k=50)


def test_ingesta_un_fragmento_por_seccion() -> None:
    chunks = load_chunks()
    assert len(chunks) == 16
    assert {c.tema for c in chunks} == {
        "otorgamiento",
        "tasas_y_costos",
        "mora_y_cobranzas",
        "refinanciacion_cancelacion",
    }
    assert len({c.id for c in chunks}) == len(chunks)
