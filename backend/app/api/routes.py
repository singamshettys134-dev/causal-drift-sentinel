"""
API routes. Thin layer over app.pipeline — frontend consumes these.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.lineage.dag import build_dag
from app.lineage.datahub_client import get_lineage_client
from app.models.schemas import LineageGraph
from app.pipeline import run_full_pipeline

router = APIRouter()


@router.get("/lineage/{model_urn:path}")
async def get_lineage(model_urn: str) -> LineageGraph:
    """Return the raw lineage DAG for the live lineage graph view."""
    client = get_lineage_client()
    try:
        graph = await client.get_ml_lineage(model_urn)
        build_dag(graph)  # validates it's a real DAG (no cycles) before returning
        return graph
    except Exception as exc:  # noqa: BLE001 - never leak internals to the client
        detail = str(exc) if settings.ENV == "development" else "Lineage fetch failed. Check server logs."
        raise HTTPException(status_code=500, detail=detail) from exc
    finally:
        await client.aclose()


@router.post("/investigate")
async def investigate(model_urn: str = "urn:li:mlModel:(demo,fraud_model_v3,PROD)", inject_drift: bool = True):
    """
    Runs the full pipeline: ingest lineage -> detect drift -> isolate root
    cause -> generate LLM report -> write back to DataHub/GitHub.

    This is what the "Replay a failure" demo button calls (spec Section 4).
    `inject_drift=False` runs the healthy-pipeline control scenario.
    """
    try:
        result = await run_full_pipeline(model_urn=model_urn, inject_drift=inject_drift)
    except Exception as exc:  # noqa: BLE001 - never leak internals to the client
        detail = str(exc) if settings.ENV == "development" else "Pipeline execution failed. Check server logs."
        raise HTTPException(status_code=500, detail=detail) from exc

    return {
        "graph": result["graph"],
        "trace": result["trace"],
        "report": result["report"],
        "writeback": result["writeback"],
    }


@router.get("/health")
async def health():
    return {"status": "ok"}
