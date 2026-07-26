from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import settings

app = FastAPI(
    title="Causal Drift Sentinel",
    description="Autonomous agent that detects ML feature/prediction drift, "
    "traces it to a causal upstream root cause via DataHub's ML lineage graph, "
    "explains the finding with an LLM, and writes the diagnosis back.",
    version="0.1.0",
)

_origins = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",")] if settings.ALLOWED_ORIGINS != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,  # set ALLOWED_ORIGINS in .env before deploying publicly
    allow_credentials=_origins != ["*"],  # credentials + wildcard origin is invalid per browser spec
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {"name": "Causal Drift Sentinel", "status": "running", "docs": "/docs"}
