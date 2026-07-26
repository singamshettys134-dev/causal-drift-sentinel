from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

app = FastAPI(
    title="Causal Drift Sentinel",
    description="Autonomous agent that detects ML feature/prediction drift, "
    "traces it to a causal upstream root cause via DataHub's ML lineage graph, "
    "explains the finding with an LLM, and writes the diagnosis back.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo/dev only; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {"name": "Causal Drift Sentinel", "status": "running", "docs": "/docs"}
