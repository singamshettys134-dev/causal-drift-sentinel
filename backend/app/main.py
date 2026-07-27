from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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

# --- Minimal rate limiter -------------------------------------------------
# /api/investigate burns a real LLM call (Groq quota) and, if write-back is
# ever enabled, real GitHub/DataHub writes. A public judge-facing demo URL
# with no limiter could be hit repeatedly and exhaust that quota. This is a
# simple in-memory fixed-window limiter per client IP — sufficient for a
# single-instance hackathon deployment; a multi-instance production
# deployment would need a shared store (e.g. Redis) instead.
_RATE_LIMIT_WINDOW_SECONDS = 60
_RATE_LIMIT_MAX_REQUESTS = 5
_request_log: dict[str, deque] = defaultdict(deque)


@app.middleware("http")
async def rate_limit_investigate(request: Request, call_next):
    if request.url.path == "/api/investigate":
        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        log = _request_log[client_ip]
        while log and now - log[0] > _RATE_LIMIT_WINDOW_SECONDS:
            log.popleft()
        if len(log) >= _RATE_LIMIT_MAX_REQUESTS:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please wait a minute before trying again."},
            )
        log.append(now)
    return await call_next(request)


app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {"name": "Causal Drift Sentinel", "status": "running", "docs": "/docs"}
