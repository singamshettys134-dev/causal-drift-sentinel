# Causal Drift Sentinel

**Autonomous root-cause diagnosis for silent ML model drift, powered by DataHub's ML lineage graph.**

Built for The DataHub Agent Hackathon — Production ML Agents Track.

---

## The problem

A model's accuracy can degrade for weeks before anyone notices, and even once
someone does, answering *"which upstream data change actually caused this"*
today means manually tracing lineage across wikis, Slack threads, and tribal
knowledge — often taking days. See [`causal-drift-sentinel-spec.md`](./causal-drift-sentinel-spec.md)
for the full problem statement and design rationale this project was built to.

## What it does

1. **Watches** a model's live prediction distribution.
2. **Detects** drift statistically — KS-test, PSI, embedding centroid drift — not "does this look different to an LLM."
3. **Traces backward** through DataHub's ML lineage DAG and runs an intervention-style
   check on every upstream node to isolate the *genuine* cause, distinguishing
   it from nodes that merely drifted around the same time.
4. **Explains** the finding in plain language via a frontier LLM reasoning
   *over* the statistical evidence — never inventing a cause on its own.
5. **Writes back** the diagnosis as a structured DataHub incident and opens a
   GitHub issue with a concrete suggested fix.

```
DataHub lineage  →  Drift Detection  →  Causal Root-Cause Engine  →  LLM Reasoning  →  Write-Back
   (MCP Server)      (scipy stats)      (graph walk + intervention)    (Groq)       (DataHub + GitHub)
```

See [`causal-drift-sentinel-spec.md`](./causal-drift-sentinel-spec.md) for the full architecture diagram and design rationale.

## Repository layout

```
backend/     FastAPI service — lineage ingestion, drift engine, causal isolator, LLM layer, write-back agent
frontend/    React (Vite) console — lineage graph view, drift timeline, root-cause report, "Replay a failure" demo mode
examples/    Real sample outputs (trace, LLM report, GitHub issue, DataHub incident payload) — see examples/README.md
```

## Quickstart (demo mode — no live DataHub instance required)

The backend ships with `USE_MOCK_DATAHUB=true` by default, which synthesizes
a realistic fraud-detection lineage graph and a scripted drift-injection
scenario, so the full pipeline runs end-to-end without a live DataHub instance.

```bash
# 1. Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: set GROQ_API_KEY for the LLM reasoning layer
# (optionally GITHUB_TOKEN / GITHUB_REPO to enable the live GitHub write-back)
uvicorn app.main:app --reload
# → http://localhost:8000/docs

# 2. Frontend (separate terminal)
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

Open the frontend and click **"▶ Replay a failure"** — this is the demo mode
described in spec Section 4, letting you (or a judge) trigger the scripted
drift scenario live and watch detect → trace → explain → write-back run
end-to-end. **"Run control (no drift)"** runs the same pipeline against a
healthy scenario, to show the agent correctly reporting nothing wrong.

## Connecting to a real DataHub instance

Set `USE_MOCK_DATAHUB=false` and configure `DATAHUB_MCP_URL` / `DATAHUB_TOKEN`
in `.env`. `backend/app/lineage/datahub_client.py::DataHubMCPClient` talks to
DataHub's MCP Server for both lineage reads and incident write-backs — no
other code needs to change, since the mock and real clients share the same
`LineageGraph` contract.

## Swapping the LLM model

One config value: `LLM_MODEL` in `backend/app/config.py` / `.env`. Nothing
else in the pipeline needs to change to upgrade models (spec Section 4 & 7).

## Running tests

```bash
cd backend
pytest tests/ -v
```

Covers the drift engine's statistical correctness, and — the most important
test — that the causal isolator correctly pinpoints the injected root cause
in the demo scenario and does **not** misattribute the downstream symptom
as an independent cause.

## License

Apache 2.0 — see [LICENSE](./LICENSE).
