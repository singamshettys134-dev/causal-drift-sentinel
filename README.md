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

Set `USE_MOCK_DATAHUB=false` and configure `DATAHUB_GMS_URL` / `DATAHUB_GMS_TOKEN`
in `.env`. `backend/app/lineage/datahub_client.py::DataHubMCPClient` spawns the
official `mcp-server-datahub` as a local subprocess (the same way Claude
Desktop/Cursor connect to it) and talks to it over stdio using the real tool
set: `search`, `get_entities`, `get_lineage`, `list_schema_fields`,
`get_lineage_paths_between` for reads. Write-back uses `add_tags` +
`update_description` to annotate the model entity (DataHub's MCP server has
no native "incident" tool) — this requires `DATAHUB_MUTATION_ENABLED=true`
here *and* `TOOLS_IS_MUTATION_ENABLED=true` on the server itself.

Requires `uv`/`uvx` installed locally (`pip install uv` or see
[astral.sh/uv](https://astral.sh/uv)), which is what launches the
`mcp-server-datahub` subprocess. If you installed the server differently,
adjust `DATAHUB_MCP_COMMAND`/`DATAHUB_MCP_ARGS` accordingly.

If instead you're using DataHub Cloud's managed MCP server, set
`DATAHUB_MCP_URL` to its SSE endpoint — this takes priority over the
subprocess path.

No other code needs to change either way, since the mock and real clients
share the same `LineageGraph` contract.

**Before your demo:** tool parameter names have shifted across
`mcp-server-datahub` releases. Run a quick smoke test against your installed
version — call `session.list_tools()` and one real `get_lineage` call — to
confirm the argument names in `datahub_client.py` still match, and adjust if
not.

## Swapping the LLM model

One config value: `LLM_MODEL` in `backend/app/config.py` / `.env`. Nothing
else in the pipeline needs to change to upgrade models (spec Section 4 & 7).

## Running tests

```bash
cd backend
pytest tests/ -v
```

Covers the drift engine's statistical correctness; the causal isolator
correctly pinpointing the injected root cause and not misattributing the
downstream symptom as an independent cause; the bootstrap confidence
interval behind that decision; the LLM reasoning layer's grounding contract
(it cannot report a root cause the algorithm didn't isolate, even if the
model hallucinates one, and falls back deterministically if the model
returns unparseable output); write-back graceful degradation on failure;
and DataHub lineage-response parsing across the response shapes different
`mcp-server-datahub` versions have used.

```bash
cd frontend
npm test
```

Covers the lineage-graph node-coloring logic, the root-cause report
rendering (empty and populated states), and the API client's request
construction/error handling.

Both suites run in CI on every push/PR — see `.github/workflows/ci.yml`.

## Deploying to separate hosts (frontend + backend)

Locally, Vite's dev-server proxy forwards `/api` requests to `localhost:8000`
automatically — no extra config needed for `npm run dev`.

In production, the frontend and backend are typically deployed on separate
hosts (e.g. frontend on Vercel/Netlify, backend on Render/Railway/Fly.io).
In that case, set two things:

**Backend** (set as environment variables on your hosting platform, not in a
committed file):
```
GROQ_API_KEY=<your key>
USE_MOCK_DATAHUB=true
WRITEBACK_ENABLED=false      # keep off for a public judge-facing demo
ENV=production
ALLOWED_ORIGINS=https://your-frontend-domain.vercel.app
```
Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT` (no `--reload`).

**Frontend** — set at build time (see `frontend/.env.example`):
```
VITE_API_BASE_URL=https://your-backend-domain.onrender.com
```
Build command: `npm install && npm run build`, output directory `dist`.

## License

Apache 2.0 — see [LICENSE](./LICENSE).
