"""
Centralized configuration.

Per the spec (Section 4 & 7): "Keep this swappable via a single config
value so the model can be upgraded without touching the pipeline."
That single value lives here: LLM_MODEL.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM reasoning layer -------------------------------------------------
    # The ONE place to change which model powers the reasoning/explanation
    # layer described in spec Section 4. Swap this string to upgrade models
    # without touching any pipeline code. Backed by Groq's OpenAI-compatible
    # chat completions API.
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_API_KEY: str = ""
    LLM_MAX_TOKENS: int = 2000

    # --- DataHub integration ---------------------------------------------------
    # Self-hosted mcp-server-datahub is normally launched as a stdio subprocess
    # (the same way Claude Desktop / Cursor connect to it), configured via these
    # two env vars passed straight through to that subprocess.
    DATAHUB_GMS_URL: str = "http://localhost:8080"
    DATAHUB_GMS_TOKEN: str = ""
    # Command used to launch the MCP server subprocess. Default assumes `uv`
    # is installed (matches the official quickstart); override if you installed
    # it differently (e.g. "mcp-server-datahub" if installed via pip).
    DATAHUB_MCP_COMMAND: str = "uvx"
    DATAHUB_MCP_ARGS: str = "mcp-server-datahub"
    # DataHub Cloud's managed MCP server is reached over URL/SSE instead of a
    # local subprocess. If set, this takes priority over the stdio subprocess mode.
    DATAHUB_MCP_URL: str = ""
    # Mutation tools (add_tags, update_description, etc.) are opt-in on the
    # DataHub side via TOOLS_IS_MUTATION_ENABLED — mirror that here so our
    # write-back path fails clearly instead of silently no-op-ing.
    DATAHUB_MUTATION_ENABLED: bool = False

    # --- GitHub integration ------------------------------------------------
    GITHUB_TOKEN: str = ""
    GITHUB_REPO: str = ""  # "owner/repo"
    # Explicit safety gate: even if GITHUB_TOKEN/GITHUB_REPO are set, a public
    # demo deployment must opt in to live write-backs, so a stranger hitting
    # /api/investigate can't spam real GitHub issues on your repo.
    WRITEBACK_ENABLED: bool = False

    # --- CORS ----------------------------------------------------------------
    # Comma-separated list of allowed origins for the deployed frontend.
    # Defaults to "*" for local dev; set this explicitly before deploying.
    ALLOWED_ORIGINS: str = "*"

    # --- Drift detection thresholds -----------------------------------------
    KS_PVALUE_ALERT_THRESHOLD: float = 0.05
    PSI_LOW_THRESHOLD: float = 0.1
    PSI_MODERATE_THRESHOLD: float = 0.2
    PSI_HIGH_THRESHOLD: float = 0.3
    EMBEDDING_COSINE_DRIFT_THRESHOLD: float = 0.15

    # --- Causal isolation ------------------------------------------------
    INTERVENTION_DELTA_MIN: float = 0.05  # min effect size to call something a genuine cause

    # --- App -----------------------------------------------------------------
    USE_MOCK_DATAHUB: bool = True  # demo mode: synthesize lineage instead of hitting a real DataHub instance
    ENV: str = "development"


settings = Settings()
