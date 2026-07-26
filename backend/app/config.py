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
    DATAHUB_GMS_URL: str = "http://localhost:8080"
    DATAHUB_TOKEN: str = ""
    DATAHUB_MCP_URL: str = ""  # DataHub MCP Server / Agent Context Kit endpoint

    # --- GitHub integration ------------------------------------------------
    GITHUB_TOKEN: str = ""
    GITHUB_REPO: str = ""  # "owner/repo"

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
