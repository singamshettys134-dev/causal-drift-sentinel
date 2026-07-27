"""
Core data models shared across the Causal Drift Sentinel backend.

These models are the contract between layers:
  Lineage Ingestion -> Drift Detection -> Causal Isolation -> LLM Reasoning -> Write-Back

Keeping them centralized means every layer speaks the same structured
language, which is what lets the LLM reasoning layer stay grounded in
real evidence instead of free-form guessing.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class _NoProtectedNamespace(BaseModel):
    """Base class silencing pydantic's 'model_' protected-namespace warning,
    since our domain naturally uses `model_urn` (an ML model's URN)."""

    model_config = ConfigDict(protected_namespaces=())


# ---------------------------------------------------------------------------
# Lineage graph
# ---------------------------------------------------------------------------

class NodeType(str, Enum):
    DATASET = "dataset"
    FEATURE = "feature"
    MODEL = "model"
    DEPLOYMENT = "deployment"


class LineageNode(_NoProtectedNamespace):
    urn: str = Field(..., description="DataHub URN, e.g. urn:li:dataset:(...)")
    name: str
    node_type: NodeType
    platform: Optional[str] = None
    description: Optional[str] = None
    schema_fields: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class LineageEdge(_NoProtectedNamespace):
    upstream_urn: str
    downstream_urn: str
    relationship: str = Field(default="derives_from")


class LineageGraph(_NoProtectedNamespace):
    nodes: list[LineageNode]
    edges: list[LineageEdge]
    root_model_urn: str = Field(..., description="The model/deployment under investigation")


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

class DriftMethod(str, Enum):
    KS_TEST = "ks_test"
    PSI = "psi"
    EMBEDDING_COSINE = "embedding_cosine_drift"


class DriftSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class FeatureDriftResult(_NoProtectedNamespace):
    node_urn: str
    feature_name: str
    method: DriftMethod
    statistic: float
    p_value: Optional[float] = None
    psi_score: Optional[float] = None
    severity: DriftSeverity
    baseline_window: str = Field(description="e.g. 'training' or '2026-06-01/2026-06-07'")
    current_window: str = Field(description="e.g. '2026-07-18/2026-07-25'")
    sample_size_baseline: int
    sample_size_current: int


class PredictionDriftResult(_NoProtectedNamespace):
    model_urn: str
    method: DriftMethod
    statistic: float
    p_value: Optional[float] = None
    severity: DriftSeverity
    detected_at: datetime


# ---------------------------------------------------------------------------
# Causal root-cause isolation
# ---------------------------------------------------------------------------

class CausalCandidate(_NoProtectedNamespace):
    node_urn: str
    node_name: str
    hops_from_model: int
    drift_result: FeatureDriftResult
    is_genuine_cause: bool
    intervention_delta: float = Field(
        description=(
            "Mean change in downstream drift signal when this node's contribution "
            "is held constant vs. observed, across bootstrap resamples. Larger "
            "magnitude = stronger causal evidence that this node is driving the "
            "downstream drift, rather than merely having changed around the same time."
        )
    )
    intervention_delta_lower_ci: float = Field(
        default=0.0,
        description="5th percentile of intervention_delta across bootstrap resamples. "
        "is_genuine_cause requires THIS (not the mean) to clear the threshold — "
        "a robust effect, not a lucky draw.",
    )
    intervention_delta_upper_ci: float = Field(
        default=0.0,
        description="95th percentile of intervention_delta across bootstrap resamples.",
    )
    confounded_with: list[str] = Field(
        default_factory=list,
        description="Other upstream URNs this node's drift is correlated with, "
        "making causal attribution ambiguous unless disentangled.",
    )


class RootCauseTrace(_NoProtectedNamespace):
    model_urn: str
    prediction_drift: PredictionDriftResult
    candidates_examined: list[CausalCandidate]
    isolated_root_causes: list[CausalCandidate] = Field(
        description="Subset of candidates_examined judged to be genuine causes, ranked by intervention_delta"
    )
    graph_path: list[str] = Field(description="URNs from root cause to model, in order")


# ---------------------------------------------------------------------------
# LLM reasoning output
# ---------------------------------------------------------------------------

class SuggestedFix(_NoProtectedNamespace):
    action: str
    target_urn: str
    rationale: str


class RootCauseReport(_NoProtectedNamespace):
    model_urn: str
    generated_at: datetime
    summary: str
    detailed_explanation: str
    root_causes: list[str] = Field(description="Human-readable names of isolated root cause nodes")
    confidence: str = Field(description="'low' | 'moderate' | 'high', based on intervention_delta magnitude & confounding")
    suggested_fixes: list[SuggestedFix]
    raw_trace: RootCauseTrace


# ---------------------------------------------------------------------------
# Write-back
# ---------------------------------------------------------------------------

class WriteBackResult(_NoProtectedNamespace):
    datahub_incident_urn: Optional[str] = None
    github_issue_url: Optional[str] = None
    github_pr_url: Optional[str] = None
    status: str
