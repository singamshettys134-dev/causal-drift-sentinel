"""
Tests covering the statistical/causal core (the parts spec Section 7 says
must be correct and defensible before the LLM layer is trusted) plus the
LLM layer's grounding contract via a mocked Groq client.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.causal.isolator import isolate_root_causes
from app.drift.engine import ks_test_drift, prediction_output_drift, psi_drift
from app.lineage.dag import build_dag
from app.lineage.datahub_client import MockDataHubClient
from app.llm.reasoning import LLMReasoningLayer
from app.models.schemas import DriftSeverity
from app.utils.demo_data import MODEL_URN, generate_demo_samples, generate_prediction_samples


# ---------------------------------------------------------------------------
# Drift engine
# ---------------------------------------------------------------------------

def test_ks_test_no_drift_on_identical_distributions():
    rng = np.random.default_rng(0)
    baseline = rng.normal(0, 1, 1000)
    current = rng.normal(0, 1, 1000)
    result = ks_test_drift(baseline, current, "urn:x", "feat", "train", "now")
    assert result.severity == DriftSeverity.NONE


def test_ks_test_detects_severe_shift():
    rng = np.random.default_rng(0)
    baseline = rng.normal(0, 1, 1000)
    current = rng.normal(5, 1, 1000)  # large mean shift
    result = ks_test_drift(baseline, current, "urn:x", "feat", "train", "now")
    assert result.severity in (DriftSeverity.HIGH, DriftSeverity.CRITICAL)


def test_psi_increases_with_shift_magnitude():
    rng = np.random.default_rng(1)
    baseline = rng.normal(0, 1, 2000)
    small_shift = rng.normal(0.2, 1, 2000)
    large_shift = rng.normal(2.0, 1, 2000)
    small = psi_drift(baseline, small_shift, "urn:x", "feat", "train", "now")
    large = psi_drift(baseline, large_shift, "urn:x", "feat", "train", "now")
    assert large.psi_score > small.psi_score


def test_prediction_output_drift_flags_severe_case():
    rng = np.random.default_rng(2)
    baseline = rng.beta(2, 5, 1000)
    current = rng.beta(5, 2, 1000)
    result = prediction_output_drift(baseline, current, MODEL_URN)
    assert result.severity != DriftSeverity.NONE


# ---------------------------------------------------------------------------
# Causal isolation - the core correctness requirement per spec Section 7
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_causal_isolation_pinpoints_injected_root_cause():
    client = MockDataHubClient()
    graph = await client.get_ml_lineage(MODEL_URN)
    dag = build_dag(graph)

    samples = generate_demo_samples(inject_drift=True)
    pb, pc = generate_prediction_samples(samples)
    pred_drift = prediction_output_drift(pb, pc, MODEL_URN)

    trace = isolate_root_causes(graph, dag, MODEL_URN, pred_drift, samples)

    isolated_names = {c.node_name for c in trace.isolated_root_causes}
    assert "raw_user_profiles" in isolated_names, (
        "The scripted drift was injected into raw_user_profiles; the causal "
        "engine must isolate it, not the downstream feature that merely "
        "inherited the drift."
    )
    # The downstream feature that inherited the drift (not the true cause)
    # must NOT be misidentified as an independent genuine cause once the
    # true upstream cause is accounted for.
    downstream_inherited = next(
        (c for c in trace.candidates_examined if c.node_name == "feature_user_risk_score"),
        None,
    )
    assert downstream_inherited is not None
    assert downstream_inherited.is_genuine_cause is False


@pytest.mark.asyncio
async def test_no_drift_scenario_yields_no_root_causes():
    client = MockDataHubClient()
    graph = await client.get_ml_lineage(MODEL_URN)
    dag = build_dag(graph)

    samples = generate_demo_samples(inject_drift=False)
    pb, pc = generate_prediction_samples(samples)
    pred_drift = prediction_output_drift(pb, pc, MODEL_URN)

    trace = isolate_root_causes(graph, dag, MODEL_URN, pred_drift, samples)
    assert trace.isolated_root_causes == []


# ---------------------------------------------------------------------------
# LLM reasoning layer - grounding contract (mocked, no live API call)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_layer_only_reports_isolated_causes():
    client = MockDataHubClient()
    graph = await client.get_ml_lineage(MODEL_URN)
    dag = build_dag(graph)
    samples = generate_demo_samples(inject_drift=True)
    pb, pc = generate_prediction_samples(samples)
    pred_drift = prediction_output_drift(pb, pc, MODEL_URN)
    trace = isolate_root_causes(graph, dag, MODEL_URN, pred_drift, samples)

    fake_json_reply = json.dumps({
        "summary": "raw_user_profiles drifted, causing fraud_model_v3 predictions to shift.",
        "detailed_explanation": "account_age_days shifted sharply younger, propagating through feature_user_risk_score.",
        "root_causes": [c.node_name for c in trace.isolated_root_causes],
        "confidence": "high",
        "suggested_fixes": [
            {"action": "add validation gate", "target_urn": trace.isolated_root_causes[0].node_urn,
             "rationale": "prevent silent cohort shifts from reaching the feature store"}
        ],
    })

    mock_message = MagicMock()
    mock_message.content = fake_json_reply
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    layer = LLMReasoningLayer(model="llama-3.3-70b-versatile")
    with patch.object(layer._client.chat.completions, "create", return_value=mock_response):
        report = layer.generate_report(trace)

    assert report.root_causes == [c.node_name for c in trace.isolated_root_causes]
    assert report.confidence == "high"
    assert report.suggested_fixes[0].target_urn == trace.isolated_root_causes[0].node_urn
