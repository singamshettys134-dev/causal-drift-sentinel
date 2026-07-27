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


@pytest.mark.asyncio
async def test_llm_layer_filters_hallucinated_root_cause():
    """
    Code-level enforcement of the grounding contract: even if the LLM
    ignores its system prompt and names a root cause the algorithm never
    isolated, the final report must not include it.
    """
    client = MockDataHubClient()
    graph = await client.get_ml_lineage(MODEL_URN)
    dag = build_dag(graph)
    samples = generate_demo_samples(inject_drift=True)
    pb, pc = generate_prediction_samples(samples)
    pred_drift = prediction_output_drift(pb, pc, MODEL_URN)
    trace = isolate_root_causes(graph, dag, MODEL_URN, pred_drift, samples)

    real_cause = trace.isolated_root_causes[0].node_name
    hallucinated_reply = json.dumps({
        "summary": "Multiple factors drifted.",
        "detailed_explanation": "Everything changed at once, hard to say why.",
        # Hallucinated: names a node the algorithm never isolated as a root cause.
        "root_causes": [real_cause, "raw_device_signals"],
        "confidence": "high",
        "suggested_fixes": [
            {"action": "investigate", "target_urn": "urn:li:dataset:(demo,raw_device_signals,PROD)",
             "rationale": "seems suspicious"},  # off-target: not an isolated cause or the model
        ],
    })

    mock_message = MagicMock()
    mock_message.content = hallucinated_reply
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    layer = LLMReasoningLayer(model="llama-3.3-70b-versatile")
    with patch.object(layer._client.chat.completions, "create", return_value=mock_response):
        report = layer.generate_report(trace)

    assert "raw_device_signals" not in report.root_causes, (
        "LLM named a node the causal engine never isolated as a root cause; "
        "the report must filter this out rather than trust the LLM's phrasing."
    )
    assert real_cause in report.root_causes
    assert report.suggested_fixes == [], (
        "The only suggested fix targeted a URN outside isolated_root_causes/model "
        "and must be dropped, not passed through."
    )


@pytest.mark.asyncio
async def test_llm_layer_falls_back_deterministically_on_malformed_response():
    """If the LLM never returns valid JSON, the pipeline must not crash —
    it should fall back to a deterministic report built from the trace."""
    client = MockDataHubClient()
    graph = await client.get_ml_lineage(MODEL_URN)
    dag = build_dag(graph)
    samples = generate_demo_samples(inject_drift=True)
    pb, pc = generate_prediction_samples(samples)
    pred_drift = prediction_output_drift(pb, pc, MODEL_URN)
    trace = isolate_root_causes(graph, dag, MODEL_URN, pred_drift, samples)

    mock_message = MagicMock()
    mock_message.content = "not valid json at all"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    layer = LLMReasoningLayer(model="llama-3.3-70b-versatile")
    with patch.object(layer._client.chat.completions, "create", return_value=mock_response):
        report = layer.generate_report(trace, _retries=0)

    assert report.confidence == "low"
    assert set(report.root_causes) == {c.node_name for c in trace.isolated_root_causes}


# ---------------------------------------------------------------------------
# Write-back agent - graceful degradation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_writeback_survives_datahub_failure():
    """A DataHub write-back failure (e.g. mutation disabled, network error)
    must not crash the request — the causal finding is already valid and
    should still be returned with a 'partial' status."""
    from app.writeback.agent import WriteBackAgent
    from app.models.schemas import RootCauseReport
    from datetime import datetime, timezone

    class FailingLineageClient(MockDataHubClient):
        async def write_incident(self, model_urn, incident_payload):
            raise RuntimeError("DATAHUB_MUTATION_ENABLED is false")

    report = RootCauseReport(
        model_urn=MODEL_URN,
        generated_at=datetime.now(timezone.utc),
        summary="test",
        detailed_explanation="test",
        root_causes=["raw_user_profiles"],
        confidence="high",
        suggested_fixes=[],
        raw_trace=(await _build_trace()),
    )

    agent = WriteBackAgent(FailingLineageClient())
    result = await agent.run(report)

    assert result.datahub_incident_urn is None
    assert result.status == "partial"


@pytest.mark.asyncio
async def test_intervention_delta_has_bootstrap_confidence_interval():
    """
    The causal engine's genuine-cause decision must be based on a bootstrap
    confidence interval, not a single lucky random draw — this asserts the
    CI fields are actually populated and internally consistent
    (lower <= mean <= upper), and that is_genuine_cause is gated on the
    conservative lower bound rather than the point estimate.
    """
    client = MockDataHubClient()
    graph = await client.get_ml_lineage(MODEL_URN)
    dag = build_dag(graph)
    samples = generate_demo_samples(inject_drift=True)
    pb, pc = generate_prediction_samples(samples)
    pred_drift = prediction_output_drift(pb, pc, MODEL_URN)
    trace = isolate_root_causes(graph, dag, MODEL_URN, pred_drift, samples)

    tested_multi_ancestor_candidate = False
    for c in trace.candidates_examined:
        assert c.intervention_delta_lower_ci <= c.intervention_delta <= c.intervention_delta_upper_ci
        if c.intervention_delta_lower_ci != c.intervention_delta_upper_ci:
            tested_multi_ancestor_candidate = True
        # The decision must key off the conservative lower bound.
        from app.config import settings
        expected = c.intervention_delta_lower_ci >= settings.INTERVENTION_DELTA_MIN
        assert c.is_genuine_cause == expected

    assert tested_multi_ancestor_candidate, (
        "Expected at least one candidate with multiple co-ancestors, where "
        "the bootstrap actually produces a nontrivial interval (not a "
        "degenerate single-point one)."
    )


# ---------------------------------------------------------------------------
# DataHub client - response parsing (exercised without a live MCP server)
# ---------------------------------------------------------------------------

def test_parses_entities_relationships_lineage_shape():
    from app.lineage.datahub_client import _parse_datahub_lineage_response

    raw = {
        "entities": [
            {"urn": "urn:li:dataset:(demo,a,PROD)", "name": "a", "entityType": "DATASET"},
            {"urn": "urn:li:mlModel:(demo,m,PROD)", "name": "m", "entityType": "ML_MODEL"},
        ],
        "relationships": [
            {"upstreamUrn": "urn:li:dataset:(demo,a,PROD)", "downstreamUrn": "urn:li:mlModel:(demo,m,PROD)", "type": "derives_from"},
        ],
    }
    graph = _parse_datahub_lineage_response(raw, "urn:li:mlModel:(demo,m,PROD)")
    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
    assert graph.edges[0].upstream_urn == "urn:li:dataset:(demo,a,PROD)"


def test_parses_results_paths_lineage_shape():
    """Some mcp-server-datahub versions return a flatter {results: [{urn, paths}]}
    shape (per the third-party client examples found during review) — the parser
    must handle this fallback without crashing, since the exact shape can drift
    across server versions."""
    from app.lineage.datahub_client import _parse_datahub_lineage_response

    raw = {
        "results": [
            {"urn": "urn:li:dataset:(demo,b,PROD)", "hops": 1,
             "paths": [["urn:li:dataset:(demo,b,PROD)", "urn:li:mlModel:(demo,m,PROD)"]]},
        ],
    }
    graph = _parse_datahub_lineage_response(raw, "urn:li:mlModel:(demo,m,PROD)")
    assert len(graph.nodes) == 1
    assert len(graph.edges) == 1


def test_parses_empty_or_unrecognized_shape_without_crashing():
    from app.lineage.datahub_client import _parse_datahub_lineage_response

    graph = _parse_datahub_lineage_response({"unexpected": "shape"}, "urn:li:mlModel:(demo,m,PROD)")
    assert graph.nodes == []
    assert graph.edges == []
    assert graph.root_model_urn == "urn:li:mlModel:(demo,m,PROD)"


async def _build_trace():
    client = MockDataHubClient()
    graph = await client.get_ml_lineage(MODEL_URN)
    dag = build_dag(graph)
    samples = generate_demo_samples(inject_drift=True)
    pb, pc = generate_prediction_samples(samples)
    pred_drift = prediction_output_drift(pb, pc, MODEL_URN)
    return isolate_root_causes(graph, dag, MODEL_URN, pred_drift, samples)
