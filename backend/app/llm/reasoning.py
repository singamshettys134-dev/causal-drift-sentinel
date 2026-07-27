"""
LLM Reasoning & Explanation Layer (spec Section 3 & 6, step 5; Section 7).

CRITICAL per spec Section 7: "The LLM reasoning layer should never be asked
to invent a root cause; it should only be asked to explain and phrase a
root cause that the statistical/graph engine has already isolated."

So this module:
  - Never sends raw data to the model, only the already-computed
    RootCauseTrace (structured evidence: drift stats + graph trace).
  - Uses a strict system prompt forbidding the model from naming any root
    cause not present in `isolated_root_causes`.
  - Parses the response into RootCauseReport via structured JSON output,
    per the pattern in <structured_outputs_in_xml>.
  - Model is swappable via the single `settings.LLM_MODEL` config value
    (spec Section 4 & 7) — nothing else in this file needs to change to
    upgrade models.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from groq import Groq

from app.config import settings
from app.models.schemas import RootCauseReport, RootCauseTrace, SuggestedFix

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are the reasoning layer of Causal Drift Sentinel, an ML \
observability agent. You will be given a structured RootCauseTrace: statistical \
drift evidence and a causal graph trace that an algorithmic engine has ALREADY \
computed and isolated.

STRICT RULES:
1. You must NEVER invent, guess, or add a root cause that is not present in \
`isolated_root_causes` in the input. Your job is to explain and phrase \
findings the algorithm already isolated, not to do new causal reasoning.
2. If `isolated_root_causes` is empty, say so plainly — do not speculate \
about a cause anyway.
3. Ground every claim in the numeric evidence provided (p-values, PSI \
scores, intervention_delta). Do not cite numbers you were not given.
4. Confidence must reflect the evidence: high confidence requires a large \
intervention_delta (>0.15) with no confounding; moderate for smaller deltas \
or partial confounding; low otherwise.
5. Suggested fixes must be concrete and tied to a specific node URN from the \
trace (e.g. "add a validation gate on <urn>", "retrain on refreshed <urn>").

Respond with ONLY a JSON object (no markdown fences, no preamble) matching \
this exact shape:
{
  "summary": "one or two sentence plain-language summary",
  "detailed_explanation": "several sentences walking through the evidence and graph trace",
  "root_causes": ["human-readable node names, from isolated_root_causes only"],
  "confidence": "low | moderate | high",
  "suggested_fixes": [
    {"action": "...", "target_urn": "...", "rationale": "..."}
  ]
}"""


def _enforce_grounding(parsed: dict, trace: RootCauseTrace) -> dict:
    """
    Code-level enforcement of the grounding contract described in the
    system prompt above — instructions alone don't guarantee an LLM won't
    hallucinate, so this filters the parsed response against the actual
    isolated_root_causes rather than trusting it blindly.
    """
    allowed_names = {c.node_name for c in trace.isolated_root_causes}
    allowed_urns = {c.node_urn for c in trace.isolated_root_causes} | {trace.model_urn}

    root_causes = [name for name in parsed.get("root_causes", []) if name in allowed_names]
    if not root_causes and allowed_names:
        # LLM omitted or hallucinated every name — fall back to the
        # algorithm's own ground truth rather than reporting nothing.
        logger.warning(
            "LLM response named no valid root causes from %s; falling back "
            "to isolated_root_causes directly.", allowed_names,
        )
        root_causes = sorted(allowed_names)
    parsed["root_causes"] = root_causes

    fixes = parsed.get("suggested_fixes", [])
    filtered_fixes = [f for f in fixes if f.get("target_urn") in allowed_urns]
    if len(filtered_fixes) != len(fixes):
        logger.warning(
            "Dropped %d suggested fix(es) targeting a URN outside the "
            "isolated root causes / model.", len(fixes) - len(filtered_fixes),
        )
    parsed["suggested_fixes"] = filtered_fixes

    return parsed


class LLMReasoningLayer:
    def __init__(self, model: str | None = None):
        # Single swappable config point per spec Section 4/7.
        self.model = model or settings.LLM_MODEL
        self._client = Groq(api_key=settings.GROQ_API_KEY)

    def generate_report(self, trace: RootCauseTrace, _retries: int = 1) -> RootCauseReport:
        evidence_payload = trace.model_dump(mode="json")

        last_error: Exception | None = None
        for attempt in range(_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    max_tokens=settings.LLM_MAX_TOKENS,
                    temperature=0,  # deterministic, grounded phrasing over creative variation
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                "Here is the RootCauseTrace evidence:\n\n"
                                f"{json.dumps(evidence_payload, indent=2)}"
                            ),
                        },
                    ],
                )

                text = response.choices[0].message.content
                text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                parsed = json.loads(text)
                parsed = _enforce_grounding(parsed, trace)

                return RootCauseReport(
                    model_urn=trace.model_urn,
                    generated_at=datetime.now(timezone.utc),
                    summary=parsed["summary"],
                    detailed_explanation=parsed["detailed_explanation"],
                    root_causes=parsed["root_causes"],
                    confidence=parsed["confidence"],
                    suggested_fixes=[SuggestedFix(**f) for f in parsed["suggested_fixes"]],
                    raw_trace=trace,
                )
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                last_error = exc
                logger.warning("LLM response malformed on attempt %d/%d: %s", attempt + 1, _retries + 1, exc)

        # Every attempt failed to produce valid, parseable JSON. Rather than
        # 500 the whole /investigate request purely because the LLM phrasing
        # layer misbehaved, fall back to a deterministic report built
        # directly from the algorithmic trace — the causal finding itself
        # is still valid even if we couldn't get a polished explanation.
        logger.error("LLM reasoning layer failed after %d attempts; using deterministic fallback report.", _retries + 1)
        return _deterministic_fallback_report(trace)


def _deterministic_fallback_report(trace: RootCauseTrace) -> RootCauseReport:
    """Built directly from the algorithmic trace, no LLM involved — used only
    when the reasoning layer itself is unavailable/misbehaving, so a
    statistically-valid finding is never lost to an LLM/formatting failure."""
    names = [c.node_name for c in trace.isolated_root_causes]
    return RootCauseReport(
        model_urn=trace.model_urn,
        generated_at=datetime.now(timezone.utc),
        summary=(
            f"Root cause isolated algorithmically: {', '.join(names)}."
            if names else "No root cause could be isolated from the current evidence."
        ),
        detailed_explanation=(
            "The LLM explanation layer was unavailable or returned an unparseable "
            "response, so this is a deterministic summary generated directly from "
            "the causal isolation engine's trace, not an LLM-authored explanation."
        ),
        root_causes=names,
        confidence="low",  # unexplained by the reasoning layer, so surfaced conservatively
        suggested_fixes=[],
        raw_trace=trace,
    )


def get_reasoning_layer() -> LLMReasoningLayer:
    return LLMReasoningLayer()
