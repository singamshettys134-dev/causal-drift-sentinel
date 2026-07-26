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
from datetime import datetime, timezone

from groq import Groq

from app.config import settings
from app.models.schemas import RootCauseReport, RootCauseTrace, SuggestedFix


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


class LLMReasoningLayer:
    def __init__(self, model: str | None = None):
        # Single swappable config point per spec Section 4/7.
        self.model = model or settings.LLM_MODEL
        self._client = Groq(api_key=settings.GROQ_API_KEY)

    def generate_report(self, trace: RootCauseTrace) -> RootCauseReport:
        evidence_payload = trace.model_dump(mode="json")

        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=settings.LLM_MAX_TOKENS,
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


def get_reasoning_layer() -> LLMReasoningLayer:
    return LLMReasoningLayer()
