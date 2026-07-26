# Sample GitHub Issue — Opened Automatically by the Write-Back Agent

This is the exact, code-generated Markdown body that
`backend/app/writeback/agent.py::_render_github_issue_body()` produces from
`sample_root_cause_report.json` (which is itself grounded in the real
statistical/causal trace in `sample_root_cause_trace.json`, produced by
actually running the drift engine and causal isolator against the scripted
demo scenario in `backend/app/utils/demo_data.py`).

In a live deployment with `GITHUB_TOKEN` / `GITHUB_REPO` configured, this
exact body is POSTed to `/repos/{owner}/{repo}/issues` by
`WriteBackAgent.open_github_issue()`, labeled `drift-detected`,
`auto-generated`.

---

**Title:** `[Drift Detected] fraud_model_v3'''s predictions have drifted critically (KS=0.54, p<1e-`

**Body:**

**Model:** `urn:li:mlModel:(demo,fraud_model_v3,PROD)`
**Confidence:** high

## Summary
fraud_model_v3's predictions have drifted critically (KS=0.54, p<1e-200) because raw_user_profiles.account_age_days shifted sharply toward brand-new accounts, propagating through feature_user_risk_score.

## Root Cause Analysis
The prediction-output KS-test flagged critical drift in fraud_model_v3's live output distribution (statistic=0.5405, p=1.26e-201) relative to its training-time baseline. Walking the lineage DAG backward, two upstream nodes showed statistically significant drift: raw_user_profiles (account_age_days, KS=0.8948, p≈0) and its direct downstream feature, feature_user_risk_score (KS=0.8948, p≈0, identical statistic because it is a near-deterministic transform of account_age_days). The intervention-style check is what separates these: holding raw_user_profiles at its baseline behavior while replaying all other current upstream signals collapses the downstream drift by an estimated 0.7767 (intervention_delta), whereas holding feature_user_risk_score's own contribution at baseline barely moves the downstream signal at all (intervention_delta=0.0005). That means feature_user_risk_score's drift is entirely inherited from raw_user_profiles rather than an independent cause, and it is correctly marked as confounded rather than causal. raw_user_profiles is the sole isolated root cause, at 2 hops from the model and with a graph path of raw_user_profiles -> feature_user_risk_score -> fraud_model_v3.

## Isolated Root Cause(s)
- `raw_user_profiles`

## Suggested Fixes
- **Add a validation gate (e.g. a PSI/KS threshold check in the ingestion job) on account_age_days before it reaches the feature store** on `urn:li:dataset:(demo,raw_user_profiles,PROD)` — The upstream table itself is the point of failure — likely a new signup cohort or a broken join surfacing mostly very-new accounts. Gating here prevents the bad distribution from ever reaching feature_user_risk_score.
- **Investigate the raw_user_profiles ingestion pipeline for a recent schema-compatible but semantically different change (e.g. a join key change or a new upstream source merged in)** on `urn:li:dataset:(demo,raw_user_profiles,PROD)` — account_age_days kept its name and type but its distribution moved by nearly a full KS statistic point — exactly the 'silent schema-compatible breakage' pattern this agent exists to catch.
- **Do not retrain fraud_model_v3 on the current feature_user_risk_score distribution until the upstream issue is fixed** on `urn:li:mlFeatureTable:(demo,feature_user_risk_score)` — Retraining now would just teach the model to treat the corrupted cohort as normal, masking the real data problem rather than fixing it.

## Statistical Evidence
- Prediction drift: `ks_test` statistic=0.5405, p=1.26e-201, severity=critical

| Candidate | Hops | Method | Statistic | Intervention Δ | Genuine Cause? |
|---|---|---|---|---|---|
| `raw_user_profiles` | 2 | ks_test | 0.8948 | 0.7767 | ✅ |
| `feature_user_risk_score` | 1 | ks_test | 0.8948 | 0.0005 | ❌ |

---
_Opened automatically by Causal Drift Sentinel — root cause isolated algorithmically via lineage graph traversal + intervention-style drift testing; this text is the LLM explanation layer phrasing that finding._
