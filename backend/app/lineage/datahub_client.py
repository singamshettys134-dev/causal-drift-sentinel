"""
Lineage Graph Ingestion Layer (spec Section 3 & 6, step 2).

Responsible for pulling ML lineage from DataHub (via the DataHub MCP Server /
Agent Context Kit) and materializing it into our internal LineageGraph model.

Two implementations are provided behind the same interface:
  - DataHubMCPClient: talks to a real DataHub MCP Server over HTTP/SSE.
  - MockDataHubClient: synthesizes a realistic demo lineage graph so the
    full pipeline can run end-to-end without a live DataHub instance
    (used for `USE_MOCK_DATAHUB=true`, e.g. local dev / hackathon demo).

Swapping between them is a one-line change in api/dependencies.py — nothing
downstream (drift engine, causal engine, LLM layer) needs to know which one
is active, since both return the same LineageGraph shape.
"""
from __future__ import annotations

import abc
import json
from typing import Any

import httpx

from app.config import settings
from app.models.schemas import LineageEdge, LineageGraph, LineageNode, NodeType


class BaseLineageClient(abc.ABC):
    @abc.abstractmethod
    async def get_ml_lineage(self, model_urn: str) -> LineageGraph:
        """Return the full multi-hop upstream lineage DAG for a given model URN."""
        raise NotImplementedError

    @abc.abstractmethod
    async def write_incident(self, model_urn: str, incident_payload: dict[str, Any]) -> str:
        """Write a structured incident annotation back onto the model entity. Returns incident URN."""
        raise NotImplementedError


class DataHubMCPClient(BaseLineageClient):
    """
    Talks to a real DataHub MCP Server (or Agent Context Kit) to pull ML
    lineage and write back incidents.

    DataHub's MCP server exposes tools for entity search, lineage traversal,
    and metadata mutation. We call those tools over the MCP protocol rather
    than hitting the raw GMS REST API directly, so this stays aligned with
    however DataHub evolves that surface.
    """

    def __init__(self, mcp_url: str, token: str):
        self.mcp_url = mcp_url
        self.token = token
        self._client = httpx.AsyncClient(
            base_url=mcp_url,
            headers={"Authorization": f"Bearer {token}"} if token else {},
            timeout=30.0,
        )

    async def _call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Minimal MCP JSON-RPC 'tools/call' invocation. A production build would
        use a full MCP client SDK (session init, capability negotiation);
        this is intentionally the thin transport needed for our two tools.
        """
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }
        resp = await self._client.post("/", json=payload)
        resp.raise_for_status()
        result = resp.json()
        if "error" in result:
            raise RuntimeError(f"MCP tool '{tool_name}' failed: {result['error']}")
        return result["result"]

    async def get_ml_lineage(self, model_urn: str) -> LineageGraph:
        raw = await self._call_tool(
            "get_lineage",
            {"urn": model_urn, "direction": "UPSTREAM", "degree": "MULTI_HOP", "types": [
                "DATASET", "ML_FEATURE_TABLE", "ML_MODEL", "ML_MODEL_DEPLOYMENT"
            ]},
        )
        return _parse_datahub_lineage_response(raw, model_urn)

    async def write_incident(self, model_urn: str, incident_payload: dict[str, Any]) -> str:
        result = await self._call_tool(
            "create_incident",
            {
                "entity_urn": model_urn,
                "type": "DATA_QUALITY",
                "title": incident_payload.get("summary", "Model drift incident"),
                "description": json.dumps(incident_payload, default=str),
                "status": "OPEN",
                "source": "causal-drift-sentinel-agent",
            },
        )
        return result.get("incident_urn", f"urn:li:incident:(mock,{model_urn})")


def _parse_datahub_lineage_response(raw: dict[str, Any], root_model_urn: str) -> LineageGraph:
    """Translate DataHub's lineage response shape into our internal LineageGraph."""
    nodes: list[LineageNode] = []
    edges: list[LineageEdge] = []
    type_map = {
        "DATASET": NodeType.DATASET,
        "ML_FEATURE_TABLE": NodeType.FEATURE,
        "ML_MODEL": NodeType.MODEL,
        "ML_MODEL_DEPLOYMENT": NodeType.DEPLOYMENT,
    }
    for entity in raw.get("entities", []):
        nodes.append(
            LineageNode(
                urn=entity["urn"],
                name=entity.get("name", entity["urn"]),
                node_type=type_map.get(entity.get("entityType", "DATASET"), NodeType.DATASET),
                platform=entity.get("platform"),
                description=entity.get("description"),
                schema_fields=entity.get("schemaFieldNames", []),
                tags=entity.get("tags", []),
            )
        )
    for rel in raw.get("relationships", []):
        edges.append(
            LineageEdge(
                upstream_urn=rel["upstreamUrn"],
                downstream_urn=rel["downstreamUrn"],
                relationship=rel.get("type", "derives_from"),
            )
        )
    return LineageGraph(nodes=nodes, edges=edges, root_model_urn=root_model_urn)


class MockDataHubClient(BaseLineageClient):
    """
    Synthesizes a realistic fraud-detection-style ML lineage graph for demo
    purposes (spec Section 4, "Data for the Demo"):

        raw_transactions, raw_user_profiles, raw_device_signals
              -> feature_txn_velocity, feature_user_risk_score, feature_device_trust
                    -> fraud_model_v3
                          -> fraud_model_v3_prod_deployment

    This lets the full pipeline (ingestion -> drift -> causal -> LLM ->
    write-back -> frontend) run end-to-end without a live DataHub instance,
    while keeping the exact same LineageGraph contract a real DataHub client
    would produce.
    """

    async def get_ml_lineage(self, model_urn: str) -> LineageGraph:
        nodes = [
            LineageNode(
                urn="urn:li:dataset:(demo,raw_transactions,PROD)",
                name="raw_transactions",
                node_type=NodeType.DATASET,
                platform="snowflake",
                description="Raw transaction events ingested from the payments service.",
                schema_fields=["txn_id", "user_id", "amount", "currency", "merchant_category", "ts"],
            ),
            LineageNode(
                urn="urn:li:dataset:(demo,raw_user_profiles,PROD)",
                name="raw_user_profiles",
                node_type=NodeType.DATASET,
                platform="postgres",
                description="User account and KYC profile data.",
                schema_fields=["user_id", "account_age_days", "kyc_tier", "country"],
            ),
            LineageNode(
                urn="urn:li:dataset:(demo,raw_device_signals,PROD)",
                name="raw_device_signals",
                node_type=NodeType.DATASET,
                platform="kafka",
                description="Device fingerprint and network signals per session.",
                schema_fields=["session_id", "user_id", "device_id", "ip_risk_score"],
            ),
            LineageNode(
                urn="urn:li:mlFeatureTable:(demo,feature_txn_velocity)",
                name="feature_txn_velocity",
                node_type=NodeType.FEATURE,
                platform="feast",
                description="Rolling 1h/24h transaction count and sum per user.",
                schema_fields=["txn_count_1h", "txn_sum_24h"],
            ),
            LineageNode(
                urn="urn:li:mlFeatureTable:(demo,feature_user_risk_score)",
                name="feature_user_risk_score",
                node_type=NodeType.FEATURE,
                platform="feast",
                description="Composite user risk score derived from profile + history.",
                schema_fields=["user_risk_score"],
            ),
            LineageNode(
                urn="urn:li:mlFeatureTable:(demo,feature_device_trust)",
                name="feature_device_trust",
                node_type=NodeType.FEATURE,
                platform="feast",
                description="Device trust score derived from device signals.",
                schema_fields=["device_trust_score"],
            ),
            LineageNode(
                urn="urn:li:mlModel:(demo,fraud_model_v3,PROD)",
                name="fraud_model_v3",
                node_type=NodeType.MODEL,
                platform="mlflow",
                description="Gradient-boosted fraud classifier, trained 2026-06-01.",
            ),
            LineageNode(
                urn="urn:li:mlModelDeployment:(demo,fraud_model_v3_prod)",
                name="fraud_model_v3_prod_deployment",
                node_type=NodeType.DEPLOYMENT,
                platform="sagemaker",
                description="Live production endpoint for fraud_model_v3.",
            ),
        ]
        edges = [
            LineageEdge(upstream_urn="urn:li:dataset:(demo,raw_transactions,PROD)",
                        downstream_urn="urn:li:mlFeatureTable:(demo,feature_txn_velocity)"),
            LineageEdge(upstream_urn="urn:li:dataset:(demo,raw_user_profiles,PROD)",
                        downstream_urn="urn:li:mlFeatureTable:(demo,feature_user_risk_score)"),
            LineageEdge(upstream_urn="urn:li:dataset:(demo,raw_transactions,PROD)",
                        downstream_urn="urn:li:mlFeatureTable:(demo,feature_user_risk_score)"),
            LineageEdge(upstream_urn="urn:li:dataset:(demo,raw_device_signals,PROD)",
                        downstream_urn="urn:li:mlFeatureTable:(demo,feature_device_trust)"),
            LineageEdge(upstream_urn="urn:li:mlFeatureTable:(demo,feature_txn_velocity)",
                        downstream_urn="urn:li:mlModel:(demo,fraud_model_v3,PROD)"),
            LineageEdge(upstream_urn="urn:li:mlFeatureTable:(demo,feature_user_risk_score)",
                        downstream_urn="urn:li:mlModel:(demo,fraud_model_v3,PROD)"),
            LineageEdge(upstream_urn="urn:li:mlFeatureTable:(demo,feature_device_trust)",
                        downstream_urn="urn:li:mlModel:(demo,fraud_model_v3,PROD)"),
            LineageEdge(upstream_urn="urn:li:mlModel:(demo,fraud_model_v3,PROD)",
                        downstream_urn="urn:li:mlModelDeployment:(demo,fraud_model_v3_prod)"),
        ]
        return LineageGraph(
            nodes=nodes,
            edges=edges,
            root_model_urn=model_urn or "urn:li:mlModelDeployment:(demo,fraud_model_v3_prod)",
        )

    async def write_incident(self, model_urn: str, incident_payload: dict[str, Any]) -> str:
        # In demo mode we just fabricate a plausible incident URN; no network call.
        return f"urn:li:incident:(demo,{abs(hash(json.dumps(incident_payload, default=str))) % 10**8})"


def get_lineage_client() -> BaseLineageClient:
    if settings.USE_MOCK_DATAHUB:
        return MockDataHubClient()
    return DataHubMCPClient(mcp_url=settings.DATAHUB_MCP_URL, token=settings.DATAHUB_TOKEN)
