"""
Smoke test for the real DataHub MCP connection (backend/README.md #
"Before your demo"). Run this once against your actual DataHub instance
before recording the demo video or presenting live — it will catch tool
name/parameter mismatches early instead of during the demo.

Usage:
    cd backend
    # .env must have USE_MOCK_DATAHUB=false and real DATAHUB_GMS_URL/TOKEN
    python -m scripts.smoke_test_datahub urn:li:mlModelDeployment:(your,model,PROD)
"""
from __future__ import annotations

import asyncio
import sys

from app.lineage.datahub_client import DataHubMCPClient


async def main(model_urn: str) -> None:
    client = DataHubMCPClient()
    try:
        session = await client._ensure_session()

        print("Connected. Listing available tools on this server...\n")
        tools = await session.list_tools()
        names = sorted(t.name for t in tools.tools)
        print(f"Found {len(names)} tools: {', '.join(names)}\n")

        expected = {"get_lineage", "search", "get_entities"}
        missing = expected - set(names)
        if missing:
            print(f"⚠️  Expected tools not found: {missing} — server version may differ from what "
                  f"datahub_client.py assumes. Check tool names above and adjust _call_tool calls.\n")

        print(f"Calling get_lineage for {model_urn} ...\n")
        graph = await client.get_ml_lineage(model_urn)
        print(f"Got {len(graph.nodes)} nodes, {len(graph.edges)} edges.")
        if not graph.nodes:
            print("⚠️  Zero nodes returned — check _parse_datahub_lineage_response() against the "
                  "raw tool output shape (add a print(raw) in get_ml_lineage to inspect it).")
        else:
            print("✅ Lineage parsing looks functional. Sample node:", graph.nodes[0])
    finally:
        await client.aclose()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
