"""
dens_city.server: Production FastAPI and Model Context Protocol (MCP) server integration
for autonomous LLM harnesses, with dedicated spawn GPU worker and stateful artifact pools.
"""

from __future__ import annotations

__all__ = [
    "models",
    "validators",
    "diagnostics",
    "pools",
    "jobs",
    "worker",
    "app",
    "mcp_server",
]
