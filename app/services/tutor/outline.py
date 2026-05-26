"""Tutor outline helpers — T14 partial port.

The PoC's `search_topics` + `get_aamc_outline` returned the AAMC-shaped
section/fc/cc/topic tree. Those tables are gone (T1) — the outline now lives
in `outline_nodes` with a single recursive shape (V-O1). T14 follow-up
rebuilds these on top of `OutlineLookup` so MCP/tutor can list/search nodes.

Stub keeps the public surface returning empty payloads until that port.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession  # noqa: F401 — kept for signature

logger = logging.getLogger(__name__)


async def search_topics(
    session: AsyncSession, *, query: str, limit: int = 20
) -> list[dict[str, Any]]:
    """Stub — TODO(T14 follow-up): port to OutlineNode name search via OutlineLookup."""
    logger.warning("search_topics stub: returns empty pending OutlineNode search port")
    return []


async def get_aamc_outline(session: AsyncSession) -> dict[str, Any]:
    """Stub — TODO(T14 follow-up): port to a node-tree dump via OutlineLookup."""
    logger.warning("get_aamc_outline stub: returns empty pending node-tree port")
    return {"sections": []}
