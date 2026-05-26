"""Tutor outline helpers — FENCED (T17, V-RB1, V-O5).

`search_topics` + `get_aamc_outline` previously returned the AAMC-shaped
section/fc/cc/topic tree. Those tables are gone (T1) — the outline now
lives in `outline_nodes` (V-O1). T22 is the planned port that rebuilds
both on top of `OutlineLookup` so MCP/tutor can list/search nodes.

Until T22:

  - the consuming routes (`GET /api/v1/tutor/outline`,
    `GET /api/v1/tutor/outline/topics/search`) are unmounted in
    `app/api/v1/tutor.py`,
  - both functions return empty payloads so any in-process caller does
    not crash.

This file is FENCED, not a stub: behavior is deliberate, not in-progress.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession  # noqa: F401 — kept for signature

logger = logging.getLogger(__name__)


_FENCED_MSG = (
    "tutor.outline is FENCED (T17, V-RB1) — routes unmounted; "
    "restoration tracked in T22 (OutlineLookup-backed port)"
)


async def search_topics(
    session: AsyncSession, *, query: str, limit: int = 20
) -> list[dict[str, Any]]:
    """FENCED — see module docstring. Returns an empty list."""
    logger.warning(_FENCED_MSG)
    return []


async def get_aamc_outline(session: AsyncSession) -> dict[str, Any]:
    """FENCED — see module docstring. Returns an empty payload."""
    logger.warning(_FENCED_MSG)
    return {"sections": []}
