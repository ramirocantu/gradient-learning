"""Resolve `AnkiAssignment.scope_kind/scope_value` → human label + URL.

FENCED (T17, V-RB1, V-O5). The PoC built the topic label + drilldown URL
by joining `Topic`/`ContentCategory` and walking the topic ancestor
chain. Those tables are dropped (T1). Restoration moves to
`OutlineLookup.path_of(node_id)` + a node-shaped drilldown URL and is
tracked in T18 (anki node-id port) + T22 (OutlineLookup-backed surface).

Until then this helper renders a raw value-based label and
`scope_url=None`. The dashboard anki route still mounts (it has
real-data uses outside this label resolver), so the placeholder is
deliberately non-raising. This file is FENCED, not a stub: behavior is
deliberate, not in-progress.
"""

from __future__ import annotations

import logging
from typing import Sequence

from sqlalchemy.ext.asyncio import AsyncSession  # noqa: F401 — kept for signature

from app.models.anki import AnkiAssignment

logger = logging.getLogger(__name__)


_FENCED_MSG = (
    "dashboard.services.anki_scope.attach_scope_labels is FENCED "
    "(T17, V-RB1) — raw scope_value labels; restoration tied to T18+T22"
)


async def attach_scope_labels(session: AsyncSession, assignments: Sequence[AnkiAssignment]) -> None:
    """FENCED — mutate `assignments` in place with placeholder labels."""
    for a in assignments:
        a.scope_label = f"{a.scope_kind}:{a.scope_value}"  # type: ignore[attr-defined]
        a.scope_url = None  # type: ignore[attr-defined]
    if assignments:
        logger.warning(_FENCED_MSG)
