"""Resolve `AnkiAssignment.scope_kind/scope_value` → human label + URL.

T14 stub. The PoC built the topic label + drilldown URL by joining
`Topic`/`ContentCategory` and walking the topic ancestor chain. With those
tables dropped, scope resolution moves to `OutlineLookup.path_of(node_id)`
+ a node-shaped drilldown URL. T14 follow-up wires that in; for now scopes
render with a raw value-based label and `scope_url=None`.
"""

from __future__ import annotations

import logging
from typing import Sequence

from sqlalchemy.ext.asyncio import AsyncSession  # noqa: F401 — kept for signature

from app.models.anki import AnkiAssignment

logger = logging.getLogger(__name__)


async def attach_scope_labels(session: AsyncSession, assignments: Sequence[AnkiAssignment]) -> None:
    """Mutate `assignments` in place, setting `scope_label` + `scope_url`.

    TODO(T14 follow-up): resolve `scope_value` → outline node → `path_of`.
    """
    for a in assignments:
        a.scope_label = f"{a.scope_kind}:{a.scope_value}"  # type: ignore[attr-defined]
        a.scope_url = None  # type: ignore[attr-defined]
    if assignments:
        logger.warning(
            "attach_scope_labels stub: rendering raw scope_value labels until "
            "node_id resolution port lands"
        )
