"""Mastery heatmap data builder — T14 stub.

The PoC's mastery rollups joined `Section`/`FoundationalConcept`/
`ContentCategory`/`Topic` + the 3-target `question_tags` + Anki state /
retention helpers. All four outline tables are dropped (T1), the 3-target
columns are dropped (T2), and the Anki subtree helpers are stubbed in T13.

Restoring the heatmap + drilldown headers needs:
  - node_id subtree rollup via `app.services.outline_subtree.subtree_node_ids`,
  - the anki state / retention helpers ported onto node_id (T13 follow-up),
  - a domain-pack flag for the "is_cars" discriminator (no AAMC section codes).

This stub keeps the public surface so the dashboard routes load and
templates render empty data instead of 500-ing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession  # noqa: F401 — kept for signature

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# View-model dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HeatmapCell:
    cc_id: int
    code: str
    name: str
    label: str
    section_code: str
    section_name: str
    is_cars: bool
    attempts: int
    accuracy: float
    wilson_lower: float
    color_bucket: str
    is_low_signal: bool
    arrow: str | None
    unlock_pct: float | None
    retention_30d: float | None


@dataclass(frozen=True)
class CCHeader:
    cc_code: str
    is_cars: bool
    attempts: int
    correct: int
    accuracy: float
    wilson_lower: float
    unlock_pct: float | None
    retention_30d: float | None
    effective_mastery: float | None


@dataclass(frozen=True)
class StateBreakdown:
    total_cards: int
    assigned: int
    suspended: int
    new: int
    learning: int
    young: int
    mature: int
    unlock_pct: float | None
    retention_7d: float | None
    retention_30d: float | None
    retention_all: float | None


@dataclass(frozen=True)
class TopicTreeRow:
    topic_id: int
    name: str
    depth: int
    has_children: bool
    drilldown_url: str
    attempts: int
    correct: int
    accuracy: float
    wilson_lower: float
    arrow: str | None
    unlock_pct: float | None
    retention_30d: float | None
    due_count: int


@dataclass(frozen=True)
class TopicHeader:
    cc_code: str
    topic_id: int
    topic_name: str
    attempts: int
    correct: int
    accuracy: float
    wilson_lower: float
    unlock_pct: float | None
    retention_30d: float | None
    effective_mastery: float | None


@dataclass(frozen=True)
class BreadcrumbItem:
    label: str
    href: str


def _empty_state_breakdown() -> StateBreakdown:
    return StateBreakdown(
        total_cards=0,
        assigned=0,
        suspended=0,
        new=0,
        learning=0,
        young=0,
        mature=0,
        unlock_pct=None,
        retention_7d=None,
        retention_30d=None,
        retention_all=None,
    )


# --------------------------------------------------------------------------- #
# Public surface — stubs until node_id rollup port lands.
# --------------------------------------------------------------------------- #


async def build_heatmap(session: AsyncSession) -> dict[str, list[HeatmapCell]]:
    """Stub — TODO(T14 follow-up): rebuild on node_id subtree rollup."""
    logger.warning("build_heatmap stub: empty pending node_id port")
    return {}


async def cc_header(session: AsyncSession, *, cc_code: str) -> CCHeader:
    logger.warning("cc_header stub: empty pending node_id port")
    return CCHeader(
        cc_code=cc_code,
        is_cars=False,
        attempts=0,
        correct=0,
        accuracy=0.0,
        wilson_lower=0.0,
        unlock_pct=None,
        retention_30d=None,
        effective_mastery=None,
    )


async def cc_anki_overview(session: AsyncSession, *, cc_code: str) -> StateBreakdown:
    logger.warning("cc_anki_overview stub: empty pending node_id port")
    return _empty_state_breakdown()


async def cc_topics_tree(
    session: AsyncSession, *, cc_id: int, cc_code: str
) -> list[TopicTreeRow]:
    logger.warning("cc_topics_tree stub: empty pending node_id port")
    return []


async def topic_header(session: AsyncSession, *, cc_code: str, topic: Any) -> TopicHeader:
    """`topic` is the OutlineNode-or-equivalent — kept Any for the stub period."""
    logger.warning("topic_header stub: empty pending node_id port")
    return TopicHeader(
        cc_code=cc_code,
        topic_id=getattr(topic, "id", 0),
        topic_name=getattr(topic, "name", ""),
        attempts=0,
        correct=0,
        accuracy=0.0,
        wilson_lower=0.0,
        unlock_pct=None,
        retention_30d=None,
        effective_mastery=None,
    )


async def topic_anki_overview(session: AsyncSession, *, topic_id: int) -> StateBreakdown:
    logger.warning("topic_anki_overview stub: empty pending node_id port")
    return _empty_state_breakdown()


async def topic_children_tree(
    session: AsyncSession, *, cc_code: str, root_topic_id: int
) -> list[TopicTreeRow]:
    logger.warning("topic_children_tree stub: empty pending node_id port")
    return []


async def validate_topic_chain(
    session: AsyncSession, *, cc_code: str, ids: list[int]
) -> tuple[list[Any], list[BreadcrumbItem]] | None:
    """Stub — TODO(T14 follow-up): port the §V32 id-path chain check onto
    `outline_nodes.parent_id`."""
    logger.warning("validate_topic_chain stub: returns None pending node_id port")
    return None
