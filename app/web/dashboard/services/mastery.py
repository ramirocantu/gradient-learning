"""Mastery heatmap data builder — FENCED (T17, V-RB1, V-O5).

The PoC's mastery rollups joined `Section`/`FoundationalConcept`/
`ContentCategory`/`Topic` + the 3-target `question_tags` + Anki state /
retention helpers. All four outline tables are dropped (T1) and the
3-target columns are dropped (T2).

The Jinja mastery / drilldown / topics / recommendations / insights
dashboard surface is not on the PKM critical loop and is gated by the
T34 SPA reassessment (§P, §C frontend-stack carve-out). Until then this
module is FENCED:

  - the dashboard `mastery`, `topics`, `recommendations`, `insights`
    routes are unmounted in `app/web/dashboard/main.py`,
  - all public functions return empty / placeholder values so any
    direct import does not crash,
  - related tests are collect-ignored in `tests/conftest.py`.

This file is FENCED, not a stub: behavior is deliberate, not in-progress.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession  # noqa: F401 — kept for signature

logger = logging.getLogger(__name__)


_FENCED_MSG = (
    "dashboard.services.mastery is FENCED (T17, V-RB1) — dashboard routes "
    "unmounted; restoration tied to T34 SPA reassessment"
)


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
# Public surface — FENCED placeholders.
# --------------------------------------------------------------------------- #


async def build_heatmap(session: AsyncSession) -> dict[str, list[HeatmapCell]]:
    """FENCED — returns empty mapping."""
    logger.warning(_FENCED_MSG)
    return {}


async def cc_header(session: AsyncSession, *, cc_code: str) -> CCHeader:
    """FENCED — returns zeroed header."""
    logger.warning(_FENCED_MSG)
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
    """FENCED — returns empty StateBreakdown."""
    logger.warning(_FENCED_MSG)
    return _empty_state_breakdown()


async def cc_topics_tree(
    session: AsyncSession, *, cc_id: int, cc_code: str
) -> list[TopicTreeRow]:
    """FENCED — returns empty list."""
    logger.warning(_FENCED_MSG)
    return []


async def topic_header(session: AsyncSession, *, cc_code: str, topic: Any) -> TopicHeader:
    """FENCED — returns zeroed header."""
    logger.warning(_FENCED_MSG)
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
    """FENCED — returns empty StateBreakdown."""
    logger.warning(_FENCED_MSG)
    return _empty_state_breakdown()


async def topic_children_tree(
    session: AsyncSession, *, cc_code: str, root_topic_id: int
) -> list[TopicTreeRow]:
    """FENCED — returns empty list."""
    logger.warning(_FENCED_MSG)
    return []


async def validate_topic_chain(
    session: AsyncSession, *, cc_code: str, ids: list[int]
) -> tuple[list[Any], list[BreadcrumbItem]] | None:
    """FENCED — returns None."""
    logger.warning(_FENCED_MSG)
    return None
