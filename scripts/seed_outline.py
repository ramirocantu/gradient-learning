"""AAMC outline seed — T14 stub.

Original PoC seeded the four AAMC tables (sections/fcs/ccs/topics) from
`app/seeds/aamc_outline.json`. Those tables are gone (T1); the AAMC outline
is now an uploaded schema materialized into `outline_nodes` (V-O3) — that
import path lands as T9 (`POST /api/v1/courses/{id}/outline:import` + the
shipped `seeds/aamc_outline.schema.json`).

Stub keeps `SeedReport` + `seed()` callable so `app.startup.ensure_outline_seeded`
imports without breaking the FastAPI startup chain. `seed()` returns an
empty report and a startup-time WARN.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession  # noqa: F401 — kept for signature

logger = logging.getLogger(__name__)


@dataclass
class SeedReport:
    sections_upserted: int = 0
    fcs_upserted: int = 0
    ccs_upserted: int = 0
    topics_upserted: int = 0
    max_depth_observed: int = 0


async def seed(session: AsyncSession) -> SeedReport:
    """Stub — TODO(T9): replace with outline-schema import + materialize."""
    logger.warning(
        "scripts.seed_outline.seed stub: no-op pending T9 outline-schema importer; "
        "boot will continue with an empty outline until a course is created via "
        "POST /api/v1/courses + POST /api/v1/courses/{id}/outline:import"
    )
    return SeedReport()
