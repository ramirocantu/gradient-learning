"""Startup helpers invoked from app lifespan and standalone entrypoints.

Auto-seeding the AAMC outline on every boot keeps the reference tables in sync
with backend/app/seeds/aamc_outline.json. The seed itself is upsert-based and
idempotent (see scripts/seed_outline.py), so re-runs on already-seeded databases
no-op semantically.
"""

from __future__ import annotations

import logging
from typing import Callable

from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AsyncSessionLocal
from scripts.seed_outline import SeedReport, seed

logger = logging.getLogger(__name__)


async def ensure_outline_seeded(
    session_factory: Callable[[], AsyncSession] = AsyncSessionLocal,
) -> SeedReport:
    """Upsert the AAMC outline. Safe to call on every entrypoint init.

    Steady-state cost on a populated DB: ~1554 row-level UPSERTs, ~1-2s wall
    time on local Postgres. The seed commits its own transaction.

    `session_factory` is overridable so tests can target the test DB without
    touching the production `DATABASE_URL`-bound `AsyncSessionLocal`.
    """
    async with session_factory() as session:
        report = await seed(session)
    logger.info(
        "ensure_outline_seeded: sections=%d fcs=%d ccs=%d topics=%d max_depth=%d",
        report.sections_upserted,
        report.fcs_upserted,
        report.ccs_upserted,
        report.topics_upserted,
        report.max_depth_observed,
    )
    return report
