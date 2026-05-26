"""Auto-seed behavior for app.startup.ensure_outline_seeded."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import async_sessionmaker

from app.startup import ensure_outline_seeded


async def test_ensure_outline_seeded_returns_expected_counts(test_engine):
    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    report = await ensure_outline_seeded(session_factory=factory)
    assert report.sections_upserted == 4
    assert report.ccs_upserted >= 10
    assert report.topics_upserted > 100


async def test_ensure_outline_seeded_idempotent_on_repeat(test_engine):
    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    r1 = await ensure_outline_seeded(session_factory=factory)
    r2 = await ensure_outline_seeded(session_factory=factory)
    assert r1.sections_upserted == r2.sections_upserted
    assert r1.ccs_upserted == r2.ccs_upserted
    assert r1.topics_upserted == r2.topics_upserted
