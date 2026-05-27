"""T12 — OutlineLookup resolves node by `>>` path (V-O4) over outline_nodes."""

import os

import asyncpg
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.database import Base
from app.models.outline import Course, OutlineNode
from app.services.outline.lookup import (
    OutlineLookup,
    OutlineNotSeededError,
)

_HOST_PORT = os.environ.get("HOST_POSTGRES_PORT", "5432")
_DB_URL = f"postgresql+asyncpg://gradient:gradient_secret@localhost:{_HOST_PORT}/gradient_test"
_ADMIN_DSN = f"postgresql://gradient:gradient_secret@localhost:{_HOST_PORT}/gradient"

_TABLES = [Course.__table__, OutlineNode.__table__]


@pytest.fixture
async def engine():
    conn = await asyncpg.connect(_ADMIN_DSN)
    try:
        await conn.execute("CREATE DATABASE gradient_test")
    except asyncpg.exceptions.DuplicateDatabaseError:
        pass
    finally:
        await conn.close()

    eng = create_async_engine(_DB_URL)
    async with eng.begin() as c:
        await c.execute(text("DROP SCHEMA public CASCADE"))
        await c.execute(text("CREATE SCHEMA public"))
        await c.run_sync(Base.metadata.create_all, tables=_TABLES)
    yield eng
    await eng.dispose()


async def _seed_aamc_shape(eng) -> dict[str, int]:
    """4-deep AAMC instance: section >> fc >> cc >> topic."""
    async with AsyncSession(eng) as s:
        c = Course(slug="aamc", name="AAMC MCAT")
        s.add(c)
        await s.flush()

        def mk(parent_id, kind, name, depth, pos):
            n = OutlineNode(
                course_id=c.id, parent_id=parent_id, kind=kind, name=name,
                depth=depth, position=pos,
            )
            s.add(n)
            return n

        sec = mk(None, "section", "CP", 0, 0); await s.flush()
        fc1 = mk(sec.id, "fc", "FC1", 1, 0); await s.flush()
        cc1a = mk(fc1.id, "cc", "1A", 2, 0); await s.flush()
        t_amino = mk(cc1a.id, "topic", "Amino acids", 3, 0)
        t_protein = mk(cc1a.id, "topic", "Proteins", 3, 1); await s.flush()
        fc2 = mk(sec.id, "fc", "FC2", 1, 1); await s.flush()
        ids = {
            "sec": sec.id, "fc1": fc1.id, "cc1a": cc1a.id,
            "amino": t_amino.id, "proteins": t_protein.id, "fc2": fc2.id,
        }
        await s.commit()
        return ids


async def test_load_raises_when_course_missing(engine):
    async with AsyncSession(engine) as s:
        with pytest.raises(OutlineNotSeededError):
            await OutlineLookup.load(s, course_slug="aamc")


async def test_load_raises_when_course_has_no_nodes(engine):
    async with AsyncSession(engine) as s:
        s.add(Course(slug="aamc", name="AAMC"))
        await s.commit()
    async with AsyncSession(engine) as s:
        with pytest.raises(OutlineNotSeededError):
            await OutlineLookup.load(s, course_slug="aamc")


async def test_node_id_by_path_resolves_each_depth(engine):
    ids = await _seed_aamc_shape(engine)
    async with AsyncSession(engine) as s:
        lk = await OutlineLookup.load(s, course_slug="aamc")
    assert lk.node_id_by_path("CP") == ids["sec"]
    assert lk.node_id_by_path("CP >> FC1") == ids["fc1"]
    assert lk.node_id_by_path("CP >> FC1 >> 1A") == ids["cc1a"]
    assert lk.node_id_by_path("CP >> FC1 >> 1A >> Amino acids") == ids["amino"]
    assert lk.node_id_by_path("CP >> FC1 >> 1A >> Proteins") == ids["proteins"]
    assert lk.node_id_by_path("CP >> FC2") == ids["fc2"]


async def test_node_id_by_path_missing_segment_returns_none(engine):
    await _seed_aamc_shape(engine)
    async with AsyncSession(engine) as s:
        lk = await OutlineLookup.load(s, course_slug="aamc")
    assert lk.node_id_by_path("CP >> FC1 >> 1A >> Nope") is None
    assert lk.node_id_by_path("NoSuchSection") is None
    # Wrong parent — "Proteins" lives under 1A, not under FC1 directly.
    assert lk.node_id_by_path("CP >> FC1 >> Proteins") is None


async def test_path_of_round_trips(engine):
    ids = await _seed_aamc_shape(engine)
    async with AsyncSession(engine) as s:
        lk = await OutlineLookup.load(s, course_slug="aamc")
    assert lk.path_of(ids["amino"]) == "CP >> FC1 >> 1A >> Amino acids"
    assert lk.path_of(ids["sec"]) == "CP"
    assert lk.path_of(99999) is None


def test_malformed_path_returns_none():
    lk = OutlineLookup(course_id=1, nodes=[])
    assert lk.node_id_by_path("") is None
    assert lk.node_id_by_path("   ") is None
