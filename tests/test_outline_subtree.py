"""T14 — shared subtree-set rollup helper (V-O1)."""

import os

import asyncpg
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.database import Base
from app.models.outline import Course, OutlineNode
from app.services.outline_subtree import subtree_node_ids, subtree_node_ids_many

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


async def _seed(eng) -> dict[str, int]:
    async with AsyncSession(eng) as s:
        c = Course(slug="aamc", name="AAMC")
        s.add(c); await s.flush()
        def mk(parent_id, kind, name, depth, pos):
            n = OutlineNode(course_id=c.id, parent_id=parent_id, kind=kind, name=name, depth=depth, position=pos)
            s.add(n); return n
        sec = mk(None, "section", "CP", 0, 0); await s.flush()
        fc1 = mk(sec.id, "fc", "FC1", 1, 0); await s.flush()
        cc1 = mk(fc1.id, "cc", "1A", 2, 0); await s.flush()
        t1 = mk(cc1.id, "topic", "Amino acids", 3, 0)
        t2 = mk(cc1.id, "topic", "Proteins", 3, 1); await s.flush()
        fc2 = mk(sec.id, "fc", "FC2", 1, 1); await s.flush()
        ids = {"sec": sec.id, "fc1": fc1.id, "cc1": cc1.id, "t1": t1.id, "t2": t2.id, "fc2": fc2.id}
        await s.commit()
        return ids


async def test_subtree_node_ids_is_union_of_descendants_and_self(engine):
    ids = await _seed(engine)
    async with AsyncSession(engine) as s:
        assert await subtree_node_ids(s, ids["fc1"]) == {ids["fc1"], ids["cc1"], ids["t1"], ids["t2"]}
        assert await subtree_node_ids(s, ids["sec"]) == set(ids.values())
        assert await subtree_node_ids(s, ids["t1"]) == {ids["t1"]}


async def test_subtree_node_ids_unknown_root_returns_empty(engine):
    await _seed(engine)
    async with AsyncSession(engine) as s:
        assert await subtree_node_ids(s, 999999) == set()


async def test_subtree_node_ids_many_unions(engine):
    ids = await _seed(engine)
    async with AsyncSession(engine) as s:
        result = await subtree_node_ids_many(s, [ids["t1"], ids["fc2"]])
    assert result == {ids["t1"], ids["fc2"]}
    async with AsyncSession(engine) as s:
        assert await subtree_node_ids_many(s, []) == set()
