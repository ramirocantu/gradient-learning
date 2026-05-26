"""T1 — courses + outline_nodes schema (V-O1 sole hierarchy, V-O4 delimiter).

Self-contained: builds only the two new tables on their own engine so the
suite-wide ``create_all`` (still referencing pre-T2 topic FKs) is not needed.
"""

import os

import asyncpg
import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.models.outline import OUTLINE_PATH_DELIMITER, Course, OutlineNode

_HOST_PORT = os.environ.get("HOST_POSTGRES_PORT", "5432")
_DB_URL = f"postgresql+asyncpg://mcat:mcat_secret@localhost:{_HOST_PORT}/mcat_coach_test"
_ADMIN_DSN = f"postgresql://mcat:mcat_secret@localhost:{_HOST_PORT}/mcat_coach"


@pytest.fixture
async def engine():
    conn = await asyncpg.connect(_ADMIN_DSN)
    try:
        await conn.execute("CREATE DATABASE mcat_coach_test")
    except asyncpg.exceptions.DuplicateDatabaseError:
        pass
    finally:
        await conn.close()

    eng = create_async_engine(_DB_URL)
    # Clean slate (cf conftest): the shared test DB may carry tables from a
    # prior full create_all whose FKs block a subset drop.
    async with eng.begin() as c:
        await c.execute(text("DROP SCHEMA public CASCADE"))
        await c.execute(text("CREATE SCHEMA public"))
        await c.run_sync(Course.__table__.create)
        await c.run_sync(OutlineNode.__table__.create)
    yield eng
    await eng.dispose()


async def _subtree_ids(session: AsyncSession, root_id: int) -> set[int]:
    """V-O1 rollup at the schema level: a node's subtree = itself + all descendants."""
    rows = await session.execute(
        text(
            """
            WITH RECURSIVE sub AS (
                SELECT id FROM outline_nodes WHERE id = :root
                UNION ALL
                SELECT n.id FROM outline_nodes n JOIN sub ON n.parent_id = sub.id
            )
            SELECT id FROM sub
            """
        ),
        {"root": root_id},
    )
    return {r[0] for r in rows}


def test_path_delimiter_reserved_and_ascii():
    # V-O4: ` >> ` reserved, ASCII; never `/`, `-`, `.`, `,`.
    assert OUTLINE_PATH_DELIMITER == " >> "
    assert OUTLINE_PATH_DELIMITER.isascii()
    for bad in ("/", "-", ".", ","):
        assert bad not in OUTLINE_PATH_DELIMITER


async def test_arbitrary_depth_tree_and_subtree_rollup(engine):
    # V-O1: AAMC as kinds on a four-deep instance; rollup = union of
    # descendants + self, sibling branches excluded.
    async with AsyncSession(engine) as s:
        course = Course(slug="aamc", name="AAMC MCAT")
        s.add(course)
        await s.flush()

        sec = OutlineNode(
            course_id=course.id, parent_id=None, kind="section", name="CP", depth=0, position=0
        )
        s.add(sec)
        await s.flush()
        fc = OutlineNode(
            course_id=course.id, parent_id=sec.id, kind="fc", name="FC1", depth=1, position=0
        )
        s.add(fc)
        await s.flush()
        cc = OutlineNode(
            course_id=course.id, parent_id=fc.id, kind="cc", name="1A", depth=2, position=0
        )
        s.add(cc)
        await s.flush()
        t1 = OutlineNode(
            course_id=course.id, parent_id=cc.id, kind="topic", name="Amino acids", depth=3, position=0
        )
        t2 = OutlineNode(
            course_id=course.id, parent_id=cc.id, kind="topic", name="Proteins", depth=3, position=1
        )
        s.add_all([t1, t2])
        await s.flush()
        # Sibling branch that must be EXCLUDED from fc's subtree.
        other_fc = OutlineNode(
            course_id=course.id, parent_id=sec.id, kind="fc", name="FC2", depth=1, position=1
        )
        s.add(other_fc)
        await s.flush()

        assert await _subtree_ids(s, fc.id) == {fc.id, cc.id, t1.id, t2.id}
        assert other_fc.id not in await _subtree_ids(s, fc.id)
        assert await _subtree_ids(s, t1.id) == {t1.id}  # leaf = only itself
        assert await _subtree_ids(s, sec.id) == {
            sec.id,
            fc.id,
            cc.id,
            t1.id,
            t2.id,
            other_fc.id,
        }


async def test_unique_sibling_name_per_parent(engine):
    # §I UQ(course_id, parent_id, name): duplicate sibling name rejected.
    async with AsyncSession(engine) as s:
        course = Course(slug="uq", name="UQ")
        s.add(course)
        await s.flush()
        root = OutlineNode(
            course_id=course.id, parent_id=None, kind="section", name="Root", depth=0, position=0
        )
        s.add(root)
        await s.flush()
        s.add(
            OutlineNode(
                course_id=course.id, parent_id=root.id, kind="cc", name="Dup", depth=1, position=0
            )
        )
        await s.flush()
        s.add(
            OutlineNode(
                course_id=course.id, parent_id=root.id, kind="cc", name="Dup", depth=1, position=1
            )
        )
        with pytest.raises(IntegrityError):
            await s.flush()


async def test_deleting_course_cascades_nodes(engine):
    # V-O1 integrity: outline_nodes is owned by its course; delete cascades.
    async with AsyncSession(engine) as s:
        course = Course(slug="cascade", name="Cascade")
        s.add(course)
        await s.flush()
        root = OutlineNode(
            course_id=course.id, parent_id=None, kind="section", name="R", depth=0, position=0
        )
        s.add(root)
        await s.flush()
        child = OutlineNode(
            course_id=course.id, parent_id=root.id, kind="cc", name="C", depth=1, position=0
        )
        s.add(child)
        await s.flush()

        await s.delete(course)
        await s.flush()

        remaining = (await s.execute(select(OutlineNode.id))).scalars().all()
        assert remaining == []
