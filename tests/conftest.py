import os
from collections.abc import AsyncIterator
from pathlib import Path

import asyncpg
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.models  # noqa: F401  — registers all models on Base.metadata
from app.database import Base


# T36 (V-RB5): the previous collect_ignore_glob workaround for
# tests/test_eval_script.py was removed once T36 deleted the test
# along with the 7 stale anthropic harness scripts in scripts/.
# New collect-ignores should re-introduce this list only when a
# FENCED-but-undeletable surface appears.

_HOST_PORT = os.environ.get("HOST_POSTGRES_PORT", "5432")
TEST_DB_URL = f"postgresql+asyncpg://gradient:gradient_secret@localhost:{_HOST_PORT}/gradient_test"
_ADMIN_DSN = f"postgresql://gradient:gradient_secret@localhost:{_HOST_PORT}/gradient"


@pytest.fixture(scope="session")
def test_media_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Shared session-scoped tmp dir for media-touching tests.

    Patches ``settings.MEDIA_ROOT`` to point at the tmp dir so the media
    handler reads from the per-session location instead of production.
    """
    path = tmp_path_factory.mktemp("test_media")
    os.environ["MEDIA_ROOT"] = str(path)
    from app.config import settings

    settings.MEDIA_ROOT = path
    return path


@pytest.fixture(scope="session")
async def test_engine(test_media_root):
    conn = await asyncpg.connect(_ADMIN_DSN)
    try:
        await conn.execute("CREATE DATABASE gradient_test")
    except asyncpg.exceptions.DuplicateDatabaseError:
        pass
    finally:
        await conn.close()

    engine = create_async_engine(TEST_DB_URL)
    async with engine.begin() as conn:
        await conn.execute(text("DROP SCHEMA public CASCADE"))
        await conn.execute(text("CREATE SCHEMA public"))
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture(scope="session")
async def seeded_report(test_engine):
    # No-op: outline seeding is not a startup/seed step (V-O6). The AAMC
    # outline is an uploaded schema materialized via
    # POST /api/v1/courses/{id}/outline:import. Kept as a session-scoped
    # fixture so existing dependents order after `test_engine`.
    return None


@pytest.fixture
async def _test_connection(seeded_report, test_engine):
    """One connection + outer transaction per test, rolled back on teardown.

    All ``db_session``/``session`` and the ``client``-overridden ``get_session``
    bind to this single connection. Inner ``session.commit()`` calls become
    nested-savepoint releases (via ``join_transaction_mode="create_savepoint"``),
    so test rows are visible across sessions for the duration of the test but
    are wiped when the outer transaction rolls back. This removes cross-test
    pollution without truncating tables.

    Idempotent ``create_all`` up front patches pre-T22 tech debt: several
    test modules (``test_outline_lookup``, ``test_outline_subtree``,
    ``test_outline_nodes``, ``test_node_id_tags``, ``test_anki_queries_smoke``,
    ``test_source_adapter``) open their own engine, ``DROP SCHEMA public
    CASCADE``, and rebuild only a subset of tables — leaving the shared
    ``gradient_test`` DB with missing tables for any later test that goes
    through this fixture. Re-running ``create_all`` (checkfirst=True
    default) restores anything they dropped without disturbing tests that
    intentionally rebuild from scratch.
    """
    async with test_engine.begin() as repair_conn:
        await repair_conn.run_sync(Base.metadata.create_all)

    async with test_engine.connect() as conn:
        outer_trans = await conn.begin()
        try:
            yield conn
        finally:
            if outer_trans.is_active:
                await outer_trans.rollback()


@pytest.fixture
async def db_session(_test_connection):
    Sm = async_sessionmaker(
        bind=_test_connection,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )
    async with Sm() as session:
        yield session


@pytest.fixture
async def session(db_session: AsyncSession) -> AsyncSession:
    """Alias for ``db_session``.

    Some dashboard-suite tests import ``session`` as the fixture name; new
    code should prefer ``db_session``. Both yield the same session.
    """
    return db_session


@pytest.fixture
async def client(_test_connection) -> AsyncIterator[AsyncClient]:
    """Unified HTTP client targeting ``app/main.py:app``.

    The app is backend-only — every route lives on the main app
    (``/api/v1/*``, ``/healthz``, ``/media/*``). Overrides ``get_session``
    to bind to the per-test connection — so commits inside route handlers
    become savepoint releases and roll back with the outer transaction.
    """
    from app.api.deps import get_session as api_get_session
    from app.main import app

    Sm = async_sessionmaker(
        bind=_test_connection,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )

    async def override_get_session():
        # Mirror production `get_session`: commit on clean exit so a
        # POST's INSERT releases its savepoint into the outer test
        # transaction (visible to subsequent requests in the same test);
        # rollback on exception so a 4xx/5xx does not persist garbage.
        # Outer-transaction rollback in `_test_connection` still owns
        # cross-test cleanup.
        async with Sm() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[api_get_session] = override_get_session

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
    finally:
        app.dependency_overrides.pop(api_get_session, None)
