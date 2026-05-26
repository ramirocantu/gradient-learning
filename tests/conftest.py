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


# T20 (V-RB4): the previous collect_ignore_glob covered legacy-schema
# test modules left over from T17/T18 fences. Those files have all been
# deleted, so no glob is needed. New collect-ignores should re-introduce
# this list only when a FENCED-but-undeletable surface appears.

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
    from scripts.seed_outline import seed

    async with AsyncSession(test_engine) as session:
        report = await seed(session)
    return report


@pytest.fixture
async def _test_connection(seeded_report, test_engine):
    """One connection + outer transaction per test, rolled back on teardown.

    All ``db_session``/``session`` and the ``client``-overridden ``get_session``
    bind to this single connection. Inner ``session.commit()`` calls become
    nested-savepoint releases (via ``join_transaction_mode="create_savepoint"``),
    so test rows are visible across sessions for the duration of the test but
    are wiped when the outer transaction rolls back. This removes cross-test
    pollution without truncating tables.
    """
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
    """Unified HTTP client targeting ``backend/app/main.py:app``.

    Routes through the real mount paths:
      - dashboard tests hit ``/``
      - viewer tests hit ``/viewer/*``
      - API tests hit ``/api/v1/*``

    Overrides ``get_session`` on the parent app and on both sub-apps to
    bind to the per-test connection — so commits inside route handlers
    become savepoint releases and roll back with the outer transaction.
    """
    from app.api.deps import get_session as api_get_session
    from app.main import app
    from app.web.dashboard.db import get_session as dashboard_get_session
    from app.web.dashboard.main import app as dashboard_app
    from app.web.viewer.db import get_session as viewer_get_session
    from app.web.viewer.main import app as viewer_app

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
    dashboard_app.dependency_overrides[dashboard_get_session] = override_get_session
    viewer_app.dependency_overrides[viewer_get_session] = override_get_session

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
    finally:
        app.dependency_overrides.pop(api_get_session, None)
        dashboard_app.dependency_overrides.pop(dashboard_get_session, None)
        viewer_app.dependency_overrides.pop(viewer_get_session, None)
