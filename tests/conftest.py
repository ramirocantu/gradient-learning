import os

# T40 / V-TC1: isolate test config from a developer's .env BEFORE any
# app.config import builds the `settings` singleton. GRADIENT_DISABLE_DOTENV
# makes Settings skip its env_file (see app/config.py), so:
#   - COACH_TOKEN falls back to its "change_me_before_use" default (the value
#     the auth tests hardcode), instead of a real .env token → no spurious 401s;
#   - NOTION_API_TOKEN / OPENAI_API_KEY stay unset → /admin/status probes report
#     unconfigured and never hit a live API (V16);
#   - DATABASE_URL is pinned to the test DB here (the field has no default).
# Popping the others guards against a shell that exports them.
os.environ["GRADIENT_DISABLE_DOTENV"] = "1"
for _leak in ("COACH_TOKEN", "NOTION_API_TOKEN", "NOTION_WIKI_DB_ID", "OPENAI_API_KEY"):
    os.environ.pop(_leak, None)
_HOST_PORT = os.environ.get("HOST_POSTGRES_PORT", "5432")
os.environ["DATABASE_URL"] = (
    f"postgresql+asyncpg://gradient:gradient_secret@localhost:{_HOST_PORT}/gradient_test"
)

import json
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

# _HOST_PORT already resolved at module top (before the app imports above).
TEST_DB_URL = f"postgresql+asyncpg://gradient:gradient_secret@localhost:{_HOST_PORT}/gradient_test"
_ADMIN_DSN = f"postgresql://gradient:gradient_secret@localhost:{_HOST_PORT}/gradient"
_AAMC_SEED_SCHEMA = (
    Path(__file__).resolve().parent.parent / "app" / "seeds" / "aamc_outline.schema.json"
)


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
async def seed_aamc_outline(db_session: AsyncSession):
    """Materialize the bundled AAMC outline (course slug ``aamc``) into the
    test DB so ``OutlineLookup.load`` resolves.

    V-O6: this is the *explicit* validate→materialize import path (the same
    one the upload route drives), invoked by a fixture — NOT a revived
    implicit startup seed. Function-scoped + flushed into the per-test
    savepoint, so it survives the subset-rebuild tests that DROP SCHEMA and
    rolls back with the test like everything else.
    """
    from app.services.outline import materialize_outline, validate_outline_schema

    payload = json.loads(_AAMC_SEED_SCHEMA.read_text())
    validated = validate_outline_schema(payload)
    await materialize_outline(db_session, validated)
    await db_session.flush()
    return validated


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


# --------------------------------------------------------------------------- #
# RCA-10 §T1 — shared workflow-test fixtures.
#
# Reusable building blocks for the E2E workflow tests (Capture→Attempt,
# PDF→Grounded Tag, Outline→Mastery). Each is a *factory* (V3): a test calls
# it to seed its own course / payload, so no state leaks across tests and the
# per-test savepoint rolls everything back.
# --------------------------------------------------------------------------- #


@pytest.fixture
def coach_headers() -> dict[str, str]:
    """The `X-Coach-Token` header that unlocks gated `/api/v1/*` routes.

    Reads the test-config default (`change_me_before_use`) — conftest pops a
    real `COACH_TOKEN` from env before the settings singleton builds, so this
    matches what `verify_coach_token` checks against in the suite.
    """
    from app.config import settings

    return {"X-Coach-Token": settings.COACH_TOKEN}


@pytest.fixture
def make_course(db_session: AsyncSession):
    """Factory → insert + flush a `Course`, returning the persisted row.

    Capture and PDF workflows need a course to attach to; outline import
    creates its own via the route. Flushed into the per-test savepoint (V3).
    """

    async def _make(
        slug: str = "biochem", name: str | None = None, description: str | None = None
    ):
        from app.models.outline import Course

        course = Course(slug=slug, name=name or slug.title(), description=description)
        db_session.add(course)
        await db_session.flush()
        return course

    return _make


@pytest.fixture
def uworld_capture_payload():
    """Factory → a JSON-serializable uworld `CapturePayload` body for
    `POST /api/v1/captures`. Override any field via kwargs.

    Builds through the strict `CapturePayload`/`ParsedCapture` schemas first,
    so a drifted schema fails *here* (in the fixture) rather than deep inside
    the route. Returns `model_dump(mode="json")` — the wire dict the client posts.
    """

    def _make(**over):
        from datetime import datetime, timezone

        from app.schemas.captures import CapturePayload, ChoiceItem, ParsedCapture

        parsed = over.pop(
            "parsed",
            ParsedCapture(
                stem_html="<p>stem</p>",
                stem_plain="stem",
                choices=[
                    ChoiceItem(key="A", html="<p>a</p>", plain="a"),
                    ChoiceItem(key="B", html="<p>b</p>", plain="b"),
                ],
                correct_choice="A",
                selected_choice="A",
                is_correct=True,
            ),
        )
        payload = CapturePayload(
            source=over.pop("source", "uworld"),
            course_slug=over.pop("course_slug", None),
            qid=over.pop("qid", "q-fixture-1"),
            captured_at=over.pop("captured_at", datetime.now(timezone.utc)),
            html=over.pop("html", "<p>raw</p>"),
            parsed=parsed,
            extension_version=over.pop("extension_version", "0.1.0"),
            **over,
        )
        return payload.model_dump(mode="json")

    return _make


@pytest.fixture
def fake_renderer():
    """Factory → an injectable `Renderer` (`Path -> list[RenderedPage]`) that
    yields `pages` stub PNG pages.

    PDF-ingest (`ingest_pdf(renderer=...)`) is mock-friendly: this lets a test
    skip PyMuPDF and a real PDF file entirely (V2 — vision is mocked, so the
    page bytes are never decoded for real).
    """

    def _make(pages: int = 1):
        from app.services.kb.pdf_ingest import RenderedPage

        def _render(_path):
            return [
                RenderedPage(page=i, image_png=b"\x89PNG-stub")
                for i in range(1, pages + 1)
            ]

        return _render

    return _make
