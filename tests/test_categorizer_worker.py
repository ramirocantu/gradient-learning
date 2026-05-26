"""Tests for the categorization orchestrator + worker + endpoints (Ticket 3.3).

Hits the real test DB (per-test outer transaction rolls back). The Anthropic
SDK is mocked at the boundary — `AsyncAnthropic().messages.create` is patched
to return a forged Message object. No real API calls.

Convention: any test whose assertions depend on per-token pricing (cost math,
budget caps) MUST pin `settings.CATEGORIZER_MODEL` via monkeypatch. The
production default may change (it has — 3.5 flipped Sonnet → Haiku); pinning
keeps these tests true regardless of what the runtime default is.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session
from app.api.v1.admin import _anthropic_client, _categorizer_cache
from app.config import settings
from app.main import app
from app.models.captures import Question, QuestionTag
from app.models.outline import ContentCategory, Topic
from app.services.categorizer import (
    QuestionNotFoundError,
    tag_question,
)
from app.services.categorizer import llm
from app.services.categorizer.cache import CategorizerCache
from app.services.categorizer.outline_lookup import OutlineLookup
from scripts import run_categorizer


# --------------------------------------------------------------------------- #
# Fixtures + helpers
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _reset_llm_cache():
    """Compat shim — persistent cache is per-test (tmp file), no global to clear."""
    llm._clear_cache_for_tests()
    yield
    llm._clear_cache_for_tests()


def _tool_block(**input_data):
    from anthropic.types import ToolUseBlock

    return ToolUseBlock(
        id="toolu_x",
        name="submit_aamc_categorization",
        input=input_data,
        type="tool_use",
    )


def _forge_message(tags: list[dict], section: str = "CP"):
    content = [
        _tool_block(primary_aamc_section=section, tags=tags),
    ]
    usage = SimpleNamespace(
        input_tokens=1000,
        output_tokens=100,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    return SimpleNamespace(content=content, usage=usage)


def _make_client(message_or_factory):
    """Build a fake AsyncAnthropic. message_or_factory may be a single Message
    or a callable taking no args and returning a Message (for per-call variation)."""
    client = MagicMock()
    client.messages = MagicMock()
    if callable(message_or_factory):
        client.messages.create = AsyncMock(side_effect=lambda **_kw: message_or_factory())
    else:
        client.messages.create = AsyncMock(return_value=message_or_factory)
    return client


@pytest.fixture
async def env(seeded_report, test_engine, tmp_path):
    """ASGI client + session factory + per-test CategorizerCache (tmp file).

    Overrides `_anthropic_client` and `_categorizer_cache` so endpoint tests
    use controllable mocks and an isolated SQLite file.
    """
    conn = await test_engine.connect()
    await conn.begin()

    def make_session() -> AsyncSession:
        return AsyncSession(
            bind=conn,
            expire_on_commit=False,
            join_transaction_mode="create_savepoint",
        )

    async def _override_session():
        session = make_session()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    mock_client = _make_client(
        _forge_message(
            [
                {
                    "kind": "topic",
                    "identifier": "Work",
                    "under_content_category": "4A",
                    "confidence": 0.9,
                    "rationale": "W=Fd",
                },
                {
                    "kind": "skill",
                    "identifier": 2,
                    "confidence": 0.9,
                    "rationale": "calc",
                },
            ]
        )
    )
    test_cache = CategorizerCache(tmp_path / "categorizer-cache.db")

    app.dependency_overrides[get_session] = _override_session
    app.dependency_overrides[_anthropic_client] = lambda: mock_client
    app.dependency_overrides[_categorizer_cache] = lambda: test_cache
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Stash on object for tests that want direct access without
            # changing the existing 3-tuple unpack contract.
            client._test_cache = test_cache  # type: ignore[attr-defined]
            yield client, make_session, mock_client
    finally:
        app.dependency_overrides.pop(get_session, None)
        app.dependency_overrides.pop(_anthropic_client, None)
        app.dependency_overrides.pop(_categorizer_cache, None)
        test_cache.close()
        await conn.rollback()
        await conn.close()


def _question(*, qid: str | None = None, tags=None, stem: str | None = None):
    """Stems vary per question by default so cache keys are distinct.

    Cache keys hash stem+explanation+tags+model. Tests that exercise caching
    semantics (max_cost stop, cost_saved tally) rely on each question
    producing a fresh cache miss, so the default stem includes the qid.
    """
    real_qid = qid or f"q-{uuid.uuid4().hex[:10]}"
    return Question(
        qid=real_qid,
        passage_id=None,
        stem_html="<p>What is the work done?</p>",
        stem_plain=stem or f"Work calculation for qid={real_qid}",
        choices=[{"key": "A", "html": "<p>a</p>", "plain": "a", "media_content_hashes": []}],
        correct_choice="A",
        explanation_html=None,
        explanation_plain="W = F*d = 10 J",
        uworld_aamc_tags=tags
        if tags is not None
        else ["Subject: Physics", "Chapter: 1. Motion, Force, and Energy"],
        needs_categorization=True,
    )


async def _cc_id(session: AsyncSession, code: str) -> int:
    return (
        await session.execute(select(ContentCategory.id).where(ContentCategory.code == code))
    ).scalar_one()


async def _topic_id(session: AsyncSession, name: str, cc_code: str) -> int:
    return (
        await session.execute(
            select(Topic.id)
            .join(ContentCategory, Topic.content_category_id == ContentCategory.id)
            .where(Topic.name == name, ContentCategory.code == cc_code)
        )
    ).scalar_one()


# --------------------------------------------------------------------------- #
# 1. tag_question persists LLM targets
# --------------------------------------------------------------------------- #


async def test_tag_question_persists_llm_targets(env):
    _, make_session, _ = env
    async with make_session() as s:
        q = _question()
        s.add(q)
        await s.commit()
        q_id = q.id

    mock_client = _make_client(
        _forge_message(
            [
                {
                    "kind": "topic",
                    "identifier": "Work",
                    "under_content_category": "4A",
                    "confidence": 0.95,
                    "rationale": "Question asks W=Fd.",
                },
                {
                    "kind": "content_category",
                    "identifier": "4A",
                    "confidence": 0.9,
                    "rationale": "Mechanics.",
                },
                {
                    "kind": "skill",
                    "identifier": 2,
                    "confidence": 0.9,
                    "rationale": "Calculation.",
                },
            ]
        )
    )

    async with make_session() as s:
        lookup = await OutlineLookup.load(s)
        result = await tag_question(q_id, s, lookup=lookup, anthropic_client=mock_client)
        await s.commit()

    assert result.targets_persisted == 3
    assert result.extractor_version == llm.EXTRACTOR_VERSION
    assert result.cache_hit is False

    async with make_session() as s:
        rows = (
            (await s.execute(select(QuestionTag).where(QuestionTag.question_id == q_id)))
            .scalars()
            .all()
        )
        assert len(rows) == 3
        for r in rows:
            assert r.source == "llm"
            assert r.extractor_version == llm.EXTRACTOR_VERSION
            assert r.rationale and len(r.rationale) > 0
        kinds = {
            "topic" if r.topic_id else "content_category" if r.content_category_id else "skill"
            for r in rows
        }
        assert kinds == {"topic", "content_category", "skill"}
        q = (await s.execute(select(Question).where(Question.id == q_id))).scalar_one()
        assert q.needs_categorization is False


# --------------------------------------------------------------------------- #
# 2. No suggestions still clears flag
# --------------------------------------------------------------------------- #


async def test_tag_question_with_no_suggestions_still_clears_flag(env):
    _, make_session, _ = env
    async with make_session() as s:
        q = _question(tags=["Subject: Underwater Basket Weaving"])  # fails fast in llm.py
        s.add(q)
        await s.commit()
        q_id = q.id

    mock_client = _make_client(_forge_message([]))

    async with make_session() as s:
        lookup = await OutlineLookup.load(s)
        result = await tag_question(q_id, s, lookup=lookup, anthropic_client=mock_client)
        await s.commit()

    assert result.targets_persisted == 0
    # SDK was never called — unrecognized subject short-circuits.
    mock_client.messages.create.assert_not_awaited()
    async with make_session() as s:
        rows = (
            (await s.execute(select(QuestionTag).where(QuestionTag.question_id == q_id)))
            .scalars()
            .all()
        )
        assert rows == []
        q = (await s.execute(select(Question).where(Question.id == q_id))).scalar_one()
        assert q.needs_categorization is False


# --------------------------------------------------------------------------- #
# 3. Idempotent: rerun replaces prior LLM rows (DELETE-then-INSERT)
# --------------------------------------------------------------------------- #


async def test_tag_question_replaces_llm_rows_on_rerun(env):
    _, make_session, _ = env
    async with make_session() as s:
        q = _question()
        s.add(q)
        await s.commit()
        q_id = q.id

    first_msg = _forge_message(
        [
            {
                "kind": "content_category",
                "identifier": "4A",
                "confidence": 0.9,
                "rationale": "first",
            }
        ]
    )
    second_msg = _forge_message(
        [
            {
                "kind": "content_category",
                "identifier": "4B",
                "confidence": 0.9,
                "rationale": "second",
            },
            {
                "kind": "skill",
                "identifier": 1,
                "confidence": 0.85,
                "rationale": "more",
            },
        ]
    )

    async with make_session() as s:
        lookup = await OutlineLookup.load(s)
        first = await tag_question(q_id, s, lookup=lookup, anthropic_client=_make_client(first_msg))
        await s.commit()
    assert first.targets_persisted == 1
    assert first.targets_replaced == 0

    llm._clear_cache_for_tests()  # force a real second call

    async with make_session() as s:
        lookup = await OutlineLookup.load(s)
        second = await tag_question(
            q_id, s, lookup=lookup, anthropic_client=_make_client(second_msg)
        )
        await s.commit()
    assert second.targets_persisted == 2
    assert second.targets_replaced == 1  # the first row got wiped

    async with make_session() as s:
        rows = (
            (await s.execute(select(QuestionTag).where(QuestionTag.question_id == q_id)))
            .scalars()
            .all()
        )
        assert len(rows) == 2
        cc_codes_present = set()
        for r in rows:
            if r.content_category_id:
                cc_codes_present.add(r.content_category_id)
        cc_4b = await _cc_id(s, "4B")
        assert cc_4b in cc_codes_present


# --------------------------------------------------------------------------- #
# 4. Manual tags survive an LLM re-run
# --------------------------------------------------------------------------- #


async def test_tag_question_preserves_manual_tags(env):
    _, make_session, _ = env
    async with make_session() as s:
        q = _question()
        s.add(q)
        await s.commit()
        q_id = q.id
        cc_5d = await _cc_id(s, "5D")
        s.add(
            QuestionTag(
                question_id=q_id,
                content_category_id=cc_5d,
                confidence=1.0,
                source="manual",
            )
        )
        await s.commit()

    mock_client = _make_client(
        _forge_message(
            [
                {
                    "kind": "skill",
                    "identifier": 2,
                    "confidence": 0.9,
                    "rationale": "calc",
                }
            ]
        )
    )
    async with make_session() as s:
        lookup = await OutlineLookup.load(s)
        result = await tag_question(q_id, s, lookup=lookup, anthropic_client=mock_client)
        await s.commit()

    assert result.manual_tags_preserved == 1
    async with make_session() as s:
        manual_rows = (
            (
                await s.execute(
                    select(QuestionTag).where(
                        QuestionTag.question_id == q_id,
                        QuestionTag.source == "manual",
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(manual_rows) == 1


# --------------------------------------------------------------------------- #
# 5. Unresolvable LLM suggestions are silently dropped
# --------------------------------------------------------------------------- #


async def test_tag_question_drops_unresolvable_suggestions(env):
    _, make_session, _ = env
    async with make_session() as s:
        q = _question()
        s.add(q)
        await s.commit()
        q_id = q.id

    mock_client = _make_client(
        _forge_message(
            [
                {
                    "kind": "topic",
                    # 3.5 canonical shape: well-formed path, but topic name
                    # absent from the AAMC outline → orchestrator resolves to None.
                    "topic_path": "4A >> Totally Made Up Topic",
                    "confidence": 0.7,
                    "rationale": "hallucinated",
                },
                {
                    "kind": "content_category",
                    "content_category_code": "ZZ",  # not in outline
                    "confidence": 0.7,
                    "rationale": "fake",
                },
                {
                    "kind": "skill",
                    "skill_number": 2,
                    "confidence": 0.9,
                    "rationale": "real",
                },
            ]
        )
    )
    async with make_session() as s:
        lookup = await OutlineLookup.load(s)
        result = await tag_question(q_id, s, lookup=lookup, anthropic_client=mock_client)
        await s.commit()

    assert result.targets_persisted == 1
    assert result.suggestions_unresolved == 2

    async with make_session() as s:
        rows = (
            (await s.execute(select(QuestionTag).where(QuestionTag.question_id == q_id)))
            .scalars()
            .all()
        )
        assert len(rows) == 1
        assert rows[0].skill == 2


# --------------------------------------------------------------------------- #
# 6. Worker processes only pending
# --------------------------------------------------------------------------- #


async def test_worker_processes_only_pending(env):
    _, make_session, _ = env
    ids: list[int] = []
    async with make_session() as s:
        for _ in range(3):
            q = _question()
            s.add(q)
            await s.flush()
            ids.append(q.id)
        already_done_id = ids[1]
        q1 = (await s.execute(select(Question).where(Question.id == already_done_id))).scalar_one()
        q1.needs_categorization = False
        await s.commit()

    mock_client = _make_client(
        _forge_message(
            [
                {
                    "kind": "skill",
                    "identifier": 2,
                    "confidence": 0.9,
                    "rationale": "x",
                }
            ]
        )
    )
    async with make_session() as s:
        summary = await run_categorizer.run(s, anthropic_client=mock_client, batch_size=10)
        await s.commit()

    assert summary.processed == 2
    assert summary.succeeded == 2
    assert summary.failed == 0

    async with make_session() as s:
        rows = (
            (
                await s.execute(
                    select(QuestionTag).where(
                        QuestionTag.question_id == already_done_id,
                        QuestionTag.source == "llm",
                    )
                )
            )
            .scalars()
            .all()
        )
        assert rows == []


# --------------------------------------------------------------------------- #
# 7. Worker continues on per-question failure
# --------------------------------------------------------------------------- #


async def test_worker_continues_on_per_question_failure(env):
    _, make_session, _ = env
    ids: list[int] = []
    async with make_session() as s:
        for _ in range(3):
            q = _question()
            s.add(q)
            await s.flush()
            ids.append(q.id)
        await s.commit()

    failing_id = ids[1]

    async def failing_tag_fn(question_id, session, *, lookup, anthropic_client, cache=None):
        if question_id == failing_id:
            raise RuntimeError("simulated failure")
        return await tag_question(
            question_id,
            session,
            lookup=lookup,
            anthropic_client=anthropic_client,
            cache=cache,
        )

    mock_client = _make_client(
        _forge_message(
            [
                {
                    "kind": "skill",
                    "identifier": 2,
                    "confidence": 0.9,
                    "rationale": "x",
                }
            ]
        )
    )
    async with make_session() as s:
        summary = await run_categorizer.run(
            s,
            anthropic_client=mock_client,
            batch_size=10,
            tag_fn=failing_tag_fn,
        )
        await s.commit()

    assert summary.processed == 3
    assert summary.succeeded == 2
    assert summary.failed == 1

    async with make_session() as s:
        for q_id in ids:
            q = (await s.execute(select(Question).where(Question.id == q_id))).scalar_one()
            if q_id == failing_id:
                assert q.needs_categorization is True
            else:
                assert q.needs_categorization is False


# --------------------------------------------------------------------------- #
# 8. Worker summary includes cost + cache fields
# --------------------------------------------------------------------------- #


async def test_worker_summary_includes_cost_and_cache(env):
    _, make_session, _ = env
    async with make_session() as s:
        for _ in range(2):
            s.add(_question())
        await s.commit()

    mock_client = _make_client(
        _forge_message(
            [
                {
                    "kind": "skill",
                    "identifier": 2,
                    "confidence": 0.9,
                    "rationale": "x",
                }
            ]
        )
    )
    async with make_session() as s:
        summary = await run_categorizer.run(s, anthropic_client=mock_client, batch_size=10)

    assert summary.total_cost_usd > 0
    assert summary.cache_hit_count + summary.cache_miss_count == 2
    text = summary.as_text()
    assert "total_cost_usd" in text
    assert "cache_hits" in text


# --------------------------------------------------------------------------- #
# 9. Recategorize endpoint: returns extended result
# --------------------------------------------------------------------------- #


async def test_recategorize_endpoint_returns_result(env):
    client, make_session, _ = env
    async with make_session() as s:
        q = _question()
        s.add(q)
        await s.commit()
        q_id = q.id

    r = await client.post(f"/api/v1/admin/questions/{q_id}/recategorize")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["question_id"] == q_id
    assert body["targets_persisted"] >= 1
    assert "cost_estimate_usd" in body
    assert "cache_hit" in body
    assert body["extractor_version"] == llm.EXTRACTOR_VERSION


# --------------------------------------------------------------------------- #
# 10. Recategorize 404
# --------------------------------------------------------------------------- #


async def test_recategorize_endpoint_404_on_unknown_question(env):
    client, _, _ = env
    r = await client.post("/api/v1/admin/questions/999999999/recategorize")
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
# 11. Recategorize: DELETE-then-INSERT replaces old LLM rows
# --------------------------------------------------------------------------- #


async def test_recategorize_endpoint_replaces_llm_rows(env):
    client, make_session, mock_client = env
    async with make_session() as s:
        q = _question()
        s.add(q)
        await s.commit()
        q_id = q.id

    # First call uses the env fixture's default mock (Work topic + skill 2).
    r1 = await client.post(f"/api/v1/admin/questions/{q_id}/recategorize")
    assert r1.status_code == 200
    assert r1.json()["targets_persisted"] >= 1

    # Swap mock to produce different output; clear cache so the second call
    # genuinely re-hits the (mocked) SDK rather than returning a cached result.
    mock_client.messages.create.return_value = _forge_message(
        [
            {
                "kind": "content_category",
                "identifier": "4B",
                "confidence": 0.9,
                "rationale": "different",
            }
        ]
    )
    # Clear the env's per-test cache so the next recategorize call re-hits
    # the (mocked) SDK rather than returning a cached result.
    client._test_cache.clear()  # type: ignore[attr-defined]

    r2 = await client.post(f"/api/v1/admin/questions/{q_id}/recategorize")
    assert r2.status_code == 200
    body = r2.json()
    assert body["targets_replaced"] >= 1

    async with make_session() as s:
        rows = (
            (
                await s.execute(
                    select(QuestionTag).where(
                        QuestionTag.question_id == q_id,
                        QuestionTag.source == "llm",
                    )
                )
            )
            .scalars()
            .all()
        )
        cc_4b = await _cc_id(s, "4B")
        assert len(rows) == 1
        assert rows[0].content_category_id == cc_4b


# --------------------------------------------------------------------------- #
# 12-15. Manual tag endpoints (carry-over from 3.2)
# --------------------------------------------------------------------------- #


async def test_manual_tag_endpoint_creates_row(env):
    client, make_session, _ = env
    async with make_session() as s:
        q = _question(tags=None)
        s.add(q)
        await s.commit()
        q_id = q.id
        cc_4a = await _cc_id(s, "4A")

    r = await client.post(
        f"/api/v1/admin/questions/{q_id}/tags",
        json={"content_category_id": cc_4a},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["source"] == "manual"
    assert body["content_category_id"] == cc_4a


async def test_manual_tag_endpoint_rejects_zero_targets(env):
    client, make_session, _ = env
    async with make_session() as s:
        q = _question(tags=None)
        s.add(q)
        await s.commit()
        q_id = q.id
    r = await client.post(f"/api/v1/admin/questions/{q_id}/tags", json={})
    assert r.status_code == 422


async def test_manual_tag_endpoint_rejects_two_targets(env):
    client, make_session, _ = env
    async with make_session() as s:
        q = _question(tags=None)
        s.add(q)
        await s.commit()
        q_id = q.id
        cc_4a = await _cc_id(s, "4A")
    r = await client.post(
        f"/api/v1/admin/questions/{q_id}/tags",
        json={"content_category_id": cc_4a, "skill": 1},
    )
    assert r.status_code == 422


async def test_manual_tag_endpoint_409_on_duplicate(env):
    client, make_session, _ = env
    async with make_session() as s:
        q = _question(tags=None)
        s.add(q)
        await s.commit()
        q_id = q.id
        cc_4a = await _cc_id(s, "4A")
    r1 = await client.post(
        f"/api/v1/admin/questions/{q_id}/tags",
        json={"content_category_id": cc_4a},
    )
    assert r1.status_code == 201
    r2 = await client.post(
        f"/api/v1/admin/questions/{q_id}/tags",
        json={"content_category_id": cc_4a},
    )
    assert r2.status_code == 409


async def test_delete_manual_tag_succeeds(env):
    client, make_session, _ = env
    async with make_session() as s:
        q = _question(tags=None)
        s.add(q)
        await s.commit()
        q_id = q.id
        cc_4a = await _cc_id(s, "4A")
    r = await client.post(
        f"/api/v1/admin/questions/{q_id}/tags",
        json={"content_category_id": cc_4a},
    )
    tag_id = r.json()["tag_id"]
    r2 = await client.delete(f"/api/v1/admin/tags/{tag_id}")
    assert r2.status_code == 204


async def test_delete_llm_tag_soft_deletes(env):
    """LLM-assigned tags are soft-deleted (is_overridden=True), not hard-deleted."""
    client, make_session, _ = env
    async with make_session() as s:
        q = _question()
        s.add(q)
        await s.commit()
        q_id = q.id

    r = await client.post(f"/api/v1/admin/questions/{q_id}/recategorize")
    assert r.status_code == 200

    async with make_session() as s:
        row = (
            (
                await s.execute(
                    select(QuestionTag).where(
                        QuestionTag.question_id == q_id,
                        QuestionTag.source == "llm",
                    )
                )
            )
            .scalars()
            .first()
        )
        assert row is not None
        tag_id = row.id

    r = await client.delete(f"/api/v1/admin/tags/{tag_id}")
    assert r.status_code == 204

    async with make_session() as s:
        row = (
            await s.execute(select(QuestionTag).where(QuestionTag.id == tag_id))
        ).scalar_one_or_none()
        assert row is not None
        assert row.is_overridden is True
        assert row.overridden_at is not None


# --------------------------------------------------------------------------- #
# Sanity
# --------------------------------------------------------------------------- #


async def test_delete_manual_tag_hard_deletes(env):
    """Manual tags are hard-deleted (row gone from DB), not soft-deleted."""
    client, make_session, _ = env
    async with make_session() as s:
        q = _question(tags=None)
        s.add(q)
        await s.commit()
        q_id = q.id
        cc_4a = await _cc_id(s, "4A")

    r = await client.post(
        f"/api/v1/admin/questions/{q_id}/tags",
        json={"content_category_id": cc_4a},
    )
    assert r.status_code == 201
    tag_id = r.json()["tag_id"]

    r2 = await client.delete(f"/api/v1/admin/tags/{tag_id}")
    assert r2.status_code == 204

    async with make_session() as s:
        row = (
            await s.execute(select(QuestionTag).where(QuestionTag.id == tag_id))
        ).scalar_one_or_none()
        assert row is None


async def test_delete_tag_404_when_missing(env):
    client, _, _ = env
    r = await client.delete("/api/v1/admin/tags/999999")
    assert r.status_code == 404
    assert "tag_id=999999 not found" in r.json()["detail"]


async def test_delete_tag_forbidden_when_source_unknown(env):
    """Tags with source='uworld_map' return 403; row is untouched."""
    client, make_session, _ = env
    async with make_session() as s:
        q = _question(tags=None)
        s.add(q)
        await s.commit()
        q_id = q.id
        cc_4a = await _cc_id(s, "4A")
        row = QuestionTag(
            question_id=q_id,
            content_category_id=cc_4a,
            confidence=1.0,
            source="uworld_map",
        )
        s.add(row)
        await s.commit()
        tag_id = row.id

    r = await client.delete(f"/api/v1/admin/tags/{tag_id}")
    assert r.status_code == 403

    async with make_session() as s:
        reloaded = (
            await s.execute(select(QuestionTag).where(QuestionTag.id == tag_id))
        ).scalar_one_or_none()
        assert reloaded is not None
        assert reloaded.is_overridden is False


async def test_soft_deleted_llm_tag_overridden_at_is_recent(env):
    """Soft-deleted LLM tag has overridden_at set to roughly now."""
    from datetime import datetime, timezone

    client, make_session, _ = env
    async with make_session() as s:
        q = _question()
        s.add(q)
        await s.commit()
        q_id = q.id

    r = await client.post(f"/api/v1/admin/questions/{q_id}/recategorize")
    assert r.status_code == 200

    async with make_session() as s:
        llm_tag = (
            (
                await s.execute(
                    select(QuestionTag).where(
                        QuestionTag.question_id == q_id,
                        QuestionTag.source == "llm",
                    )
                )
            )
            .scalars()
            .first()
        )
        assert llm_tag is not None
        tag_id = llm_tag.id

    before_delete = datetime.now(timezone.utc)
    r2 = await client.delete(f"/api/v1/admin/tags/{tag_id}")
    assert r2.status_code == 204

    async with make_session() as s:
        row = (
            await s.execute(select(QuestionTag).where(QuestionTag.id == tag_id))
        ).scalar_one_or_none()
        assert row is not None
        assert row.overridden_at is not None
        ts = row.overridden_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        delta = abs((ts - before_delete).total_seconds())
        assert delta < 5


async def test_rationale_persisted_on_llm_tag(env):
    """LLM-returned rationale is written to QuestionTag.rationale."""
    _, make_session, _ = env
    async with make_session() as s:
        q = _question()
        s.add(q)
        await s.commit()
        q_id = q.id

    expected_rationale = "Work equals force times displacement"
    mock_client = _make_client(
        _forge_message(
            [
                {
                    "kind": "content_category",
                    "identifier": "4A",
                    "confidence": 0.9,
                    "rationale": expected_rationale,
                }
            ]
        )
    )
    async with make_session() as s:
        lookup = await OutlineLookup.load(s)
        await tag_question(q_id, s, lookup=lookup, anthropic_client=mock_client)
        await s.commit()

    async with make_session() as s:
        row = (
            (
                await s.execute(
                    select(QuestionTag).where(
                        QuestionTag.question_id == q_id,
                        QuestionTag.source == "llm",
                    )
                )
            )
            .scalars()
            .first()
        )
        assert row is not None
        assert row.rationale == expected_rationale

    # Empty-string rationale is allowed — seeds without error.
    async with make_session() as s:
        q2 = _question()
        s.add(q2)
        await s.flush()
        q2_id = q2.id
        cc_4a = await _cc_id(s, "4A")
        s.add(
            QuestionTag(
                question_id=q2_id,
                content_category_id=cc_4a,
                confidence=0.9,
                source="llm",
                rationale="",
            )
        )
        await s.commit()


async def test_tag_question_raises_on_missing(env):
    _, make_session, _ = env
    async with make_session() as s:
        lookup = await OutlineLookup.load(s)
        mock_client = _make_client(_forge_message([]))
        with pytest.raises(QuestionNotFoundError):
            await tag_question(999_999_999, s, lookup=lookup, anthropic_client=mock_client)


# --------------------------------------------------------------------------- #
# Ticket 6.8 additions
# --------------------------------------------------------------------------- #


async def test_worker_persists_deep_topic_tag(env):
    """LLM returns a depth-2 path; orchestrator persists the deep topic's DB ID."""
    _, make_session, _ = env
    deep_path = "5A >> Solubility >> Solubility product constant; the equilibrium expression Ksp"
    async with make_session() as s:
        q = _question(tags=["Subject: General Chemistry", "Chapter: X"])
        s.add(q)
        await s.commit()
        q_id = q.id

    mock_client = _make_client(
        _forge_message(
            [
                {
                    "kind": "topic",
                    "topic_path": deep_path,
                    "confidence": 0.9,
                    "rationale": "Ksp expression",
                },
                {
                    "kind": "skill",
                    "skill_number": 2,
                    "confidence": 0.9,
                    "rationale": "calc",
                },
            ],
            section="CP",
        )
    )

    async with make_session() as s:
        lookup = await OutlineLookup.load(s)
        result = await tag_question(q_id, s, lookup=lookup, anthropic_client=mock_client)
        await s.commit()

    assert result.suggestions_unresolved == 0
    assert result.extractor_version == llm.EXTRACTOR_VERSION

    async with make_session() as s:
        # Verify the deep topic ID was persisted, not the parent "Solubility" ID
        parent_id = (
            await s.execute(
                select(Topic.id)
                .join(ContentCategory, Topic.content_category_id == ContentCategory.id)
                .where(
                    Topic.name == "Solubility",
                    ContentCategory.code == "5A",
                    Topic.parent_topic_id.is_(None),
                )
            )
        ).scalar_one()
        deep_topic_id = (
            await s.execute(
                select(Topic.id)
                .join(ContentCategory, Topic.content_category_id == ContentCategory.id)
                .where(
                    Topic.name == "Solubility product constant; the equilibrium expression Ksp",
                    ContentCategory.code == "5A",
                )
            )
        ).scalar_one()

        tag_row = (
            (
                await s.execute(
                    select(QuestionTag).where(
                        QuestionTag.question_id == q_id,
                        QuestionTag.topic_id.is_not(None),
                    )
                )
            )
            .scalars()
            .first()
        )
        assert tag_row is not None
        assert tag_row.topic_id == deep_topic_id
        assert tag_row.topic_id != parent_id
        assert tag_row.extractor_version == llm.EXTRACTOR_VERSION


async def test_worker_logs_unresolved_path_and_skips(env, caplog):
    import logging

    _, make_session, _ = env
    async with make_session() as s:
        q = _question(tags=["Subject: General Chemistry", "Chapter: X"])
        s.add(q)
        await s.commit()
        q_id = q.id

    mock_client = _make_client(
        _forge_message(
            [
                {
                    "kind": "topic",
                    "topic_path": "5A >> Fakery >> Notreal",
                    "confidence": 0.9,
                    "rationale": "hallucinated",
                },
                {
                    "kind": "skill",
                    "skill_number": 2,
                    "confidence": 0.9,
                    "rationale": "real",
                },
            ],
            section="CP",
        )
    )

    with caplog.at_level(logging.WARNING):
        async with make_session() as s:
            lookup = await OutlineLookup.load(s)
            result = await tag_question(q_id, s, lookup=lookup, anthropic_client=mock_client)
            await s.commit()

    assert result.suggestions_unresolved == 1
    assert result.targets_persisted == 1  # skill persisted

    async with make_session() as s:
        topic_rows = (
            (
                await s.execute(
                    select(QuestionTag).where(
                        QuestionTag.question_id == q_id,
                        QuestionTag.topic_id.is_not(None),
                    )
                )
            )
            .scalars()
            .all()
        )
        assert topic_rows == []


# --------------------------------------------------------------------------- #
# Ticket 3.4 additions
# --------------------------------------------------------------------------- #


def _per_call_costly_message():
    """Forge a high-cost response so `--max-cost-usd 0.01` trips after 2 calls."""
    from anthropic.types import ToolUseBlock

    content = [
        ToolUseBlock(
            id="toolu_costly",
            name="submit_aamc_categorization",
            input={
                "primary_aamc_section": "CP",
                "tags": [
                    {
                        "kind": "skill",
                        "identifier": 2,
                        "confidence": 0.9,
                        "rationale": "x",
                    }
                ],
            },
            type="tool_use",
        )
    ]
    # Each call costs ~$0.006 (2000 input * $3/M + 200 output * $15/M).
    usage = SimpleNamespace(
        input_tokens=2000,
        output_tokens=200,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    return SimpleNamespace(content=content, usage=usage)


async def test_worker_respects_max_cost_usd(env, tmp_path, monkeypatch):
    """Cap stops the drain; remaining questions stay queued."""
    # Pin model so per-call cost math below is independent of the production default.
    monkeypatch.setattr(settings, "CATEGORIZER_MODEL", "claude-sonnet-4-6")

    _, make_session, _ = env
    ids: list[int] = []
    async with make_session() as s:
        for _ in range(5):
            q = _question()
            s.add(q)
            await s.flush()
            ids.append(q.id)
        await s.commit()

    mock_client = _make_client(_per_call_costly_message)
    cache = CategorizerCache(tmp_path / "max-cost.db")

    try:
        async with make_session() as s:
            summary = await run_categorizer.run(
                s,
                anthropic_client=mock_client,
                batch_size=10,
                cache=cache,
                max_cost_usd=0.01,
            )
            await s.commit()
    finally:
        cache.close()

    # Per-call cost ≈ $0.006 → cap of $0.01 trips on the 2nd call.
    assert summary.cost_limit_hit is True
    assert summary.succeeded == 2
    assert summary.processed == 2
    assert summary.total_cost_usd >= 0.01

    async with make_session() as s:
        remaining = (
            await s.execute(select(Question.id).where(Question.needs_categorization.is_(True)))
        ).all()
        # 3 of the 5 questions should still be pending.
        assert len(remaining) == 3


async def test_worker_without_max_cost_runs_to_completion(env, tmp_path):
    _, make_session, _ = env
    async with make_session() as s:
        for _ in range(3):
            s.add(_question())
        await s.commit()

    mock_client = _make_client(_per_call_costly_message)
    cache = CategorizerCache(tmp_path / "no-cap.db")

    try:
        async with make_session() as s:
            summary = await run_categorizer.run(
                s, anthropic_client=mock_client, batch_size=10, cache=cache
            )
            await s.commit()
    finally:
        cache.close()

    assert summary.cost_limit_hit is False
    assert summary.processed == 3
    assert summary.succeeded == 3

    async with make_session() as s:
        remaining = (
            await s.execute(select(Question.id).where(Question.needs_categorization.is_(True)))
        ).all()
        assert remaining == []


async def test_worker_logs_cost_saved(env, tmp_path):
    """Pre-populate cache; worker reports total_cost_saved_usd > 0."""
    _, make_session, _ = env
    ids: list[int] = []
    async with make_session() as s:
        for _ in range(2):
            q = _question()
            s.add(q)
            await s.flush()
            ids.append(q.id)
        await s.commit()

    cache = CategorizerCache(tmp_path / "preheat.db")
    mock_client = _make_client(_per_call_costly_message)

    try:
        # First pass populates the cache.
        async with make_session() as s:
            first = await run_categorizer.run(
                s, anthropic_client=mock_client, batch_size=10, cache=cache
            )
            await s.commit()
        assert first.cache_hit_count == 0
        assert first.cache_miss_count == 2

        # Re-queue both questions and run again — every call should be a hit.
        async with make_session() as s:
            for q_id in ids:
                q = (await s.execute(select(Question).where(Question.id == q_id))).scalar_one()
                q.needs_categorization = True
            await s.commit()

        async with make_session() as s:
            second = await run_categorizer.run(
                s, anthropic_client=mock_client, batch_size=10, cache=cache
            )
            await s.commit()
        assert second.cache_hit_count == 2
        assert second.cache_miss_count == 0
        assert second.total_cost_usd == 0.0
        assert second.total_cost_saved_usd > 0
    finally:
        cache.close()


# --------------------------------------------------------------------------- #
# Ticket 3.5 additions
# --------------------------------------------------------------------------- #


async def test_orchestrator_resolves_qualified_topic_path(env):
    """LLM emits canonical topic_path; orchestrator persists the right topic_id."""
    _, make_session, _ = env
    async with make_session() as s:
        q = _question()
        s.add(q)
        await s.commit()
        q_id = q.id

    mock_client = _make_client(
        _forge_message(
            [
                {
                    "kind": "topic",
                    "topic_path": "4A >> Work",
                    "confidence": 0.95,
                    "rationale": "W=Fd",
                },
                {
                    "kind": "skill",
                    "skill_number": 2,
                    "confidence": 0.9,
                    "rationale": "calc",
                },
            ]
        )
    )

    async with make_session() as s:
        lookup = await OutlineLookup.load(s)
        result = await tag_question(q_id, s, lookup=lookup, anthropic_client=mock_client)
        await s.commit()
    assert result.targets_persisted == 2
    assert result.suggestions_unresolved == 0

    async with make_session() as s:
        expected_topic_id = await _topic_id(s, "Work", "4A")
        rows = (
            (
                await s.execute(
                    select(QuestionTag).where(
                        QuestionTag.question_id == q_id,
                        QuestionTag.topic_id.is_not(None),
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(rows) == 1
        assert rows[0].topic_id == expected_topic_id


async def test_orchestrator_drops_unknown_topic_path(env):
    """LLM emits topic_path absent from outline → suggestion unresolved + dropped."""
    _, make_session, _ = env
    async with make_session() as s:
        q = _question()
        s.add(q)
        await s.commit()
        q_id = q.id

    mock_client = _make_client(
        _forge_message(
            [
                {
                    "kind": "topic",
                    "topic_path": "4A >> Definitely Not A Real Topic",
                    "confidence": 0.7,
                    "rationale": "hallucinated",
                },
                {
                    "kind": "skill",
                    "skill_number": 2,
                    "confidence": 0.9,
                    "rationale": "real",
                },
            ]
        )
    )

    async with make_session() as s:
        lookup = await OutlineLookup.load(s)
        result = await tag_question(q_id, s, lookup=lookup, anthropic_client=mock_client)
        await s.commit()
    assert result.targets_persisted == 1
    assert result.suggestions_unresolved == 1

    async with make_session() as s:
        rows = (
            (await s.execute(select(QuestionTag).where(QuestionTag.question_id == q_id)))
            .scalars()
            .all()
        )
        assert len(rows) == 1
        assert rows[0].skill == 2
        assert rows[0].topic_id is None


# --------------------------------------------------------------------------- #
# T55 Step 2 — V42: worker dispatches section-grouped so Anthropic prompt-cache
# prefix stays hot across consecutive calls within the same section.
# --------------------------------------------------------------------------- #


async def test_worker_dispatches_sections_grouped(env):
    """V42: candidates in a batch ordered by derived section."""
    _, make_session, _ = env

    # Insert questions interleaved across sections: BB, CP, BB, PS, CP.
    interleaved_tags = [
        ["Subject: Biology", "Chapter: Cell"],
        ["Subject: Physics", "Chapter: Motion"],
        ["Subject: Biochemistry", "Chapter: Enzymes"],
        ["Subject: Psychology", "Chapter: Cognition"],
        ["Subject: General Chemistry", "Chapter: Thermo"],
    ]
    qid_to_section: dict[str, str] = {}
    section_by_subject = {
        "Biology": "BB",
        "Biochemistry": "BB",
        "Physics": "CP",
        "General Chemistry": "CP",
        "Psychology": "PS",
        "Sociology": "PS",
    }
    async with make_session() as s:
        for tags in interleaved_tags:
            q = _question(tags=tags)
            s.add(q)
            await s.flush()
            subject = tags[0][len("Subject: ") :]
            qid_to_section[q.qid] = section_by_subject[subject]
        await s.commit()

    # Recorder tag_fn captures the order in which questions are tagged.
    invocation_qids: list[str] = []

    async def recording_tag_fn(question_id, session, *, lookup, anthropic_client, cache):
        q = (await session.execute(select(Question).where(Question.id == question_id))).scalar_one()
        invocation_qids.append(q.qid)
        q.needs_categorization = False
        from app.services.categorizer import TagQuestionResult
        from app.services.categorizer.llm import CategorizeResult

        return TagQuestionResult(
            cache_hit=False,
            categorize_result=CategorizeResult(
                suggestions=[],
                primary_aamc_section=None,
                cache_hit=False,
                input_tokens=0,
                output_tokens=0,
                estimated_cost_usd=0.0,
                extractor_version=llm.EXTRACTOR_VERSION,
                parse_warnings=[],
            ),
            targets_persisted=0,
            targets_replaced=0,
            suggestions_unresolved=0,
            manual_preserved=0,
            cost_estimate_usd=0.0,
            cost_saved_usd=0.0,
            extractor_version=llm.EXTRACTOR_VERSION,
        )

    async with make_session() as s:
        await run_categorizer.run(
            s,
            anthropic_client=MagicMock(),
            batch_size=10,
            tag_fn=recording_tag_fn,
        )
        await s.commit()

    sections_visited = [qid_to_section[q] for q in invocation_qids]
    # Each section appears contiguously — no interleaving across sections
    # within the same batch.
    seen_sections: list[str] = []
    for sec in sections_visited:
        if not seen_sections or seen_sections[-1] != sec:
            seen_sections.append(sec)
    assert len(seen_sections) == len(set(seen_sections)), (
        f"sections interleaved: {sections_visited!r}"
    )
