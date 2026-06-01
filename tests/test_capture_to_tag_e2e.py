"""RCA-10 §T3 — E2E: Capture → Attempt → Grounded Tag.

Drives the real HTTP surface (POST /api/v1/captures, GET /api/v1/tutor/...)
and the shared grounded arc (embed_pending → recall → tag_pending) in-process,
codifying what the T2/T9 manual passes proved:

- capture persists Question + Attempt (+ media), dedup on `qid` (§I);
- V6: ABSENT course_slug → NULL-course fallback; UNKNOWN slug → 422;
- V9: re-capture omitting uworld_aamc_tags clobbers the stored tags + re-flags
  needs_categorization (confirmed-intended snapshot semantics, B1);
- V10: a captured question runs the grounded arc → question_tags persisted
  (source='llm', node_id, confidence, manual_review) + needs_categorization
  flipped false, surfaced via GET /tutor/questions/by-qid; ambiguous-course
  questions are skipped untouched.

Mock altitude (V2): embeddings go through the SDK-boundary mock
(make_embeddings_client); the grounded LLM pass is patched at the jobs layer
(generate_grounded_tags) — its SDK internals + V12 calibrator are owned by
test_grounded / test_calibrator. Media depth is owned by test_ingest; here we
assert media_ids resolve once. Each test self-seeds (V3); the per-test
savepoint rolls back.
"""

from __future__ import annotations

import base64
import hashlib
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.captures import Attempt, Question, QuestionTag
from app.models.media import Media
from app.models.outline import OutlineNode
from app.services.kb.jobs import embed_pending, tag_pending
from app.services.llm.grounded import GroundedResult, GroundedTag
from tests._openai_mocks import make_embeddings_client

_PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6300010000000500010d0a2db40000000049454e44ae426082"
)
_PNG_B64 = base64.b64encode(_PNG_BYTES).decode("ascii")
_PNG_HASH = hashlib.sha256(_PNG_BYTES).hexdigest()

_CAPTURES = "/api/v1/captures"


async def _node(session: AsyncSession, course_id: int, name: str) -> OutlineNode:
    n = OutlineNode(
        course_id=course_id, parent_id=None, kind="concept", name=name, depth=0, position=0
    )
    session.add(n)
    await session.flush()
    return n


def _grounded(node_id: int, *, conf: float = 0.9, manual_review: bool = False) -> GroundedResult:
    """A controlled grounded result (stands in for the real LLM pass)."""
    return GroundedResult(
        tags=[
            GroundedTag(
                node_id=node_id,
                path=None,
                candidate_index=1,
                via="embedding",
                rationale="matches the captured stem",
                calibrated_confidence=conf,
                manual_review=manual_review,
            )
        ],
        extractor_version="grounded-v1",
        model="m",
        calibrator_model="c",
        input_tokens=10,
        output_tokens=2,
        cached_tokens=0,
    )


async def _question_by_qid(session: AsyncSession, qid: str) -> Question:
    return (
        await session.execute(select(Question).where(Question.qid == qid))
    ).scalar_one()


# --------------------------------------------------------------------------- #
# capture → Question + Attempt + media
# --------------------------------------------------------------------------- #


async def test_capture_persists_question_attempt_and_media(
    client: AsyncClient,
    coach_headers: dict[str, str],
    make_course,
    uworld_capture_payload,
    db_session: AsyncSession,
    test_media_root,
):
    await make_course(slug="cap-e2e", name="Capture E2E")
    qid = f"q-{uuid.uuid4().hex[:8]}"
    body = uworld_capture_payload(qid=qid, course_slug="cap-e2e")
    # attach one media item + reference it from choice A (media depth = test_ingest)
    body["media"] = [{"content_hash": _PNG_HASH, "mime_type": "image/png", "bytes_b64": _PNG_B64}]
    body["parsed"]["choices"][0]["media_content_hashes"] = [_PNG_HASH]

    resp = await client.post(_CAPTURES, json=body, headers=coach_headers)
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert set(payload) == {"capture_id", "question_id", "attempt_id", "passage_id", "media_ids"}
    assert len(payload["media_ids"]) == 1

    q = await _question_by_qid(db_session, qid)
    assert q.course_id is not None
    attempts = (
        (await db_session.execute(select(Attempt).where(Attempt.question_id == q.id)))
        .scalars()
        .all()
    )
    assert len(attempts) == 1
    media = (await db_session.execute(select(Media).where(Media.content_hash == _PNG_HASH))).scalars().all()
    assert len(media) == 1


async def test_capture_dedup_same_qid_new_attempt(
    client: AsyncClient, coach_headers, make_course, uworld_capture_payload, db_session: AsyncSession
):
    await make_course(slug="dedup-e2e", name="Dedup")
    qid = f"q-{uuid.uuid4().hex[:8]}"
    first = await client.post(
        _CAPTURES, json=uworld_capture_payload(qid=qid, course_slug="dedup-e2e"), headers=coach_headers
    )
    second = await client.post(
        _CAPTURES, json=uworld_capture_payload(qid=qid, course_slug="dedup-e2e"), headers=coach_headers
    )
    assert first.status_code == second.status_code == 200
    # Same question (qid UNIQUE), distinct attempts.
    assert first.json()["question_id"] == second.json()["question_id"]
    assert first.json()["attempt_id"] != second.json()["attempt_id"]

    q = await _question_by_qid(db_session, qid)
    attempts = (
        (await db_session.execute(select(Attempt).where(Attempt.question_id == q.id)))
        .scalars()
        .all()
    )
    assert len(attempts) == 2


# --------------------------------------------------------------------------- #
# V6 — absent slug fallback vs unknown slug 422 (distinct paths)
# --------------------------------------------------------------------------- #


async def test_capture_absent_slug_falls_back_to_null_course(
    client: AsyncClient, coach_headers, uworld_capture_payload, db_session: AsyncSession
):
    qid = f"q-{uuid.uuid4().hex[:8]}"
    resp = await client.post(_CAPTURES, json=uworld_capture_payload(qid=qid), headers=coach_headers)
    assert resp.status_code == 200, resp.text
    q = await _question_by_qid(db_session, qid)
    assert q.course_id is None  # single-course / unscoped fallback


async def test_capture_unknown_slug_rejected_422(
    client: AsyncClient, coach_headers, uworld_capture_payload
):
    resp = await client.post(
        _CAPTURES,
        json=uworld_capture_payload(qid=f"q-{uuid.uuid4().hex[:8]}", course_slug="no-such-course"),
        headers=coach_headers,
    )
    assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# V9 — re-capture omitting tags clobbers + re-flags (intended snapshot, B1)
# --------------------------------------------------------------------------- #


async def test_recapture_omitting_tags_clobbers_and_reflags(
    client: AsyncClient, coach_headers, make_course, uworld_capture_payload, db_session: AsyncSession
):
    await make_course(slug="clobber-e2e", name="Clobber")
    qid = f"q-{uuid.uuid4().hex[:8]}"

    body1 = uworld_capture_payload(qid=qid, course_slug="clobber-e2e")
    body1["parsed"]["uworld_aamc_tags"] = ["Biochemistry"]
    await client.post(_CAPTURES, json=body1, headers=coach_headers)

    q = await _question_by_qid(db_session, qid)
    assert q.uworld_aamc_tags == ["Biochemistry"]
    # Simulate a downstream consumer having categorized it.
    q.needs_categorization = False
    await db_session.commit()

    # Re-capture the SAME qid WITHOUT tags → snapshot semantics clear them.
    body2 = uworld_capture_payload(qid=qid, course_slug="clobber-e2e")  # no uworld_aamc_tags
    await client.post(_CAPTURES, json=body2, headers=coach_headers)

    await db_session.refresh(q)
    assert q.uworld_aamc_tags is None  # clobbered
    assert q.needs_categorization is True  # re-flagged for re-tag


# --------------------------------------------------------------------------- #
# V10 — captured question runs the grounded arc; readable via the tutor route
# --------------------------------------------------------------------------- #


async def test_captured_question_grounded_tagged_and_readable(
    client: AsyncClient, coach_headers, make_course, uworld_capture_payload, db_session: AsyncSession
):
    course = await make_course(slug="tag-e2e", name="Tag E2E")
    node = await _node(db_session, course.id, "Glycolysis")
    await db_session.commit()

    qid = f"q-{uuid.uuid4().hex[:8]}"
    await client.post(
        _CAPTURES, json=uworld_capture_payload(qid=qid, course_slug="tag-e2e"), headers=coach_headers
    )
    q = await _question_by_qid(db_session, qid)
    assert q.needs_categorization is True

    # embed node + question (same mocked vector → recall ranks the node #1)
    await embed_pending(db_session, openai_client=make_embeddings_client())

    with patch(
        "app.services.kb.jobs.generate_grounded_tags",
        new=AsyncMock(return_value=_grounded(node.id, conf=0.8)),
    ):
        report = await tag_pending(
            db_session, tagging_client=MagicMock(), calibrator_client=MagicMock()
        )
    await db_session.commit()

    assert report.questions_tagged == 1
    await db_session.refresh(q)
    assert q.needs_categorization is False
    qtags = (
        (await db_session.execute(select(QuestionTag).where(QuestionTag.question_id == q.id)))
        .scalars()
        .all()
    )
    assert [t.node_id for t in qtags] == [node.id]
    assert qtags[0].source == "llm"
    assert qtags[0].confidence is not None

    # readable via the public tutor route
    resp = await client.get(f"/api/v1/tutor/questions/by-qid/{qid}", headers=coach_headers)
    assert resp.status_code == 200, resp.text
    tags = resp.json()["tags"]
    assert any(t["node_id"] == node.id and t["source"] == "llm" for t in tags)


async def test_grounded_tag_skips_ambiguous_course_question(
    client: AsyncClient, coach_headers, make_course, uworld_capture_payload, db_session: AsyncSession
):
    # Two courses + an unscoped (no course_slug) captured question → ambiguous.
    await make_course(slug="amb-a", name="A")
    await make_course(slug="amb-b", name="B")
    qid = f"q-{uuid.uuid4().hex[:8]}"
    await client.post(_CAPTURES, json=uworld_capture_payload(qid=qid), headers=coach_headers)
    q = await _question_by_qid(db_session, qid)
    assert q.course_id is None

    await embed_pending(db_session, openai_client=make_embeddings_client())
    gen = AsyncMock(return_value=_grounded(1))
    with patch("app.services.kb.jobs.generate_grounded_tags", new=gen):
        report = await tag_pending(db_session, tagging_client=MagicMock())
    await db_session.commit()

    assert report.questions_skipped == 1
    gen.assert_not_called()
    await db_session.refresh(q)
    assert q.needs_categorization is True  # untouched
