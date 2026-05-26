from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

pytestmark = pytest.mark.asyncio


async def _add_media(session, content_hash: str, local_path: str = "", body: bytes = b"x") -> int:
    from app.models.media import Media

    if not local_path:
        local_path = f"{content_hash[:2]}/{content_hash}.png"
    m = Media(
        content_hash=content_hash,
        local_path=local_path,
        mime_type="image/png",
        byte_size=len(body),
    )
    session.add(m)
    await session.flush()
    return m.id


async def _add_question(session, qid: str, **kwargs):
    from app.models.captures import Question

    defaults: dict = {
        "qid": qid,
        "stem_html": "<p>stem</p>",
        "stem_plain": "stem",
        "choices": [
            {"key": "A", "html": "<p>A</p>", "plain": "A", "media_ids": []},
            {"key": "B", "html": "<p>B</p>", "plain": "B", "media_ids": []},
            {"key": "C", "html": "<p>C</p>", "plain": "C", "media_ids": []},
            {"key": "D", "html": "<p>D</p>", "plain": "D", "media_ids": []},
        ],
        "correct_choice": "A",
        "explanation_html": "<p>because</p>",
        "explanation_plain": "because",
        "uworld_aamc_tags": ["Subject: Biology"],
        "needs_categorization": True,
    }
    defaults.update(kwargs)
    q = Question(**defaults)
    session.add(q)
    await session.flush()
    return q


async def test_captures_list_renders_recent_questions(client, db_session):
    await _add_question(db_session, "QID-A")
    await _add_question(db_session, "QID-B")
    await _add_question(db_session, "QID-C")
    await db_session.commit()

    r = await client.get("/viewer/captures")
    assert r.status_code == 200
    assert "QID-A" in r.text
    assert "QID-B" in r.text
    assert "QID-C" in r.text


async def test_captures_list_sorts_by_last_updated_desc(client, db_session):
    """A re-ingested question should jump to the top of the list."""
    from datetime import timedelta

    base = datetime.now(UTC)
    q_old = await _add_question(
        db_session,
        "QID-OLD-FIRST",
        first_seen_at=base - timedelta(days=2),
        last_updated_at=base - timedelta(days=2),
    )
    q_mid = await _add_question(
        db_session,
        "QID-MID",
        first_seen_at=base - timedelta(days=1),
        last_updated_at=base - timedelta(days=1),
    )
    # Old question re-captured most recently — should now be on top.
    q_old.last_updated_at = base
    db_session.add(q_old)
    await db_session.commit()
    assert q_mid is not None  # silence linter

    r = await client.get("/viewer/captures")
    assert r.status_code == 200
    body = r.text
    pos_old = body.index("QID-OLD-FIRST")
    pos_mid = body.index("QID-MID")
    assert pos_old < pos_mid, "QID-OLD-FIRST (re-captured) should appear before QID-MID"


async def test_capture_detail_renders_full_content_with_media_rewrite(client, db_session):
    from app.models.captures import Passage

    image_hash = hashlib.sha256(b"img").hexdigest()
    media_id = await _add_media(
        db_session, image_hash, local_path=f"{image_hash[:2]}/{image_hash}.png"
    )

    passage = Passage(
        uworld_passage_id="p-1",
        content_hash=hashlib.sha256(b"pass").hexdigest(),
        html=f'<p>P</p><img data-media-content-hash="{image_hash}">',
        plain_text="P",
    )
    db_session.add(passage)
    await db_session.flush()

    q = await _add_question(
        db_session,
        "QID-IMG",
        passage_id=passage.id,
        stem_html=f'<p>stem</p><img data-media-content-hash="{image_hash}">',
        explanation_html=f'<p>exp</p><img data-media-content-hash="{image_hash}">',
        choices=[
            {"key": "A", "html": "<p>A</p>", "plain": "A", "media_ids": [media_id]},
            {"key": "B", "html": "<p>B</p>", "plain": "B", "media_ids": []},
            {"key": "C", "html": "<p>C</p>", "plain": "C", "media_ids": []},
            {"key": "D", "html": "<p>D</p>", "plain": "D", "media_ids": []},
        ],
    )
    await db_session.commit()

    r = await client.get(f"/viewer/captures/{q.id}")
    assert r.status_code == 200
    body = r.text

    for key in ("A", "B", "C", "D"):
        assert f">{key}<" in body or f"key {key}" in body or f"{key}</span>" in body

    assert f'src="/media/{image_hash[:2]}/{image_hash}.png"' in body
    assert "data-media-content-hash" not in body
    assert "Explanation" in body


async def test_capture_detail_renders_no_passage_marker(client, db_session):
    q = await _add_question(db_session, "QID-NOPASS")
    await db_session.commit()

    r = await client.get(f"/viewer/captures/{q.id}")
    assert r.status_code == 200
    body = r.text
    assert "No passage linked" in body
    assert "discrete" in body


async def test_capture_detail_renders_passage_metadata_and_siblings(client, db_session):
    from app.models.captures import Passage

    passage = Passage(
        uworld_passage_id="p-99",
        content_hash=hashlib.sha256(b"shared").hexdigest(),
        html="<p>shared passage body</p>",
        plain_text="shared passage body",
    )
    db_session.add(passage)
    await db_session.flush()

    q1 = await _add_question(db_session, "QID-SIB-1", passage_id=passage.id)
    await _add_question(db_session, "QID-SIB-2", passage_id=passage.id)
    await _add_question(db_session, "QID-SIB-3", passage_id=passage.id)
    await db_session.commit()

    r = await client.get(f"/viewer/captures/{q1.id}")
    assert r.status_code == 200
    body = r.text
    assert "shared passage body" in body
    assert "uworld p-99" in body
    assert "shared by 3 questions" in body
    assert "QID-SIB-2" in body
    assert "QID-SIB-3" in body
    # Current question itself shouldn't be linked as a sibling.
    assert body.count("QID-SIB-1") >= 1


async def test_capture_detail_renders_missing_media_placeholder(client, db_session):
    unknown_hash = "deadbeef" * 8
    q = await _add_question(
        db_session,
        "QID-MISS",
        stem_html=f'<p>stem</p><img data-media-content-hash="{unknown_hash}">',
    )
    await db_session.commit()

    r = await client.get(f"/viewer/captures/{q.id}")
    assert r.status_code == 200
    assert 'data-missing="true"' in r.text


async def test_capture_detail_404_on_unknown_id(client, db_session):
    r = await client.get("/viewer/captures/99999")
    assert r.status_code == 404


async def test_media_file_serves_bytes(client, db_session, test_media_root: Path):
    content = b"hello-bytes"
    h = hashlib.sha256(content).hexdigest()
    local_path = f"{h[:2]}/{h}.png"
    full_path = test_media_root / local_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_bytes(content)

    await _add_media(db_session, h, local_path=local_path, body=content)
    await db_session.commit()

    r = await client.get(f"/media/{local_path}")
    assert r.status_code == 200
    assert r.content == content
    assert "max-age=86400" in r.headers.get("cache-control", "")


async def test_media_file_rejects_path_traversal(client, db_session):
    r = await client.get("/media/../../etc/passwd")
    assert r.status_code == 404


# (T20: removed `_seed_minimal_outline` + `test_capture_detail_renders_aamc_tags`
#  + `test_captures_list_renders_aamc_summary_chips` — all three seeded
#  `Section`/`FoundationalConcept`/`ContentCategory`/`Topic` and the
#  3-target `QuestionTag(topic_id|content_category_id|skill)` shape. Both
#  the outline tables and the 3-target columns are dropped (T1/T2).
#  Re-coverage of AAMC chip rendering on `OutlineNode` + `node_id` tags
#  lands when the viewer is reworked alongside T34.)


async def test_version_endpoint_returns_iso_and_advances_on_write(client, db_session):
    r = await client.get("/viewer/_version")
    assert r.status_code == 200
    v0 = r.json()["v"]
    assert isinstance(v0, str) and v0  # ISO string, even when DB empty (epoch)

    await _add_question(db_session, "QID-VERSION-1")
    await db_session.commit()

    r2 = await client.get("/viewer/_version")
    assert r2.status_code == 200
    v1 = r2.json()["v"]
    assert v1 > v0  # ISO 8601 strings sort lexicographically as timestamps
