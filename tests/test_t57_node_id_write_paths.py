"""T57 gate (B6+B7, V-RB6): the two ported write paths exercised green.

B6 — ``POST /api/v1/admin/questions/{id}/tags`` forwards ``node_id`` to the
node_id-only ``admin_tags.create_manual_tag`` and echoes ``row.node_id``.
B7 — ``POST /api/v1/anki/assignments`` resolves candidate cards via the
``node_id`` subtree rollup (``outline_subtree``, V-O1) over
``anki_note_tags.node_id`` — no legacy ``topics``/``content_categories`` join.

Both paths used to raise (TypeError/AttributeError; UndefinedTable/Column) the
moment they ran, hidden by the absence of any exercising test.
"""

from __future__ import annotations

from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.anki import AnkiCard, AnkiNote, AnkiNoteTag
from app.models.captures import Question, QuestionTag
from app.models.outline import Course, OutlineNode

_AUTH = {"X-Coach-Token": settings.COACH_TOKEN}


async def _seed_outline(db: AsyncSession) -> dict[str, int]:
    """Proteins (section) → Amino acids (topic). Returns name→node_id."""
    course = Course(slug="biochem-t57", name="Biochem")
    db.add(course)
    await db.flush()
    ids: dict[str, int] = {}
    parent = None
    for name, kind, depth in [("Proteins", "section", 0), ("Amino acids", "topic", 1)]:
        n = OutlineNode(
            course_id=course.id,
            parent_id=parent,
            kind=kind,
            name=name,
            depth=depth,
            position=0,
        )
        db.add(n)
        await db.flush()
        ids[name] = n.id
        parent = n.id
    return ids


# ── B6: admin manual-tag write path ─────────────────────────────────────── #


async def test_admin_manual_tag_forwards_node_id(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    ids = await _seed_outline(db_session)
    q = Question(
        source="uworld",
        qid="T57-Q1",
        stem_html="<p>q</p>",
        stem_plain="q",
        choices=[{"label": "A", "text": "a"}],
        correct_choice="A",
    )
    db_session.add(q)
    await db_session.flush()
    qid = q.id
    await db_session.commit()

    r = await client.post(
        f"/api/v1/admin/questions/{qid}/tags",
        headers=_AUTH,
        json={"node_id": ids["Amino acids"]},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["node_id"] == ids["Amino acids"]
    assert body["question_id"] == qid
    assert body["source"] == "manual"
    assert body["confidence"] is None  # V-T3: manual → NULL

    rows = (await db_session.execute(select(QuestionTag))).scalars().all()
    assert any(row.node_id == ids["Amino acids"] and row.question_id == qid for row in rows)


async def test_admin_manual_tag_unknown_question_404(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    ids = await _seed_outline(db_session)
    await db_session.commit()
    r = await client.post(
        "/api/v1/admin/questions/999999/tags",
        headers=_AUTH,
        json={"node_id": ids["Proteins"]},
    )
    assert r.status_code == 404, r.text


# ── B7: anki assignment candidate selection ─────────────────────────────── #


async def test_create_assignment_resolves_node_subtree(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    ids = await _seed_outline(db_session)
    # One suspended card whose note is tagged to the CHILD node; assign at the
    # PARENT → subtree rollup (V-O1) must pull it in.
    note = AnkiNote(note_id=1_700_000_000_001)
    db_session.add(note)
    await db_session.flush()
    db_session.add(
        AnkiCard(
            anki_card_id=9_000_001,
            deck_name="MileDown",
            note_id=note.note_id,
            queue=-1,  # suspended → eligible
        )
    )
    db_session.add(
        AnkiNoteTag(
            note_id=note.note_id,
            tag_raw="biochem::amino-acids",
            node_id=ids["Amino acids"],
            parsed_kind="resolved",
            source="schema_map",
            confidence=None,
        )
    )
    await db_session.flush()
    await db_session.commit()

    r = await client.post(
        "/api/v1/anki/assignments",
        headers=_AUTH,
        json={
            "scope_kind": "topic",
            "scope_value": str(ids["Proteins"]),  # parent node id
            "scheduled_unlock_at": "2099-01-01T00:00:00+00:00",
        },
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["card_ids"] == [9_000_001]
    assert body["status"] == "pending"


async def test_create_assignment_empty_subtree_resolves_no_cards(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    ids = await _seed_outline(db_session)
    await db_session.commit()
    # Leaf node with no tagged cards → empty card_ids, still a valid 201.
    r = await client.post(
        "/api/v1/anki/assignments",
        headers=_AUTH,
        json={
            "scope_kind": "topic",
            "scope_value": str(ids["Amino acids"]),
            "scheduled_unlock_at": "2099-01-01T00:00:00+00:00",
        },
    )
    assert r.status_code == 201, r.text
    assert r.json()["card_ids"] == []


async def test_create_assignment_non_int_scope_value_422(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    await _seed_outline(db_session)
    await db_session.commit()
    r = await client.post(
        "/api/v1/anki/assignments",
        headers=_AUTH,
        json={
            "scope_kind": "cc",
            "scope_value": "4C",  # legacy CC code, not a node_id → 422
            "scheduled_unlock_at": "2099-01-01T00:00:00+00:00",
        },
    )
    assert r.status_code == 422, r.text
