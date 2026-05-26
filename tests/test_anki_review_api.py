"""Tests for SPEC T67 + T76 + T77 review API (V53 amended)."""

from __future__ import annotations

from datetime import date, timedelta

from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.anki import AnkiReview


_AUTH = {"X-Coach-Token": "change_me_before_use"}


async def test_review_endpoints_require_coach_token(client: AsyncClient) -> None:
    r = await client.post("/api/v1/anki/reviews", json={})
    assert r.status_code == 401
    r = await client.get("/api/v1/anki/reviews")
    assert r.status_code == 401


async def test_post_review_happy(client: AsyncClient, db_session: AsyncSession) -> None:
    review_date = (date.today() + timedelta(days=1)).isoformat()
    r = await client.post(
        "/api/v1/anki/reviews",
        headers=_AUTH,
        json={
            "card_ids": [1735689600001, 1735689600002],
            "review_date": review_date,
        },
    )
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "pending"
    # V53 amended: deck name uses the new row's own PK.
    assert body["deck_name"] == f"mcat-coach::review::{body['id']}"
    assert body["card_ids"] == [1735689600001, 1735689600002]
    assert body["review_date"] == review_date


async def test_post_review_dup_same_day_creates_new_row(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """V53 amended: no UNIQUE constraint — dup same-day creates two
    distinct rows w/ distinct deck names per the tags-as-log design."""
    payload = {
        "card_ids": [1, 2],
        "review_date": (date.today() + timedelta(days=1)).isoformat(),
    }
    r1 = await client.post("/api/v1/anki/reviews", headers=_AUTH, json=payload)
    assert r1.status_code == 201
    r2 = await client.post("/api/v1/anki/reviews", headers=_AUTH, json=payload)
    assert r2.status_code == 201
    assert r1.json()["id"] != r2.json()["id"]
    assert r1.json()["deck_name"] != r2.json()["deck_name"]


async def test_post_review_empty_card_ids_422(
    client: AsyncClient,
) -> None:
    r = await client.post(
        "/api/v1/anki/reviews",
        headers=_AUTH,
        json={
            "card_ids": [],
            "review_date": date.today().isoformat(),
        },
    )
    assert r.status_code == 422


async def test_list_reviews_filters(client: AsyncClient, db_session: AsyncSession) -> None:
    today = date.today()
    db_session.add_all(
        [
            AnkiReview(
                review_date=today + timedelta(days=1),
                card_ids=[1],
                deck_name="mcat-coach::review::seed-soon",
                status="pending",
            ),
            AnkiReview(
                review_date=today + timedelta(days=20),
                card_ids=[2],
                deck_name="mcat-coach::review::seed-far",
                status="pending",
            ),
            AnkiReview(
                review_date=today,
                card_ids=[3],
                deck_name="mcat-coach::review::seed-done",
                status="pushed",
            ),
        ]
    )
    await db_session.flush()

    r = await client.get("/api/v1/anki/reviews", headers=_AUTH)
    assert r.status_code == 200
    assert len(r.json()) == 3

    r = await client.get("/api/v1/anki/reviews?status=pending", headers=_AUTH)
    assert {row["status"] for row in r.json()} == {"pending"}

    r = await client.get("/api/v1/anki/reviews?window_days=5", headers=_AUTH)
    # window_days=5 keeps today + 5d range; "done" today, "soon" +1d in,
    # "far" +20d out.
    deck_names = {row["deck_name"] for row in r.json()}
    assert "mcat-coach::review::seed-soon" in deck_names
    assert "mcat-coach::review::seed-done" in deck_names
    assert "mcat-coach::review::seed-far" not in deck_names
