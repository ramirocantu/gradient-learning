"""Tests for SPEC T67 load-config + load-adherence API (V54, V59, V60)."""

from __future__ import annotations

from decimal import Decimal

from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.anki import AnkiLoadConfig


_AUTH = {"X-Coach-Token": "change_me_before_use"}


async def test_load_routes_require_coach_token(client: AsyncClient) -> None:
    r = await client.get("/api/v1/anki/load-config")
    assert r.status_code == 401
    r = await client.post("/api/v1/anki/load-config", json={})
    assert r.status_code == 401
    r = await client.get("/api/v1/anki/load-adherence")
    assert r.status_code == 401


async def test_get_load_config_seeds_defaults_on_miss(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """V59 read-or-create: first GET inserts the seed row (200, 60)."""
    r = await client.get("/api/v1/anki/load-config", headers=_AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["daily_card_review_budget"] == 200
    assert Decimal(str(body["daily_minutes_budget"])) == Decimal("60")


async def test_post_load_config_upserts(client: AsyncClient, db_session: AsyncSession) -> None:
    r = await client.post(
        "/api/v1/anki/load-config",
        headers=_AUTH,
        json={
            "daily_card_review_budget": 300,
            "daily_minutes_budget": "75.5",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["daily_card_review_budget"] == 300
    assert Decimal(str(body["daily_minutes_budget"])) == Decimal("75.5")

    # Second POST updates the same row (singleton).
    r = await client.post(
        "/api/v1/anki/load-config",
        headers=_AUTH,
        json={
            "daily_card_review_budget": 150,
            "daily_minutes_budget": "45",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["daily_card_review_budget"] == 150
    assert Decimal(str(body["daily_minutes_budget"])) == Decimal("45")


async def test_post_load_config_rejects_non_positive_422(
    client: AsyncClient,
) -> None:
    r = await client.post(
        "/api/v1/anki/load-config",
        headers=_AUTH,
        json={"daily_card_review_budget": 0, "daily_minutes_budget": "30"},
    )
    assert r.status_code == 422
    r = await client.post(
        "/api/v1/anki/load-config",
        headers=_AUTH,
        json={"daily_card_review_budget": 100, "daily_minutes_budget": "0"},
    )
    assert r.status_code == 422


async def test_get_load_adherence_shape_matches_v54(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    db_session.add(
        AnkiLoadConfig(
            id=1,
            daily_card_review_budget=200,
            daily_minutes_budget=Decimal("60"),
        )
    )
    await db_session.flush()

    r = await client.get("/api/v1/anki/load-adherence", headers=_AUTH)
    assert r.status_code == 200
    body = r.json()
    expected_keys = {
        "window_days",
        "projected_daily_load",
        "projected_daily_minutes",
        "daily_card_review_budget",
        "daily_minutes_budget",
        "headroom_card_review_pct",
        "headroom_minutes_pct",
        "status_label",
    }
    assert set(body.keys()) == expected_keys
    # V60: payload must not carry advisory.
    assert "recommended_changes" not in body
    assert body["window_days"] == 30
    assert body["status_label"] == "feasible"  # empty state, no load


async def test_get_load_adherence_window_days_query(
    client: AsyncClient,
) -> None:
    r = await client.get("/api/v1/anki/load-adherence?window_days=7", headers=_AUTH)
    assert r.status_code == 200
    assert r.json()["window_days"] == 7


async def test_get_load_adherence_invalid_window_422(
    client: AsyncClient,
) -> None:
    r = await client.get("/api/v1/anki/load-adherence?window_days=0", headers=_AUTH)
    assert r.status_code == 422
