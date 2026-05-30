"""V-AUTH: auth is enforced globally, not per-route.

Every ``/api/v1/*`` route and ``/media/*`` requires a valid ``X-Coach-Token``;
only ``/healthz`` is public. This locks the default-secure contract: a new
router added without its own ``Depends`` is still gated, because the gate
lives on the ``v1`` router + the media include (see ``app/main.py``), never
on individual routes.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from app.config import settings

_AUTH = {"X-Coach-Token": settings.COACH_TOKEN}

# One representative GET per router — spans every include in app/main.py plus
# the app-root media mount. GETs avoid body-validation races with the auth dep.
_GATED_PATHS = [
    "/api/v1/courses",  # outline (router-level → now global)
    "/api/v1/concept-edges",  # kb_reads (router-level → now global)
    "/api/v1/admin/status",  # admin (per-route → now global)
    "/api/v1/tutor/sessions/recent",  # tutor (inline param → now global)
    "/api/v1/anki/review-queue",  # anki
    "/api/v1/anki/load-adherence",  # anki_load
    "/api/v1/anki/reviews",  # anki_review
    "/api/v1/anki/assignments",  # anki_assign
    "/api/v1/attempts/1/notes",  # attempts
    "/media/does-not-exist.pdf",  # media (was OPEN → now gated)
]


@pytest.mark.parametrize("path", _GATED_PATHS)
async def test_route_requires_coach_token(client: AsyncClient, path: str) -> None:
    """No token anywhere on the API surface → 401, before any handler runs."""
    resp = await client.get(path)
    assert resp.status_code == 401, f"{path} should be gated, got {resp.status_code}"


@pytest.mark.parametrize("path", _GATED_PATHS)
async def test_route_passes_auth_with_token(client: AsyncClient, path: str) -> None:
    """With a valid token the auth gate is cleared — never a 401.

    The handler may still 404/422/etc. (missing rows, etc.); we only assert
    that authentication itself no longer rejects the request.
    """
    resp = await client.get(path, headers=_AUTH)
    assert resp.status_code != 401, f"{path} rejected a valid token"


async def test_healthz_is_public(client: AsyncClient) -> None:
    """Liveness probe stays open — no token required."""
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
