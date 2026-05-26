"""V-D1 structural guards (§T23).

The dashboard is the Jinja thin client for P0–P3; SPA reassessment lives
at T34. V-D1 says the public `/api/v1/*` JSON API is its sole data seam,
which has three concrete teeth:

1. No dashboard-only / private / internal-prefixed endpoint may exist on
   the public API — every JSON route is part of the contract a future
   SPA / external consumer can call.
2. The dashboard sub-app (``app/web/dashboard/main.py``) mounts only
   HTML-rendering routers. It must not re-export a router from
   ``app.api.v1.*`` (the JSON API lives at one URL, period) and must not
   define JSON routes itself.
3. The dashboard sub-app must not invent its own JSON seam — every
   handler returns HTML / Redirect / a non-JSON ``Response``.

Heavy data-fetch refactors (replacing in-process service calls with
public-API view-function calls) ride the T34 SPA work; T23 locks the
*structural* contract so any drift fails CI before the swap.
"""

from __future__ import annotations

import inspect

from fastapi import FastAPI
from fastapi.datastructures import Default, DefaultPlaceholder
from fastapi.responses import (
    HTMLResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
)
from fastapi.routing import APIRoute

import app.main as main_mod
import app.web.dashboard.main as dashboard_main_mod


# Responses the dashboard sub-app may legitimately return. JSONResponse is
# deliberately excluded — JSON belongs on `/api/v1/*`.
_ALLOWED_DASHBOARD_RESPONSE_CLASSES: tuple[type[Response], ...] = (
    HTMLResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,  # raw 204/302 etc.
)


def _api_v1_paths() -> set[str]:
    return {r.path for r in main_mod.app.routes if isinstance(r, APIRoute)}


def _resolve_response_class(route: APIRoute) -> type[Response]:
    """Unwrap FastAPI's ``Default(...)`` placeholder if present."""
    rc = route.response_class
    if isinstance(rc, DefaultPlaceholder):
        return rc.value
    return rc


# ---------- Clause 1: no dashboard-only / private prefix on the public API ----------


def test_no_dashboard_only_api_v1_prefix() -> None:
    """V-D1 clause 1 — extend `/api/v1/*`, ⊥ private dashboard-only routes."""
    forbidden_substrings = ("/dashboard", "/internal", "/private")
    bad = sorted(
        p
        for p in _api_v1_paths()
        if p.startswith("/api/v1") and any(tok in p.lower() for tok in forbidden_substrings)
    )
    assert not bad, (
        f"V-D1 forbids dashboard-only / private prefixes on /api/v1/*; "
        f"found: {bad}"
    )


# ---------- Clause 2: dashboard sub-app mounts no JSON-API router ----------


def test_dashboard_app_does_not_remount_api_v1_routers() -> None:
    """V-D1 clause 2 — the JSON API lives on the main app only.

    Re-exporting an ``app.api.v1.*`` router under the dashboard sub-app
    would create two URLs serving the same payload, splitting the seam
    the future SPA must speak.
    """
    dashboard_app: FastAPI = dashboard_main_mod.app
    leaked = sorted(
        r.path
        for r in dashboard_app.routes
        if isinstance(r, APIRoute) and r.path.startswith("/api/v1")
    )
    assert not leaked, (
        f"dashboard sub-app exposes `/api/v1/*` paths: {leaked} — V-D1 "
        f"requires those live solely on the main app"
    )


# ---------- Clause 3: dashboard handlers respond non-JSON ----------


def test_dashboard_routes_return_non_json_responses() -> None:
    """V-D1 clause 3 — no JSON seam in the dashboard sub-app.

    Each ``APIRoute`` either declares ``response_class=HTMLResponse``
    (rendering Jinja) or returns a non-JSON ``Response`` subtype. A
    default ``JSONResponse`` indicates an accidental JSON endpoint and
    must be ported to ``/api/v1/*`` first.
    """
    dashboard_app: FastAPI = dashboard_main_mod.app
    bad: list[tuple[str, str]] = []
    for r in dashboard_app.routes:
        if not isinstance(r, APIRoute):
            continue
        rc = _resolve_response_class(r)
        if not issubclass(rc, _ALLOWED_DASHBOARD_RESPONSE_CLASSES):
            bad.append((r.path, rc.__name__))
    assert not bad, (
        f"dashboard routes returning non-HTML responses (V-D1 violation): {bad}"
    )


# ---------- Self-documenting contract ----------


def test_dashboard_main_documents_v_d1_contract() -> None:
    """The contract is load-bearing for future contributors — keep it
    explicit in the module docstring so the next person adding a route
    has the rule in front of them."""
    src = inspect.getsource(dashboard_main_mod)
    assert "V-D1" in src, "dashboard sub-app must reference V-D1 in its docstring"
    assert "/api/v1" in src, (
        "dashboard sub-app docstring should name `/api/v1/*` as the data seam"
    )


# ---------- Sanity: public-API view-function re-use is allowed ----------


def test_dashboard_admin_route_imports_public_api_view_function() -> None:
    """Affirmative pattern check — `admin.py` already calls
    ``app.api.v1.admin.list_jobs_payload`` / ``trigger_job_logic``
    instead of duplicating the scheduler poke. This is the V-D1-shaped
    way to reuse JSON-API logic inside a Jinja handler; keep it in place
    so it's the obvious template for future routes."""
    import app.web.dashboard.routes.admin as admin_mod

    src = inspect.getsource(admin_mod)
    assert "from app.api.v1.admin import" in src, (
        "dashboard /admin route should import its scheduler helpers from "
        "the public API view (`app.api.v1.admin`), not redefine them"
    )
    assert "list_jobs_payload" in src and "trigger_job_logic" in src, (
        "dashboard /admin should reuse list_jobs_payload + trigger_job_logic"
    )
