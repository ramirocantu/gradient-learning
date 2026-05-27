"""V-D1 (backend-only) structural guards.

The repo ships **no view layer**. `/api/v1/*` + `/healthz` + `/media/*` are
the entire surface; clients (native macOS app, Chrome extension, MCP host)
are external and consume the JSON API over HTTP. These guards lock that
contract so a Jinja/HTML view layer can't drift back in:

1. No route renders HTML — the app has no `HTMLResponse` endpoints.
2. No catch-all / sub-app mounts — the dashboard + viewer sub-apps are gone.
3. The `app/web` package holds only the media file-server (no templates,
   no dashboard/viewer packages).
"""

from __future__ import annotations

from pathlib import Path

from fastapi.datastructures import DefaultPlaceholder
from fastapi.responses import HTMLResponse, Response
from fastapi.routing import APIRoute
from starlette.routing import Mount

import app as app_pkg
import app.main as main_mod


def _resolve_response_class(route: APIRoute) -> type[Response]:
    rc = route.response_class
    if isinstance(rc, DefaultPlaceholder):
        return rc.value
    return rc


def test_no_html_rendering_routes() -> None:
    """V-D1 — no endpoint serves HTML; the app is JSON + static assets only."""
    bad = [
        route.path
        for route in main_mod.app.routes
        if isinstance(route, APIRoute)
        and issubclass(_resolve_response_class(route), HTMLResponse)
    ]
    assert not bad, f"backend-only app must not render HTML; HTMLResponse routes: {bad}"


def test_no_sub_app_mounts() -> None:
    """V-D1 — no mounted sub-apps (the dashboard `/` + viewer `/viewer`
    catch-alls were removed). Everything lives on the main app."""
    mounts = [r.path for r in main_mod.app.routes if isinstance(r, Mount)]
    assert not mounts, f"backend-only app must not mount sub-apps; found mounts: {mounts}"


def test_web_package_is_media_only() -> None:
    """V-D1 — `app/web` contains only the media file-server, no view layer."""
    web_dir = Path(app_pkg.__file__).resolve().parent / "web"
    py_files = {p.name for p in web_dir.glob("*.py")}
    assert py_files == {"__init__.py", "media.py"}, (
        f"app/web should hold only __init__.py + media.py; found {sorted(py_files)}"
    )
    for gone in ("dashboard", "viewer"):
        assert not (web_dir / gone).exists(), (
            f"app/web/{gone} should be deleted (backend-only)"
        )
