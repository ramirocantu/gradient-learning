"""Dashboard sub-app (Jinja thin client).

V-D1 contract:
- Dashboard is a *client* over the public `/api/v1/*` JSON API — that
  surface is the sole data seam, and the boundary the future SPA
  (T34 reassessment) will swap in over.
- This sub-app mounts only HTML-rendering routers. ⊥ JSON routes here;
  ⊥ a dashboard-only backend endpoint (extend `/api/v1/*` instead).
- Route handlers may call public-API view functions in-process (e.g.
  `from app.api.v1.admin import list_jobs_payload`) — same contract,
  no httpx loopback overhead. New data needs land in `/api/v1/*` first.

Structural enforcement lives in `tests/test_v_d1_dashboard_seam.py`.
"""

from pathlib import Path

import markdown as md_lib
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from markupsafe import Markup

from app.web.dashboard.routes import (
    admin,
    anki,
    home,
    questions,
    sessions,
)
# FENCED (T17, V-RB1, V-O5): mastery/topics/recommendations/insights routes
# consume FENCED services (app/services/{analytics,recommender,analyzer}
# + app/web/dashboard/services/{mastery,drilldown}). Restoration is tied to
# the T34 SPA reassessment.
# from app.web.dashboard.routes import insights, mastery, recommendations, topics
from app.web.dashboard.routes.questions import tags_router

app = FastAPI(
    title="MCAT Coach",
    docs_url=None,
    redoc_url=None,
)


def _markdown_filter(text: str) -> Markup:
    return Markup(md_lib.markdown(text, extensions=["nl2br"]))


# Stash templates on app state for routes to use
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))
templates.env.filters["markdown_to_html"] = _markdown_filter
app.state.templates = templates

app.include_router(home.router)
app.include_router(sessions.router)
# FENCED (T17, V-RB1): mastery/topics/recommendations/insights include disabled.
# app.include_router(mastery.router)
# app.include_router(topics.router)
# app.include_router(recommendations.router)
# app.include_router(insights.router)
app.include_router(questions.router)
app.include_router(tags_router)
app.include_router(admin.router)
app.include_router(anki.router)
