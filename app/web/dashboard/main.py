from pathlib import Path

import markdown as md_lib
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from markupsafe import Markup

from app.web.dashboard.routes import (
    admin,
    anki,
    home,
    insights,
    mastery,
    questions,
    recommendations,
    sessions,
    topics,
)
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
app.include_router(mastery.router)
app.include_router(topics.router)
app.include_router(recommendations.router)
app.include_router(insights.router)
app.include_router(questions.router)
app.include_router(tags_router)
app.include_router(admin.router)
app.include_router(anki.router)
