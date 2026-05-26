from pathlib import Path

from fastapi import FastAPI
from fastapi.templating import Jinja2Templates

from app.web.media import router as media_router
from app.web.viewer.routes import captures, version

_HERE = Path(__file__).resolve().parent
_TEMPLATES_DIR = _HERE / "templates"


def create_app() -> FastAPI:
    app = FastAPI(
        title="MCAT Coach — Capture Browser",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    app.include_router(version.router)
    app.include_router(captures.router)
    app.include_router(media_router)

    return app


app = create_app()
