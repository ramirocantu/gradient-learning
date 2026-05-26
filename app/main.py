import json
import logging
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.admin import router as admin_router
# FENCED (T17, V-RB1, V-O5): analytics/analyzer/recommendations routers consume
# FENCED services (app/services/{analytics,recommender,analyzer}). Restoration
# is tied to the post-P0.5 node_id rollup port + T34 SPA reassessment.
# from app.api.v1.analytics import router as analytics_router
# from app.api.v1.analyzer import router as analyzer_router
from app.api.v1.anki import router as anki_router
from app.api.v1.anki_assign import router as anki_assign_router
from app.api.v1.anki_load import router as anki_load_router
from app.api.v1.anki_review import router as anki_review_router
from app.api.v1.attempts import router as attempts_router
from app.api.v1.captures import router as captures_router
from app.api.v1.outline import router as outline_router
# from app.api.v1.recommendations import router as recommendations_router
from app.api.v1.tutor import router as tutor_router
from app.config import settings
from app.kb_config import validate_kb_config
from app.scheduler import start_scheduler, stop_scheduler
from app.web.dashboard.main import app as dashboard_app
from app.web.media import router as media_router
from app.web.viewer.main import app as viewer_app


@asynccontextmanager
async def lifespan(app: FastAPI):
    # T25 / V-KB2: validate P2 KB substrate env at startup. Missing
    # optional values WARN-log but do not block boot — the matching
    # service raises on first use if its var is unset.
    validate_kb_config(settings)
    start_scheduler()
    yield
    stop_scheduler()


app = FastAPI(title="MCAT Coach", version="0.1.0", lifespan=lifespan)

# Local-only service: allow the Chrome extension and any localhost origin.
# The extension origin changes per install, so we allow the full scheme prefix.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"(chrome-extension://.*|https?://localhost(:\d+)?)",
    allow_methods=["*"],
    allow_headers=["*"],
)


v1 = APIRouter(prefix="/api/v1")
v1.include_router(captures_router)
v1.include_router(outline_router)
v1.include_router(admin_router)
# FENCED (T17, V-RB1): analytics/analyzer/recommendations include disabled.
# v1.include_router(analytics_router)
# v1.include_router(analyzer_router)
v1.include_router(attempts_router)
# v1.include_router(recommendations_router)
v1.include_router(tutor_router)
v1.include_router(anki_router)
v1.include_router(anki_assign_router)
v1.include_router(anki_review_router)
v1.include_router(anki_load_router)
app.include_router(v1)


@app.get("/healthz")
async def health_check():
    return {"status": "ok"}


# Media — root-mounted, serves /media/{file_path}. Registered before the
# dashboard's catch-all "/" mount so the dashboard router doesn't shadow it.
app.include_router(media_router)

# Viewer sub-app (capture browser, dev tool) mounted at /viewer/*.
app.mount("/viewer", viewer_app)

# Dashboard mounted last at "/" — catch-all for the user-facing UI.
app.mount("/", dashboard_app)


_ingest_logger = logging.getLogger("app.ingest.validation")


@app.exception_handler(RequestValidationError)
async def _capture_validation_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    if request.url.path.startswith("/api/v1/captures"):
        errors = exc.errors()
        try:
            payload = json.dumps(errors, default=str)
            if len(payload) > 4000:
                payload = payload[:4000] + "...<truncated>"
        except (TypeError, ValueError):
            payload = repr(errors)
        _ingest_logger.warning("capture payload validation failed: %s", payload)
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder({"detail": exc.errors()}),
    )
