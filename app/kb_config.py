"""P2 KB substrate config validation (T25, V-KB2).

Called from the FastAPI lifespan in `app/main.py` at process startup.
The validator inspects the loaded ``Settings`` object and emits a WARN
per missing optional value — it does NOT fail fast, because the
substrate services (PDF ingest, Notion write-out, embedding writes) are
opt-in: a user running only the question/Anki pipelines need not set
the Notion / PDF-inbox vars.

Hard failures (raised) are reserved for state that is already broken
at the type level (e.g. a non-Path PDF_INBOX_DIR), which
pydantic-settings catches earlier; this function only validates the
*semantic* preconditions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.config import Settings

_logger = logging.getLogger("app.kb_config")


def validate_kb_config(settings: "Settings") -> list[str]:
    """Return a list of human-readable warnings about KB substrate config.

    Each warning is also emitted at WARN level on ``app.kb_config``.
    Empty list = all checks passed. Does not raise — caller may proceed
    even with warnings; the matching service raises when it actually
    needs the value (e.g. Notion sync raises if NOTION_API_TOKEN is None).
    """

    warnings: list[str] = []

    if not settings.NOTION_API_TOKEN:
        warnings.append(
            "NOTION_API_TOKEN unset — Notion write-out (V-N1) disabled "
            "until set in .env. Other features unaffected."
        )

    if not settings.NOTION_WIKI_DB_ID:
        warnings.append(
            "NOTION_WIKI_DB_ID unset — Notion write-out (V-N2) cannot "
            "target a database until set in .env."
        )

    pdf_dir: Path = settings.PDF_INBOX_DIR
    if not pdf_dir.exists():
        warnings.append(
            f"PDF_INBOX_DIR ({pdf_dir}) does not exist — PDF ingest "
            "(T26) will create it lazily on first use, but you can "
            "pre-create it now."
        )

    if not settings.EMBEDDING_MODEL:
        warnings.append(
            "EMBEDDING_MODEL unset — embedding writes (V-E1) require a "
            "model name to stamp `embedding_version`."
        )

    for msg in warnings:
        _logger.warning("kb_config: %s", msg)

    return warnings
