import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


_BACKEND_ROOT = Path(__file__).resolve().parent.parent

# T40 / V-TC1: when GRADIENT_DISABLE_DOTENV is set (the test suite sets it at
# tests/conftest.py import, before this module first loads), skip the real
# .env so a developer's secrets/tokens never bleed into tests. Config then
# comes from process env + field defaults only.
_ENV_FILE = None if os.environ.get("GRADIENT_DISABLE_DOTENV") == "1" else _BACKEND_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        # If a shell exports an env var as empty (e.g. Claude Code sets
        # OPENAI_API_KEY="" for sandbox safety), prefer the .env file value
        # over the empty OS env var.
        env_ignore_empty=True,
    )

    DATABASE_URL: str
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str | None = None
    COACH_TOKEN: str = "change_me_before_use"
    MEDIA_ROOT: Path = _BACKEND_ROOT / "data" / "media"

    # 9.0: rendered into the get_flagged_attempts response as absolute deep-links.
    # 9.5 R.4: dashboard now lives in the same FastAPI process as the JSON API,
    # so a single BACKEND_BASE_URL covers both. Override via .env when the
    # process is reachable at a non-default origin.
    BACKEND_BASE_URL: str = "http://localhost:8000"

    # OpenAI model selection (P0 pivot). Single tagging/facts model + a
    # logprobs-capable calibrator. Per §C, the calibrator MUST be a standard
    # (non-reasoning) chat model — o-series models don't expose logprobs.
    # Picked in the T5 spike; override via .env if a re-eval rotates them.
    OPENAI_MODEL: str = "gpt-4.1-mini"
    # V-KB3: optional vision-capable chat model for PDF page transcription.
    # None → fall back to OPENAI_MODEL (gpt-4.1-mini is multimodal).
    OPENAI_VISION_MODEL: str | None = None
    OPENAI_CALIBRATOR_MODEL: str = "gpt-4.1-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Scheduler (Ticket 6.9b)
    SCHEDULER_ENABLED: bool = True

    # AnkiConnect (SPEC §T1, P11). Read-only HTTP client to a locally running
    # Anki desktop with the AnkiConnect addon. Sync job (T4) hits this URL.
    # Pinned to 127.0.0.1 (NOT localhost) because AnkiConnect binds IPv4 only;
    # macOS /etc/hosts maps `localhost` to both 127.0.0.1 and ::1, and the
    # resolver inside uvicorn occasionally picks IPv6 first -> ConnectError
    # surfaced as `error="anki_not_running"` even though Anki is running.
    ANKICONNECT_URL: str = "http://127.0.0.1:8765"
    ANKI_DECK_NAME: str = "MileDown"
    ANKI_SYNC_INTERVAL_MINUTES: int = 15

    # Anki deck prefix (SPEC §V50, §V58, T73). mcat-coach-created filtered review
    # decks live under `<ANKI_DECK_PREFIX>::review::*`; the AnkiConnect write
    # allowlist (V50) constrains createFilteredDeck to this namespace.
    ANKI_DECK_PREFIX: str = "mcat-coach"

    # Anki assignment unlock scheduler (SPEC §T63, V51, V55). Hourly by
    # default — spec calls for `0 * * * *`, but the existing scheduler
    # uses interval triggers so we round to 60 minutes.
    ANKI_ASSIGNMENT_UNLOCK_INTERVAL_MINUTES: int = 60

    # Anki assignment auto-completion scheduler (SPEC §T64, V51). Daily
    # cron — spec default `15 5 * * *` (05:15 UTC). Override the hour or
    # minute independently via env (e.g. ANKI_ASSIGNMENT_COMPLETE_CRON_HOUR=6).
    ANKI_ASSIGNMENT_COMPLETE_CRON_HOUR: int = 5
    ANKI_ASSIGNMENT_COMPLETE_CRON_MINUTE: int = 15

    # Anki review-push scheduler (SPEC §T65, V53, V55). Hourly so a push
    # scheduled for "today" fires within ~1h of due time.
    ANKI_REVIEW_PUSH_INTERVAL_MINUTES: int = 60

    # P2 KB substrate (V-KB2, §I.env). PDF inbox = local directory the
    # ingest poller watches; Notion vars carry the write-out target
    # (V-N1: one-way mirror only). Tokens are optional at boot — only
    # required when the matching service runs. `validate_kb_config`
    # (app/kb_config.py) logs WARN for missing optionals at startup.
    PDF_INBOX_DIR: Path = _BACKEND_ROOT / "data" / "pdf_inbox"
    NOTION_API_TOKEN: str | None = None
    NOTION_WIKI_DB_ID: str | None = None

    # T51 KB job cadence. PDF inbox polled for new <slug>/*.pdf; Notion
    # write-out mirrors tagged atomic facts one-way (V-N1). Both no-op when
    # their config (OpenAI key / Notion token) is absent.
    PDF_INGEST_INTERVAL_MINUTES: int = 30
    NOTION_SYNC_INTERVAL_MINUTES: int = 60


settings = Settings()


def ensure_media_root() -> Path:
    """Create MEDIA_ROOT if missing; return absolute path."""
    settings.MEDIA_ROOT.mkdir(parents=True, exist_ok=True)
    return settings.MEDIA_ROOT
