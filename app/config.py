from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


_BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # If a shell exports an env var as empty (e.g. Claude Code sets
        # ANTHROPIC_API_KEY="" for sandbox safety), prefer the .env file value
        # over the empty OS env var.
        env_ignore_empty=True,
    )

    DATABASE_URL: str
    ANTHROPIC_API_KEY: str
    COACH_TOKEN: str = "change_me_before_use"
    MEDIA_ROOT: Path = _BACKEND_ROOT / "data" / "media"

    # 9.0: rendered into the get_flagged_attempts response as absolute deep-links.
    # 9.5 R.4: dashboard now lives in the same FastAPI process as the JSON API,
    # so a single BACKEND_BASE_URL covers both. Override via .env when the
    # process is reachable at a non-default origin.
    BACKEND_BASE_URL: str = "http://localhost:8000"

    # Categorizer (Ticket 3.4). Switched to Haiku after 3.5's eval showed 80% perfect-match
    # and 90% strong-match agreement with Sonnet on canonical outputs at ~2× lower cost.
    CATEGORIZER_MODEL: str = "claude-haiku-4-5-20251001"
    CATEGORIZER_CACHE_PATH: Path = _BACKEND_ROOT / "data" / "categorizer-cache.db"

    # Feature extractor (Ticket 4.2). Sonnet by default per CLAUDE.md — feature
    # extraction makes subjective judgment calls (distractor difficulty, jargon,
    # trap presence) where Sonnet's reasoning pays off. A future ticket may run
    # a Haiku eval, but the categorizer eval doesn't generalize: different task,
    # different signal-to-noise.
    FEATURE_EXTRACTOR_MODEL: str = "claude-sonnet-4-6"
    FEATURE_EXTRACTOR_CACHE_PATH: Path = _BACKEND_ROOT / "data" / "feature-extractor-cache.db"

    # Insight synthesizer (Ticket 4.5). Sonnet per CLAUDE.md convention.
    SYNTHESIZER_CACHE_PATH: Path = _BACKEND_ROOT / "data" / "synthesizer-cache.db"

    # Scheduler (Ticket 6.9b)
    SCHEDULER_ENABLED: bool = True
    CATEGORIZER_INTERVAL_MINUTES: int = 15
    CATEGORIZER_PER_RUN_BUDGET_USD: float = 0.50
    FEATURE_EXTRACTION_INTERVAL_MINUTES: int = 60

    # AnkiConnect (SPEC §T1, P11). Read-only HTTP client to a locally running
    # Anki desktop with the AnkiConnect addon. Sync job (T4) hits this URL.
    # Pinned to 127.0.0.1 (NOT localhost) because AnkiConnect binds IPv4 only;
    # macOS /etc/hosts maps `localhost` to both 127.0.0.1 and ::1, and the
    # resolver inside uvicorn occasionally picks IPv6 first -> ConnectError
    # surfaced as `error="anki_not_running"` even though Anki is running.
    ANKICONNECT_URL: str = "http://127.0.0.1:8765"
    ANKI_DECK_NAME: str = "MileDown"
    ANKI_SYNC_INTERVAL_MINUTES: int = 15

    # Anki topic resolver (SPEC §T32). LLM pass over cards already parsed as
    # aamc_cc — emits a topic_id suggestion under the parsed CC. Mirrors the
    # UWorld categorizer pattern (Haiku, structured output, SQLite cache).
    ANKI_TOPIC_RESOLVER_MODEL: str = "claude-haiku-4-5-20251001"
    ANKI_TOPIC_RESOLVER_CACHE_PATH: Path = _BACKEND_ROOT / "data" / "anki-topic-resolver-cache.db"
    ANKI_TOPIC_RESOLVER_INTERVAL_MINUTES: int = 60
    ANKI_TOPIC_RESOLVER_PER_RUN_BUDGET_USD: float = 0.50
    ANKI_TOPIC_RESOLVER_CONFIDENCE_THRESHOLD: float = 0.5

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


settings = Settings()


def ensure_media_root() -> Path:
    """Create MEDIA_ROOT if missing; return absolute path."""
    settings.MEDIA_ROOT.mkdir(parents=True, exist_ok=True)
    return settings.MEDIA_ROOT
