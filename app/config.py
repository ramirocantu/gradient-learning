from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


_BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_BACKEND_ROOT / ".env",
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
    OPENAI_CALIBRATOR_MODEL: str = "gpt-4.1-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Per-call model overrides. Default to OPENAI_MODEL; .env can split them.
    CATEGORIZER_MODEL: str = "gpt-4.1-mini"
    CATEGORIZER_CACHE_PATH: Path = _BACKEND_ROOT / "data" / "categorizer-cache.db"

    # Feature extractor (Ticket 4.2). Heavier judgment calls — pin to the
    # full GPT-4.1 by default; the spike (T5) may bump this to a thinking
    # model after a re-eval. Override via .env without code change.
    FEATURE_EXTRACTOR_MODEL: str = "gpt-4.1"
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
    # UWorld categorizer pattern (cheap chat model, structured output, SQLite cache).
    ANKI_TOPIC_RESOLVER_MODEL: str = "gpt-4.1-mini"
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
