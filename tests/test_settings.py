"""Settings tests (SPEC §V58 amended — disjointness validator dropped per T73)."""

from app.config import Settings


_REQUIRED = {
    "DATABASE_URL": "postgresql+asyncpg://mcat:mcat_secret@localhost:5432/x",
    "ANTHROPIC_API_KEY": "sk-test",
}


def test_anki_deck_prefix_default():
    s = Settings(**_REQUIRED)
    assert s.ANKI_DECK_PREFIX == "mcat-coach"


def test_settings_accept_deck_name_under_prefix():
    """Per V58 amend (T73): disjointness validator dropped. Sync queries
    `findCards("deck:<name>")` (exact match), so filtered decks under
    `<prefix>::review::*` are siblings of the source deck — no overlap
    check needed at settings load."""
    s = Settings(
        **_REQUIRED,
        ANKI_DECK_PREFIX="mcat-coach",
        ANKI_DECK_NAME="mcat-coach::review::2026-05-21",
    )
    assert s.ANKI_DECK_NAME == "mcat-coach::review::2026-05-21"
