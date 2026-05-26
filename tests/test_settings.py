"""Settings tests (SPEC §V58 amended — disjointness validator dropped per T73)."""

from pathlib import Path

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


# --------------------------------------------------------------------------- #
# T25 / V-KB2 / §I.env — P2 KB substrate fields
# --------------------------------------------------------------------------- #


def test_kb_substrate_notion_vars_default_none():
    s = Settings(**_REQUIRED)
    assert s.NOTION_API_TOKEN is None
    assert s.NOTION_WIKI_DB_ID is None


def test_kb_substrate_pdf_inbox_default_is_path():
    s = Settings(**_REQUIRED)
    assert isinstance(s.PDF_INBOX_DIR, Path)
    assert s.PDF_INBOX_DIR.name == "pdf_inbox"


def test_kb_substrate_embedding_model_default():
    # §C / §O: text-embedding-3-small is the single-provider default
    # (dim 1536). BGE-local is a config swap, not the default.
    s = Settings(**_REQUIRED)
    assert s.EMBEDDING_MODEL == "text-embedding-3-small"


def test_kb_substrate_pdf_inbox_overridable():
    s = Settings(**_REQUIRED, PDF_INBOX_DIR="/tmp/custom_inbox")
    assert s.PDF_INBOX_DIR == Path("/tmp/custom_inbox")


def test_kb_substrate_notion_vars_overridable():
    s = Settings(
        **_REQUIRED,
        NOTION_API_TOKEN="secret_abc",
        NOTION_WIKI_DB_ID="db-id-123",
    )
    assert s.NOTION_API_TOKEN == "secret_abc"
    assert s.NOTION_WIKI_DB_ID == "db-id-123"
