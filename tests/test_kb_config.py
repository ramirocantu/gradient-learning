"""T25 — KB substrate config validation (V-KB2, §I.env).

Covers `validate_kb_config`: emits one WARN per missing optional value,
returns the warning list, does not raise. The lifespan in `app/main.py`
calls it at startup so missing-optional state is logged but boot
proceeds — substrate services raise on first use instead.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from app.config import Settings
from app.kb_config import validate_kb_config


_REQUIRED = {
    "DATABASE_URL": "postgresql+asyncpg://gradient:gradient_secret@localhost:5432/x",
}


def _make_settings(**overrides) -> Settings:
    return Settings(**_REQUIRED, **overrides)


def test_all_optional_unset_yields_warnings(tmp_path: Path):
    # Default state: Notion vars are None, PDF_INBOX_DIR is the
    # default (likely non-existent on a fresh checkout). EMBEDDING_MODEL
    # has a hardcoded default in Settings, so that one stays clean.
    bogus_inbox = tmp_path / "does-not-exist"
    s = _make_settings(
        NOTION_API_TOKEN=None,
        NOTION_WIKI_DB_ID=None,
        PDF_INBOX_DIR=bogus_inbox,
    )
    warnings = validate_kb_config(s)

    joined = "\n".join(warnings)
    assert "NOTION_API_TOKEN unset" in joined
    assert "NOTION_WIKI_DB_ID unset" in joined
    assert "PDF_INBOX_DIR" in joined and "does not exist" in joined
    # EMBEDDING_MODEL has a default; no warning for it.
    assert "EMBEDDING_MODEL unset" not in joined


def test_all_optional_set_yields_no_warnings(tmp_path: Path):
    inbox = tmp_path / "pdf_inbox"
    inbox.mkdir()
    s = _make_settings(
        NOTION_API_TOKEN="secret_abc",
        NOTION_WIKI_DB_ID="11111111-2222-3333-4444-555555555555",
        PDF_INBOX_DIR=inbox,
        EMBEDDING_MODEL="text-embedding-3-small",
    )
    warnings = validate_kb_config(s)
    assert warnings == []


def test_empty_embedding_model_warns(tmp_path: Path):
    inbox = tmp_path / "pdf_inbox"
    inbox.mkdir()
    s = _make_settings(
        NOTION_API_TOKEN="secret_abc",
        NOTION_WIKI_DB_ID="db-id",
        PDF_INBOX_DIR=inbox,
        EMBEDDING_MODEL="",
    )
    warnings = validate_kb_config(s)
    assert any("EMBEDDING_MODEL unset" in w for w in warnings)


def test_warnings_emitted_on_logger(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    bogus_inbox = tmp_path / "missing"
    s = _make_settings(
        NOTION_API_TOKEN=None,
        NOTION_WIKI_DB_ID=None,
        PDF_INBOX_DIR=bogus_inbox,
    )
    with caplog.at_level(logging.WARNING, logger="app.kb_config"):
        validate_kb_config(s)

    records = [r for r in caplog.records if r.name == "app.kb_config"]
    assert len(records) == 3  # NOTION_API_TOKEN, NOTION_WIKI_DB_ID, PDF_INBOX_DIR
    assert all("kb_config:" in r.getMessage() for r in records)


def test_validator_never_raises(tmp_path: Path):
    # Even with everything unset the validator returns; it never raises.
    s = _make_settings(PDF_INBOX_DIR=tmp_path / "x")
    validate_kb_config(s)  # must not raise
