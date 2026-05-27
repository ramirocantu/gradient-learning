"""T40 / V-TC1: the test suite must not read the real .env or hit the network
for config — settings come from process env + field defaults only.

These guards fail if a developer's .env ever bleeds back into the suite
(the B3 regression): a real COACH_TOKEN would flip auth tests to 401, a real
NOTION_API_TOKEN would point /admin/status probes at the live API, etc.
"""

from __future__ import annotations

import os

from app.config import Settings, settings


def test_dotenv_disabled_during_tests():
    # conftest sets this at import, before the settings singleton is built.
    assert os.environ.get("GRADIENT_DISABLE_DOTENV") == "1"


def test_singleton_uses_test_defaults_not_real_env():
    # COACH_TOKEN must be the field default — the value the auth tests hardcode.
    assert settings.COACH_TOKEN == "change_me_before_use"
    # External-service creds unset → probes report unconfigured (V16: no live calls).
    assert not settings.OPENAI_API_KEY
    assert settings.NOTION_API_TOKEN is None
    # DB pinned to the test database.
    assert settings.DATABASE_URL.endswith("/gradient_test")


def test_fresh_settings_ignores_real_dotenv():
    # A fresh Settings() (as test_settings.py constructs) must also skip the
    # real .env — proving the isolation is at the Settings layer, not a
    # one-off mutation of the singleton.
    s = Settings(DATABASE_URL="postgresql+asyncpg://x:y@localhost:5432/z")
    assert s.NOTION_API_TOKEN is None
    assert s.NOTION_WIKI_DB_ID is None
    assert s.COACH_TOKEN == "change_me_before_use"
