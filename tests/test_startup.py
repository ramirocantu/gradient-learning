"""V-RB3 / V-O6 guard: startup ⊥ implicit-seed.

After T19, `app/startup.py` is gone and `app.main.lifespan` no longer
invokes any auto-seed path. Seed restoration = explicit re-upload of
`seeds/aamc_outline.schema.json` via
`POST /api/v1/courses/{id}/outline:import`.
"""

from __future__ import annotations

import importlib
import inspect

import pytest


def test_app_startup_module_removed():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("app.startup")


def test_main_lifespan_does_not_auto_seed():
    import app.main as main_mod

    src = inspect.getsource(main_mod.lifespan)
    assert "ensure_outline_seeded" not in src
    assert "seed_outline" not in src
    assert "app.startup" not in src
