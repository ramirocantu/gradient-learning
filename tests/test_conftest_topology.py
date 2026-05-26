"""Regression tests for the unified-conftest fixture topology (Ticket R.2c).

R.2c collapsed three per-suite conftest files into a single root conftest
with one engine + one test DB (``gradient_test``). These tests guard
against a sibling re-introducing a second engine or pointing a fixture at
a different test database.
"""

from __future__ import annotations

from pathlib import Path


def test_one_test_engine_per_session() -> None:
    """Exactly one conftest.py defines a ``test_engine`` fixture under ``tests/``.

    Counts ``def test_engine`` occurrences across every ``conftest.py`` under
    ``backend/tests/``. The unified conftest is the sole source of truth — a
    drift back to per-suite conftests would surface here.
    """
    tests_root = Path(__file__).resolve().parent
    hits = []
    for path in tests_root.rglob("conftest.py"):
        text = path.read_text()
        if "def test_engine" in text:
            hits.append(str(path.relative_to(tests_root)))

    assert hits == ["conftest.py"], (
        f"Expected exactly one conftest.py to define ``test_engine`` "
        f"(the root one); found {hits!r}."
    )


def test_test_engine_points_at_gradient_test(test_engine) -> None:
    """The session-scoped engine binds to ``gradient_test`` exactly.

    R.2c retired the per-suite orphan databases. A regression that re-binds
    the engine to a different test database (e.g. with a suite suffix) would
    surface here.
    """
    url = str(test_engine.url)
    assert url.endswith("/gradient_test"), (
        f"test_engine URL should target gradient_test; got {url!r}"
    )
    # No suite-suffixed test DBs survive the consolidation.
    suite_suffixes = ("_web_" + "dashboard", "_web_" + "viewer")
    for suffix in suite_suffixes:
        assert ("gradient_test" + suffix) not in url
