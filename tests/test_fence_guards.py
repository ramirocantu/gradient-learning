"""V-RB1 guard: T17 fenced services do not self-document as stub / partial port.

Per V-RB1, the listed service modules must either be ported onto
OutlineNode + outline_subtree or explicitly fenced. "Fenced" rules out
docstrings / log messages that label the surface as a stub or
partial port — fenced surfaces are deliberately disabled, not
in-progress.

Also confirms:
  - the consuming routes are unmounted (V-RB1 "route-disabled" clause),
  - the feature-extraction scheduler entry is not registered (V-RB1
    "route-disabled" extended to background work).
"""

from __future__ import annotations

import inspect
import re

import pytest

import app.main as main_mod
import app.scheduler as scheduler_mod
import app.api.v1.tutor as tutor_api_mod
import app.web.dashboard.main as dashboard_main_mod

import app.services.analytics as analytics_mod
import app.services.recommender as recommender_mod
import app.services.tutor.outline as tutor_outline_mod
import app.services.analyzer as analyzer_mod
import app.services.analyzer.patterns as analyzer_patterns_mod
import app.services.analyzer.trajectory as analyzer_trajectory_mod
import app.web.dashboard.services.mastery as dash_mastery_mod
import app.web.dashboard.services.drilldown as dash_drilldown_mod
import app.web.dashboard.services.anki_scope as dash_anki_scope_mod
import app.services.anki.queries as anki_queries_mod
import app.services.anki.state as anki_state_mod
import app.services.anki.retention as anki_retention_mod
import app.api.v1.anki as anki_api_mod


# Match the literal words "stub" or "partial port" as standalone tokens
# (not "stubs" inside "stubs are fenced" — the V-RB1 check is about
# self-labelling the module's *purpose*, not banning the word).
_STUB_PATTERNS = [
    re.compile(r"\bT14 stub\b", re.IGNORECASE),
    re.compile(r"\bT13 stub\b", re.IGNORECASE),
    re.compile(r"\bT14 partial port\b", re.IGNORECASE),
    re.compile(r"\bpartial port\b", re.IGNORECASE),
    re.compile(r'"""Stub\b'),
    re.compile(r"^Stub —", re.MULTILINE),
    re.compile(r"TODO\(T(?:4|13|14)[^)]*\)"),
    # `<func_name> stub: ...` log/msg patterns — bans the legacy log shape
    # without flagging the allowed "FENCED, not a stub:" disclaimer.
    # Identifier ≥ 3 chars so "a stub:" / "an stub:" do not match.
    re.compile(r"\b[a-z_]{3,} stub: "),
]


_FENCED_MODULES = [
    analytics_mod,
    recommender_mod,
    tutor_outline_mod,
    analyzer_mod,
    analyzer_patterns_mod,
    analyzer_trajectory_mod,
    dash_mastery_mod,
    dash_drilldown_mod,
    dash_anki_scope_mod,
    anki_queries_mod,
    anki_state_mod,
    anki_retention_mod,
]


# V-RB2 SQL-pattern absence check — these substrings indicate live raw-SQL
# joins against the dropped `topics` / `content_categories` tables, or
# legacy column references on `anki_note_tags`. They must not appear in
# active code inside the V-RB2-scoped anki services.
_FORBIDDEN_LEGACY_SQL = [
    "FROM topics",
    "JOIN topics",
    "FROM content_categories",
    "JOIN content_categories",
    "topics.parent_topic_id",
    "t.topic_id",
    "tp.topic_id",
    "t.content_category_id",
    "tp.content_category_id",
]


_VRB2_MODULES = [anki_queries_mod, anki_state_mod, anki_retention_mod]


@pytest.mark.parametrize("module", _FENCED_MODULES, ids=lambda m: m.__name__)
def test_module_does_not_self_document_as_stub(module):
    """V-RB1: fenced modules ⊥ self-document as stub / partial port."""
    src = inspect.getsource(module)
    hits = [pat.pattern for pat in _STUB_PATTERNS if pat.search(src)]
    assert not hits, (
        f"{module.__name__} still contains stub/partial-port self-doc "
        f"patterns: {hits}"
    )


@pytest.mark.parametrize("module", _FENCED_MODULES, ids=lambda m: m.__name__)
def test_module_declares_fence(module):
    """V-RB1: fenced modules carry an explicit FENCED marker + a P0.5 task
    reference (T17 fenced rescope-out services; T18 extended the fence to
    the anki queries/state/retention surfaces)."""
    src = inspect.getsource(module)
    assert "FENCED" in src, f"{module.__name__} missing FENCED marker"
    assert ("T17" in src) or ("T18" in src), (
        f"{module.__name__} missing T17/T18 reference"
    )


def test_api_routers_for_fenced_surfaces_unmounted():
    """V-RB1 route-disabled clause — analytics/analyzer/recommendations
    routers are not mounted under `/api/v1/*`."""
    paths = {route.path for route in main_mod.app.routes}
    # Sub-routers contribute to `app.routes` flattened; check known paths.
    forbidden_prefixes = (
        "/api/v1/analytics",
        "/api/v1/analyzer",
        "/api/v1/recommendations",
    )
    bad = [p for p in paths if any(p.startswith(pre) for pre in forbidden_prefixes)]
    assert not bad, f"FENCED API routes still mounted: {bad}"


def test_tutor_outline_routes_disabled():
    """Tutor outline endpoints (search_topics + get_aamc_outline) are
    commented out in `app/api/v1/tutor.py`."""
    src = inspect.getsource(tutor_api_mod)
    # The decorators for the two FENCED routes must be inside comments.
    assert re.search(r"#\s*@router\.get\(\"/outline/topics/search\"\)", src), (
        "tutor /outline/topics/search route should be commented out (FENCED)"
    )
    assert re.search(r"#\s*@router\.get\(\"/outline\"\)", src), (
        "tutor /outline route should be commented out (FENCED)"
    )


def test_dashboard_routers_for_fenced_surfaces_unmounted():
    """V-RB1 route-disabled clause — dashboard mastery/topics/
    recommendations/insights routes are not registered."""
    src = inspect.getsource(dashboard_main_mod)
    for name in ("mastery", "topics", "recommendations", "insights"):
        active = re.search(rf"^app\.include_router\({name}\.router\)", src, re.MULTILINE)
        assert not active, (
            f"dashboard `{name}` route still mounted; expected FENCED comment-out"
        )


@pytest.mark.parametrize("module", _VRB2_MODULES, ids=lambda m: m.__name__)
def test_vrb2_no_legacy_sql_patterns(module):
    """V-RB2: anki queries/state/retention contain no live raw-SQL joins
    against `topics` / `content_categories` and no legacy `t.topic_id`
    / `t.content_category_id` column references."""
    src = inspect.getsource(module)
    hits = [pat for pat in _FORBIDDEN_LEGACY_SQL if pat in src]
    assert not hits, (
        f"{module.__name__} still contains forbidden legacy-SQL patterns: {hits}"
    )


def test_vrb2_anki_routes_disabled():
    """V-RB2 route-disabled clause — `/api/v1/anki/cards` (topic_id) and
    `/api/v1/anki/performance` are unmounted."""
    paths = {route.path for route in main_mod.app.routes}
    forbidden = {"/api/v1/anki/cards", "/api/v1/anki/performance"}
    bad = forbidden & paths
    assert not bad, f"FENCED anki routes still mounted: {sorted(bad)}"


def test_vrb2_anki_api_module_drops_fenced_imports():
    """V-RB2: `app.api.v1.anki` no longer imports the FENCED helpers in
    active code (all such imports are commented out)."""
    src = inspect.getsource(anki_api_mod)
    forbidden_imports = [
        "list_cards_for_topic",
        "state_for_cc",
        "state_for_topic",
        "retention_for_cc",
        "retention_for_topic",
    ]
    for name in forbidden_imports:
        for match in re.finditer(re.escape(name), src):
            line_start = src.rfind("\n", 0, match.start()) + 1
            line = src[line_start:match.start()]
            # Allow only commented-out occurrences.
            assert line.lstrip().startswith("#"), (
                f"`{name}` referenced in active code of app.api.v1.anki — "
                f"expected only commented-out occurrences (FENCED)"
            )


def test_scheduler_feature_extraction_unregistered():
    """V-RB1 — `run_feature_extraction` scheduler entry is not registered."""
    src = inspect.getsource(scheduler_mod.start_scheduler)
    # Only commented (line starts with #) occurrences of the job id are allowed.
    for match in re.finditer(r'id="run_feature_extraction"', src):
        line_start = src.rfind("\n", 0, match.start()) + 1
        line = src[line_start:match.start()]
        assert line.lstrip().startswith("#"), (
            "run_feature_extraction scheduler entry must be commented out (FENCED)"
        )
