"""V-RB4 guard: tests/ contains no references to the dropped legacy
outline models (`Topic`, `ContentCategory`, `FoundationalConcept`,
`Section`) or the legacy `cc_code` parameter shape.

Per T20, all such test modules have been deleted (their surfaces are
either fully pruned or covered by FENCED service stubs). Any new
re-coverage must use `OutlineNode` + `node_id` / `outline_subtree`.

This file itself references the dropped symbols inside string literals
for the regex; the check whitelists this module and `test_fence_guards.py`
(which mentions legacy column names inside SQL-pattern allowlists).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


_TESTS_DIR = Path(__file__).parent

# `cc_code` is intentionally NOT in this list because:
#   - `tests/test_anki_queries_smoke.py` notes the kwarg's legacy status
#     in a comment without using it (T20 dropped the assertion lines that
#     called it).
#   - `tests/test_fence_guards.py` uses the bare string `cc_code` inside
#     V-RB2 forbidden-pattern messages.
# Both files are exempt via _EXEMPT below.
_FORBIDDEN_IMPORT_RE = re.compile(
    r"from\s+app\.models\.outline\s+import\s+[^\n]*\b(Topic|ContentCategory|FoundationalConcept|Section)\b"
)

_EXEMPT = {
    # This file references the forbidden tokens in the regex itself.
    "test_legacy_schema_purge.py",
    # Fence-guard test mentions legacy column names inside its SQL-pattern
    # allowlist (V-RB2 forbidden patterns).
    "test_fence_guards.py",
}


def _all_test_files() -> list[Path]:
    return sorted(_TESTS_DIR.rglob("test_*.py"))


@pytest.mark.parametrize(
    "path",
    [p for p in _all_test_files() if p.name not in _EXEMPT],
    ids=lambda p: str(p.relative_to(_TESTS_DIR)),
)
def test_no_dropped_outline_model_imports(path: Path) -> None:
    """V-RB4: no test imports the dropped legacy outline models."""
    src = path.read_text()
    hits = _FORBIDDEN_IMPORT_RE.findall(src)
    assert not hits, (
        f"{path.relative_to(_TESTS_DIR)} imports dropped outline models "
        f"({sorted(set(hits))}); either rewrite onto OutlineNode + node_id "
        f"or delete the test (V-RB4)."
    )


def test_no_cc_code_kwarg_in_active_test_code() -> None:
    """V-RB4: no test passes the legacy `cc_code=` kwarg to a service
    helper or API call. (Comments / docstrings are allowed — they may
    explain the legacy shape historically.)"""
    bad: list[str] = []
    for path in _all_test_files():
        if path.name in _EXEMPT:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            # Skip comment-only lines + lines inside docstrings (best-effort:
            # only flag occurrences of literal `cc_code=` not preceded by `#`
            # on the same line).
            if stripped.startswith("#"):
                continue
            if "cc_code=" in line:
                # Allow comment-suffix lines (`... # cc_code=...`) by
                # checking what precedes the token.
                idx = line.index("cc_code=")
                preceding = line[:idx]
                if "#" in preceding:
                    continue
                bad.append(f"{path.relative_to(_TESTS_DIR)}:{lineno}: {stripped}")
    assert not bad, (
        "tests still pass legacy `cc_code=` kwarg (V-RB4); rewrite onto "
        "OutlineNode + node_id or delete:\n  " + "\n  ".join(bad)
    )
