"""V-RB5 fence guard: ⊥ anthropic imports anywhere in app/ or scripts/.

T4 made OpenAI the single LLM provider (§C, V16). T36 cleaned the
residual `from anthropic import …` imports from the seven harness
scripts that survived the pivot. This guard ensures no new code
reintroduces an anthropic dependency — the SDK is no longer pinned in
`pyproject.toml` and a stray import would either:
  (a) collection-error the suite as soon as `uv sync` trims the
      transient install (the B2 trap), or
  (b) silently re-introduce a second LLM provider, breaking the V16
      single-provider boundary.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCAN_ROOTS = [_REPO_ROOT / "app", _REPO_ROOT / "scripts"]
_ANTHROPIC_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+anthropic(?:\.[\w.]+)?\s+import\s|import\s+anthropic(?:\s|$|\.))",
    re.MULTILINE,
)


def _gather_py_files() -> list[Path]:
    files: list[Path] = []
    for root in _SCAN_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            # Skip caches.
            if "__pycache__" in path.parts:
                continue
            files.append(path)
    return files


def test_v_rb5_no_anthropic_imports_in_app_or_scripts():
    offenders: list[str] = []
    for path in _gather_py_files():
        src = path.read_text(encoding="utf-8", errors="replace")
        for match in _ANTHROPIC_IMPORT_RE.finditer(src):
            line_no = src.count("\n", 0, match.start()) + 1
            rel = path.relative_to(_REPO_ROOT)
            offenders.append(f"{rel}:{line_no}: {match.group(0).strip()}")

    assert not offenders, (
        "V-RB5: anthropic imports must not appear in app/ or scripts/ "
        "post-pivot (T4). OpenAI is the single LLM provider — port the "
        "offender to the OpenAI SDK (see app/services/llm/) or delete it.\n" + "\n".join(offenders)
    )


def test_v_rb5_anthropic_not_in_pyproject():
    pyproject = (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    # Allow the word inside comments or unrelated metadata but reject
    # `"anthropic..."` in a dependency-list entry. We match the canonical
    # PEP 508 prefix.
    bad_lines = [
        line for line in pyproject.splitlines() if re.search(r"^\s*[\"']anthropic[\"\s<>=!~]", line)
    ]
    assert not bad_lines, (
        "V-RB5: pyproject.toml must not depend on `anthropic` post-pivot. "
        "Offending lines:\n" + "\n".join(bad_lines)
    )
