"""Guard against accidental re-introduction of DASHBOARD_BASE_URL.

The setting was collapsed into BACKEND_BASE_URL in Phase 9.5 R.4 because
the dashboard now serves on the same origin as the JSON API.
"""

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_no_dashboard_base_url_in_tracked_code():
    result = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "grep",
            "-l",
            "-i",
            "dashboard_base_url",
        ],
        capture_output=True,
        text=True,
    )
    # `git grep -l` returns 0 if matches found, 1 if none. We want 1.
    hits = result.stdout.strip().splitlines()
    # Allow this test file itself + PLAN docs + the R.* kickoffs that
    # document the collapse itself. The kickoffs are dev artifacts; the
    # regression target is real config.
    allowed = {
        "backend/tests/test_no_dashboard_base_url.py",
        "project-management/PLAN.md",
        "project-management/PLAN-archive.md",
        "project-management/R0-findings.md",
        "project-management/kickoffs/ticket-R.0-investigation.md",
        "project-management/kickoffs/ticket-R.1-mechanical-merge.md",
        "project-management/kickoffs/ticket-R.2a-drop-admin-httpx-proxy.md",
        "project-management/kickoffs/ticket-R.2b-collapse-viewer-html-rewriter.md",
        "project-management/kickoffs/ticket-R.2c-conftest-unification.md",
        "project-management/kickoffs/ticket-R.2d-llm-cache-base.md",
        "project-management/kickoffs/ticket-R.3-dead-code-deletion.md",
        "project-management/kickoffs/ticket-R.4-distribution-doc-sweep.md",
        # Pre-9.5 kickoffs predate the collapse and reference the old field
        # name as part of their original design. Out of scope for R.4.
        "project-management/kickoffs/ticket-7.3-send-to-things-button.md",
        "project-management/kickoffs/ticket-8.2-mcp-end-of-session-planning-tools.md",
        "project-management/kickoffs/ticket-9.1-mcpb-bundle-healthcheck.md",
        "project-management/kickoffs/ticket-9.3-install-script-smoke.md",
    }
    unexpected = [h for h in hits if h not in allowed]
    assert not unexpected, f"DASHBOARD_BASE_URL references found in: {unexpected}"
