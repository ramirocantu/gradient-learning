"""V-RB6 guard: no LIVE write path references the retired 3-target tag shape.

The V-T1 migration retired the PoC `(topic_id | content_category_id | skill)`
3-target — `node_id` is the only tag target. B4 ported the anki sync/tag_parser
path and widened V-RB2, but V-RB2's list is anki-only and missed two more live
write paths (B6, B7):

  B6  ``app/api/v1/admin.py`` — ``ManualTagBody`` / the ``create_manual_tag``
      route / ``_tag_row_payload`` still build, forward, and read
      ``topic_id`` / ``content_category_id`` / ``skill``. The service
      ``app/services/admin_tags.py:create_manual_tag`` is node_id-only, so the
      route raises ``TypeError`` (bad kwargs) / ``AttributeError`` (dropped
      ``QuestionTag`` columns) on any call.
  B7  ``app/services/anki/assignment.py`` — ``_CANDIDATE_SQL_TOPIC`` /
      ``_CANDIDATE_SQL_CC`` JOIN the dropped ``topics`` / ``content_categories``
      tables and reference ``anki_note_tags.topic_id`` / ``content_category_id``
      / ``parsed_kind``, so ``POST /api/v1/anki/assignments`` fails at the SQL
      layer.

V-RB6 is behavior-scoped: no retired-3-target token in any live (non-fenced)
write path. The fix is T57. Until then these assertions ``xfail`` (strict) so
the suite stays green AND the tripwire flips to a hard failure the moment the
files are ported — at which point delete the ``xfail`` markers.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Retired 3-target identifiers + the dropped tables/enums. `node_*` is excluded
# deliberately; bare `skill` is too generic for a line regex (the structural
# `ManualTagBody` field is caught by the admin.py scan via `skill:`-style hits
# on the other two targets, which always co-occur in that write path).
_RETIRED_RE = re.compile(
    r"\b(?:topic_id|content_category_id|cc_code|skill_number|parsed_kind"
    r"|content_categories|aamc_topic|aamc_cc)\b"
)


def _retired_hits(rel_path: str) -> list[str]:
    text = (_REPO_ROOT / rel_path).read_text(encoding="utf-8")
    return [
        f"{rel_path}:{lineno}: {line.strip()}"
        for lineno, line in enumerate(text.splitlines(), start=1)
        if _RETIRED_RE.search(line)
    ]


def test_v_rb6_admin_manual_tag_is_node_id_only() -> None:
    hits = _retired_hits("app/api/v1/admin.py")
    assert not hits, (
        "retired 3-target tokens in admin manual-tag write path (B6, V-RB6):\n" + "\n".join(hits)
    )


def test_v_rb6_anki_assignment_is_node_id_only() -> None:
    hits = _retired_hits("app/services/anki/assignment.py")
    assert not hits, (
        "retired 3-target tokens in anki assignment candidate SQL (B7, V-RB6):\n" + "\n".join(hits)
    )
