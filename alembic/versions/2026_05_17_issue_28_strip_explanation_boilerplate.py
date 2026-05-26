"""Issue #28 — strip UWorld "Explanation/User Id" header + standards/copyright
footer from existing questions.explanation_html / explanation_plain.

Revision ID: 20260517_iss28
Revises: 20260516_t69d
Create Date: 2026-05-17

The scraper now targets `#first-explanation` (the active tab-pane that holds
only the body) rather than the outer `#explanation-container`. New captures
land clean; this migration rewrites historical rows so the LLM categorizer
and feature extractor see the same shape after the next `extractor_version`
bump triggers re-derivation.

Pure-Python rewriter (no new deps). For each row whose `explanation_html`
contains a `<div ... id="first-explanation" ...>`, extract the inner content
via balanced `<div>` counting and rebuild `explanation_plain` via stdlib
`html.parser`.

Downgrade is a no-op: the original boilerplate-laden text isn't recoverable
from the rewritten row. `raw_captures.raw_html` retains the full original
DOM for any forensic re-derivation.
"""

from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser
from typing import Sequence, Union

from alembic import op

revision: str = "20260517_iss28"
down_revision: Union[str, None] = "20260516_t69d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_FIRST_EXP_OPEN = re.compile(r'<div\b[^>]*\bid="first-explanation"[^>]*>', re.IGNORECASE)
_DIV_OPEN = re.compile(r"<div\b[^>]*>", re.IGNORECASE)
_DIV_CLOSE = re.compile(r"</div\s*>", re.IGNORECASE)
_SCRIPT_OR_STYLE = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_WS_RUN = re.compile(r"[ \t]+")
_BLANK_RUN = re.compile(r"\n{3,}")


def _extract_first_explanation_inner(html: str) -> str | None:
    """Return inner HTML of `<div id="first-explanation" ...>` or None on miss."""
    m = _FIRST_EXP_OPEN.search(html)
    if m is None:
        return None
    start = m.end()
    depth = 1
    i = start
    while i < len(html):
        open_m = _DIV_OPEN.search(html, i)
        close_m = _DIV_CLOSE.search(html, i)
        if close_m is None:
            return None
        if open_m is not None and open_m.start() < close_m.start():
            depth += 1
            i = open_m.end()
            continue
        depth -= 1
        if depth == 0:
            return html[start : close_m.start()]
        i = close_m.end()
    return None


class _PlainTextExtractor(HTMLParser):
    """Best-effort HTML → plain text; mirrors text.ts htmlToPlainText behaviour
    closely enough for the categorizer/feature extractor prompts.
    """

    _BLOCK = {
        "p",
        "div",
        "br",
        "li",
        "tr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "blockquote",
        "pre",
        "section",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._buf: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs):  # noqa: ANN001
        if tag in {"script", "style"}:
            self._skip_depth += 1
            return
        if tag in self._BLOCK:
            self._buf.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if tag in self._BLOCK:
            self._buf.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        self._buf.append(data)

    def get_text(self) -> str:
        raw = "".join(self._buf)
        raw = unescape(raw)
        # Normalize: collapse runs of spaces, cap blank lines, trim each line.
        lines = [_WS_RUN.sub(" ", ln).strip() for ln in raw.splitlines()]
        out = "\n".join(lines)
        out = _BLANK_RUN.sub("\n\n", out).strip()
        return out


def _html_to_plain(html: str) -> str:
    parser = _PlainTextExtractor()
    parser.feed(_SCRIPT_OR_STYLE.sub("", html))
    parser.close()
    return parser.get_text()


def upgrade() -> None:
    bind = op.get_bind()
    rows = bind.exec_driver_sql(
        "SELECT id, explanation_html FROM questions "
        "WHERE explanation_html IS NOT NULL "
        "AND explanation_html LIKE '%id=\"first-explanation\"%'"
    ).fetchall()

    cleaned = 0
    skipped = 0
    for row in rows:
        qid, html = row[0], row[1]
        inner = _extract_first_explanation_inner(html)
        if inner is None or not inner.strip():
            skipped += 1
            continue
        new_plain = _html_to_plain(inner)
        bind.exec_driver_sql(
            "UPDATE questions SET explanation_html = $1, explanation_plain = $2 WHERE id = $3",
            (inner, new_plain, qid),
        )
        cleaned += 1

    print(f"issue_28: rewrote {cleaned} rows; skipped {skipped} (no #first-explanation found)")

    # Re-trigger LLM workers against the cleaned text. Preserve manual overrides
    # (is_overridden=true) — those reflect user corrections that should survive
    # a re-categorization pass.
    bind.exec_driver_sql("DELETE FROM question_tags WHERE source = 'llm' AND is_overridden = false")
    bind.exec_driver_sql("UPDATE questions SET needs_categorization = true")
    bind.exec_driver_sql("DELETE FROM question_features")


def downgrade() -> None:
    # Not recoverable from the rewritten row. raw_captures.raw_html retains
    # the original DOM for any future re-derivation.
    pass
