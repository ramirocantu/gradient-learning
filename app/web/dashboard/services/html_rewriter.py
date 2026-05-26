"""Regex-based media-ref rewriter for the dashboard.

Captured HTML carries ``data-media-content-hash="HASH"`` on each ``<img>``;
we rewrite those to ``src="{prefix}{local_path}"``. Post-9.5, media is
served at ``/media/*`` on the same origin as the dashboard, so the prefix
defaults to the relative ``/media/`` path.

Regex-only — no bs4 dep — because the only mutation we need is one attribute
substitution per matching ``<img>``.
"""

from __future__ import annotations

import re

MEDIA_PREFIX = "/media/"

_MEDIA_HASH_ATTR_RE = re.compile(r'data-media-content-hash="([^"]+)"')


def rewrite_media_refs(
    html: str | None,
    media_by_hash: dict[str, str],
    *,
    prefix: str = MEDIA_PREFIX,
) -> str:
    """Replace ``data-media-content-hash="H"`` with ``src="{prefix}{local_path}"``.

    Hashes absent from ``media_by_hash`` are stamped ``data-missing="true"``.
    Pending placeholders (``"pending:*"``) emitted by the scraper for choice
    images are also marked missing — the dashboard does not have positional
    choice-media context like the viewer does.
    """
    if not html:
        return html or ""

    def replace(match: re.Match[str]) -> str:
        h = match.group(1)
        if h.startswith("pending:"):
            return 'data-missing="true"'
        local_path = media_by_hash.get(h)
        if local_path is None:
            return 'data-missing="true"'
        return f'src="{prefix}{local_path}"'

    return _MEDIA_HASH_ATTR_RE.sub(replace, html)
