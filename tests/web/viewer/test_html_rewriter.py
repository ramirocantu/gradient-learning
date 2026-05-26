"""Tests for the viewer's html_rewriter module after R.2b collapse.

R.2b deleted the bs4 ``rewrite_media_refs`` from the viewer module —
the dashboard's regex variant is now the single implementation. Only
``rewrite_choice_html`` survives in the viewer module.
"""

from __future__ import annotations


def test_rewrite_media_refs_viewer_path_uses_dashboard_impl() -> None:
    """The viewer route now imports rewrite_media_refs from the dashboard module.

    Asserts that the dashboard implementation handles the stem-HTML shape the
    viewer used to handle locally: a hash attr resolves to a same-origin
    ``/media/<local_path>`` src.
    """
    from app.web.dashboard.services.html_rewriter import rewrite_media_refs

    stem = '<p>see <img data-media-content-hash="abc123" alt="fig"></p>'
    out = rewrite_media_refs(stem, {"abc123": "ab/abc123.png"})

    assert 'src="/media/ab/abc123.png"' in out
    assert "data-media-content-hash" not in out


def test_rewrite_choice_html_unchanged() -> None:
    """Regression insurance: rewrite_choice_html still positionally resolves imgs.

    R.2b shouldn't touch this function — this confirms it.
    """
    from app.web.viewer.services.html_rewriter import rewrite_choice_html

    choice = '<p>A) <img data-media-content-hash="pending:0"></p>'
    out = rewrite_choice_html(choice, ["xy/choice0.png"])

    assert 'src="/media/xy/choice0.png"' in out
    assert "data-media-content-hash" not in out


def test_no_duplicate_rewrite_media_refs_export() -> None:
    """Regression guard: ``rewrite_media_refs`` lives in exactly one module."""
    from app.web.dashboard.services import html_rewriter as dash_hr
    from app.web.viewer.services import html_rewriter as viewer_hr

    assert hasattr(dash_hr, "rewrite_media_refs")
    assert not hasattr(viewer_hr, "rewrite_media_refs")
