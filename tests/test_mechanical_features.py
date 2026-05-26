"""Tests for app.services.analyzer.mechanical_features.

Pure Python — no LLM, no DB. Question / Passage are duck-typed via
SimpleNamespace.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.analyzer.mechanical_features import (
    compute_mechanical_features,
)


def _q(
    *,
    passage_id=None,
    stem_plain="A box slides 2 m.",
    stem_html="<p>A box slides 2 m.</p>",
    explanation_html=None,
    choices=None,
):
    return SimpleNamespace(
        qid="q-test",
        passage_id=passage_id,
        stem_plain=stem_plain,
        stem_html=stem_html,
        explanation_html=explanation_html,
        choices=choices
        or [
            {"key": "A", "html": "<p>a</p>", "plain": "a", "media_ids": []},
            {"key": "B", "html": "<p>b</p>", "plain": "b", "media_ids": []},
        ],
    )


def _p(*, plain_text="passage text", html="<p>passage text</p>"):
    return SimpleNamespace(plain_text=plain_text, html=html)


def test_question_format_discrete_when_passage_id_null():
    f = compute_mechanical_features(_q(passage_id=None), passage=None)
    assert f.question_format == "discrete"
    assert f.passage_length_bucket is None


def test_question_format_passage_based_when_passage_present():
    f = compute_mechanical_features(_q(passage_id=42), passage=_p(plain_text="x" * 800))
    assert f.question_format == "passage_based"
    assert f.passage_length_bucket == "medium"


def test_has_negative_phrasing_matches_EXCEPT_NOT_LEAST():
    for stem in (
        "All of the following are true EXCEPT:",
        "Which of these is NOT a consequence?",
        "Which option is LEAST likely?",
    ):
        f = compute_mechanical_features(_q(stem_plain=stem), passage=None)
        assert f.has_negative_phrasing is True, stem


def test_has_negative_phrasing_does_not_match_cannot_knot():
    for stem in (
        "The reaction cannot proceed without a catalyst.",
        "The boatswain tied a knot.",
        "Several notes were attached to the paper.",
        "Not enough oxygen reaches the cells.",  # lowercase 'Not' starting a sentence shouldn't match either-case-only pattern.
    ):
        f = compute_mechanical_features(_q(stem_plain=stem), passage=None)
        assert f.has_negative_phrasing is False, stem


@pytest.mark.parametrize(
    "char_count,expected",
    [
        (0, "short"),
        (100, "short"),
        (499, "short"),
        (500, "medium"),
        (1000, "medium"),
        (1500, "medium"),
        (1501, "long"),
        (3000, "long"),
    ],
)
def test_passage_length_bucket_thresholds(char_count, expected):
    f = compute_mechanical_features(_q(passage_id=1), passage=_p(plain_text="x" * char_count))
    assert f.passage_length_bucket == expected
