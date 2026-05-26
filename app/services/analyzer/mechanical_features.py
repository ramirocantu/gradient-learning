"""Mechanical feature computation — pure Python, no LLM, no DB.

Pre-computes the features of a Question that don't need LLM judgment:
question_format, has_negative_phrasing, passage_length_bucket. These are passed
to the LLM as facts in the prompt preamble so it doesn't re-derive them.

involves_graph_or_figure and involves_data_table moved to LLM judgment (Ticket 4.3):
HTML-tag scanning produced false positives because UWorld embeds icons and
MathML on every page regardless of whether a substantive visual is present.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.models.captures import Passage, Question


PassageLengthBucket = Literal["short", "medium", "long"]
QuestionFormat = Literal["discrete", "passage_based"]


# Case-sensitive: UWorld ALL-CAPS these words to mark negative phrasing.
# `\b` boundaries avoid matching "cannot", "knot", "notes".
_NEGATIVE_PATTERN = re.compile(r"\b(?:EXCEPT|NOT|LEAST)\b")


@dataclass(frozen=True)
class MechanicalFeatures:
    question_format: QuestionFormat
    has_negative_phrasing: bool
    passage_length_bucket: PassageLengthBucket | None


def _bucket_passage_length(plain_text: str | None) -> PassageLengthBucket:
    length = len(plain_text or "")
    if length < 500:
        return "short"
    if length <= 1500:
        return "medium"
    return "long"


def compute_mechanical_features(
    question: Question,
    passage: Passage | None,
) -> MechanicalFeatures:
    """Pre-compute mechanical features. See module docstring."""
    is_passage_based = question.passage_id is not None and passage is not None
    question_format: QuestionFormat = "passage_based" if is_passage_based else "discrete"

    has_negative = bool(_NEGATIVE_PATTERN.search(question.stem_plain or ""))

    bucket: PassageLengthBucket | None
    if is_passage_based:
        bucket = _bucket_passage_length(passage.plain_text if passage else None)
    else:
        bucket = None

    return MechanicalFeatures(
        question_format=question_format,
        has_negative_phrasing=has_negative,
        passage_length_bucket=bucket,
    )
