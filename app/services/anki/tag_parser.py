"""AnkiHub tag parser for the AnKing MCAT Deck.

Implements the SPEC §V3 parse rules as amended 2026-05-19 (§B3) after the
first live sync revealed that AnKing tags do NOT use the bare `aamc::` /
`uworld::qid::` prefixes the original spec conjectured. The empirical
shape (verified across 31 distinct AAMC tags + thousands of qid tags):

- AAMC CC: `#AK_MCAT_v2::#AAMC::Concepts::<SECTION-SLASH>::Foundational_Concept_NN::<CC>-<text>`
  Section slashes (`C/P`, `B/B`, `P/S`) normalize to no-slash (`CP`, `BB`, `PS`);
  `CARS` passes through. AnKing tops out at CC granularity — there are no
  topic-level AAMC tags. Resolution therefore targets `content_category_id`,
  NOT `topic_id`. T32 will add an LLM categorizer pass to derive topic_id
  from card stem text + the parsed CC scope.
- UWorld qid: `#AK_MCAT_v2::#UWorld::<digits>`
- Anything else → `parsed_kind='unparsed'`.

`parsed_kind` is a strong claim: `'aamc_cc'` rows always carry a usable
`content_category_id`. AAMC-shaped tags that fail CC resolution are
demoted to `'unparsed'` rather than persisted with a null content_category_id.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from app.services.categorizer.outline_lookup import OutlineLookup


logger = logging.getLogger(__name__)


_AAMC_CC_RE = re.compile(
    r"^#AK_MCAT_v2::#AAMC::Concepts::(C/P|CARS|B/B|P/S)::Foundational_Concept_\d+::([0-9A-Z]+)-.+$"
)
_AAMC_SKILL_RE = re.compile(r"^#AK_MCAT_v2::#AAMC::Skills::Skill_([1-4])-.+$")
_UWORLD_QID_RE = re.compile(r"^#AK_MCAT_v2::#UWorld::(\d+)$")


_SECTION_NORMALIZE = {"C/P": "CP", "B/B": "BB", "P/S": "PS", "CARS": "CARS"}


@dataclass(frozen=True)
class ParsedTag:
    tag_raw: str
    topic_id: Optional[int]
    content_category_id: Optional[int]
    skill_number: Optional[int]
    question_qid: Optional[str]
    parsed_kind: str  # 'aamc_topic' | 'aamc_cc' | 'aamc_skill' | 'uworld_qid' | 'unparsed'


def parse_tag(tag: str, *, outline_lookup: OutlineLookup) -> ParsedTag:
    qid_match = _UWORLD_QID_RE.match(tag)
    if qid_match:
        return ParsedTag(
            tag_raw=tag,
            topic_id=None,
            content_category_id=None,
            skill_number=None,
            question_qid=qid_match.group(1),
            parsed_kind="uworld_qid",
        )

    skill_match = _AAMC_SKILL_RE.match(tag)
    if skill_match:
        return ParsedTag(
            tag_raw=tag,
            topic_id=None,
            content_category_id=None,
            skill_number=int(skill_match.group(1)),
            question_qid=None,
            parsed_kind="aamc_skill",
        )

    aamc_match = _AAMC_CC_RE.match(tag)
    if aamc_match:
        section_raw = aamc_match.group(1)
        cc_code = aamc_match.group(2)
        # Validate section normalization is sane; the regex group already
        # constrains the alternation, but keep the mapping explicit so the
        # invariant is greppable from a code review.
        section_code = _SECTION_NORMALIZE.get(section_raw)
        if section_code is None:  # pragma: no cover - regex already filters
            logger.debug("anki tag aamc-shaped but unexpected section %r in %r", section_raw, tag)
            return ParsedTag(
                tag_raw=tag,
                topic_id=None,
                content_category_id=None,
                skill_number=None,
                question_qid=None,
                parsed_kind="unparsed",
            )
        cc_id = outline_lookup.content_category_id(cc_code)
        if cc_id is None:
            logger.debug(
                "anki tag aamc-shaped but unknown CC %r in %r (section=%s)",
                cc_code,
                tag,
                section_code,
            )
            return ParsedTag(
                tag_raw=tag,
                topic_id=None,
                content_category_id=None,
                skill_number=None,
                question_qid=None,
                parsed_kind="unparsed",
            )
        return ParsedTag(
            tag_raw=tag,
            topic_id=None,
            content_category_id=cc_id,
            skill_number=None,
            question_qid=None,
            parsed_kind="aamc_cc",
        )

    logger.debug("anki tag unmatched by known rules: %r", tag)
    return ParsedTag(
        tag_raw=tag,
        topic_id=None,
        content_category_id=None,
        skill_number=None,
        question_qid=None,
        parsed_kind="unparsed",
    )
