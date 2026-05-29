"""AnkiHub tag parser for the AnKing MCAT Deck (reference tag-shape plugin).

Implements the SPEC §V3 parse rules as amended 2026-05-19 (§B3) after the
first live sync revealed that AnKing tags do NOT use the bare `aamc::` /
`uworld::qid::` prefixes the original spec conjectured. The empirical
shape (verified across 31 distinct AAMC tags + thousands of qid tags):

- AAMC CC: `#AK_MCAT_v2::#AAMC::Concepts::<SECTION-SLASH>::Foundational_Concept_NN::<CC>-<text>`
  Section slashes (`C/P`, `B/B`, `P/S`) normalize to no-slash (`CP`, `BB`, `PS`);
  `CARS` passes through. AnKing tops out at CC granularity. T41: resolution
  targets the AAMC course's outline `node_id` (V-T1/V-O5 — the retired
  topic/cc/skill 3-target is gone). The tag's section + FC number + CC code
  reconstruct the outline path `"<SECTION> >> FC<NN> >> <CC>"`, resolved via
  `OutlineLookup.node_id_by_path` (V-O4).
- AAMC Skill: `#AK_MCAT_v2::#AAMC::Skills::Skill_N-...` — recognized but maps
  to NO outline node (the AAMC outline is Section>>FC>>CC, skills are not
  nodes), so `node_id=None`, `parsed_kind='aamc_skill'`.
- UWorld qid: `#AK_MCAT_v2::#UWorld::<digits>` → `parsed_kind='uworld_qid'`.
- Anything else → `parsed_kind='unparsed'`.

`parsed_kind` is the plugin's provenance claim; the only persisted resolution
target is `node_id` (V-T1). An aamc-shaped tag whose CC path fails to resolve
(or with no `OutlineLookup` available — e.g. the AAMC outline is unseeded) is
demoted to `'unparsed'` with `node_id=None` rather than guessed at.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from app.models.outline import OUTLINE_PATH_DELIMITER
from app.services.outline.lookup import OutlineLookup


logger = logging.getLogger(__name__)


_AAMC_CC_RE = re.compile(
    r"^#AK_MCAT_v2::#AAMC::Concepts::(C/P|CARS|B/B|P/S)::"
    r"Foundational_Concept_0*(\d+)::([0-9A-Z]+)-.+$"
)
_AAMC_SKILL_RE = re.compile(r"^#AK_MCAT_v2::#AAMC::Skills::Skill_([1-4])-.+$")
_UWORLD_QID_RE = re.compile(r"^#AK_MCAT_v2::#UWorld::(\d+)$")


_SECTION_NORMALIZE = {"C/P": "CP", "B/B": "BB", "P/S": "PS", "CARS": "CARS"}


@dataclass(frozen=True)
class ParsedTag:
    tag_raw: str
    node_id: Optional[int]
    question_qid: Optional[str]
    parsed_kind: str  # 'aamc_cc' | 'aamc_skill' | 'uworld_qid' | 'unparsed'


def _unparsed(tag: str) -> ParsedTag:
    return ParsedTag(tag_raw=tag, node_id=None, question_qid=None, parsed_kind="unparsed")


def parse_tag(tag: str, *, outline_lookup: Optional[OutlineLookup]) -> ParsedTag:
    """Classify one AnKing tag string → `ParsedTag` (node_id-targeted).

    `outline_lookup` may be None (the AAMC outline is unseeded); CC tags then
    demote to `'unparsed'` instead of resolving — sync stays soft (V4).
    """
    qid_match = _UWORLD_QID_RE.match(tag)
    if qid_match:
        return ParsedTag(
            tag_raw=tag,
            node_id=None,
            question_qid=qid_match.group(1),
            parsed_kind="uworld_qid",
        )

    skill_match = _AAMC_SKILL_RE.match(tag)
    if skill_match:
        # AAMC Skills are not outline nodes — recognized, but node_id stays None.
        return ParsedTag(
            tag_raw=tag,
            node_id=None,
            question_qid=None,
            parsed_kind="aamc_skill",
        )

    aamc_match = _AAMC_CC_RE.match(tag)
    if aamc_match:
        section_raw = aamc_match.group(1)
        fc_number = aamc_match.group(2)
        cc_segment = aamc_match.group(3)  # the CC node's name in the outline, e.g. "4E"
        section_code = _SECTION_NORMALIZE.get(section_raw)
        if section_code is None:  # pragma: no cover - regex already filters
            logger.debug("anki tag aamc-shaped but unexpected section %r in %r", section_raw, tag)
            return _unparsed(tag)
        if outline_lookup is None:
            logger.debug("anki tag aamc-shaped but no outline_lookup (unseeded) in %r", tag)
            return _unparsed(tag)
        path = OUTLINE_PATH_DELIMITER.join([section_code, f"FC{fc_number}", cc_segment])
        node_id = outline_lookup.node_id_by_path(path)
        if node_id is None:
            logger.debug(
                "anki tag aamc-shaped but path %r did not resolve to a node in %r",
                path,
                tag,
            )
            return _unparsed(tag)
        return ParsedTag(
            tag_raw=tag,
            node_id=node_id,
            question_qid=None,
            parsed_kind="aamc_cc",
        )

    logger.debug("anki tag unmatched by known rules: %r", tag)
    return _unparsed(tag)
