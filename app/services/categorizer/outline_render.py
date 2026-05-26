"""Render slices of the AAMC outline as Markdown for the LLM categorizer prompt.

Reads directly from the seed JSON (source of truth, ~600 lines) rather than
the DB so the renderer works without a session. Cached at module level —
the outline never changes at runtime.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.services.categorizer._text import normalize_typographic_punctuation

_OUTLINE_PATH = Path(__file__).resolve().parent.parent.parent / "seeds" / "aamc_outline.json"

# Mapping from a UWorld `Subject:` value to the AAMC section the question
# belongs to. Unknown subjects return None and the categorizer fails fast.
SUBJECT_TO_SECTION: dict[str, str] = {
    "General Chemistry": "CP",
    "Organic Chemistry": "CP",
    "Physics": "CP",
    "Biology": "BB",
    "Biochemistry": "BB",
    "Behavioral Sciences": "PS",
    "Psychology": "PS",
    "Sociology": "PS",
    "Critical Analysis and Reasoning Skills": "CARS",
}

_MAX_TOPIC_DEPTH = 3


@functools.lru_cache(maxsize=1)
def _load_outline() -> dict[str, Any]:
    return json.loads(_OUTLINE_PATH.read_text())


def _render_topic(topic: dict[str, Any], depth: int = 1) -> list[str]:
    if depth > _MAX_TOPIC_DEPTH:
        return []
    indent = "  " * (depth - 1)
    prefix = "Topic" if depth == 1 else "Subtopic"
    lines = [f"{indent}- {prefix}: {topic['name']}"]
    for child in topic.get("children", []) or []:
        lines.extend(_render_topic(child, depth=depth + 1))
    return lines


@functools.lru_cache(maxsize=8)
def render_outline_for_section(section_code: str) -> str:
    """Markdown rendering of one section's outline subtree, capped at 3 topic levels."""
    outline = _load_outline()
    section = next((s for s in outline["sections"] if s["code"] == section_code), None)
    if section is None:
        return f"# Unknown section {section_code!r}\n(no outline available)\n"

    if section_code == "CARS":
        return (
            "# Section CARS — Critical Analysis and Reasoning Skills\n"
            "\n"
            "CARS tests reading comprehension and reasoning. The AAMC publishes "
            "**no** content categories, topics, or subtopics for this section.\n"
            "\n"
            "When categorizing a CARS question, emit only `kind='skill'` tags "
            "(1=Knowledge of Scientific Principles is rare here; 2=Reasoning is "
            "the typical CARS skill). Do not emit topic or content_category tags.\n"
        )

    lines: list[str] = [f"# Section {section['code']} — {section['name']}", ""]

    for fc in section.get("foundational_concepts", []) or []:
        lines.append(f"## Foundational Concept {fc['code']} — {fc['name']}")
        lines.append("")
        for cc in fc.get("content_categories", []) or []:
            lines.append(f"### Content Category {cc['code']} — {cc['name']}")
            description = cc.get("description")
            if description:
                lines.append(f"_{description}_")
            for topic in cc.get("topics", []) or []:
                lines.extend(_render_topic(topic, depth=1))
            lines.append("")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# Ticket 3.5 / 6.8: canonical identifier lists.
#
# The LLM gets a closed enum of valid identifiers. Each topic is addressed by
# its fully-qualified path "<CC_CODE> / <Parent> / <Child>", recursed to
# _MAX_TOPIC_DEPTH levels. Depth-0 (top-level) paths are also included so
# the LLM can fall back to the parent when no child fits.
#
# Why recurse to depth 3 (not every leaf):
#   - The AAMC outline has up to 5 levels of nesting. Emitting all leaves
#     would produce ~1500+ enum values and push the tool schema well past
#     token budget for a single call.
#   - Depth 3 captures the most clinically actionable granularity (e.g.
#     "5A / Solubility / Ksp") without enumerating obscure depth-4/5
#     terminology that the LLM already maps via topic-name semantics.
#   - The enum is cached in the Anthropic prompt cache (cache_control=ephemeral
#     on the system block), so the ~10× growth in enum size is a one-time
#     cache-miss cost, not a per-question cost.
# --------------------------------------------------------------------------- #


def _collect_paths(topics: list[Any], prefix: str, depth: int = 1) -> list[str]:
    """Recursively collect fully-qualified topic paths up to _MAX_TOPIC_DEPTH.

    Topic names are ASCII-normalized (curly quotes → straight) so the LLM enum
    contains only ASCII apostrophes and the lookup side can match regardless of
    which apostrophe variant the LLM echoes back.
    """
    if depth > _MAX_TOPIC_DEPTH:
        return []
    paths: list[str] = []
    for topic in topics:
        name = normalize_typographic_punctuation(topic["name"])
        path = f"{prefix} >> {name}"
        paths.append(path)
        children = topic.get("children", []) or []
        if children:
            paths.extend(_collect_paths(children, path, depth + 1))
    return paths


@dataclass(frozen=True)
class CanonicalIdentifiers:
    """Closed sets of valid identifiers for one section."""

    topic_paths: tuple[str, ...] = field(default_factory=tuple)
    content_category_codes: tuple[str, ...] = field(default_factory=tuple)
    skill_numbers: tuple[int, ...] = (1, 2, 3, 4)


@functools.lru_cache(maxsize=8)
def canonical_identifiers_for_section(section_code: str) -> CanonicalIdentifiers:
    """Return the closed lists of valid identifiers for a given section.

    Topic paths recurse to _MAX_TOPIC_DEPTH so the LLM can choose the most
    specific matching path. Both parent and child paths are included; the LLM
    must prefer the deepest (leaf-first).

    CARS: topic_paths empty, content_category_codes=('CARS',). Skill always
    1-4.
    """
    if section_code == "CARS":
        return CanonicalIdentifiers(
            topic_paths=(),
            content_category_codes=("CARS",),
        )

    outline = _load_outline()
    section = next((s for s in outline["sections"] if s["code"] == section_code), None)
    if section is None:
        return CanonicalIdentifiers()

    topic_paths: list[str] = []
    cc_codes: list[str] = []
    for fc in section.get("foundational_concepts", []) or []:
        for cc in fc.get("content_categories", []) or []:
            cc_codes.append(cc["code"])
            topic_paths.extend(_collect_paths(cc.get("topics", []) or [], cc["code"]))

    return CanonicalIdentifiers(
        topic_paths=tuple(topic_paths),
        content_category_codes=tuple(cc_codes),
    )


def render_canonical_identifiers_block(section_code: str) -> str:
    """Markdown block of canonical identifier lists, appended to the prompt.

    The topic list is leaf-first: both parent and child paths are present.
    The LLM must prefer the deepest matching path (most specific topic).
    """
    ids = canonical_identifiers_for_section(section_code)
    lines = [
        "=== CANONICAL IDENTIFIERS FOR THIS SECTION ===",
        "",
        "You MUST pick identifiers verbatim from these lists. Do not paraphrase, "
        "abbreviate, or invent identifiers. If a concept does not appear in these "
        "lists, do not tag it.",
        "",
        "IMPORTANT — leaf-first rule: the list contains BOTH parent topics and "
        "their children. Always prefer the most specific (deepest) matching path. "
        "Only use a parent path when none of its children genuinely apply.",
        "",
    ]
    if ids.topic_paths:
        lines.append(
            "Valid topic identifiers (fully-qualified, "
            "`<content_category_code> >> <topic_name>` or deeper):"
        )
        lines.extend(f"- {p}" for p in ids.topic_paths)
        lines.append("")
    else:
        lines.append("Valid topic identifiers: (none — this section has no AAMC topics)")
        lines.append("")
    lines.append("Valid content category codes:")
    lines.extend(f"- {c}" for c in ids.content_category_codes)
    lines.append("")
    lines.append(f"Valid skill numbers: {', '.join(str(n) for n in ids.skill_numbers)}")
    return "\n".join(lines) + "\n"
