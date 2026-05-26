"""Tests for outline_render — Ticket 6.8 recursion + path format.

These tests run without a DB session (the renderer reads the seed JSON
directly). All assertions are against real AAMC outline data.
"""

from __future__ import annotations

import re

from app.services.categorizer.outline_render import (
    canonical_identifiers_for_section,
    render_canonical_identifiers_block,
)


def test_canonical_identifiers_include_subtopics():
    ids = canonical_identifiers_for_section("CP")
    assert len(ids.topic_paths) > 150, (
        f"Expected more than 150 paths (depth-0 count), got {len(ids.topic_paths)}"
    )
    assert (
        "5A >> Solubility >> Solubility product constant; the equilibrium expression Ksp"
        in ids.topic_paths
    ), "Depth-1 child path missing from canonical list"


def test_canonical_identifiers_cap_at_max_depth():
    ids = canonical_identifiers_for_section("CP")
    # CC 5B has a depth-4 item: Covalent Bond >> Stereochemistry... >> Isomers >> Structural isomers
    depth4_path = "5B >> Covalent Bond >> Stereochemistry of covalently bonded molecules >> Isomers >> Structural isomers"
    assert depth4_path not in ids.topic_paths, (
        "Depth-4 path should be excluded (cap at _MAX_TOPIC_DEPTH=3)"
    )
    # But its parent (depth-3) IS included
    depth3_path = "5B >> Covalent Bond >> Stereochemistry of covalently bonded molecules >> Isomers"
    assert depth3_path in ids.topic_paths, "Depth-3 path should be included"


def test_canonical_identifiers_cars_unchanged():
    cars = canonical_identifiers_for_section("CARS")
    assert cars.topic_paths == ()
    assert cars.content_category_codes == ("CARS",)
    assert cars.skill_numbers == (1, 2, 3, 4)


def test_canonical_identifiers_path_format():
    cc_pattern = re.compile(r"^[A-Z0-9]+$")
    for section in ("CP", "BB", "PS"):
        ids = canonical_identifiers_for_section(section)
        for path in ids.topic_paths:
            parts = path.split(" >> ")
            assert len(parts) >= 2, f"Path {path!r} has fewer than 2 segments"
            assert cc_pattern.match(parts[0]), (
                f"First segment {parts[0]!r} in path {path!r} is not a CC code"
            )
            assert all(p for p in parts[1:]), f"Empty segment in path {path!r}"


def test_render_canonical_identifiers_block_mentions_leaf_first():
    block = render_canonical_identifiers_block("CP")
    assert "leaf" in block.lower() or "most specific" in block.lower(), (
        "Identifier block must instruct LLM to prefer leaf/most-specific paths"
    )
    assert "4A >> Translational Motion" in block


# --------------------------------------------------------------------------- #
# Ticket 6.8b — Unicode typographic apostrophe normalization
# --------------------------------------------------------------------------- #


def test_canonical_identifiers_normalize_typographic_apostrophes():
    """Piaget path in PS enum must use ASCII apostrophe, not U+2019."""
    ps = canonical_identifiers_for_section("PS")
    piaget_paths = [p for p in ps.topic_paths if "Piaget" in p]
    assert piaget_paths, "Piaget topic must appear in PS canonical paths"
    for path in piaget_paths:
        assert "'" in path, f"Piaget path must contain ASCII apostrophe: {path!r}"
    # No path in the entire enum should contain a Unicode curly apostrophe
    curly = "’"
    assert not any(curly in p for p in ps.topic_paths), (
        "U+2019 must not appear in any canonical path (normalization broke)"
    )


# --------------------------------------------------------------------------- #
# §V40 — reserved-delimiter guard
# --------------------------------------------------------------------------- #


def test_outline_topic_names_do_not_contain_reserved_delimiter():
    """§V40: no AAMC topic name may contain the reserved path delimiter ` >> `.

    The renderer joins ancestor names with ` >> ` and the parser splits on the
    same string; a topic name containing the delimiter would mis-segment and
    fail to resolve (cf §B5, where ` / ` collided with `Resistivity: ρ = R•A / L`).
    Walks the raw seed JSON so this catches future AAMC additions before they
    ever reach the renderer.
    """
    import json
    from pathlib import Path

    seed_path = Path(__file__).resolve().parents[1] / "app" / "seeds" / "aamc_outline.json"
    data = json.loads(seed_path.read_text(encoding="utf-8"))

    reserved = " >> "
    offenders: list[str] = []

    def walk(topics: list[dict]) -> None:
        for topic in topics or []:
            if reserved in topic.get("name", ""):
                offenders.append(topic["name"])
            walk(topic.get("children", []) or [])

    for section in data.get("sections", []):
        for fc in section.get("foundational_concepts", []) or []:
            for cc in fc.get("content_categories", []) or []:
                walk(cc.get("topics", []) or [])

    assert offenders == [], (
        f"§V40 violation — topic name contains reserved delimiter {reserved!r}: {offenders}"
    )
