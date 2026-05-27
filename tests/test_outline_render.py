"""§V40 — reserved-delimiter guard over the raw AAMC seed JSON.

The legacy outline_render / canonical-identifier tests were removed with the
MCAT categorizer (T53). This guard reads the seed JSON directly and has no
dependency on deleted code, so it survives.
"""

from __future__ import annotations


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
