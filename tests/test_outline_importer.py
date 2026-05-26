"""Tests for the validate-then-materialize outline importer (§T9, V-O2, V-O4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.outline.importer import (
    OutlineSchemaValidationError,
    validate_outline_schema,
)


def _minimal_payload(extra_nodes: list[dict] | None = None) -> dict:
    """Smallest valid upload: one course, one root node."""
    return {
        "course": {"slug": "test-course", "name": "Test Course"},
        "nodes": [
            {"path": ["A"], "kind": "section", "name": "A", "position": 1},
            *(extra_nodes or []),
        ],
    }


def test_validate_minimal_ok():
    validated = validate_outline_schema(_minimal_payload())
    assert validated.course.slug == "test-course"
    assert len(validated.nodes_in_order) == 1
    assert validated.nodes_in_order[0].path == ("A",)


def test_validate_rejects_missing_course_slug():
    payload = _minimal_payload()
    payload["course"].pop("slug")
    with pytest.raises(OutlineSchemaValidationError) as exc:
        validate_outline_schema(payload)
    assert any("slug" in e for e in exc.value.errors)


def test_validate_rejects_duplicate_path():
    """V-O2: duplicate node path → whole-upload rejection."""
    extras = [
        {"path": ["A", "B"], "kind": "topic", "name": "B", "position": 1},
        {"path": ["A", "B"], "kind": "topic", "name": "B", "position": 2},
    ]
    with pytest.raises(OutlineSchemaValidationError) as exc:
        validate_outline_schema(_minimal_payload(extras))
    assert any("duplicates" in e for e in exc.value.errors)


def test_validate_rejects_broken_parent_chain():
    """V-O2: descendant without parent in upload → reject."""
    extras = [
        {"path": ["A", "B", "C"], "kind": "topic", "name": "C", "position": 1},
        # No "A >> B" — only "A" exists as a root.
    ]
    with pytest.raises(OutlineSchemaValidationError) as exc:
        validate_outline_schema(_minimal_payload(extras))
    assert any("parent" in e for e in exc.value.errors)


def test_validate_rejects_reserved_delimiter_in_name():
    """V-O4: ` >> ` is reserved; node names must not contain it."""
    extras = [
        {
            "path": ["A", "weird >> name"],
            "kind": "topic",
            "name": "weird >> name",
            "position": 1,
        }
    ]
    with pytest.raises(OutlineSchemaValidationError) as exc:
        validate_outline_schema(_minimal_payload(extras))
    assert any("delimiter" in e for e in exc.value.errors)


def test_validate_rejects_kind_depth_contradiction():
    """Same-depth nodes must share a kind."""
    payload = {
        "course": {"slug": "x", "name": "x"},
        "nodes": [
            {"path": ["A"], "kind": "section", "name": "A", "position": 1},
            {"path": ["B"], "kind": "unit", "name": "B", "position": 2},
        ],
    }
    with pytest.raises(OutlineSchemaValidationError) as exc:
        validate_outline_schema(payload)
    assert any("contradiction" in e for e in exc.value.errors)


def test_validate_rejects_name_path_mismatch():
    """`name` must match the last segment of `path`."""
    payload = _minimal_payload(
        [
            {"path": ["A", "B"], "kind": "topic", "name": "C", "position": 1},
        ]
    )
    with pytest.raises(OutlineSchemaValidationError) as exc:
        validate_outline_schema(payload)
    assert any("must match the last segment" in e for e in exc.value.errors)


def test_validate_rejects_depth_disagrees_with_path():
    """Explicit `depth` must equal `len(path) - 1`."""
    payload = _minimal_payload(
        [
            {
                "path": ["A", "B"],
                "kind": "topic",
                "name": "B",
                "position": 1,
                "depth": 5,
            }
        ]
    )
    with pytest.raises(OutlineSchemaValidationError) as exc:
        validate_outline_schema(payload)
    assert any("depth" in e for e in exc.value.errors)


def test_aamc_seed_validates():
    """V-O3: the bundled AAMC seed must round-trip through the validator."""
    payload = json.loads(
        Path("app/seeds/aamc_outline.schema.json").read_text()
    )
    validated = validate_outline_schema(payload)
    assert validated.course.slug == "aamc"
    # Spot-check: AAMC has 4 top-level sections.
    roots = [n for n in validated.nodes_in_order if len(n.path) == 1]
    assert {n.name for n in roots} == {"CP", "CARS", "BB", "PS"}
    # And many nodes overall (1554 at this version, give some headroom).
    assert len(validated.nodes_in_order) >= 1500


def test_validate_collects_multiple_errors():
    """Whole-upload-or-reject means the API can show ALL problems at once."""
    payload = {
        "course": {"name": "no slug"},
        "nodes": [
            {"path": [], "kind": "x", "name": ""},
            {"path": ["X"], "kind": "x", "name": "Y"},
        ],
    }
    with pytest.raises(OutlineSchemaValidationError) as exc:
        validate_outline_schema(payload)
    assert len(exc.value.errors) >= 3  # slug + each bad node


def test_materialization_order_puts_parents_first():
    """Materializer expects parents before children — sort key is (depth, path)."""
    payload = {
        "course": {"slug": "z", "name": "z"},
        "nodes": [
            {"path": ["A", "B"], "kind": "topic", "name": "B", "position": 1},
            {"path": ["A"], "kind": "section", "name": "A", "position": 1},
        ],
    }
    # Mixed kinds at different depths is OK.
    validated = validate_outline_schema(payload)
    ordered_paths = [n.path for n in validated.nodes_in_order]
    assert ordered_paths[0] == ("A",)
    assert ordered_paths[1] == ("A", "B")
