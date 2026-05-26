"""Tests for the SQLite-backed CategorizerCache (Ticket 3.4)."""

from __future__ import annotations

from pathlib import Path

from app.services.categorizer.cache import CategorizerCache
from app.services.categorizer.llm import (
    CategorizeResult,
    LlmTagSuggestion,
    make_cache_key,
)


def _result(*, model: str = "claude-sonnet-4-6", cost: float = 0.0125) -> CategorizeResult:
    return CategorizeResult(
        suggestions=[
            LlmTagSuggestion(
                kind="topic",
                identifier="Work",
                under_content_category="4A",
                confidence=0.9,
                rationale="W=Fd",
            ),
            LlmTagSuggestion(
                kind="skill",
                identifier=2,
                under_content_category=None,
                confidence=0.85,
                rationale="calc",
            ),
        ],
        primary_aamc_section="CP",
        cache_hit=False,
        input_tokens=1200,
        output_tokens=150,
        estimated_cost_usd=cost,
        extractor_version="v1",
        parse_warnings=[],
        cost_saved_usd=0.0,
        model=model,
    )


def test_cache_round_trip(tmp_path: Path):
    cache = CategorizerCache(tmp_path / "cache.db")
    key = make_cache_key("stem", "expl", ["Subject: Physics"], "m1")
    cache.put(key, _result(model="m1"), "v1", model="m1")
    got = cache.get(key, "v1")
    assert got is not None
    assert got.cache_hit is True
    assert got.estimated_cost_usd == 0.0
    assert len(got.suggestions) == 2
    assert got.suggestions[0].identifier == "Work"
    assert got.suggestions[0].under_content_category == "4A"
    assert got.suggestions[1].kind == "skill"
    assert got.suggestions[1].identifier == 2
    cache.close()


def test_cache_miss_on_unknown_key(tmp_path: Path):
    cache = CategorizerCache(tmp_path / "cache.db")
    assert cache.get("nonsense-key", "v1") is None
    cache.close()


def test_cache_miss_on_version_mismatch(tmp_path: Path):
    cache = CategorizerCache(tmp_path / "cache.db")
    key = make_cache_key("stem", "expl", ["t"], "m1")
    cache.put(key, _result(model="m1"), "v1", model="m1")
    assert cache.get(key, "v1") is not None
    assert cache.get(key, "v2") is None
    cache.close()


def test_cache_persists_across_instances(tmp_path: Path):
    path = tmp_path / "cache.db"
    a = CategorizerCache(path)
    key = make_cache_key("stem", "expl", ["t"], "m1")
    a.put(key, _result(model="m1"), "v1", model="m1")
    a.close()

    b = CategorizerCache(path)
    got = b.get(key, "v1")
    assert got is not None
    assert got.suggestions[0].identifier == "Work"
    b.close()


def test_cache_clear_by_version(tmp_path: Path):
    cache = CategorizerCache(tmp_path / "cache.db")
    k1 = make_cache_key("s1", "e1", ["t"], "m1")
    k2 = make_cache_key("s2", "e2", ["t"], "m1")
    k3 = make_cache_key("s3", "e3", ["t"], "m1")
    cache.put(k1, _result(model="m1"), "v1", model="m1")
    cache.put(k2, _result(model="m1"), "v1", model="m1")
    cache.put(k3, _result(model="m1"), "v2", model="m1")

    deleted = cache.clear(extractor_version="v1")
    assert deleted == 2
    assert cache.get(k1, "v1") is None
    assert cache.get(k3, "v2") is not None
    cache.close()


def test_cache_stats(tmp_path: Path):
    cache = CategorizerCache(tmp_path / "cache.db")
    for i, (ver, model) in enumerate([("v1", "m1"), ("v1", "m1"), ("v2", "m2")]):
        k = make_cache_key(f"s{i}", "e", ["t"], model)
        cache.put(k, _result(model=model, cost=0.01), ver, model=model)
    stats = cache.stats()
    assert stats["total_entries"] == 3
    assert stats["by_version"] == {"v1": 2, "v2": 1}
    assert stats["by_model"] == {"m1": 2, "m2": 1}
    assert abs(stats["total_cost_saved_usd"] - 0.03) < 1e-9
    cache.close()


def test_cache_key_includes_model(tmp_path: Path):
    """Different models on same content should produce distinct keys."""
    sonnet_key = make_cache_key("stem", "expl", ["t"], "claude-sonnet-4-6")
    haiku_key = make_cache_key("stem", "expl", ["t"], "claude-haiku-4-5-20251001")
    assert sonnet_key != haiku_key

    cache = CategorizerCache(tmp_path / "cache.db")
    cache.put(sonnet_key, _result(model="claude-sonnet-4-6"), "v1", model="claude-sonnet-4-6")
    cache.put(
        haiku_key,
        _result(model="claude-haiku-4-5-20251001"),
        "v1",
        model="claude-haiku-4-5-20251001",
    )
    assert cache.get(sonnet_key, "v1") is not None
    assert cache.get(haiku_key, "v1") is not None
    cache.close()
