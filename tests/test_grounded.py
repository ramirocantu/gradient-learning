"""T29 — app/services/llm/grounded.py contract tests (V-L3, V69, V45, V44).

V-L3: the model picks ONLY from the recall candidate set; ⊥ free-form
over the full outline. V44: dual surface (numbered list + int enum).
V45: strict json_schema. V69: each pick re-scored via the logprob
calibrator; <0.5 ⇒ manual_review. V16: OpenAI mocked at the SDK boundary.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from app.services.kb.recall import Candidate, RecallResult
from app.services.llm.grounded import (
    EXTRACTOR_VERSION,
    GroundedResult,
    build_pick_schema,
    build_system_prompt,
    generate_grounded_tags,
)
from tests._openai_mocks import make_client, make_completion


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _recall(*candidates: Candidate, version: str = "v1") -> RecallResult:
    return RecallResult(candidates=list(candidates), embedding_version=version)


def _cand(node_id: int, *, path: str | None, score: float = 0.9, via: str = "embedding") -> Candidate:
    return Candidate(node_id=node_id, path=path, score=score, via=via)


def _tagging_client(tags: list[dict]) -> object:
    """Structured-output tagging client — returns one json_schema body."""
    return make_client(make_completion(content=json.dumps({"tags": tags})))


def _logprobs_completion(top: list[tuple[str, float]]) -> SimpleNamespace:
    """Forge a calibrator completion exposing `choice.logprobs.content[0]
    .top_logprobs` (the shape calibrator.grade_yes_no reads)."""
    candidates = [SimpleNamespace(token=t, logprob=lp, bytes=None) for t, lp in top]
    first = SimpleNamespace(
        token=top[0][0] if top else "",
        logprob=top[0][1] if top else 0.0,
        bytes=None,
        top_logprobs=candidates,
    )
    choice = SimpleNamespace(
        index=0,
        message=SimpleNamespace(role="assistant", content=top[0][0] if top else "", tool_calls=[]),
        finish_reason="stop",
        logprobs=SimpleNamespace(content=[first]),
    )
    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=1,
        total_tokens=11,
        prompt_tokens_details=SimpleNamespace(cached_tokens=0),
    )
    return SimpleNamespace(id="c", choices=[choice], usage=usage, model="gpt-4.1-mini")


def _calibrator_client(*, yes: float, no: float) -> object:
    return make_client(_logprobs_completion([("Yes", yes), ("No", no)]))


# --------------------------------------------------------------------------- #
# V45 — strict schema shape
# --------------------------------------------------------------------------- #


def test_schema_is_strict_and_closed():
    schema = build_pick_schema(3)
    assert schema["type"] == "json_schema"
    js = schema["json_schema"]
    assert js["strict"] is True
    root = js["schema"]
    assert root["additionalProperties"] is False
    assert root["required"] == ["tags"]
    item = root["properties"]["tags"]["items"]
    assert item["additionalProperties"] is False
    # V45: every property listed in required.
    assert set(item["required"]) == set(item["properties"].keys())
    # V45: no numeric bounds anywhere.
    blob = json.dumps(schema)
    assert "minimum" not in blob and "maximum" not in blob and "minItems" not in blob


def test_schema_node_index_is_int_enum_1_to_n():
    """V44: int enum [1..N] is the grammar constraint."""
    schema = build_pick_schema(4)
    idx = schema["json_schema"]["schema"]["properties"]["tags"]["items"]["properties"][
        "node_index"
    ]
    assert idx["type"] == "integer"
    assert idx["enum"] == [1, 2, 3, 4]


def test_schema_empty_candidates_has_no_enum():
    idx = build_pick_schema(0)["json_schema"]["schema"]["properties"]["tags"]["items"][
        "properties"
    ]["node_index"]
    assert "enum" not in idx


# --------------------------------------------------------------------------- #
# V44 — numbered candidate list is the reasoning surface
# --------------------------------------------------------------------------- #


def test_system_prompt_numbers_candidates():
    res = _recall(
        _cand(10, path="root >> alpha"),
        _cand(20, path="root >> beta", via="edge", score=0.7),
    )
    prompt = build_system_prompt(res)
    assert "Candidate outline nodes" in prompt
    assert "1. " in prompt and "root >> alpha" in prompt
    assert "2. " in prompt and "root >> beta" in prompt
    # V-L3: instruction forbids inventing nodes.
    assert "MUST pick from the numbered candidate list" in prompt


# --------------------------------------------------------------------------- #
# V-L3 — empty candidates short-circuit, no LLM call
# --------------------------------------------------------------------------- #


async def test_empty_candidates_skips_llm():
    tagging = make_client()  # would error shape if called meaningfully
    result = await generate_grounded_tags(
        entity_text="glycolysis converts glucose to pyruvate",
        recall_result=_recall(),
        tagging_client=tagging,
        calibrator_client=_calibrator_client(yes=-0.1, no=-3.0),
    )
    assert isinstance(result, GroundedResult)
    assert result.tags == []
    assert result.extractor_version == EXTRACTOR_VERSION
    tagging.chat.completions.create.assert_not_awaited()


# --------------------------------------------------------------------------- #
# Happy path — pick mapped to node + V69 calibration applied
# --------------------------------------------------------------------------- #


async def test_pick_maps_to_node_and_calibrates():
    res = _recall(
        _cand(10, path="root >> alpha"),
        _cand(20, path="root >> beta", via="edge", score=0.7),
    )
    tagging = _tagging_client(
        [{"node_index": 2, "rationale": "beta fits"}]
    )
    # Calibrator: Yes dominates → high confidence, no manual_review.
    calib = _calibrator_client(yes=-0.05, no=-4.0)

    result = await generate_grounded_tags(
        entity_text="some fact",
        recall_result=res,
        tagging_client=tagging,
        calibrator_client=calib,
    )

    assert len(result.tags) == 1
    tag = result.tags[0]
    assert tag.node_id == 20            # node_index 2 → second candidate
    assert tag.path == "root >> beta"
    assert tag.candidate_index == 2
    assert tag.via == "edge"
    assert tag.calibrated_confidence > 0.9     # V69 logprob grade — sole confidence
    assert tag.manual_review is False
    calib.chat.completions.create.assert_awaited()  # V69 ran


async def test_service_tier_threaded_into_tagging_and_calibrator(monkeypatch):
    """V-L5: settings.OPENAI_SERVICE_TIER flows into BOTH the tagging chat call
    and the calibrator call."""
    from app.services.llm import grounded as grounded_mod

    monkeypatch.setattr(grounded_mod.settings, "OPENAI_SERVICE_TIER", "flex")

    res = _recall(_cand(10, path="root >> alpha"))
    tagging = _tagging_client([{"node_index": 1, "rationale": "fits"}])
    calib = _calibrator_client(yes=-0.05, no=-4.0)

    await generate_grounded_tags(
        entity_text="some fact",
        recall_result=res,
        tagging_client=tagging,
        calibrator_client=calib,
    )

    assert tagging.chat.completions.create.await_args.kwargs["service_tier"] == "flex"
    assert calib.chat.completions.create.await_args.kwargs["service_tier"] == "flex"


async def test_calibration_below_half_sets_manual_review():
    """V69 / V-T3: calibrated <0.5 ⇒ manual_review."""
    res = _recall(_cand(10, path="root >> alpha"))
    tagging = _tagging_client(
        [{"node_index": 1, "rationale": "sure"}]
    )
    calib = _calibrator_client(yes=-3.0, no=-0.1)  # No dominates → <0.5

    result = await generate_grounded_tags(
        entity_text="x",
        recall_result=res,
        tagging_client=tagging,
        calibrator_client=calib,
    )
    tag = result.tags[0]
    assert tag.calibrated_confidence < 0.5
    assert tag.manual_review is True


# --------------------------------------------------------------------------- #
# Server-side belt — V-L3 out-of-range reject + dedupe + clip
# --------------------------------------------------------------------------- #


async def test_out_of_range_index_rejected():
    """V-L3 belt: an index past N is dropped even if the SDK lets it through."""
    res = _recall(_cand(10, path="root >> alpha"))
    tagging = _tagging_client(
        [
            {"node_index": 1, "rationale": "ok"},
            {"node_index": 5, "rationale": "bogus"},  # N=1
        ]
    )
    calib = _calibrator_client(yes=-0.1, no=-3.0)

    result = await generate_grounded_tags(
        entity_text="x",
        recall_result=res,
        tagging_client=tagging,
        calibrator_client=calib,
    )
    assert [t.node_id for t in result.tags] == [10]
    assert any("out of range" in w for w in result.parse_warnings)


async def test_duplicate_index_deduped():
    res = _recall(_cand(10, path="root >> alpha"), _cand(20, path="root >> beta"))
    tagging = _tagging_client(
        [
            {"node_index": 1, "rationale": "a"},
            {"node_index": 1, "rationale": "dupe"},
        ]
    )
    calib = _calibrator_client(yes=-0.1, no=-3.0)

    result = await generate_grounded_tags(
        entity_text="x",
        recall_result=res,
        tagging_client=tagging,
        calibrator_client=calib,
    )
    assert len(result.tags) == 1
    assert result.tags[0].node_id == 10


async def test_unparsed_node_index_warns_and_skips():
    res = _recall(_cand(10, path="root >> alpha"))
    tagging = _tagging_client(
        [{"node_index": "two", "rationale": "bad"}]
    )
    calib = _calibrator_client(yes=-0.1, no=-3.0)

    result = await generate_grounded_tags(
        entity_text="x",
        recall_result=res,
        tagging_client=tagging,
        calibrator_client=calib,
    )
    assert result.tags == []
    assert any("not an int" in w for w in result.parse_warnings)


# --------------------------------------------------------------------------- #
# Path fallback + token accounting
# --------------------------------------------------------------------------- #


async def test_pathless_candidate_uses_node_fallback_label():
    res = _recall(_cand(10, path=None, via="edge"))
    tagging = _tagging_client(
        [{"node_index": 1, "rationale": "ok"}]
    )
    calib = _calibrator_client(yes=-0.1, no=-3.0)

    result = await generate_grounded_tags(
        entity_text="x",
        recall_result=res,
        tagging_client=tagging,
        calibrator_client=calib,
    )
    tag = result.tags[0]
    assert tag.node_id == 10
    assert tag.path is None


async def test_token_usage_recorded():
    res = _recall(_cand(10, path="root >> alpha"))
    tagging = make_client(
        make_completion(
            content=json.dumps(
                {"tags": [{"node_index": 1, "rationale": "ok"}]}
            ),
            prompt_tokens=512,
            completion_tokens=40,
            cached_tokens=128,
        )
    )
    calib = _calibrator_client(yes=-0.1, no=-3.0)

    result = await generate_grounded_tags(
        entity_text="x",
        recall_result=res,
        tagging_client=tagging,
        calibrator_client=calib,
    )
    assert result.input_tokens == 512
    assert result.output_tokens == 40
    assert result.cached_tokens == 128  # V-L1 read, not inferred
