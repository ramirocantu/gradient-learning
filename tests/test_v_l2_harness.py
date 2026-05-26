"""V-L2 harness unit tests (§T10) — pure-function metric logic.

The harness's *math* is what the gate hinges on; the LLM-driven runner
(`scripts/run_v_l2_gate.py`) is glue. We test the metric primitives in
isolation so the gate can be trusted without burning real API calls.
"""

from __future__ import annotations

from app.services.eval import (
    EvalCase,
    jaccard,
    record_eval_run,
    regression_blocks_pivot,
)
from app.services.eval.metrics import (
    MEAN_JACCARD_TOLERANCE,
    SET_EQUALITY_TOLERANCE,
    EvalRunResult,
)


def test_jaccard_empty_pair_is_perfect_agreement():
    """No tags on either side ⇒ jaccard=1.0. Avoids zero-division."""
    assert jaccard(frozenset(), frozenset()) == 1.0


def test_jaccard_disjoint_pair():
    assert jaccard(frozenset({"a"}), frozenset({"b"})) == 0.0


def test_jaccard_partial_overlap():
    a = frozenset({"x", "y", "z"})
    b = frozenset({"y", "z", "w"})
    # |a ∩ b| = 2, |a ∪ b| = 4 → 0.5
    assert jaccard(a, b) == 0.5


def test_record_eval_run_aggregates():
    cases = [
        EvalCase(qid="q1", gold_tags=frozenset({"a"}), predicted_tags=frozenset({"a"})),
        EvalCase(qid="q2", gold_tags=frozenset({"a", "b"}), predicted_tags=frozenset({"a"})),
    ]
    result = record_eval_run("model-x", cases)
    assert result.n_cases == 2
    # mean_jaccard = (1.0 + 0.5) / 2
    assert abs(result.mean_jaccard - 0.75) < 1e-9
    # 1 of 2 cases is set-equal.
    assert result.set_equality_rate == 0.5


def test_record_eval_run_empty_input():
    result = record_eval_run("model-x", [])
    assert result.n_cases == 0
    assert result.mean_jaccard == 0.0


def test_gate_passes_when_within_tolerance():
    baseline = EvalRunResult(
        model="claude", n_cases=10, mean_jaccard=0.80, set_equality_rate=0.60,
        per_case_jaccard=tuple([0.8] * 10),
    )
    candidate = EvalRunResult(
        model="gpt", n_cases=10, mean_jaccard=0.79, set_equality_rate=0.58,
        per_case_jaccard=tuple([0.79] * 10),
    )
    report = regression_blocks_pivot(baseline, candidate)
    assert report.blocks_pivot is False
    assert "within tolerance" in report.reason


def test_gate_blocks_when_mean_jaccard_regresses_past_tolerance():
    baseline = EvalRunResult(
        model="claude", n_cases=10, mean_jaccard=0.80, set_equality_rate=0.60,
        per_case_jaccard=tuple([0.8] * 10),
    )
    candidate = EvalRunResult(
        model="gpt",
        n_cases=10,
        mean_jaccard=0.80 - MEAN_JACCARD_TOLERANCE - 0.02,
        set_equality_rate=0.60,
        per_case_jaccard=tuple([0.75] * 10),
    )
    report = regression_blocks_pivot(baseline, candidate)
    assert report.blocks_pivot is True
    assert "mean_jaccard" in report.reason


def test_gate_blocks_when_set_equality_regresses_past_tolerance():
    baseline = EvalRunResult(
        model="claude", n_cases=10, mean_jaccard=0.80, set_equality_rate=0.60,
        per_case_jaccard=tuple([0.8] * 10),
    )
    candidate = EvalRunResult(
        model="gpt",
        n_cases=10,
        mean_jaccard=0.80,
        set_equality_rate=0.60 - SET_EQUALITY_TOLERANCE - 0.02,
        per_case_jaccard=tuple([0.8] * 10),
    )
    report = regression_blocks_pivot(baseline, candidate)
    assert report.blocks_pivot is True
    assert "set_equality_rate" in report.reason


def test_gate_reports_both_when_both_metrics_regress():
    baseline = EvalRunResult(
        model="claude", n_cases=10, mean_jaccard=0.80, set_equality_rate=0.60,
        per_case_jaccard=tuple([0.8] * 10),
    )
    candidate = EvalRunResult(
        model="gpt",
        n_cases=10,
        mean_jaccard=0.50,
        set_equality_rate=0.30,
        per_case_jaccard=tuple([0.5] * 10),
    )
    report = regression_blocks_pivot(baseline, candidate)
    assert report.blocks_pivot is True
    assert "mean_jaccard" in report.reason
    assert "set_equality_rate" in report.reason


def test_report_as_dict_round_trips():
    baseline = EvalRunResult(
        model="claude", n_cases=10, mean_jaccard=0.80, set_equality_rate=0.60,
        per_case_jaccard=tuple([0.8] * 10),
    )
    candidate = EvalRunResult(
        model="gpt", n_cases=10, mean_jaccard=0.79, set_equality_rate=0.58,
        per_case_jaccard=tuple([0.79] * 10),
    )
    report = regression_blocks_pivot(baseline, candidate)
    d = report.as_dict()
    assert d["baseline"]["model"] == "claude"
    assert d["candidate"]["model"] == "gpt"
    assert d["blocks_pivot"] is False
