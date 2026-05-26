"""V-L2 gate runner — re-evaluates tagging quality on the chosen OpenAI
model and compares against the Claude baseline at
`tests/fixtures/v_l2_claude_baseline.json`.

Exits 0 when within tolerance, non-zero when the candidate regresses past
`MEAN_JACCARD_TOLERANCE` or `SET_EQUALITY_TOLERANCE` (§T10, V-L2).

Usage:
    python -m scripts.run_v_l2_gate \\
        --eval-set tests/fixtures/v_l2_eval_set.jsonl \\
        --report data/v_l2_report.json

`--eval-set` is a JSONL of `{qid, question_text, uworld_aamc_tags,
gold_tags}` rows. Each row triggers one `categorize()` call against the
live OpenAI client; `gold_tags` is the canonical set of expected
node-paths. The script writes the diffable JSON report next to the eval
set and prints a one-line PASS / BLOCKS verdict.

This runner intentionally lives outside `app/` so it never gets imported
in production paths.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from app.config import settings
from app.services.categorizer import llm as categorizer_llm
from app.services.categorizer.outline_lookup import OutlineLookup
from app.services.eval import (
    EvalCase,
    record_eval_run,
    regression_blocks_pivot,
)
from app.services.eval.metrics import load_baseline, write_report
from app.services.llm.client import build_openai_client

logger = logging.getLogger("v_l2_gate")


def _load_eval_set(path: Path) -> list[dict[str, Any]]:
    """Read JSONL — one object per line, no trailing-comma forgiveness."""
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"eval set line {i + 1} invalid JSON: {exc}")
    return rows


async def _predict_one(
    row: dict[str, Any],
    *,
    openai_client: Any,
    lookup: OutlineLookup,
) -> frozenset[str]:
    """Run the categorizer on one row and return the predicted node-path set."""
    question = SimpleNamespace(
        qid=row.get("qid", "?"),
        stem_plain=row.get("question_text", ""),
        explanation_plain=row.get("explanation", ""),
        uworld_aamc_tags=list(row.get("uworld_aamc_tags", [])),
    )
    result = await categorizer_llm.categorize(
        question,
        openai_client=openai_client,
        outline_lookup=lookup,
    )
    # We only compare on `topic` suggestions — the eval set's `gold_tags`
    # uses canonical topic paths.
    return frozenset(
        str(s.identifier) for s in result.suggestions if s.kind == "topic"
    )


async def run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    eval_set = _load_eval_set(Path(args.eval_set))
    if not eval_set:
        raise SystemExit("eval set is empty")

    openai_client = build_openai_client(max_retries=5)
    # Empty OutlineLookup — eval rows arrive with `uworld_aamc_tags` and the
    # `Subject:` derivation routes the model to the right section. The
    # categorizer never reads `lookup` (carries it for legacy reasons).
    lookup = OutlineLookup(course_id=0, nodes=[])

    cases: list[EvalCase] = []
    for row in eval_set:
        predicted = await _predict_one(row, openai_client=openai_client, lookup=lookup)
        cases.append(
            EvalCase(
                qid=row.get("qid", "?"),
                gold_tags=frozenset(row.get("gold_tags", [])),
                predicted_tags=predicted,
            )
        )

    candidate = record_eval_run(settings.OPENAI_MODEL, cases)
    baseline = load_baseline(Path(args.baseline))
    report = regression_blocks_pivot(baseline, candidate)
    write_report(Path(args.report), report)

    verdict = "BLOCKS PIVOT" if report.blocks_pivot else "PASS"
    print(
        f"V-L2 {verdict}: candidate_mean_jaccard={candidate.mean_jaccard:.3f} "
        f"(baseline {baseline.mean_jaccard:.3f}, Δ {report.mean_jaccard_delta:+.3f}); "
        f"set_equality={candidate.set_equality_rate:.3f} "
        f"(baseline {baseline.set_equality_rate:.3f}, Δ {report.set_equality_delta:+.3f})"
    )
    return 1 if report.blocks_pivot else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V-L2 measurement harness (§T10)")
    parser.add_argument(
        "--eval-set",
        default="tests/fixtures/v_l2_eval_set.jsonl",
        help="JSONL eval rows (qid, question_text, uworld_aamc_tags, gold_tags)",
    )
    parser.add_argument(
        "--baseline",
        default="tests/fixtures/v_l2_claude_baseline.json",
        help="PoC Claude baseline (mean_jaccard, set_equality_rate)",
    )
    parser.add_argument(
        "--report",
        default="data/v_l2_report.json",
        help="Where to write the run report",
    )
    args = parser.parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
