"""Smoke test for `scripts.eval_categorizer_models`.

Mocks both Sonnet and Haiku at the SDK boundary and asserts the output
Markdown contains the expected report sections. Does not assess report
content quality.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.captures import Question
from app.services.categorizer.cache import CategorizerCache
from scripts import eval_categorizer_models


def _tool_block(model: str, ident: str | int):
    from anthropic.types import ToolUseBlock

    suggestion = {
        "kind": "topic" if isinstance(ident, str) else "skill",
        "identifier": ident,
        "confidence": 0.85,
        "rationale": f"{model}-rationale",
    }
    if isinstance(ident, str):
        suggestion["under_content_category"] = "4A"
    return ToolUseBlock(
        id="toolu_x",
        name="submit_aamc_categorization",
        input={
            "primary_aamc_section": "CP",
            "tags": [
                suggestion,
                {
                    "kind": "skill",
                    "identifier": 2,
                    "confidence": 0.9,
                    "rationale": "calc",
                },
            ],
        },
        type="tool_use",
    )


def _forge(model: str, ident: str | int):
    usage = SimpleNamespace(
        input_tokens=1000,
        output_tokens=100,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    return SimpleNamespace(content=[_tool_block(model, ident)], usage=usage)


def _patched_anthropic_client():
    """AsyncAnthropic mock that varies output by model arg."""
    client = MagicMock()
    client.messages = MagicMock()

    async def _create(**kwargs):
        model = kwargs.get("model", "")
        if "sonnet" in model:
            return _forge(model, "Work")
        if "haiku" in model:
            # Haiku picks a different topic — to create non-1.0 Jaccard.
            return _forge(model, "Force")
        return _forge(model, "Work")

    client.messages.create = AsyncMock(side_effect=_create)
    return client


@pytest.fixture
async def two_questions(test_engine, seeded_report):
    """Insert two Physics questions into the test DB and clean up."""
    Sm = async_sessionmaker(test_engine, expire_on_commit=False)
    qids = []
    async with Sm() as s:
        for i in range(2):
            q = Question(
                qid=f"eval-{uuid.uuid4().hex[:8]}-{i}",
                passage_id=None,
                stem_html="<p>stem</p>",
                stem_plain=f"Eval question {i} stem",
                choices=[
                    {
                        "key": "A",
                        "html": "<p>a</p>",
                        "plain": "a",
                        "media_content_hashes": [],
                    }
                ],
                correct_choice="A",
                explanation_html=None,
                explanation_plain=f"Eval explanation {i}",
                uworld_aamc_tags=[
                    "Subject: Physics",
                    "Chapter: 1. Motion, Force, and Energy",
                ],
                needs_categorization=False,
            )
            s.add(q)
            await s.flush()
            qids.append(q.id)
        await s.commit()
    yield qids
    async with Sm() as s:
        for qid in qids:
            row = (await s.execute(select(Question).where(Question.id == qid))).scalar_one_or_none()
            if row is not None:
                await s.delete(row)
        await s.commit()


async def test_eval_script_produces_report_without_crashing(two_questions, test_engine, tmp_path):
    """End-to-end: eval script reads from test DB, calls mocked SDK, emits Markdown."""
    cache_path = tmp_path / "eval-cache.db"
    fake_client = _patched_anthropic_client()
    eval_version = "eval-test"
    cache = CategorizerCache(cache_path)

    try:
        async with AsyncSession(test_engine) as session:
            (
                evaluations,
                aggregate,
                started_iso,
            ) = await eval_categorizer_models.run_eval(
                session,
                sample=2,
                stratify=False,
                client=fake_client,
                cache=cache,
                eval_version=eval_version,
            )
    finally:
        cache.clear(extractor_version=eval_version)
        cache.close()

    assert len(evaluations) == 2

    markdown = eval_categorizer_models._build_markdown(
        evaluations=evaluations,
        aggregate=aggregate,
        sample=2,
        stratified=False,
        started_iso=started_iso,
        sonnet_model=eval_categorizer_models.SONNET_MODEL,
        haiku_model=eval_categorizer_models.HAIKU_MODEL,
    )
    assert "## Aggregate" in markdown
    assert "## Overlap analysis" in markdown
    assert "## Per-question detail" in markdown
    assert "Sonnet" in markdown
    assert "Haiku" in markdown
    # The mocked Sonnet/Haiku produced different topics → mean Jaccard < 1.0.
    assert "Mean Jaccard:" in markdown
