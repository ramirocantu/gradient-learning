"""V41 amended — worker partial-failure on transient OpenAI errors.

A `RateLimitError` / `InternalServerError` raised mid-drain MUST:
  - break the loop early (no further per-item retries — SDK already retried),
  - mark `summary.partial_failure=True` + record the cause in `summary.error`,
  - leave already-succeeded work intact (callers commit; we don't rollback),
  - let the scheduler set `task_run.status='succeeded'` (partial — the
    candidate filter resumes the rest on the next tick).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import openai
import pytest

from app.services.categorizer.worker import WorkerSummary, run


def _rate_limit_error() -> openai.RateLimitError:
    """Forge a real RateLimitError instance."""
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(429, request=request)
    return openai.RateLimitError(
        message="rate limited",
        response=response,
        body={"error": {"message": "rate limited"}},
    )


def _internal_server_error() -> openai.InternalServerError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(500, request=request)
    return openai.InternalServerError(
        message="boom",
        response=response,
        body={"error": {"message": "boom"}},
    )


class _FakeSession:
    """Just enough of AsyncSession to drive the worker through one row."""

    def __init__(self, rows: list[tuple[int, str, list[str]]]) -> None:
        self._rows = rows
        self._begin_nested_calls = 0
        self.rollback_called = False

    def begin_nested(self):
        outer = self

        class _Ctx:
            async def __aenter__(self_inner):
                outer._begin_nested_calls += 1
                return None

            async def __aexit__(self_inner, exc_type, exc, tb):
                # Re-raise to mirror SAVEPOINT semantics on error.
                return False

        return _Ctx()

    async def execute(self, _stmt):
        rows = self._rows
        self._rows = []  # second call drains nothing → outer loop exits

        class _Result:
            def __init__(self_inner, items):
                self_inner._items = items

            def all(self_inner):
                return list(self_inner._items)

        return _Result(rows)

    async def rollback(self) -> None:
        self.rollback_called = True


async def test_rate_limit_error_breaks_loop_with_partial_failure(monkeypatch):
    """One pending row, tag_fn raises RateLimitError → summary.partial_failure=True."""
    from app.services.categorizer import worker as worker_module

    # No real `select` lookup — just preload the session.
    session = _FakeSession(rows=[(1, "qid-1", ["Subject: Physics"])])
    tag_fn = AsyncMock(side_effect=_rate_limit_error())
    monkeypatch.setattr(worker_module, "OutlineLookup", SimpleNamespace(load=AsyncMock(return_value=SimpleNamespace())))

    summary = await run(
        session,  # type: ignore[arg-type]
        openai_client=SimpleNamespace(),  # type: ignore[arg-type]
        tag_fn=tag_fn,
        lookup=SimpleNamespace(),
    )

    assert isinstance(summary, WorkerSummary)
    assert summary.partial_failure is True
    assert summary.error is not None and "RateLimitError" in summary.error
    # We broke early — the only attempted item didn't succeed, so succeeded=0.
    assert summary.succeeded == 0
    assert summary.processed == 1


async def test_internal_server_error_also_triggers_partial_failure(monkeypatch):
    from app.services.categorizer import worker as worker_module

    session = _FakeSession(rows=[(2, "qid-2", ["Subject: Physics"])])
    tag_fn = AsyncMock(side_effect=_internal_server_error())
    monkeypatch.setattr(worker_module, "OutlineLookup", SimpleNamespace(load=AsyncMock(return_value=SimpleNamespace())))

    summary = await run(
        session,  # type: ignore[arg-type]
        openai_client=SimpleNamespace(),  # type: ignore[arg-type]
        tag_fn=tag_fn,
        lookup=SimpleNamespace(),
    )

    assert summary.partial_failure is True
    assert "InternalServerError" in (summary.error or "")


async def test_generic_exception_does_not_set_partial_failure(monkeypatch):
    """Non-transient errors keep the loop running (per-item failure counter)."""
    from app.services.categorizer import worker as worker_module

    session = _FakeSession(rows=[(3, "qid-3", ["Subject: Physics"])])
    tag_fn = AsyncMock(side_effect=RuntimeError("bug"))
    monkeypatch.setattr(worker_module, "OutlineLookup", SimpleNamespace(load=AsyncMock(return_value=SimpleNamespace())))

    summary = await run(
        session,  # type: ignore[arg-type]
        openai_client=SimpleNamespace(),  # type: ignore[arg-type]
        tag_fn=tag_fn,
        lookup=SimpleNamespace(),
    )

    assert summary.partial_failure is False
    assert summary.failed == 1
    assert "qid-3" in summary.failure_qids


async def test_dry_run_rollback_on_partial_failure(monkeypatch):
    """Dry-run + partial failure: session rolls back before returning."""
    from app.services.categorizer import worker as worker_module

    session = _FakeSession(rows=[(4, "qid-4", ["Subject: Physics"])])
    tag_fn = AsyncMock(side_effect=_rate_limit_error())
    monkeypatch.setattr(worker_module, "OutlineLookup", SimpleNamespace(load=AsyncMock(return_value=SimpleNamespace())))

    summary = await run(
        session,  # type: ignore[arg-type]
        openai_client=SimpleNamespace(),  # type: ignore[arg-type]
        tag_fn=tag_fn,
        lookup=SimpleNamespace(),
        dry_run=True,
    )

    assert summary.partial_failure is True
    assert session.rollback_called is True
