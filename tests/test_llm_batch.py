"""Unit tests for the generic batches wrapper (SPEC §T51).

Mock the Anthropic SDK at `client.messages.batches.{create,retrieve,results}`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from app.services.llm.batch import (
    BatchRequestItem,
    get_batch_status,
    iter_batch_results,
    submit_batch,
)


@dataclass
class _FakeRequestCounts:
    processing: int = 0
    succeeded: int = 0
    errored: int = 0
    canceled: int = 0
    expired: int = 0


@dataclass
class _FakeMessageBatch:
    id: str = "batch_test_abc"
    processing_status: str = "in_progress"
    request_counts: _FakeRequestCounts = None  # type: ignore[assignment]
    created_at: Any = None
    ended_at: Any = None

    def __post_init__(self) -> None:
        if self.request_counts is None:
            self.request_counts = _FakeRequestCounts()


class _FakeBatches:
    def __init__(self) -> None:
        self.create_calls: list[dict] = []
        self.retrieve_calls: list[str] = []
        self.results_calls: list[str] = []
        self.next_create_response: _FakeMessageBatch | None = None
        self.next_retrieve_response: _FakeMessageBatch | None = None
        self.results_payload: list[Any] = []

    async def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return self.next_create_response or _FakeMessageBatch()

    async def retrieve(self, batch_id, **_):
        self.retrieve_calls.append(batch_id)
        return self.next_retrieve_response or _FakeMessageBatch(id=batch_id)

    async def results(self, batch_id, **_):
        self.results_calls.append(batch_id)

        async def stream():
            for item in self.results_payload:
                yield item

        return stream()


class _FakeAnthropic:
    def __init__(self) -> None:
        self.messages = type(
            "M",
            (),
            {"batches": _FakeBatches()},
        )()


@pytest.mark.asyncio
async def test_submit_batch_passes_requests_through() -> None:
    fake = _FakeAnthropic()
    items = [
        BatchRequestItem(custom_id="a", params={"model": "haiku", "messages": []}),
        BatchRequestItem(custom_id="b", params={"model": "haiku", "messages": []}),
    ]
    await submit_batch(fake, items)  # type: ignore[arg-type]
    assert len(fake.messages.batches.create_calls) == 1
    sent = fake.messages.batches.create_calls[0]["requests"]
    assert [r["custom_id"] for r in sent] == ["a", "b"]
    assert sent[0]["params"]["model"] == "haiku"


@pytest.mark.asyncio
async def test_submit_batch_rejects_empty_items() -> None:
    fake = _FakeAnthropic()
    with pytest.raises(ValueError, match="empty"):
        await submit_batch(fake, [])  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_submit_batch_enforces_safety_cap() -> None:
    fake = _FakeAnthropic()
    items = [
        BatchRequestItem(custom_id=str(i), params={"model": "h", "messages": []}) for i in range(5)
    ]
    with pytest.raises(ValueError, match="safety cap"):
        await submit_batch(fake, items, safety_cap=2)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_get_batch_status_round_trips_id() -> None:
    fake = _FakeAnthropic()
    fake.messages.batches.next_retrieve_response = _FakeMessageBatch(
        id="batch_xyz", processing_status="ended"
    )
    batch = await get_batch_status(fake, "batch_xyz")  # type: ignore[arg-type]
    assert batch.id == "batch_xyz"
    assert batch.processing_status == "ended"
    assert fake.messages.batches.retrieve_calls == ["batch_xyz"]


@pytest.mark.asyncio
async def test_iter_batch_results_yields_payload() -> None:
    fake = _FakeAnthropic()
    fake.messages.batches.results_payload = [
        {"custom_id": "topic-1-4A", "result": {"type": "succeeded"}},
        {"custom_id": "topic-2-4A", "result": {"type": "errored"}},
    ]
    out = []
    async for item in iter_batch_results(fake, "batch_x"):  # type: ignore[arg-type]
        out.append(item)
    assert [r["custom_id"] for r in out] == ["topic-1-4A", "topic-2-4A"]
    assert fake.messages.batches.results_calls == ["batch_x"]
