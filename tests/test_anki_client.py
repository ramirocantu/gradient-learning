"""Tests for AnkiConnectClient (SPEC T1).

Uses httpx.MockTransport (built-in) to avoid a respx dev-dep on backend/.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from app.services.anki.client import (
    AnkiConnectClient,
    AnkiConnectError,
    AnkiUnreachableError,
    AnkiWriteFailed,
    AnkiWriteForbidden,
)


_URL = "http://localhost:8765"


def _ok(result: Any) -> bytes:
    return json.dumps({"result": result, "error": None}).encode()


def _err(message: str) -> bytes:
    return json.dumps({"result": None, "error": message}).encode()


def _transport(handler: Any) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_version_returns_int() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=_ok(6))

    async with AnkiConnectClient(_URL, transport=_transport(handler)) as client:
        result = await client.version()

    assert result == 6
    assert captured["body"]["action"] == "version"
    assert captured["body"]["version"] == 6


@pytest.mark.asyncio
async def test_find_cards_returns_id_list() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=_ok([1001, 1002, 1003]))

    async with AnkiConnectClient(_URL, transport=_transport(handler)) as client:
        result = await client.find_cards('deck:"MileDown"')

    assert result == [1001, 1002, 1003]
    assert captured["body"]["action"] == "findCards"
    assert captured["body"]["params"] == {"query": 'deck:"MileDown"'}


@pytest.mark.asyncio
async def test_cards_info_returns_dict_list() -> None:
    captured: dict[str, Any] = {}

    payload = [
        {"cardId": 1001, "tags": ["aamc::CP::4A"], "due": 100},
        {"cardId": 1002, "tags": ["uworld::qid::402391"], "due": 200},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=_ok(payload))

    async with AnkiConnectClient(_URL, transport=_transport(handler)) as client:
        result = await client.cards_info([1001, 1002])

    assert result == payload
    assert captured["body"]["action"] == "cardsInfo"
    assert captured["body"]["params"] == {"cards": [1001, 1002]}


@pytest.mark.asyncio
async def test_cards_info_empty_input_short_circuits() -> None:
    """Empty card_ids must not issue an HTTP call (V13 spirit: no needless traffic)."""

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - asserted unreached
        raise AssertionError("cards_info([]) must not call AnkiConnect")

    async with AnkiConnectClient(_URL, transport=_transport(handler)) as client:
        result = await client.cards_info([])

    assert result == []


@pytest.mark.asyncio
async def test_unreachable_raises_AnkiUnreachableError() -> None:
    """V4: AnkiConnect down → AnkiUnreachableError. Caller (T3) maps to error envelope."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    async with AnkiConnectClient(_URL, transport=_transport(handler)) as client:
        with pytest.raises(AnkiUnreachableError):
            await client.version()


@pytest.mark.asyncio
async def test_timeout_raises_AnkiUnreachableError() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out")

    async with AnkiConnectClient(_URL, transport=_transport(handler)) as client:
        with pytest.raises(AnkiUnreachableError):
            await client.version()


@pytest.mark.asyncio
async def test_ankiconnect_error_response_raises_AnkiConnectError() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_err("deck not found: Bogus"))

    async with AnkiConnectClient(_URL, transport=_transport(handler)) as client:
        with pytest.raises(AnkiConnectError, match="deck not found"):
            await client.find_cards('deck:"Bogus"')


@pytest.mark.asyncio
async def test_malformed_response_raises_AnkiConnectError() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps({"unexpected": "shape"}).encode())

    async with AnkiConnectClient(_URL, transport=_transport(handler)) as client:
        with pytest.raises(AnkiConnectError, match="malformed"):
            await client.version()


def test_client_action_allowlist() -> None:
    """V13 (amended T56) + V50: client surface = read methods + the V50
    write allowlist {unsuspend, createDeck, createFilteredDeck,
    deleteDecks}. Any new public method must map to either a non-mutating
    AnkiConnect action or an explicitly allowlisted write — extending
    this set without amending §V50 is a spec drift.
    """
    public = {
        name
        for name in dir(AnkiConnectClient)
        if not name.startswith("_") and callable(getattr(AnkiConnectClient, name))
    }
    lifecycle = {"aclose"}
    actions = public - lifecycle
    assert actions == {
        # read
        "version",
        "find_cards",
        "cards_info",
        "notes_info",
        "deck_names",
        "card_reviews",
        # V50 write allowlist (amended 2026-05-23 per T74)
        "unsuspend_cards",
        "add_tags",
        "create_filtered_deck",
    }, f"unexpected public methods on AnkiConnectClient: {actions}"


# --------------------------- write allowlist (V50) -------------------------- #


_PREFIX = "mcat-coach"


@pytest.mark.asyncio
async def test_unsuspend_cards_happy_path() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=_ok(True))

    async with AnkiConnectClient(
        _URL, transport=_transport(handler), write_deck_prefix=_PREFIX
    ) as client:
        result = await client.unsuspend_cards([1001, 1002])

    assert result is True
    assert captured["body"]["action"] == "unsuspend"
    assert captured["body"]["params"] == {"cards": [1001, 1002]}


@pytest.mark.asyncio
async def test_unsuspend_cards_empty_short_circuits() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("unsuspend_cards([]) must not call AnkiConnect")

    async with AnkiConnectClient(
        _URL, transport=_transport(handler), write_deck_prefix=_PREFIX
    ) as client:
        assert await client.unsuspend_cards([]) is True


@pytest.mark.asyncio
async def test_add_tags_sends_notes_param_without_cardsinfo() -> None:
    """V75 (§B12 superseded): AnkiConnect addTags is NOTE-scoped. Post-§V75
    the caller passes note_ids straight from the stored snapshot, so add_tags
    sends `notes=` directly — NOT `cards=` (that param does not exist on
    addTags) — and issues NO `cardsInfo` resolve (the prior V63 card→note
    lookup is gone)."""
    by_action: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        by_action[body["action"]] = body
        if body["action"] == "cardsInfo":  # pragma: no cover - asserted unreached
            raise AssertionError("add_tags must not issue cardsInfo post-§V75")
        return httpx.Response(200, content=_ok(None))

    async with AnkiConnectClient(
        _URL, transport=_transport(handler), write_deck_prefix=_PREFIX
    ) as client:
        await client.add_tags([5001, 5002], ["assignment:5"])

    # No card→note resolve read — note ids come straight from the snapshot.
    assert "cardsInfo" not in by_action
    # the write targets NOTES, not cards
    assert by_action["addTags"]["action"] == "addTags"
    assert by_action["addTags"]["params"] == {"notes": [5001, 5002], "tags": "assignment:5"}
    assert "cards" not in by_action["addTags"]["params"]


@pytest.mark.asyncio
async def test_add_tags_joins_multiple_tags_space_separated() -> None:
    by_action: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        by_action[body["action"]] = body
        return httpx.Response(200, content=_ok(None))

    async with AnkiConnectClient(
        _URL, transport=_transport(handler), write_deck_prefix=_PREFIX
    ) as client:
        await client.add_tags([7001], ["assignment:5", "review:3"])

    assert by_action["addTags"]["params"]["tags"] == "assignment:5 review:3"
    assert by_action["addTags"]["params"]["notes"] == [7001]


@pytest.mark.asyncio
async def test_add_tags_dedups_note_ids() -> None:
    """V64 (§B13) / §V75: duplicate note ids are de-duped order-preserving
    before the write (Anki rejects duplicate ids in an id-list action via
    its internal UNIQUE). No cardsInfo resolve — the caller passes note ids."""
    by_action: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        by_action[body["action"]] = body
        if body["action"] == "cardsInfo":  # pragma: no cover - asserted unreached
            raise AssertionError("add_tags must not issue cardsInfo post-§V75")
        return httpx.Response(200, content=_ok(None))

    async with AnkiConnectClient(
        _URL, transport=_transport(handler), write_deck_prefix=_PREFIX
    ) as client:
        await client.add_tags([5001, 5002, 5001, 5002], ["assignment:9"])

    assert "cardsInfo" not in by_action
    assert by_action["addTags"]["params"]["notes"] == [5001, 5002]  # note dedup


@pytest.mark.asyncio
async def test_unsuspend_cards_dedups_duplicate_ids() -> None:
    """V64 (§B13): duplicate cids blow AnkiConnect's internal search_cids
    UNIQUE(cid) and fail the whole batch. unsuspend_cards de-dups
    order-preserving before the call."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=_ok(True))

    async with AnkiConnectClient(
        _URL, transport=_transport(handler), write_deck_prefix=_PREFIX
    ) as client:
        result = await client.unsuspend_cards([1001, 1002, 1001, 1003, 1002])

    assert result is True
    assert captured["body"]["params"] == {"cards": [1001, 1002, 1003]}


@pytest.mark.asyncio
async def test_add_tags_empty_short_circuits() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("add_tags w/ empty input must not call AnkiConnect")

    async with AnkiConnectClient(
        _URL, transport=_transport(handler), write_deck_prefix=_PREFIX
    ) as client:
        # Empty note_ids → no-op.
        await client.add_tags([], ["assignment:5"])
        # Empty tag list → no-op.
        await client.add_tags([5001], [])
        # Both empty → no-op.
        await client.add_tags([], [])


@pytest.mark.asyncio
async def test_create_filtered_deck_happy_path_builds_cid_query() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=_ok(1735689700000))

    async with AnkiConnectClient(
        _URL, transport=_transport(handler), write_deck_prefix=_PREFIX
    ) as client:
        deck_id = await client.create_filtered_deck(
            "mcat-coach::review::2026-05-22::ad-hoc",
            [2001, 2002, 2003],
        )

    assert deck_id == 1735689700000
    assert captured["body"]["action"] == "createFilteredDeck"
    params = captured["body"]["params"]
    assert params["newDeckName"] == "mcat-coach::review::2026-05-22::ad-hoc"
    assert params["searchQuery"] == "cid:2001,2002,2003"
    assert params["reschedule"] is False
    assert params["gatherCount"] == 3


@pytest.mark.asyncio
async def test_create_filtered_deck_out_of_namespace_raises_forbidden() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("forbidden create_filtered_deck must not reach AnkiConnect")

    async with AnkiConnectClient(
        _URL, transport=_transport(handler), write_deck_prefix=_PREFIX
    ) as client:
        # Not in `mcat-coach::review::*`.
        with pytest.raises(AnkiWriteForbidden, match="review"):
            await client.create_filtered_deck("mcat-coach::assignments::2026-05-22", [1, 2])
        # Entirely outside the write prefix.
        with pytest.raises(AnkiWriteForbidden):
            await client.create_filtered_deck("Custom Study Session", [1, 2])


@pytest.mark.asyncio
async def test_create_filtered_deck_empty_card_ids_raises_forbidden() -> None:
    """V50 restricts searchQuery to `cid:<csv>`; empty ids would produce
    `cid:` (no body) which falls outside the allowed shape."""

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("empty-id filtered deck must not reach AnkiConnect")

    async with AnkiConnectClient(
        _URL, transport=_transport(handler), write_deck_prefix=_PREFIX
    ) as client:
        with pytest.raises(AnkiWriteForbidden, match="cid:"):
            await client.create_filtered_deck("mcat-coach::review::2026-05-22", [])


@pytest.mark.asyncio
async def test_write_null_body_raises_unreachable() -> None:
    """V55: malformed / null response body on a write maps to
    AnkiUnreachableError so the caller's V55 retry semantics fire."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"null")

    async with AnkiConnectClient(
        _URL, transport=_transport(handler), write_deck_prefix=_PREFIX
    ) as client:
        with pytest.raises(AnkiUnreachableError, match="malformed"):
            await client.unsuspend_cards([1001])


@pytest.mark.asyncio
async def test_write_error_field_raises_write_failed() -> None:
    """V55: AnkiConnect `error` field non-null on a write maps to
    AnkiWriteFailed (distinct from the read-side AnkiConnectError). §V75:
    no cardsInfo resolve — the error maps on the addTags write leg directly."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_err("collection is read-only"))

    async with AnkiConnectClient(
        _URL, transport=_transport(handler), write_deck_prefix=_PREFIX
    ) as client:
        with pytest.raises(AnkiWriteFailed, match="collection is read-only"):
            await client.add_tags([5001], ["assignment:1"])


@pytest.mark.asyncio
async def test_notes_info_returns_dict_list() -> None:
    captured: dict[str, Any] = {}

    payload = [
        {"noteId": 2001, "tags": ["aamc::CP::4A::Translational_motion"], "fields": {}},
        {"noteId": 2002, "tags": ["uworld::qid::402391"], "fields": {}},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=_ok(payload))

    async with AnkiConnectClient(_URL, transport=_transport(handler)) as client:
        result = await client.notes_info([2001, 2002])

    assert result == payload
    assert captured["body"]["action"] == "notesInfo"
    assert captured["body"]["params"] == {"notes": [2001, 2002]}


@pytest.mark.asyncio
async def test_notes_info_empty_input_short_circuits() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("notes_info([]) must not call AnkiConnect")

    async with AnkiConnectClient(_URL, transport=_transport(handler)) as client:
        result = await client.notes_info([])

    assert result == []
