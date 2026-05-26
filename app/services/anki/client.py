"""AnkiConnect HTTP client.

Thin async wrapper around the AnkiConnect addon's JSON-RPC interface
(localhost:8765 by default). Per SPEC §V13 / §V50 (amended 2026-05-23
per T74), this client exposes read-only actions plus a closed write
allowlist: `unsuspend` + `addTags` on source-deck cards (audit-trail
tag mutation, no scheduler impact) and `createFilteredDeck` constrained
to `<ANKI_DECK_PREFIX>::review::*`.

Per SPEC §V4, transport-level connection failure raises
AnkiUnreachableError; callers (T3 sync service) map this to the
{synced_cards: 0, error: "anki_not_running"} envelope. Per §V55,
write-side failures map malformed/null responses to
AnkiUnreachableError and non-null `error` fields to AnkiWriteFailed,
keeping the read-side error envelope unchanged.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.config import settings


_JSONRPC_VERSION = 6


class AnkiUnreachableError(RuntimeError):
    """AnkiConnect endpoint not reachable (connection refused, DNS, timeout)."""


class AnkiConnectError(RuntimeError):
    """AnkiConnect returned a non-null `error` field in the JSON-RPC response."""


class AnkiWriteForbidden(RuntimeError):
    """Write attempt violates SPEC §V50 allowlist (filtered-deck name
    outside `<ANKI_DECK_PREFIX>::review::*` namespace, or filtered-deck
    query body other than `cid:<csv>`)."""


class AnkiWriteFailed(RuntimeError):
    """AnkiConnect returned a non-null `error` field on a write action.
    Distinct from `AnkiConnectError` so callers can branch (e.g. T63
    increment `anki_assignments.failure_count` on writes only)."""


class AnkiConnectClient:
    """Async, read-only AnkiConnect client.

    Construct once per request scope; the underlying httpx.AsyncClient is
    created lazily on first call and closed by `aclose()` or via async
    context-manager protocol.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float | httpx.Timeout | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        write_deck_prefix: str | None = None,
    ) -> None:
        self._base_url = base_url
        # Default split timeout: connect should be near-instant for a healthy
        # AnkiConnect on loopback, but reads on large decks (findCards over a
        # 6k-card AnKing deck, cardsInfo over thousands of cards) can take
        # tens of seconds. A single tight 5s timeout (SPEC §B2) caused all
        # syncs against the AnKing deck to surface as
        # `error="anki_not_running"` because ReadTimeout was misclassified
        # as unreachable. Splitting connect vs read keeps the "Anki really
        # down" signal sharp while letting big reads complete.
        if timeout is None:
            timeout = httpx.Timeout(connect=5.0, read=120.0, write=30.0, pool=5.0)
        self._timeout = timeout
        self._transport = transport
        self._http: httpx.AsyncClient | None = None
        self._write_prefix = (
            write_deck_prefix if write_deck_prefix is not None else settings.ANKI_DECK_PREFIX
        )

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            kwargs: dict[str, Any] = {"timeout": self._timeout}
            if self._transport is not None:
                kwargs["transport"] = self._transport
            self._http = httpx.AsyncClient(**kwargs)
        return self._http

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def __aenter__(self) -> AnkiConnectClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def _invoke(self, action: str, **params: Any) -> Any:
        payload = {"action": action, "version": _JSONRPC_VERSION, "params": params}
        try:
            response = await self._client().post(self._base_url, json=payload)
        except httpx.ConnectError as exc:
            raise AnkiUnreachableError(str(exc)) from exc
        except httpx.ConnectTimeout as exc:
            raise AnkiUnreachableError(f"timeout connecting to {self._base_url}") from exc
        except httpx.ReadTimeout as exc:
            raise AnkiUnreachableError(f"timeout reading from {self._base_url}") from exc

        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict) or "result" not in body or "error" not in body:
            raise AnkiConnectError(f"malformed AnkiConnect response: {body!r}")
        if body["error"] is not None:
            raise AnkiConnectError(str(body["error"]))
        return body["result"]

    async def version(self) -> int:
        result = await self._invoke("version")
        return int(result)

    async def find_cards(self, query: str) -> list[int]:
        result = await self._invoke("findCards", query=query)
        if not isinstance(result, list):
            raise AnkiConnectError(f"findCards expected list, got {type(result).__name__}")
        return [int(card_id) for card_id in result]

    async def cards_info(self, card_ids: list[int]) -> list[dict[str, Any]]:
        if not card_ids:
            return []
        result = await self._invoke("cardsInfo", cards=list(card_ids))
        if not isinstance(result, list):
            raise AnkiConnectError(f"cardsInfo expected list, got {type(result).__name__}")
        return result

    async def notes_info(self, note_ids: list[int]) -> list[dict[str, Any]]:
        """Per-note metadata including `tags`. AnkiConnect's cardsInfo carries
        card-level scheduler state but no tags — tags live on the parent note,
        so the sync service issues notesInfo for the unique note IDs it
        observes in cardsInfo's responses."""
        if not note_ids:
            return []
        result = await self._invoke("notesInfo", notes=list(note_ids))
        if not isinstance(result, list):
            raise AnkiConnectError(f"notesInfo expected list, got {type(result).__name__}")
        return result

    async def card_reviews(self, deck: str, start_id: int) -> list[list[int]]:
        """Revlog rows for `deck` with `id >= start_id` (Anki revlog id is
        unix-ms, monotonically increasing). Used by the T36 sync extension
        to append to `anki_card_reviews` incrementally per §V26.

        AnkiConnect returns each row as a positional tuple:
        `[reviewTime_ms, cardId, usn, button, newInterval, prevInterval,
        newFactor, reviewDuration_ms, reviewType]`.
        """
        result = await self._invoke("cardReviews", deck=deck, startID=int(start_id))
        if not isinstance(result, list):
            raise AnkiConnectError(f"cardReviews expected list, got {type(result).__name__}")
        return result

    async def deck_names(self) -> list[str]:
        """All deck names known to Anki. Read-only — used by the sync service
        to enumerate decks in the loud-fail log when the configured deck name
        does not match anything (SPEC §V20)."""
        result = await self._invoke("deckNames")
        if not isinstance(result, list):
            raise AnkiConnectError(f"deckNames expected list, got {type(result).__name__}")
        return [str(d) for d in result]

    # ----------------------- write allowlist (V50) ------------------------- #

    async def _write_invoke(self, action: str, **params: Any) -> Any:
        """Write-side _invoke twin per §V55.

        Maps malformed / null-body responses to `AnkiUnreachableError` and
        non-null `error` fields to `AnkiWriteFailed`. The read-side `_invoke`
        keeps its existing `AnkiConnectError` semantics so unrelated read
        callers stay untouched.
        """
        payload = {"action": action, "version": _JSONRPC_VERSION, "params": params}
        try:
            response = await self._client().post(self._base_url, json=payload)
        except httpx.ConnectError as exc:
            raise AnkiUnreachableError(str(exc)) from exc
        except httpx.ConnectTimeout as exc:
            raise AnkiUnreachableError(f"timeout connecting to {self._base_url}") from exc
        except httpx.ReadTimeout as exc:
            raise AnkiUnreachableError(f"timeout reading from {self._base_url}") from exc

        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict) or "result" not in body or "error" not in body:
            raise AnkiUnreachableError(f"malformed AnkiConnect response: {body!r}")
        if body["error"] is not None:
            raise AnkiWriteFailed(str(body["error"]))
        return body["result"]

    def _require_review_namespaced(self, name: str) -> None:
        prefix = self._write_prefix
        if not name.startswith(prefix + "::review::"):
            raise AnkiWriteForbidden(
                f"filtered-deck name {name!r} must match {prefix!r}::review::* (SPEC §V50)"
            )

    async def unsuspend_cards(self, card_ids: list[int]) -> bool:
        """V50 allowlist: flip queue -1 → 0 on the given source-deck cards.
        Does NOT mutate scheduling intervals or ease (Anki semantics).
        Source-deck membership is enforced at the caller layer (T62) — the
        client trusts the id list.

        Card ids are de-duplicated before the call (V64): AnkiConnect builds
        an internal `search_cids` temp table w/ UNIQUE(cid), so a duplicate
        id raises `UNIQUE constraint failed: search_cids.cid` and fails the
        whole batch. Dedup is order-preserving (first occurrence wins)."""
        if not card_ids:
            return True
        deduped = list(dict.fromkeys(int(c) for c in card_ids))
        result = await self._write_invoke("unsuspend", cards=deduped)
        return bool(result)

    async def add_tags(self, note_ids: list[int], tags: list[str]) -> None:
        """V50 allowlist: add tags to source-deck NOTES (§V63 superseded by §V75).

        The tags are write-only audit-trail markers (`coach::assignment:{n}`
        after unlock, `coach::review:{m}` after filtered-deck push). Anki tag
        mutation does not touch SRS state, so this is safe on source-deck
        notes (V50 deliberate reversal of the prior ban; see §V50 rationale).

        AnkiConnect's `addTags` action is note-scoped — its params are `notes`
        (note-ids) + a space-separated tag string; there is no `cards` param.
        Post-§V75 callers pass note-ids straight from the stored snapshot
        (`anki_assignments.note_ids` / `anki_reviews.note_ids`), so the prior
        runtime card→note `cardsInfo` resolve (V63) is gone — no `cardsInfo`
        call on this path. Tagging a note marks all its cards, the only
        granularity Anki offers, matching the audit-trail intent.

        No namespace check — tag values are owned by mcat-coach and are
        free-form per the allowlist. Empty note_ids or empty tags short-
        circuit to a no-op (matches `unsuspend_cards` shape)."""
        if not note_ids or not tags:
            return
        # Dedup note ids (V64 belt) before the write.
        deduped_notes = list(dict.fromkeys(int(n) for n in note_ids))
        tag_string = " ".join(tags)
        await self._write_invoke("addTags", notes=deduped_notes, tags=tag_string)

    async def create_filtered_deck(self, name: str, card_ids: list[int]) -> int:
        """V50 allowlist: create a filtered (dynamic) deck inside
        `<prefix>::review::*` populated via a `cid:<csv>` search query —
        the only query shape permitted at this layer. `reschedule=False`
        forces cram mode so the filtered deck does not perturb scheduling
        when cards return home (V13)."""
        self._require_review_namespaced(name)
        if not card_ids:
            raise AnkiWriteForbidden(
                "filtered-deck card_ids empty: searchQuery must be cid:<csv> (SPEC §V50)"
            )
        # Dedup (V64): duplicate cids in the search query blow AnkiConnect's
        # internal `search_cids` UNIQUE(cid). Order-preserving, first wins.
        deduped = list(dict.fromkeys(int(cid) for cid in card_ids))
        search_query = "cid:" + ",".join(str(cid) for cid in deduped)
        result = await self._write_invoke(
            "createFilteredDeck",
            newDeckName=name,
            searchQuery=search_query,
            gatherCount=len(deduped),
            reschedule=False,
        )
        return int(result)
