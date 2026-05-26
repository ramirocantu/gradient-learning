"""Tests for SPEC §T41 — subtree-membership helper + per-request memoization.

Covers:
- §V31 — subtree includes the anchor itself; sibling isolation;
  descendants at any depth are included.
- T41 ticket — `SubtreeCache` memoizes per (session, topic_id) and
  `prime_cc` folds N+1 lookups into a single query.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.outline import ContentCategory, Topic
from app.services.topic_subtree import SubtreeCache, subtree_topic_ids


async def _first_cc(session: AsyncSession) -> ContentCategory:
    return (await session.execute(select(ContentCategory).limit(1))).scalar_one()


async def _second_cc(session: AsyncSession) -> ContentCategory:
    rows = (await session.execute(select(ContentCategory).limit(2))).scalars().all()
    return rows[1]


async def _make_tree(
    session: AsyncSession, cc: ContentCategory, *, label: str
) -> tuple[Topic, Topic, Topic, Topic]:
    """Build parent → child → grandchild + unrelated sibling under `cc`.

    Returns (parent, child, grandchild, sibling). Used to exercise
    anchor-inclusion, multi-level descent, and sibling isolation.
    """
    parent = Topic(
        content_category_id=cc.id,
        parent_topic_id=None,
        name=f"T41 parent {label}",
        disciplines=[],
        depth=0,
        position=920,
    )
    sibling = Topic(
        content_category_id=cc.id,
        parent_topic_id=None,
        name=f"T41 sibling {label}",
        disciplines=[],
        depth=0,
        position=921,
    )
    session.add_all([parent, sibling])
    await session.flush()
    child = Topic(
        content_category_id=cc.id,
        parent_topic_id=parent.id,
        name=f"T41 child {label}",
        disciplines=[],
        depth=1,
        position=922,
    )
    session.add(child)
    await session.flush()
    grandchild = Topic(
        content_category_id=cc.id,
        parent_topic_id=child.id,
        name=f"T41 grandchild {label}",
        disciplines=[],
        depth=2,
        position=923,
    )
    session.add(grandchild)
    await session.flush()
    return parent, child, grandchild, sibling


class _ExecuteCounter:
    """Wraps an AsyncSession to count execute() calls without breaking it."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self.count = 0

    async def execute(self, *args, **kwargs):  # noqa: ANN001 — passthrough
        self.count += 1
        return await self._session.execute(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._session, name)


# --- §V31 anchor + descent + sibling isolation ---


async def test_leaf_topic_returns_only_anchor(db_session: AsyncSession) -> None:
    """Childless topic → subtree = [topic_id] (anchor only)."""
    cc = await _first_cc(db_session)
    leaf = Topic(
        content_category_id=cc.id,
        parent_topic_id=None,
        name="T41 leaf only",
        disciplines=[],
        depth=0,
        position=930,
    )
    db_session.add(leaf)
    await db_session.flush()

    ids = await subtree_topic_ids(db_session, topic_id=leaf.id)
    assert ids == [leaf.id]


async def test_parent_subtree_includes_all_descendants(
    db_session: AsyncSession,
) -> None:
    """Parent → child → grandchild: subtree = {parent, child, grandchild}."""
    cc = await _first_cc(db_session)
    parent, child, grandchild, _sib = await _make_tree(db_session, cc, label="descend")

    ids = set(await subtree_topic_ids(db_session, topic_id=parent.id))
    assert ids == {parent.id, child.id, grandchild.id}


async def test_sibling_subtree_excludes_other_branch(
    db_session: AsyncSession,
) -> None:
    """Sibling subtree must NOT include parent's descendants."""
    cc = await _first_cc(db_session)
    parent, child, grandchild, sibling = await _make_tree(db_session, cc, label="isol")

    sib_ids = set(await subtree_topic_ids(db_session, topic_id=sibling.id))
    assert sib_ids == {sibling.id}
    assert child.id not in sib_ids
    assert grandchild.id not in sib_ids
    assert parent.id not in sib_ids


async def test_internal_node_subtree_excludes_ancestors(
    db_session: AsyncSession,
) -> None:
    """Subtree of an internal node = node + descendants; ancestors excluded."""
    cc = await _first_cc(db_session)
    parent, child, grandchild, _sib = await _make_tree(db_session, cc, label="internal")

    child_ids = set(await subtree_topic_ids(db_session, topic_id=child.id))
    assert child_ids == {child.id, grandchild.id}
    assert parent.id not in child_ids


# --- SubtreeCache: memoization ---


async def test_cache_get_returns_same_result_as_helper(
    db_session: AsyncSession,
) -> None:
    """Cache.get() and direct helper return the same membership set."""
    cc = await _first_cc(db_session)
    parent, _c, _g, _s = await _make_tree(db_session, cc, label="parity")

    direct = set(await subtree_topic_ids(db_session, topic_id=parent.id))
    cache = SubtreeCache(db_session)
    via_cache = set(await cache.get(parent.id))
    assert direct == via_cache


async def test_cache_memoizes_repeated_lookups(db_session: AsyncSession) -> None:
    """Second get() on same topic_id issues zero additional execute() calls."""
    cc = await _first_cc(db_session)
    parent, _c, _g, _s = await _make_tree(db_session, cc, label="memo")
    counter = _ExecuteCounter(db_session)
    cache = SubtreeCache(counter)  # type: ignore[arg-type]

    first = await cache.get(parent.id)
    after_first = counter.count
    second = await cache.get(parent.id)
    after_second = counter.count

    assert first == second
    assert after_first == 1
    assert after_second == 1  # no further query on repeat


# --- SubtreeCache.prime_cc: bulk closure load ---


async def test_prime_cc_single_query_populates_every_topic(
    db_session: AsyncSession,
) -> None:
    """prime_cc issues exactly one execute and primes every topic in CC."""
    cc = await _first_cc(db_session)
    parent, child, grandchild, sibling = await _make_tree(db_session, cc, label="prime")
    counter = _ExecuteCounter(db_session)
    cache = SubtreeCache(counter)  # type: ignore[arg-type]

    pre = counter.count
    await cache.prime_cc(cc.code)
    after_prime = counter.count
    assert after_prime - pre == 1

    # All four created topics are primed — get() for any of them must
    # not issue additional queries.
    for t in (parent, child, grandchild, sibling):
        await cache.get(t.id)
    assert counter.count == after_prime


async def test_prime_cc_subtree_membership_correct(
    db_session: AsyncSession,
) -> None:
    """After prime_cc, parent subtree includes child + grandchild;
    sibling subtree is singleton; child subtree excludes parent."""
    cc = await _first_cc(db_session)
    parent, child, grandchild, sibling = await _make_tree(db_session, cc, label="bulk")

    cache = SubtreeCache(db_session)
    await cache.prime_cc(cc.code)

    assert set(await cache.get(parent.id)) == {parent.id, child.id, grandchild.id}
    assert set(await cache.get(child.id)) == {child.id, grandchild.id}
    assert set(await cache.get(grandchild.id)) == {grandchild.id}
    assert set(await cache.get(sibling.id)) == {sibling.id}


async def test_prime_cc_only_caches_topics_in_that_cc(
    db_session: AsyncSession,
) -> None:
    """Topics in another CC are NOT primed; get() for them re-queries."""
    cc_a = await _first_cc(db_session)
    cc_b = await _second_cc(db_session)
    parent_a, _c, _g, _s = await _make_tree(db_session, cc_a, label="ccA")
    parent_b = Topic(
        content_category_id=cc_b.id,
        parent_topic_id=None,
        name="T41 lone ccB",
        disciplines=[],
        depth=0,
        position=940,
    )
    db_session.add(parent_b)
    await db_session.flush()

    counter = _ExecuteCounter(db_session)
    cache = SubtreeCache(counter)  # type: ignore[arg-type]
    await cache.prime_cc(cc_a.code)
    prime_count = counter.count

    # CC_A topic — primed, no extra query.
    await cache.get(parent_a.id)
    assert counter.count == prime_count

    # CC_B topic — NOT primed, one fallback query.
    await cache.get(parent_b.id)
    assert counter.count == prime_count + 1


async def test_cache_independent_per_instance(db_session: AsyncSession) -> None:
    """Two caches don't share state — primes don't leak across requests."""
    cc = await _first_cc(db_session)
    parent, _c, _g, _s = await _make_tree(db_session, cc, label="indep")

    cache_a = SubtreeCache(db_session)
    await cache_a.prime_cc(cc.code)
    counter_b = _ExecuteCounter(db_session)
    cache_b = SubtreeCache(counter_b)  # type: ignore[arg-type]
    # cache_b has no prior knowledge — fresh lookup must hit DB.
    await cache_b.get(parent.id)
    assert counter_b.count == 1
