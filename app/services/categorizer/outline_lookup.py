"""In-memory lookup over the seeded AAMC outline.

Built once per request (or once per worker process — see Ticket 3.2).
Resolves YAML target references (`content_category: "5E"`, `topic: "Energy"`,
etc.) to the integer primary keys held in the `sections`, `content_categories`,
and `topics` tables.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.outline import ContentCategory, Section, Topic
from app.services.categorizer._text import normalize_typographic_punctuation

logger = logging.getLogger(__name__)


class OutlineNotSeededError(RuntimeError):
    """Raised when OutlineLookup loads against an unseeded outline table set.

    Belt-and-suspenders for the auto-seed in `app.startup.ensure_outline_seeded`.
    If a future entrypoint forgets to call it, the categorizer raises here
    instead of silently dropping every topic/content_category suggestion.
    """


@dataclass(frozen=True)
class _TopicRow:
    id: int
    name: str
    content_category_id: int
    parent_topic_id: Optional[int]


class OutlineLookup:
    """Caches the outline tree on construction; provides ID lookups by code/name."""

    def __init__(
        self,
        *,
        sections_by_code: dict[str, int],
        ccs_by_code: dict[str, int],
        topics: list[_TopicRow],
    ) -> None:
        self._sections = sections_by_code
        self._ccs = ccs_by_code
        self._topics = topics
        self._cc_code_by_id = {v: k for k, v in ccs_by_code.items()}
        self._topics_by_id: dict[int, _TopicRow] = {t.id: t for t in topics}

    @classmethod
    async def load(cls, session: AsyncSession) -> "OutlineLookup":
        sections = (await session.execute(select(Section))).scalars().all()
        ccs = (await session.execute(select(ContentCategory))).scalars().all()
        topics = (await session.execute(select(Topic))).scalars().all()
        if not ccs or not topics:
            raise OutlineNotSeededError(
                f"outline tables empty (sections={len(sections)}, "
                f"content_categories={len(ccs)}, topics={len(topics)}); "
                "run `uv run python scripts/seed_outline.py` or boot via "
                "uvicorn so app.startup.ensure_outline_seeded fires"
            )
        return cls(
            sections_by_code={s.code: s.id for s in sections},
            ccs_by_code={c.code: c.id for c in ccs},
            topics=[
                _TopicRow(
                    id=t.id,
                    # Normalize so path lookups match regardless of apostrophe variant.
                    # The DB row is unchanged; only the in-memory comparison side is ASCII.
                    name=normalize_typographic_punctuation(t.name),
                    content_category_id=t.content_category_id,
                    parent_topic_id=t.parent_topic_id,
                )
                for t in topics
            ],
        )

    def section_id(self, code: str) -> int | None:
        return self._sections.get(code)

    def content_category_id(self, code: str) -> int | None:
        return self._ccs.get(code)

    def topic_id_by_path(self, path: str) -> int | None:
        """Resolve a `>>`-delimited topic path to a topic ID.

        Format: `"<CC_code> >> <name>"` or `"<CC_code> >> <parent> >> <child>"` etc.
        Walks the parent-chain in the in-memory topic list. Returns None (with
        a warning) if any segment fails to resolve uniquely.

        Per §V40, the reserved delimiter is ` >> ` because ` / ` collides with
        ÷ notation in physics-formula topic names (e.g. `Resistivity: ρ = R•A / L`).
        """
        parts = [p.strip() for p in normalize_typographic_punctuation(path).split(" >> ")]
        if len(parts) < 2 or not parts[0]:
            logger.warning("topic_id_by_path: malformed path %r", path)
            return None

        cc_code = parts[0]
        name_parts = parts[1:]

        cc_id = self._ccs.get(cc_code)
        if cc_id is None:
            logger.warning("topic_id_by_path: unknown CC %r in path %r", cc_code, path)
            return None

        current_parent_id: Optional[int] = None
        for i, name in enumerate(name_parts):
            candidates = [
                t
                for t in self._topics
                if t.name == name
                and t.content_category_id == cc_id
                and t.parent_topic_id == current_parent_id
            ]
            if len(candidates) == 0:
                logger.warning(
                    "topic_id_by_path: no topic named %r at segment %d of %r",
                    name,
                    i + 1,
                    path,
                )
                return None
            if len(candidates) > 1:
                logger.warning(
                    "topic_id_by_path: ambiguous topic %r at segment %d of %r (%d matches)",
                    name,
                    i + 1,
                    path,
                    len(candidates),
                )
                return None
            current_parent_id = candidates[0].id

        return current_parent_id

    def topic_id(
        self,
        name: str,
        *,
        under_content_category: str | None = None,
        under_topic: str | None = None,
    ) -> int | None:
        cc_id: int | None = None
        if under_content_category is not None:
            cc_id = self._ccs.get(under_content_category)
            if cc_id is None:
                logger.warning(
                    "topic_id: unknown content_category %r when resolving topic %r",
                    under_content_category,
                    name,
                )
                return None

        parent_id: int | None = None
        if under_topic is not None:
            parent_candidates = [
                t
                for t in self._topics
                if t.name == under_topic and (cc_id is None or t.content_category_id == cc_id)
            ]
            if len(parent_candidates) != 1:
                logger.warning(
                    "topic_id: parent topic %r is ambiguous or missing (cc=%s, %d matches)",
                    under_topic,
                    under_content_category,
                    len(parent_candidates),
                )
                return None
            parent_id = parent_candidates[0].id

        candidates = [
            t
            for t in self._topics
            if t.name == name
            and (cc_id is None or t.content_category_id == cc_id)
            and (under_topic is None or t.parent_topic_id == parent_id)
        ]
        if len(candidates) == 1:
            return candidates[0].id
        if not candidates:
            logger.warning(
                "topic_id: no topic named %r (cc=%s, parent=%s)",
                name,
                under_content_category,
                under_topic,
            )
            return None
        logger.warning(
            "topic_id: ambiguous topic %r (cc=%s, parent=%s, %d matches) — refine the YAML",
            name,
            under_content_category,
            under_topic,
            len(candidates),
        )
        return None
