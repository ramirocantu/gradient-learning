from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class ContentEmbedding(Base):
    """Polymorphic vector store keyed by `(entity_kind, entity_id)` (V-KB1, V-E1).

    Hosts embeddings for `question` / `atomic_fact` / `outline_node` rows.
    `embedding_version` stamps every row; provider or dim change ⇒ bump
    + full re-embed (V-E1). One UQ per `(entity_kind, entity_id, version)`
    so multiple versions can coexist briefly during a migration sweep but
    a single (entity, version) is unique.

    NOTE — pgvector swap (T25): the `embedding` column is JSONB placeholder
    pending T25 (which adds the `pgvector` Python dep + `vector` Postgres
    extension + image swap). T25 ALTERs the column to `vector(N)` where N
    matches `EMBEDDING_MODEL` (default 1536 for `text-embedding-3-small`).
    Mixed-dim vectors in one column are forbidden (V-E1) — the version
    bump rule is what protects against it.
    """

    __tablename__ = "content_embeddings"
    __table_args__ = (
        CheckConstraint(
            "entity_kind IN ('question','atomic_fact','outline_node')",
            name="ck_content_embeddings_entity_kind",
        ),
        UniqueConstraint(
            "entity_kind",
            "entity_id",
            "embedding_version",
            name="uq_content_embeddings_entity_version",
        ),
        Index(
            "ix_content_embeddings_entity",
            "entity_kind",
            "entity_id",
        ),
        Index("ix_content_embeddings_version", "embedding_version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    entity_kind: Mapped[str] = mapped_column(Text, nullable=False)
    entity_id: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(
        JSONB().with_variant(JSONB, "postgresql"), nullable=True
    )
    embedding_version: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Keep the JSONB import alias usable from the rest of the codebase.
    __mapper_args__: dict[str, Any] = {}
