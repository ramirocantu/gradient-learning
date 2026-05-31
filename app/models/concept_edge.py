from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class ConceptEdge(Base):
    """Cross-domain node↔node edge (V-KB1, V-E2, §A).

    `kind='similarity'` rows are derived (cosine over `content_embeddings`);
    `kind='manual'` rows are human-verified (V-E2). UQ on `(src, dst, kind)`
    so the same pair across both kinds is allowed but each kind dedupes.
    """

    __tablename__ = "concept_edges"
    __table_args__ = (
        CheckConstraint(
            "kind IN ('similarity','manual')",
            name="ck_concept_edges_kind",
        ),
        CheckConstraint(
            "src_node_id <> dst_node_id",
            name="ck_concept_edges_no_self_edge",
        ),
        CheckConstraint(
            "score IS NULL OR (score BETWEEN -1.0 AND 1.0)",
            name="ck_concept_edges_score_range",
        ),
        UniqueConstraint(
            "src_node_id", "dst_node_id", "kind", name="uq_concept_edges_src_dst_kind"
        ),
        Index("ix_concept_edges_src", "src_node_id"),
        Index("ix_concept_edges_dst", "dst_node_id"),
        Index("ix_concept_edges_kind", "kind"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    src_node_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("outline_nodes.id", ondelete="CASCADE"), nullable=False
    )
    dst_node_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("outline_nodes.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    score: Mapped[Optional[float]] = mapped_column(Numeric(6, 5), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
