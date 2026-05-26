from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class NotionPage(Base):
    """Pointer to a Notion page mirroring one outline node (V-N1, V-N2).

    Write-out only: Postgres → Notion, never read-back, never local content
    copy (V-N1). One page per outline node (V-N2) — `node_id` is UQ. Re-sync
    upserts on `node_id`; the stored `tags` JSONB is the last-synced tag
    snapshot for backlink rendering, ⊥ a source of truth.
    """

    __tablename__ = "notion_pages"
    __table_args__ = (
        UniqueConstraint("node_id", name="uq_notion_pages_node_id"),
        UniqueConstraint("notion_page_id", name="uq_notion_pages_notion_page_id"),
        Index("ix_notion_pages_node_id", "node_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    node_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("outline_nodes.id", ondelete="CASCADE"), nullable=False
    )
    notion_page_id: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    tags: Mapped[Optional[list[Any]]] = mapped_column(JSONB, nullable=True)
    last_synced_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
