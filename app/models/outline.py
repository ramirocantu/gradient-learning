from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

# V-O4: reserved node-path delimiter — ASCII; renderer + parser must agree.
# ` >> ` over ` / ` (collides with ÷ in physics-formula node names). The
# schema importer (T9) rejects any node name containing this substring.
OUTLINE_PATH_DELIMITER = " >> "


class Course(Base):
    """A study domain (a course: biochem, anatomy, …).

    Adding a course + importing an outline schema materializes its
    `outline_nodes` tree. Domain-blind core (SPEC §A); MCAT/AAMC is just one
    uploaded schema (V-O3), not privileged.
    """

    __tablename__ = "courses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    slug: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    nodes: Mapped[list[OutlineNode]] = relationship(
        back_populates="course", cascade="all, delete-orphan"
    )


class OutlineNode(Base):
    """Sole outline hierarchy (V-O1). Recursive tree, arbitrary depth.

    The per-course `kind` label says what a level means (section|unit|lecture|
    concept|…). AAMC is one course expressed as a 4-deep instance
    (section→fc→cc→topic as `kind` values), ⊥ dedicated tables. Rollup is
    subtree membership (a set, not a sum): each item lives once at its most
    specific node; a parent's set = union of descendants' + own direct items.
    """

    __tablename__ = "outline_nodes"
    __table_args__ = (
        UniqueConstraint(
            "course_id", "parent_id", "name", name="uq_outline_nodes_course_parent_name"
        ),
        Index("ix_outline_nodes_course_id", "course_id"),
        Index("ix_outline_nodes_parent_id", "parent_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    course_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("courses.id", ondelete="CASCADE"), nullable=False
    )
    parent_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("outline_nodes.id", ondelete="CASCADE"), nullable=True
    )
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    depth: Mapped[int] = mapped_column(Integer, nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    course: Mapped[Course] = relationship(back_populates="nodes")
    parent: Mapped[Optional[OutlineNode]] = relationship(
        "OutlineNode",
        back_populates="children",
        remote_side=lambda: [OutlineNode.id],
    )
    children: Mapped[list[OutlineNode]] = relationship(
        "OutlineNode",
        back_populates="parent",
        cascade="all, delete-orphan",
    )
