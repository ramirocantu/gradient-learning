from __future__ import annotations

from typing import Optional

from sqlalchemy import ForeignKey, Index, Integer, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Section(Base):
    __tablename__ = "sections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)

    foundational_concepts: Mapped[list[FoundationalConcept]] = relationship(
        back_populates="section", cascade="all, delete-orphan"
    )


class FoundationalConcept(Base):
    __tablename__ = "foundational_concepts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    section_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("sections.id", ondelete="CASCADE"), nullable=False
    )
    code: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)

    section: Mapped[Section] = relationship(back_populates="foundational_concepts")
    content_categories: Mapped[list[ContentCategory]] = relationship(
        back_populates="foundational_concept", cascade="all, delete-orphan"
    )


class ContentCategory(Base):
    __tablename__ = "content_categories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    foundational_concept_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("foundational_concepts.id", ondelete="CASCADE"),
        nullable=False,
    )
    code: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    position: Mapped[int] = mapped_column(Integer, nullable=False)

    foundational_concept: Mapped[FoundationalConcept] = relationship(
        back_populates="content_categories"
    )
    topics: Mapped[list[Topic]] = relationship(
        back_populates="content_category",
        foreign_keys="[Topic.content_category_id]",
        cascade="all, delete-orphan",
    )


class Topic(Base):
    __tablename__ = "topics"
    __table_args__ = (
        UniqueConstraint(
            "content_category_id",
            "parent_topic_id",
            "name",
            name="uq_topic_cc_parent_name",
        ),
        Index("ix_topic_content_category_id", "content_category_id"),
        Index("ix_topic_parent_topic_id", "parent_topic_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    content_category_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("content_categories.id", ondelete="RESTRICT"),
        nullable=False,
    )
    parent_topic_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("topics.id", ondelete="SET NULL"), nullable=True
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    disciplines: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, server_default="{}")
    depth: Mapped[int] = mapped_column(Integer, nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)

    content_category: Mapped[ContentCategory] = relationship(
        back_populates="topics", foreign_keys=lambda: [Topic.content_category_id]
    )
    parent: Mapped[Optional[Topic]] = relationship(
        "Topic",
        back_populates="children",
        foreign_keys=lambda: [Topic.parent_topic_id],
        remote_side=lambda: [Topic.id],
    )
    children: Mapped[list[Topic]] = relationship(
        "Topic",
        back_populates="parent",
        foreign_keys=lambda: [Topic.parent_topic_id],
    )
