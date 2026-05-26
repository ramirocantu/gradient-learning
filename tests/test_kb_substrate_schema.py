"""T24 — schema-level tests for the P2 KB substrate (V-KB1, §I.schema).

Covers pdf_sources, atomic_facts, content_embeddings, concept_edges,
notion_pages, discriminator_factors: migration roundtrip, insert
round-trip, UQ/CHECK guards, FK cascade behavior. Embedding column is
JSONB placeholder (T25 swaps to pgvector vector(N)).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import uuid
from pathlib import Path

import asyncpg
import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.atomic_fact import AtomicFact
from app.models.atomic_fact_tag import AtomicFactTag
from app.models.captures import Question
from app.models.concept_edge import ConceptEdge
from app.models.content_embedding import ContentEmbedding
from app.models.discriminator_factor import DiscriminatorFactor
from app.models.notion_page import NotionPage
from app.models.outline import Course, OutlineNode
from app.models.pdf_source import PdfSource


_HOST_PORT = os.environ.get("HOST_POSTGRES_PORT", "5432")
_ADMIN_DSN = f"postgresql://gradient:gradient_secret@localhost:{_HOST_PORT}/gradient"
_BACKEND_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture
async def tx_session(seeded_report, test_engine):
    conn = await test_engine.connect()
    await conn.begin()
    session = AsyncSession(
        bind=conn,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )
    try:
        yield session
    finally:
        await session.close()
        await conn.rollback()
        await conn.close()


def _slug() -> str:
    return f"c-{uuid.uuid4().hex[:8]}"


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


async def _make_course(session: AsyncSession) -> Course:
    c = Course(slug=_slug(), name="KB Test Course")
    session.add(c)
    await session.flush()
    return c


async def _make_node(session: AsyncSession, course: Course, name: str = "Node A") -> OutlineNode:
    n = OutlineNode(
        course_id=course.id,
        parent_id=None,
        kind="concept",
        name=f"{name}-{uuid.uuid4().hex[:6]}",
        depth=0,
        position=0,
    )
    session.add(n)
    await session.flush()
    return n


async def _make_pdf(session: AsyncSession, course: Course, *, sha: str | None = None) -> PdfSource:
    p = PdfSource(
        course_id=course.id,
        filename="lecture.pdf",
        sha256=sha or _hash(uuid.uuid4().hex),
        status="pending",
    )
    session.add(p)
    await session.flush()
    return p


async def _make_question(session: AsyncSession) -> Question:
    q = Question(
        qid=f"q-{uuid.uuid4().hex[:12]}",
        stem_html="<p>x</p>",
        stem_plain="x",
        choices=[{"key": "A", "html": "<p>a</p>", "plain": "a", "media_ids": []}],
        correct_choice="A",
    )
    session.add(q)
    await session.flush()
    return q


# --------------------------------------------------------------------------- #
# 1. Migration roundtrip — 0003_kb_substrate up→down→up clean
# --------------------------------------------------------------------------- #


async def test_migration_apply_and_rollback():
    db_name = "gradient_kb_substrate_migrate_test"
    admin = await asyncpg.connect(_ADMIN_DSN)
    try:
        await admin.execute(
            f"""
            SELECT pg_terminate_backend(pid)
              FROM pg_stat_activity
             WHERE datname = '{db_name}' AND pid <> pg_backend_pid();
            """
        )
        await admin.execute(f'DROP DATABASE IF EXISTS "{db_name}"')
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()

    target_url = f"postgresql+asyncpg://gradient:gradient_secret@localhost:{_HOST_PORT}/{db_name}"
    env = {**os.environ, "ALEMBIC_DATABASE_URL": target_url}

    try:
        for args in (
            ["alembic", "upgrade", "head"],
            ["alembic", "downgrade", "0002_source_discriminator"],
            ["alembic", "upgrade", "head"],
        ):
            r = subprocess.run(
                args,
                cwd=_BACKEND_DIR,
                env=env,
                capture_output=True,
                text=True,
            )
            assert r.returncode == 0, (
                f"alembic {' '.join(args[1:])} failed\n"
                f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
            )
    finally:
        admin = await asyncpg.connect(_ADMIN_DSN)
        try:
            await admin.execute(
                f"""
                SELECT pg_terminate_backend(pid)
                  FROM pg_stat_activity
                 WHERE datname = '{db_name}' AND pid <> pg_backend_pid();
                """
            )
            await admin.execute(f'DROP DATABASE IF EXISTS "{db_name}"')
        finally:
            await admin.close()


# --------------------------------------------------------------------------- #
# 2. pdf_sources
# --------------------------------------------------------------------------- #


async def test_pdf_source_insert_round_trip(tx_session: AsyncSession):
    course = await _make_course(tx_session)
    course_id = course.id
    pdf = await _make_pdf(tx_session, course)
    pid = pdf.id

    tx_session.expire_all()
    got = (await tx_session.execute(select(PdfSource).where(PdfSource.id == pid))).scalar_one()
    assert got.course_id == course_id
    assert got.status == "pending"
    assert got.filename == "lecture.pdf"


async def test_pdf_source_sha256_unique(tx_session: AsyncSession):
    course = await _make_course(tx_session)
    sha = _hash("same-file")
    await _make_pdf(tx_session, course, sha=sha)
    await tx_session.commit()

    with pytest.raises(IntegrityError):
        await _make_pdf(tx_session, course, sha=sha)


async def test_pdf_source_status_check(tx_session: AsyncSession):
    course = await _make_course(tx_session)
    await tx_session.commit()

    with pytest.raises(IntegrityError):
        tx_session.add(
            PdfSource(
                course_id=course.id,
                filename="x.pdf",
                sha256=_hash(uuid.uuid4().hex),
                status="bogus",
            )
        )
        await tx_session.flush()


async def test_pdf_source_course_cascade(tx_session: AsyncSession):
    course = await _make_course(tx_session)
    pdf = await _make_pdf(tx_session, course)
    pid = pdf.id

    await tx_session.delete(course)
    await tx_session.flush()

    remaining = (
        await tx_session.execute(select(PdfSource).where(PdfSource.id == pid))
    ).scalar_one_or_none()
    assert remaining is None


# --------------------------------------------------------------------------- #
# 3. atomic_facts
# --------------------------------------------------------------------------- #


async def test_atomic_fact_insert_round_trip(tx_session: AsyncSession):
    course = await _make_course(tx_session)
    pdf = await _make_pdf(tx_session, course)
    node = await _make_node(tx_session, course)
    pdf_id, node_id = pdf.id, node.id
    f = AtomicFact(
        course_id=course.id,
        pdf_source_id=pdf_id,
        page=12,
        text="Glycolysis converts glucose to pyruvate.",
        node_id=node_id,
        content_hash=_hash("fact-1"),
    )
    tx_session.add(f)
    await tx_session.flush()
    fid = f.id

    tx_session.expire_all()
    got = (await tx_session.execute(select(AtomicFact).where(AtomicFact.id == fid))).scalar_one()
    assert got.pdf_source_id == pdf_id
    assert got.node_id == node_id
    assert got.page == 12


async def test_atomic_fact_dedup_within_course(tx_session: AsyncSession):
    course = await _make_course(tx_session)
    pdf = await _make_pdf(tx_session, course)
    chash = _hash("dup-fact")
    tx_session.add(
        AtomicFact(
            course_id=course.id,
            pdf_source_id=pdf.id,
            text="A",
            content_hash=chash,
        )
    )
    await tx_session.flush()
    await tx_session.commit()

    with pytest.raises(IntegrityError):
        tx_session.add(
            AtomicFact(
                course_id=course.id,
                pdf_source_id=pdf.id,
                text="A again",
                content_hash=chash,
            )
        )
        await tx_session.flush()


async def test_atomic_fact_node_set_null_on_node_delete(tx_session: AsyncSession):
    course = await _make_course(tx_session)
    pdf = await _make_pdf(tx_session, course)
    node = await _make_node(tx_session, course)
    f = AtomicFact(
        course_id=course.id,
        pdf_source_id=pdf.id,
        text="x",
        node_id=node.id,
        content_hash=_hash(uuid.uuid4().hex),
    )
    tx_session.add(f)
    await tx_session.flush()
    fid = f.id

    await tx_session.delete(node)
    await tx_session.flush()
    tx_session.expire_all()

    got = (await tx_session.execute(select(AtomicFact).where(AtomicFact.id == fid))).scalar_one()
    assert got.node_id is None


async def test_atomic_fact_pdf_cascade(tx_session: AsyncSession):
    course = await _make_course(tx_session)
    pdf = await _make_pdf(tx_session, course)
    f = AtomicFact(
        course_id=course.id,
        pdf_source_id=pdf.id,
        text="x",
        content_hash=_hash(uuid.uuid4().hex),
    )
    tx_session.add(f)
    await tx_session.flush()
    fid = f.id

    await tx_session.delete(pdf)
    await tx_session.flush()

    remaining = (
        await tx_session.execute(select(AtomicFact).where(AtomicFact.id == fid))
    ).scalar_one_or_none()
    assert remaining is None


# --------------------------------------------------------------------------- #
# 4. content_embeddings
# --------------------------------------------------------------------------- #


async def test_content_embedding_insert_round_trip(tx_session: AsyncSession):
    e = ContentEmbedding(
        entity_kind="atomic_fact",
        entity_id=42,
        embedding=[0.1, 0.2, 0.3],
        embedding_version="v1-text-embedding-3-small-1536",
    )
    tx_session.add(e)
    await tx_session.flush()
    eid = e.id

    tx_session.expire_all()
    got = (
        await tx_session.execute(select(ContentEmbedding).where(ContentEmbedding.id == eid))
    ).scalar_one()
    assert got.entity_kind == "atomic_fact"
    assert got.embedding == [0.1, 0.2, 0.3]
    assert got.embedding_version.startswith("v1-")


async def test_content_embedding_entity_kind_check(tx_session: AsyncSession):
    await tx_session.commit()
    with pytest.raises(IntegrityError):
        tx_session.add(
            ContentEmbedding(
                entity_kind="bogus",
                entity_id=1,
                embedding_version="v1",
            )
        )
        await tx_session.flush()


async def test_content_embedding_unique_per_entity_version(tx_session: AsyncSession):
    tx_session.add(
        ContentEmbedding(
            entity_kind="question",
            entity_id=7,
            embedding_version="v1",
        )
    )
    await tx_session.flush()
    await tx_session.commit()

    with pytest.raises(IntegrityError):
        tx_session.add(
            ContentEmbedding(
                entity_kind="question",
                entity_id=7,
                embedding_version="v1",
            )
        )
        await tx_session.flush()


async def test_content_embedding_version_coexistence(tx_session: AsyncSession):
    # V-E1: version bump lets two rows for the same (kind, id) coexist
    # briefly during a re-embed sweep — UQ is on the full triple.
    tx_session.add(
        ContentEmbedding(entity_kind="question", entity_id=9, embedding_version="v1")
    )
    tx_session.add(
        ContentEmbedding(entity_kind="question", entity_id=9, embedding_version="v2")
    )
    await tx_session.flush()


# --------------------------------------------------------------------------- #
# 5. concept_edges
# --------------------------------------------------------------------------- #


async def test_concept_edge_insert_round_trip(tx_session: AsyncSession):
    course = await _make_course(tx_session)
    a = await _make_node(tx_session, course, name="A")
    b = await _make_node(tx_session, course, name="B")
    a_id, b_id = a.id, b.id
    e = ConceptEdge(src_node_id=a_id, dst_node_id=b_id, kind="similarity", score=0.83)
    tx_session.add(e)
    await tx_session.flush()
    eid = e.id

    tx_session.expire_all()
    got = (
        await tx_session.execute(select(ConceptEdge).where(ConceptEdge.id == eid))
    ).scalar_one()
    assert got.kind == "similarity"
    assert float(got.score) == pytest.approx(0.83)
    assert got.src_node_id == a_id
    assert got.dst_node_id == b_id


async def test_concept_edge_kind_check(tx_session: AsyncSession):
    course = await _make_course(tx_session)
    a = await _make_node(tx_session, course, name="A")
    b = await _make_node(tx_session, course, name="B")
    await tx_session.commit()

    with pytest.raises(IntegrityError):
        tx_session.add(ConceptEdge(src_node_id=a.id, dst_node_id=b.id, kind="bogus"))
        await tx_session.flush()


async def test_concept_edge_no_self_edge(tx_session: AsyncSession):
    course = await _make_course(tx_session)
    a = await _make_node(tx_session, course, name="A")
    await tx_session.commit()

    with pytest.raises(IntegrityError):
        tx_session.add(ConceptEdge(src_node_id=a.id, dst_node_id=a.id, kind="manual"))
        await tx_session.flush()


async def test_concept_edge_unique_src_dst_kind(tx_session: AsyncSession):
    course = await _make_course(tx_session)
    a = await _make_node(tx_session, course, name="A")
    b = await _make_node(tx_session, course, name="B")
    tx_session.add(ConceptEdge(src_node_id=a.id, dst_node_id=b.id, kind="similarity"))
    await tx_session.flush()
    # Same pair, different kind = allowed.
    tx_session.add(ConceptEdge(src_node_id=a.id, dst_node_id=b.id, kind="manual"))
    await tx_session.flush()
    await tx_session.commit()

    with pytest.raises(IntegrityError):
        tx_session.add(ConceptEdge(src_node_id=a.id, dst_node_id=b.id, kind="similarity"))
        await tx_session.flush()


# --------------------------------------------------------------------------- #
# 6. notion_pages
# --------------------------------------------------------------------------- #


async def test_notion_page_insert_round_trip(tx_session: AsyncSession):
    course = await _make_course(tx_session)
    node = await _make_node(tx_session, course)
    node_id = node.id
    p = NotionPage(
        node_id=node_id,
        notion_page_id="page-abc",
        url="https://notion.so/page-abc",
        tags=["biochem", "glycolysis"],
    )
    tx_session.add(p)
    await tx_session.flush()
    pid = p.id

    tx_session.expire_all()
    got = (
        await tx_session.execute(select(NotionPage).where(NotionPage.id == pid))
    ).scalar_one()
    assert got.node_id == node_id
    assert got.tags == ["biochem", "glycolysis"]


async def test_notion_page_one_per_node(tx_session: AsyncSession):
    course = await _make_course(tx_session)
    node = await _make_node(tx_session, course)
    tx_session.add(
        NotionPage(node_id=node.id, notion_page_id="p1", url="u1")
    )
    await tx_session.flush()
    await tx_session.commit()

    with pytest.raises(IntegrityError):
        tx_session.add(
            NotionPage(node_id=node.id, notion_page_id="p2", url="u2")
        )
        await tx_session.flush()


async def test_notion_page_node_cascade(tx_session: AsyncSession):
    course = await _make_course(tx_session)
    node = await _make_node(tx_session, course)
    p = NotionPage(node_id=node.id, notion_page_id="px", url="ux")
    tx_session.add(p)
    await tx_session.flush()
    pid = p.id

    await tx_session.delete(node)
    await tx_session.flush()

    remaining = (
        await tx_session.execute(select(NotionPage).where(NotionPage.id == pid))
    ).scalar_one_or_none()
    assert remaining is None


# --------------------------------------------------------------------------- #
# 7. discriminator_factors
# --------------------------------------------------------------------------- #


async def test_discriminator_factor_insert_round_trip(tx_session: AsyncSession):
    course = await _make_course(tx_session)
    node = await _make_node(tx_session, course)
    q = await _make_question(tx_session)
    q_id, node_id = q.id, node.id
    df = DiscriminatorFactor(
        question_id=q_id,
        factor_text="Confuses Km with Vmax.",
        node_id=node_id,
        notion_block_id="block-abc",
    )
    tx_session.add(df)
    await tx_session.flush()
    dfid = df.id

    tx_session.expire_all()
    got = (
        await tx_session.execute(
            select(DiscriminatorFactor).where(DiscriminatorFactor.id == dfid)
        )
    ).scalar_one()
    assert got.question_id == q_id
    assert got.node_id == node_id


async def test_discriminator_factor_dedupe_per_question(tx_session: AsyncSession):
    # V-M3: append-only dedupe by (question_id, factor_text).
    q = await _make_question(tx_session)
    tx_session.add(DiscriminatorFactor(question_id=q.id, factor_text="X"))
    await tx_session.flush()
    await tx_session.commit()

    with pytest.raises(IntegrityError):
        tx_session.add(DiscriminatorFactor(question_id=q.id, factor_text="X"))
        await tx_session.flush()


async def test_discriminator_factor_question_cascade(tx_session: AsyncSession):
    q = await _make_question(tx_session)
    df = DiscriminatorFactor(question_id=q.id, factor_text="Y")
    tx_session.add(df)
    await tx_session.flush()
    dfid = df.id

    await tx_session.delete(q)
    await tx_session.flush()

    remaining = (
        await tx_session.execute(
            select(DiscriminatorFactor).where(DiscriminatorFactor.id == dfid)
        )
    ).scalar_one_or_none()
    assert remaining is None


# --------------------------------------------------------------------------- #
# 8. atomic_fact_tags (T30) — canonical <target>_tags shape (V-T1/V-T2/V-T3)
# --------------------------------------------------------------------------- #


async def _make_fact(tx_session: AsyncSession, course: Course, pdf: PdfSource) -> AtomicFact:
    f = AtomicFact(
        course_id=course.id,
        pdf_source_id=pdf.id,
        text="x",
        content_hash=_hash(uuid.uuid4().hex),
    )
    tx_session.add(f)
    await tx_session.flush()
    return f


async def test_atomic_fact_tag_insert_round_trip(tx_session: AsyncSession):
    course = await _make_course(tx_session)
    pdf = await _make_pdf(tx_session, course)
    node = await _make_node(tx_session, course)
    fact = await _make_fact(tx_session, course, pdf)
    fid, nid = fact.id, node.id
    t = AtomicFactTag(
        atomic_fact_id=fid,
        node_id=nid,
        source="llm",
        confidence=0.87,
        rationale="grounded pick",
        extractor_version="grounded-v1",
    )
    tx_session.add(t)
    await tx_session.flush()
    tid = t.id

    tx_session.expire_all()
    got = (
        await tx_session.execute(select(AtomicFactTag).where(AtomicFactTag.id == tid))
    ).scalar_one()
    assert got.atomic_fact_id == fid
    assert got.node_id == nid
    assert got.source == "llm"
    assert float(got.confidence) == pytest.approx(0.87)
    assert got.manual_review is False


async def test_atomic_fact_tag_confidence_required_for_llm(tx_session: AsyncSession):
    """V-T3: source='llm' with NULL confidence violates the CHECK."""
    course = await _make_course(tx_session)
    pdf = await _make_pdf(tx_session, course)
    node = await _make_node(tx_session, course)
    fact = await _make_fact(tx_session, course, pdf)
    await tx_session.commit()

    with pytest.raises(IntegrityError):
        tx_session.add(
            AtomicFactTag(
                atomic_fact_id=fact.id, node_id=node.id, source="llm", confidence=None
            )
        )
        await tx_session.flush()


async def test_atomic_fact_tag_low_conf_requires_review(tx_session: AsyncSession):
    """V-T3: confidence <0.5 without manual_review violates the CHECK."""
    course = await _make_course(tx_session)
    pdf = await _make_pdf(tx_session, course)
    node = await _make_node(tx_session, course)
    fact = await _make_fact(tx_session, course, pdf)
    await tx_session.commit()

    with pytest.raises(IntegrityError):
        tx_session.add(
            AtomicFactTag(
                atomic_fact_id=fact.id,
                node_id=node.id,
                source="llm",
                confidence=0.30,
                manual_review=False,
            )
        )
        await tx_session.flush()


async def test_atomic_fact_tag_unique_node_source(tx_session: AsyncSession):
    course = await _make_course(tx_session)
    pdf = await _make_pdf(tx_session, course)
    node = await _make_node(tx_session, course)
    fact = await _make_fact(tx_session, course, pdf)
    tx_session.add(
        AtomicFactTag(
            atomic_fact_id=fact.id, node_id=node.id, source="llm", confidence=0.9
        )
    )
    await tx_session.flush()
    await tx_session.commit()

    with pytest.raises(IntegrityError):
        tx_session.add(
            AtomicFactTag(
                atomic_fact_id=fact.id, node_id=node.id, source="llm", confidence=0.8
            )
        )
        await tx_session.flush()


async def test_atomic_fact_tag_fact_cascade(tx_session: AsyncSession):
    course = await _make_course(tx_session)
    pdf = await _make_pdf(tx_session, course)
    node = await _make_node(tx_session, course)
    fact = await _make_fact(tx_session, course, pdf)
    t = AtomicFactTag(
        atomic_fact_id=fact.id, node_id=node.id, source="manual", confidence=None
    )
    tx_session.add(t)
    await tx_session.flush()
    tid = t.id

    await tx_session.delete(fact)
    await tx_session.flush()

    remaining = (
        await tx_session.execute(select(AtomicFactTag).where(AtomicFactTag.id == tid))
    ).scalar_one_or_none()
    assert remaining is None
