"""T26 — app/services/kb/pdf_ingest.py contract tests (V-KB1).

V-KB1: substrate idempotent re-run. Re-ingesting a file with the same
SHA-256 returns the existing ``pdf_sources`` row and writes no new
``atomic_facts``. ``UQ(course_id, content_hash)`` on ``atomic_facts``
is the second line of defense if the sentence splitter changes.

The parser is injected so we never touch pdfplumber here — the seam's
``parse_pdf`` default is exercised by the smoke test below using a
tiny real PDF generated in a tmpdir.
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.atomic_fact import AtomicFact
from app.models.outline import Course
from app.models.pdf_source import PdfSource
from app.services.kb.pdf_ingest import (
    IngestReport,
    ParsedPage,
    ingest_pdf,
    split_atomic_candidates,
)


async def _make_course(session: AsyncSession) -> Course:
    c = Course(slug=f"pdf-{uuid.uuid4().hex[:8]}", name="PDF Course")
    session.add(c)
    await session.flush()
    return c


def _fake_pdf(path: Path, content: bytes) -> None:
    path.write_bytes(content)


def _forge_parser(pages: list[ParsedPage]):
    def _parser(_: Path) -> list[ParsedPage]:
        return pages

    return _parser


# --------------------------------------------------------------------------- #
# split_atomic_candidates — pure
# --------------------------------------------------------------------------- #


def test_split_drops_short_sentences():
    out = split_atomic_candidates("Too short. This sentence is long enough to keep.")
    assert "This sentence is long enough to keep." in out
    assert all(len(s) >= 20 for s in out)


def test_split_handles_multiline():
    text = (
        "Glycolysis converts glucose to pyruvate.\n"
        "It produces ATP through substrate-level phosphorylation.\n"
        "skip"
    )
    out = split_atomic_candidates(text)
    assert len(out) == 2


# --------------------------------------------------------------------------- #
# ingest_pdf — DB-backed
# --------------------------------------------------------------------------- #


async def test_first_ingest_creates_pdf_source_and_facts(
    db_session: AsyncSession, tmp_path: Path
):
    course = await _make_course(db_session)
    course_id = course.id
    pdf = tmp_path / "lecture-01.pdf"
    _fake_pdf(pdf, b"%PDF-1.4 fake")

    parser = _forge_parser(
        [
            ParsedPage(
                page=1,
                text=(
                    "Glycolysis converts glucose to pyruvate.\n"
                    "It yields net two ATP per glucose molecule."
                ),
            ),
            ParsedPage(
                page=2,
                text="The TCA cycle regenerates oxaloacetate after each turn.",
            ),
        ]
    )

    report = await ingest_pdf(db_session, course_id=course_id, path=pdf, parser=parser)

    assert isinstance(report, IngestReport)
    assert report.reused_pdf is False
    assert report.pages == 2
    assert report.new_facts == 3
    assert report.dup_facts == 0

    pdf_row = (
        await db_session.execute(
            select(PdfSource).where(PdfSource.id == report.pdf_source_id)
        )
    ).scalar_one()
    assert pdf_row.status == "ingested"
    assert pdf_row.ingested_at is not None
    assert pdf_row.filename == "lecture-01.pdf"

    facts = (
        await db_session.execute(
            select(AtomicFact).where(AtomicFact.pdf_source_id == report.pdf_source_id)
        )
    ).scalars().all()
    assert len(facts) == 3
    pages = {f.page for f in facts}
    assert pages == {1, 2}


async def test_re_ingest_same_file_is_noop(db_session: AsyncSession, tmp_path: Path):
    course = await _make_course(db_session)
    course_id = course.id
    pdf = tmp_path / "lecture-dup.pdf"
    _fake_pdf(pdf, b"%PDF-1.4 stable-bytes")
    parser = _forge_parser(
        [ParsedPage(page=1, text="Mitochondrial matrix hosts the TCA cycle.")]
    )

    r1 = await ingest_pdf(db_session, course_id=course_id, path=pdf, parser=parser)
    r2 = await ingest_pdf(db_session, course_id=course_id, path=pdf, parser=parser)

    assert r1.reused_pdf is False and r1.new_facts == 1
    assert r2.reused_pdf is True
    assert r2.new_facts == 0
    assert r2.pdf_source_id == r1.pdf_source_id

    pdf_rows = (
        await db_session.execute(
            select(PdfSource).where(PdfSource.course_id == course_id)
        )
    ).scalars().all()
    assert len(pdf_rows) == 1


async def test_content_hash_dedupes_facts_within_course(
    db_session: AsyncSession, tmp_path: Path
):
    """V-KB1: even if the sentence appears in a different PDF in the same
    course, the (course_id, content_hash) UQ keeps a single row."""

    course = await _make_course(db_session)
    course_id = course.id

    pdf_a = tmp_path / "a.pdf"
    pdf_b = tmp_path / "b.pdf"
    _fake_pdf(pdf_a, b"%PDF-1.4 a")
    _fake_pdf(pdf_b, b"%PDF-1.4 b")

    shared = "Insulin is secreted by pancreatic beta cells."
    parser_a = _forge_parser([ParsedPage(page=1, text=shared)])
    parser_b = _forge_parser([ParsedPage(page=1, text=shared)])

    r1 = await ingest_pdf(db_session, course_id=course_id, path=pdf_a, parser=parser_a)
    r2 = await ingest_pdf(db_session, course_id=course_id, path=pdf_b, parser=parser_b)

    assert r1.new_facts == 1
    assert r2.new_facts == 0
    assert r2.dup_facts == 1

    chash = hashlib.sha256(shared.encode()).hexdigest()
    rows = (
        await db_session.execute(
            select(AtomicFact).where(
                AtomicFact.course_id == course_id, AtomicFact.content_hash == chash
            )
        )
    ).scalars().all()
    assert len(rows) == 1


async def test_real_pdfplumber_smoke(db_session: AsyncSession, tmp_path: Path):
    """One end-to-end pass exercising the default ``parse_pdf`` parser.

    Generates a tiny PDF with pypdfium2/pdfplumber's dependency chain so
    we don't ship a binary fixture into the repo. If pdfplumber yields
    nothing for this minimal file (some readers do, depending on text
    extraction support), the test asserts the seam still completes
    without raising — V-KB1 is about idempotency, not parse quality.
    """

    pdfplumber = pytest.importorskip("pdfplumber")
    reportlab = pytest.importorskip(  # pragma: no cover — optional dev dep
        "reportlab",
        reason="reportlab not installed; pdf write-side smoke skipped",
    )
    from reportlab.pdfgen import canvas  # type: ignore

    course = await _make_course(db_session)
    course_id = course.id
    pdf = tmp_path / "smoke.pdf"
    c = canvas.Canvas(str(pdf))
    c.drawString(72, 720, "Mitochondria are the powerhouse of the cell.")
    c.drawString(72, 700, "ATP synthase couples proton flow to phosphorylation.")
    c.save()

    report = await ingest_pdf(db_session, course_id=course_id, path=pdf)
    assert report.reused_pdf is False
    # Parse quality varies; we only check the seam wrote a PdfSource row.
    pdf_row = (
        await db_session.execute(
            select(PdfSource).where(PdfSource.id == report.pdf_source_id)
        )
    ).scalar_one()
    assert pdf_row.status == "ingested"
