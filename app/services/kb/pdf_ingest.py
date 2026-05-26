"""PDF ingest + atomic-fact extraction (T26, V-KB1).

Parses a classroom PDF with pdfplumber, splits each page's text into
sentence-grained atomic candidates, and persists
``pdf_sources`` + ``atomic_facts`` rows. Idempotent: re-ingesting a
file with the same SHA-256 returns the existing ``pdf_sources`` row
and writes no new atomic facts (V-KB1; the
``UQ(course_id, content_hash)`` on ``atomic_facts`` is the
belt-and-suspenders guard if the sentence splitter changes).

The parser is injected so tests can drop in a forged page list without
materializing a real PDF.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.atomic_fact import AtomicFact
from app.models.pdf_source import PdfSource

_logger = logging.getLogger("app.services.kb.pdf_ingest")


@dataclass
class ParsedPage:
    page: int
    text: str


@dataclass
class IngestReport:
    pdf_source_id: int
    new_facts: int
    dup_facts: int
    pages: int
    reused_pdf: bool


# Sentence split: keep it conservative. We want atomic claims with
# enough content to embed meaningfully — bullets and titles below
# the threshold are dropped (the LLM4Tag prompt in T29 chunks larger).
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
_MIN_SENTENCE_LEN = 20


def parse_pdf(path: Path) -> list[ParsedPage]:
    """Default pdfplumber-backed parser. Tests inject their own."""

    import pdfplumber  # heavy import — only when called for real

    pages: list[ParsedPage] = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append(ParsedPage(page=i, text=text))
    return pages


def split_atomic_candidates(text: str, *, min_len: int = _MIN_SENTENCE_LEN) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for sent in _SENTENCE_RE.split(line):
            sent = sent.strip()
            if len(sent) >= min_len:
                out.append(sent)
    return out


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def ingest_pdf(
    session: AsyncSession,
    *,
    course_id: int,
    path: Path,
    parser: Callable[[Path], list[ParsedPage]] = parse_pdf,
) -> IngestReport:
    sha = _sha256_file(path)

    existing = (
        await session.execute(select(PdfSource).where(PdfSource.sha256 == sha))
    ).scalar_one_or_none()
    if existing is not None:
        return IngestReport(
            pdf_source_id=existing.id,
            new_facts=0,
            dup_facts=0,
            pages=0,
            reused_pdf=True,
        )

    pdf_row = PdfSource(
        course_id=course_id,
        filename=path.name,
        sha256=sha,
        status="parsing",
    )
    session.add(pdf_row)
    await session.flush()
    pdf_id = pdf_row.id

    pages = parser(path)
    new_facts = 0
    dup_facts = 0
    for parsed in pages:
        for sentence in split_atomic_candidates(parsed.text):
            content_hash = _sha256_text(sentence)
            existing_fact = (
                await session.execute(
                    select(AtomicFact).where(
                        AtomicFact.course_id == course_id,
                        AtomicFact.content_hash == content_hash,
                    )
                )
            ).scalar_one_or_none()
            if existing_fact is not None:
                dup_facts += 1
                continue
            session.add(
                AtomicFact(
                    course_id=course_id,
                    pdf_source_id=pdf_id,
                    page=parsed.page,
                    text=sentence,
                    content_hash=content_hash,
                )
            )
            new_facts += 1

    pdf_row.status = "ingested"
    pdf_row.ingested_at = datetime.now(timezone.utc)
    await session.flush()

    return IngestReport(
        pdf_source_id=pdf_id,
        new_facts=new_facts,
        dup_facts=dup_facts,
        pages=len(pages),
        reused_pdf=False,
    )
