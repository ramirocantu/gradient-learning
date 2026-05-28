"""PDF ingest — vision transcription + grounded atomic-fact extraction (T54).

Notes-ingress redesign (2026-05-28): lecture notes / slidedecks arrive as
PDFs that frequently have **no extractable text** — handwriting, scanned
pages, image-only slides. So we no longer trust ``pdfplumber.extract_text``.
Instead every page is rendered to an image (PyMuPDF) and transcribed by an
OpenAI **vision** call (V-KB3); the transcription is then handed to an OpenAI
structured-output call that emits atomic factual claims (V-KB4). Facts persist
to ``atomic_facts`` with ``node_id`` NULL — the grounded-tag categorizer
(V-L3/V69, T50) assigns the node later.

Idempotent (V-KB1): re-ingesting a file with the same SHA-256 returns the
existing ``pdf_sources`` row and writes no new facts. ``UQ(course_id,
content_hash)`` on ``atomic_facts`` is the second line of defense.

Both the page renderer and the OpenAI clients are injected so tests never
render a real PDF or hit the API (V16): the renderer default uses PyMuPDF;
the vision + extraction clients are mocked at the SDK boundary.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.atomic_fact import AtomicFact
from app.models.pdf_source import PdfSource

_logger = logging.getLogger("app.services.kb.pdf_ingest")

# Bump when the vision prompt / extraction schema changes meaningfully.
# Stamped onto every persisted fact (V-KB3) so a re-run under a new version
# is a clean miss once content_hash dedup is keyed differently downstream.
EXTRACTOR_VERSION = "pdf-vision-v1"

_RENDER_DPI = 150
_VISION_MAX_TOKENS = 4096
_EXTRACT_MAX_TOKENS = 2048

_VISION_SYSTEM = (
    "You transcribe a single page from a student's lecture notes or slide deck. "
    "The page may be typed, a slide image, or handwritten. Output the full "
    "readable text content of the page, faithfully and verbatim where legible. "
    "Transcribe handwriting as best you can. Do NOT summarize, interpret, or add "
    "commentary — emit only the page's own text. If the page has no legible text, "
    "output nothing."
)

_EXTRACT_SYSTEM = (
    "You extract atomic factual claims from a page of study material. "
    "Each fact must be a single, self-contained, standalone statement that makes "
    "sense without the surrounding text. Split compound sentences into separate "
    "facts. Drop slide titles, page numbers, headers, and noise. Ground every "
    "fact strictly in the provided text — do NOT invent or infer beyond it."
)


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class RenderedPage:
    page: int
    image_png: bytes


@dataclass
class PageTranscription:
    text: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0


@dataclass
class FactExtraction:
    facts: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0


@dataclass
class IngestReport:
    pdf_source_id: int
    new_facts: int
    dup_facts: int
    pages: int
    reused_pdf: bool
    extractor_version: str = EXTRACTOR_VERSION
    # V-L1: token accounting summed across every vision + extraction call.
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0


# Injection seam types.
Renderer = Callable[[Path], list[RenderedPage]]


# --------------------------------------------------------------------------- #
# Page render (PyMuPDF) — injectable
# --------------------------------------------------------------------------- #


def render_pages(path: Path, *, dpi: int = _RENDER_DPI) -> list[RenderedPage]:
    """Default renderer: rasterize each PDF page to a PNG via PyMuPDF.

    Heavy import deferred to call time so the module imports cheaply and tests
    that inject a forged renderer never load PyMuPDF.
    """

    import pymupdf  # noqa: PLC0415 — heavy import only when rendering for real

    pages: list[RenderedPage] = []
    with pymupdf.open(str(path)) as doc:
        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(dpi=dpi)
            pages.append(RenderedPage(page=i, image_png=pix.tobytes("png")))
    return pages


# --------------------------------------------------------------------------- #
# Usage accounting (V-L1)
# --------------------------------------------------------------------------- #


def _read_usage(completion: Any) -> tuple[int, int, int]:
    """Return ``(prompt_tokens, output_tokens, cached_tokens)`` from a
    ChatCompletion. Cache hits come from ``prompt_tokens_details.cached_tokens``
    — never inferred (V-L1)."""

    usage = getattr(completion, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    cached_tokens = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached_tokens = int(getattr(details, "cached_tokens", 0) or 0)
    return prompt_tokens, output_tokens, cached_tokens


def _message_content(completion: Any) -> str | None:
    choices = getattr(completion, "choices", None) or []
    choice = choices[0] if choices else None
    message = getattr(choice, "message", None) if choice is not None else None
    content = getattr(message, "content", None) if message is not None else None
    return content


# --------------------------------------------------------------------------- #
# Vision transcription (V-KB3)
# --------------------------------------------------------------------------- #


async def transcribe_page(
    image_png: bytes,
    *,
    client: Any,
    model: str,
    max_tokens: int = _VISION_MAX_TOKENS,
) -> PageTranscription:
    """One OpenAI vision call: page image → transcribed text (V-KB3).

    ``client`` is an ``AsyncOpenAI``-shaped object, injected + mocked at the
    SDK boundary in tests (V16)."""

    b64 = base64.b64encode(image_png).decode("ascii")
    completion = await client.chat.completions.create(
        model=model,
        max_completion_tokens=max_tokens,
        messages=[
            {"role": "system", "content": _VISION_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe this page."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            },
        ],
    )
    prompt_tokens, output_tokens, cached_tokens = _read_usage(completion)
    text = (_message_content(completion) or "").strip()
    return PageTranscription(
        text=text,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
    )


# --------------------------------------------------------------------------- #
# Atomic-fact extraction (V-KB4, V45 structured output)
# --------------------------------------------------------------------------- #


_EXTRACT_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "extract_atomic_facts",
        "description": "Atomic factual claims grounded in the page text.",
        "strict": True,
        "schema": {
            "type": "object",
            "required": ["facts"],
            "additionalProperties": False,
            "properties": {
                "facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["text"],
                        "additionalProperties": False,
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "One self-contained atomic claim.",
                            },
                        },
                    },
                },
            },
        },
    },
}


def _parse_facts(payload: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for raw in payload.get("facts") or []:
        if not isinstance(raw, dict):
            continue
        text = raw.get("text")
        if not isinstance(text, str):
            continue
        text = text.strip()
        if text:
            out.append(text)
    return out


async def extract_atomic_facts(
    text: str,
    *,
    client: Any,
    model: str,
    max_tokens: int = _EXTRACT_MAX_TOKENS,
) -> FactExtraction:
    """One OpenAI structured-output call: page text → atomic facts (V-KB4).

    Empty/blank input → no LLM call, empty result. Strict json_schema emits the
    document in ``choice.message.content`` (mirrors ``llm/grounded.py``)."""

    if not text.strip():
        return FactExtraction()

    completion = await client.chat.completions.create(
        model=model,
        max_completion_tokens=max_tokens,
        messages=[
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user", "content": text.strip()},
        ],
        response_format=_EXTRACT_SCHEMA,
    )
    prompt_tokens, output_tokens, cached_tokens = _read_usage(completion)

    content = _message_content(completion)
    facts: list[str] = []
    if content:
        try:
            facts = _parse_facts(json.loads(content))
        except json.JSONDecodeError as exc:
            _logger.warning("extract: response content not valid JSON: %s", exc)
    return FactExtraction(
        facts=facts,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


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
    vision_client: Any,
    extract_client: Any | None = None,
    renderer: Renderer = render_pages,
    vision_model: str | None = None,
    extract_model: str | None = None,
    extractor_version: str = EXTRACTOR_VERSION,
) -> IngestReport:
    """Render → vision-transcribe → extract facts → persist (V-KB3, V-KB4).

    Args:
        course_id: owning course (atomic_facts dedup scope).
        path: the PDF on disk.
        vision_client: injected ``AsyncOpenAI`` for page transcription (V16).
        extract_client: client for fact extraction; defaults to ``vision_client``.
        renderer: page→image renderer; defaults to PyMuPDF, injectable for tests.
        vision_model / extract_model: default to ``OPENAI_VISION_MODEL`` (falling
            back to ``OPENAI_MODEL``) / ``OPENAI_MODEL``.

    Runs inside the caller's transaction — the caller owns commit/rollback.
    """

    extract_client = extract_client or vision_client
    resolved_vision_model = vision_model or settings.OPENAI_VISION_MODEL or settings.OPENAI_MODEL
    resolved_extract_model = extract_model or settings.OPENAI_MODEL

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
            extractor_version=extractor_version,
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

    pages = renderer(path)
    new_facts = 0
    dup_facts = 0
    in_tokens = 0
    out_tokens = 0
    cached = 0
    seen_hashes: set[str] = set()

    for rendered in pages:
        transcription = await transcribe_page(
            rendered.image_png, client=vision_client, model=resolved_vision_model
        )
        in_tokens += transcription.prompt_tokens
        out_tokens += transcription.output_tokens
        cached += transcription.cached_tokens

        extraction = await extract_atomic_facts(
            transcription.text, client=extract_client, model=resolved_extract_model
        )
        in_tokens += extraction.prompt_tokens
        out_tokens += extraction.output_tokens
        cached += extraction.cached_tokens

        for fact_text in extraction.facts:
            content_hash = _sha256_text(fact_text)
            if content_hash in seen_hashes:
                dup_facts += 1
                continue
            existing_fact = (
                await session.execute(
                    select(AtomicFact).where(
                        AtomicFact.course_id == course_id,
                        AtomicFact.content_hash == content_hash,
                    )
                )
            ).scalar_one_or_none()
            if existing_fact is not None:
                seen_hashes.add(content_hash)
                dup_facts += 1
                continue
            session.add(
                AtomicFact(
                    course_id=course_id,
                    pdf_source_id=pdf_id,
                    page=rendered.page,
                    text=fact_text,
                    content_hash=content_hash,
                    extractor_version=extractor_version,
                )
            )
            seen_hashes.add(content_hash)
            new_facts += 1

    pdf_row.status = "ingested"
    pdf_row.ingested_at = datetime.now(timezone.utc)
    await session.flush()

    _logger.info(
        "pdf_ingest: pdf=%d pages=%d new_facts=%d dup_facts=%d "
        "vision_model=%s extract_model=%s prompt=%d cached=%d out=%d version=%s",
        pdf_id,
        len(pages),
        new_facts,
        dup_facts,
        resolved_vision_model,
        resolved_extract_model,
        in_tokens,
        cached,
        out_tokens,
        extractor_version,
    )

    return IngestReport(
        pdf_source_id=pdf_id,
        new_facts=new_facts,
        dup_facts=dup_facts,
        pages=len(pages),
        reused_pdf=False,
        extractor_version=extractor_version,
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        cached_tokens=cached,
    )
