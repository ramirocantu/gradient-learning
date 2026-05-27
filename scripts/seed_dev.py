"""Seed synthetic dev/test data straight into the DB — no OpenAI / Anki / Notion.

Populates `questions` + `attempts` + `question_tags(node_id)` + `attempt_notes`
so the client-facing reads (tutor captures/sessions/flagged, question-by-qid,
per-node mastery once exposed) have real rows, bypassing every external
pipeline and the legacy manual-tag route.

Also seeds the KB substrate — `pdf_sources` + `atomic_facts` (+ `atomic_fact_tags`)
+ `concept_edges` + `notion_pages` — so the T45–T48 read routes (desktop ¶T8–¶T11)
return real rows without the (unwired) ingest / similarity / Notion pipelines.

Requires the course's outline to already exist (seed it first, README §5):

    curl -XPOST localhost:8000/api/v1/courses -d '{"slug":"aamc","name":"..."}'
    curl -XPOST localhost:8000/api/v1/courses/1/outline:import \
        --data-binary @app/seeds/aamc_outline.schema.json

Then:

    python -m scripts.seed_dev                 # seed (idempotent)
    python -m scripts.seed_dev --course aamc   # pick course by slug
    python -m scripts.seed_dev --wipe          # remove dev rows (qid LIKE 'dev-%')

Idempotent: every synthetic qid is namespaced `dev-…`; re-running skips
existing questions. `--wipe` deletes only the `dev-` namespace.
"""

from __future__ import annotations

import argparse
import asyncio
import random
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AsyncSessionLocal
from app.models.atomic_fact import AtomicFact
from app.models.atomic_fact_tag import AtomicFactTag
from app.models.attempt_note import AttemptNote
from app.models.captures import Attempt, Question, QuestionTag
from app.models.concept_edge import ConceptEdge
from app.models.notion_page import NotionPage
from app.models.outline import Course, OutlineNode
from app.models.pdf_source import PdfSource

QID_PREFIX = "dev-"

# Synthetic biochem questions. `node_hint` is matched (ILIKE) against outline
# node names; unmatched hints fall back to arbitrary leaf nodes so the seed
# still tags something. `dist` = wrong-pick bias used to shape attempts.
QUESTIONS = [
    {
        "qid": "dev-12420", "node_hint": "beta-oxidation",
        "stem": "Net ATP equivalents per palmitate (C16) via β-oxidation, malate-aspartate shuttle?",
        "choices": ["96", "106", "120", "129"], "correct": "C", "common_wrong": "B",
        "explanation": "7 cycles β-ox → 7 FADH₂ + 7 NADH + 8 acetyl-CoA; malate-aspartate keeps cytosolic NADH at 2.5 ATP; minus 2 ATP activation → 120.",
    },
    {
        "qid": "dev-12418", "node_hint": "lipid",
        "stem": "Saponification of a triacylglycerol proceeds via which mechanism?",
        "choices": ["Acid-catalyzed", "Base-catalyzed", "Radical", "Enzymatic only"],
        "correct": "B", "common_wrong": "A",
        "explanation": "Saponification is base-catalyzed ester hydrolysis → glycerol + fatty-acid salts (soap).",
    },
    {
        "qid": "dev-12416", "node_hint": "bioenerget",
        "stem": "Which citric-acid-cycle step is irreversible and rate-limiting?",
        "choices": ["Citrate synthase", "Aconitase", "Isocitrate dehydrogenase", "Fumarase"],
        "correct": "C", "common_wrong": "A",
        "explanation": "Isocitrate dehydrogenase is a committed, allosterically-regulated irreversible step.",
    },
    {
        "qid": "dev-12414", "node_hint": "carbohydrate",
        "stem": "Glycogen phosphorylase is activated by which signal in muscle?",
        "choices": ["Insulin", "Epinephrine", "Glucagon", "Cortisol"],
        "correct": "B", "common_wrong": "C",
        "explanation": "Muscle lacks glucagon receptors; epinephrine → cAMP → PKA → phosphorylase activation.",
    },
    {
        "qid": "dev-12412", "node_hint": "enzyme",
        "stem": "A competitive inhibitor affects Km and Vmax how?",
        "choices": ["↑Km, Vmax same", "Km same, ↓Vmax", "↑Km, ↑Vmax", "↓Km, Vmax same"],
        "correct": "A", "common_wrong": "B",
        "explanation": "Competitive inhibition raises apparent Km; Vmax unchanged (outcompeted at high [S]).",
    },
    {
        "qid": "dev-12410", "node_hint": "amino acid",
        "stem": "Which amino acid is both glucogenic and ketogenic?",
        "choices": ["Leucine", "Lysine", "Phenylalanine", "Alanine"],
        "correct": "C", "common_wrong": "A",
        "explanation": "Phe (and Tyr, Ile, Trp) are both; Leu/Lys are purely ketogenic.",
    },
    {
        "qid": "dev-12408", "node_hint": "beta-oxidation",
        "stem": "Activation of a fatty acid to acyl-CoA costs how many ATP equivalents?",
        "choices": ["1", "2", "3", "0"], "correct": "B", "common_wrong": "A",
        "explanation": "ATP → AMP + PPᵢ ≈ 2 high-energy phosphate bonds.",
    },
    {
        "qid": "dev-12406", "node_hint": "lipid",
        "stem": "The carnitine shuttle transports what across the inner mitochondrial membrane?",
        "choices": ["Acetyl-CoA", "Long-chain acyl groups", "NADH", "Pyruvate"],
        "correct": "B", "common_wrong": "A",
        "explanation": "CPT-I/CPT-II shuttle long-chain acyl groups as acylcarnitine into the matrix.",
    },
]

# (test_id, days_ago) — each session is one day; its attempts cluster in a
# tight window that day so session wall-clock stays realistic.
SESSIONS = [("Bio-Sys 14", 0), ("Mixed 042", 1), ("Bio-Sys 13", 2)]


# ── KB substrate (desktop ¶T8–¶T11 / backend T45–T48) ───────────────────────
# Seeds concept_edges / atomic_facts(+tags) / pdf_sources / notion_pages so the
# KB read routes return real rows in dev — the population pipelines (similarity
# derivation, PDF ingest, Notion write-out) are unwired here. All rows are
# dev-namespaced (`dev-` sha256 / content_hash / notion_page_id) and removed by
# --wipe. `*_idx` fields index into the resolved KB node list (deepest nodes).
#
# Seeds only columns the §I model layer actually carries. The §T field wishlist
# diverges from §I in places, so some desktop fields have no seed source:
#   • extractor_version → seeded on atomic_fact_tags (NOT the fact row); the
#     /atomic-facts route does not surface it yet (needs a tag read-join).
#   • notion block_count / status → not modeled on notion_pages; status is
#     derivable client-side from last_synced_at (we vary it: synced vs NULL).
#   • pdf pages / node_id → not modeled on pdf_sources (course-scoped by design);
#     no seed source — client degrades these to "—".
KB_PDFS = [
    {"sha": "dev-sha-lehninger-ch17", "filename": "Lehninger-ch17-fatty-acid-catabolism.pdf", "status": "ingested", "ingested": True},
    {"sha": "dev-sha-stryer-ch16", "filename": "Stryer-ch16-glycolysis.pdf", "status": "ingested", "ingested": True},
    {"sha": "dev-sha-scan-2026", "filename": "scanned-lecture-notes-2026.pdf", "status": "parsing", "ingested": False},
]

KB_FACTS = [
    {"text": "β-oxidation of one palmitate (C16) yields 8 acetyl-CoA, 7 NADH, and 7 FADH₂.", "page": 612, "pdf_idx": 0, "node_idx": 0},
    {"text": "The carnitine shuttle (CPT-I/CPT-II) is rate-limiting for long-chain acyl entry into the matrix.", "page": 615, "pdf_idx": 0, "node_idx": 0},
    {"text": "CPT-I is inhibited by malonyl-CoA, reciprocally coupling fatty-acid synthesis and oxidation.", "page": 617, "pdf_idx": 0, "node_idx": 1},
    {"text": "Saponification is base-catalyzed hydrolysis of the ester bonds in a triacylglycerol.", "page": 88, "pdf_idx": 0, "node_idx": 1},
    {"text": "Hexokinase is feedback-inhibited by its product glucose-6-phosphate.", "page": 433, "pdf_idx": 1, "node_idx": 2},
    {"text": "PFK-1 is the committed step of glycolysis, activated by AMP and fructose-2,6-bisphosphate.", "page": 437, "pdf_idx": 1, "node_idx": 2},
    {"text": "Pyruvate kinase catalyzes a substrate-level phosphorylation that yields ATP.", "page": 441, "pdf_idx": 1, "node_idx": 3},
]

KB_EDGES = [
    {"src_idx": 0, "dst_idx": 2, "kind": "similarity", "score": 0.82},
    {"src_idx": 1, "dst_idx": 0, "kind": "similarity", "score": 0.74},
    {"src_idx": 2, "dst_idx": 3, "kind": "similarity", "score": 0.88},
    {"src_idx": 0, "dst_idx": 3, "kind": "manual", "score": None},
]

KB_NOTION = [
    {"node_idx": 0, "synced_hours_ago": 3},
    {"node_idx": 1, "synced_hours_ago": 27},
    {"node_idx": 2, "synced_hours_ago": 52},
    {"node_idx": 3, "synced_hours_ago": None},  # never synced → client shows pending
]

KB_EXTRACTOR_VERSION = "dev-extract-v1"


def _choices_json(letters_to_text: list[str]) -> list[dict]:
    keys = ["A", "B", "C", "D", "E"]
    return [
        {"key": keys[i], "html": f"<p>{t}</p>", "plain": t, "media_content_hashes": []}
        for i, t in enumerate(letters_to_text)
    ]


async def _resolve_node_pool(session: AsyncSession, course_id: int) -> dict[str, int]:
    """Map each distinct node_hint → an outline_node.id (ILIKE match), with a
    leaf-node fallback so unmatched hints still tag something real."""
    pool: dict[str, int] = {}
    hints = {q["node_hint"] for q in QUESTIONS}
    for hint in hints:
        row = (
            await session.execute(
                select(OutlineNode.id)
                .where(OutlineNode.course_id == course_id)
                .where(OutlineNode.name.ilike(f"%{hint}%"))
                .order_by(OutlineNode.depth.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if row is not None:
            pool[hint] = row
    if len(pool) < len(hints):
        # fallback: deepest nodes fill the gaps deterministically
        leaves = (
            await session.execute(
                select(OutlineNode.id)
                .where(OutlineNode.course_id == course_id)
                .order_by(OutlineNode.depth.desc(), OutlineNode.id)
                .limit(len(hints))
            )
        ).scalars().all()
        for hint in hints:
            if hint not in pool and leaves:
                pool[hint] = leaves[len(pool) % len(leaves)]
    return pool


async def _kb_nodes(session: AsyncSession, course_id: int, n: int) -> list[int]:
    """Up to `n` distinct deepest outline nodes for the course — the targets KB
    rows tag/link. Deterministic order (depth desc, id)."""
    return (
        await session.execute(
            select(OutlineNode.id)
            .where(OutlineNode.course_id == course_id)
            .order_by(OutlineNode.depth.desc(), OutlineNode.id)
            .limit(n)
        )
    ).scalars().all()


async def seed_kb(session: AsyncSession, course: Course) -> tuple[int, int, int, int]:
    """Seed the KB substrate (pdf_sources, atomic_facts(+tags), concept_edges,
    notion_pages) so the T45–T48 read routes return real rows. Idempotent:
    pdfs key on sha256, facts on content_hash, edges on (src,dst,kind), pages
    on node_id — each skips if present. Returns (pdfs, facts, edges, pages)."""
    nodes = await _kb_nodes(session, course.id, 8)
    if len(nodes) < 2:
        return (0, 0, 0, 0)  # outline too small to link/tag meaningfully
    name_rows = (
        await session.execute(
            select(OutlineNode.id, OutlineNode.name).where(OutlineNode.id.in_(nodes))
        )
    ).all()
    node_names = {nid: nm for nid, nm in name_rows}
    now = datetime.now(timezone.utc)
    made_p = made_f = made_e = made_n = 0

    # pdf_sources — keep an idx→id map for fact attachment
    pdf_id_by_idx: dict[int, int] = {}
    for i, pd in enumerate(KB_PDFS):
        existing = (
            await session.execute(select(PdfSource.id).where(PdfSource.sha256 == pd["sha"]))
        ).scalar_one_or_none()
        if existing is not None:
            pdf_id_by_idx[i] = existing
            continue
        pdf = PdfSource(
            course_id=course.id,
            filename=pd["filename"],
            sha256=pd["sha"],
            status=pd["status"],
            ingested_at=(now - timedelta(days=i + 1)) if pd["ingested"] else None,
        )
        session.add(pdf)
        await session.flush()
        pdf_id_by_idx[i] = pdf.id
        made_p += 1

    # atomic_facts (+ one grounded llm tag carrying extractor_version per §I)
    for i, fd in enumerate(KB_FACTS):
        if fd["node_idx"] >= len(nodes):
            continue
        content_hash = f"dev-fact-{i}"
        existing = (
            await session.execute(
                select(AtomicFact.id).where(
                    AtomicFact.course_id == course.id,
                    AtomicFact.content_hash == content_hash,
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            continue
        node_id = nodes[fd["node_idx"]]
        fact = AtomicFact(
            course_id=course.id,
            pdf_source_id=pdf_id_by_idx[fd["pdf_idx"]],
            page=fd["page"],
            text=fd["text"],
            node_id=node_id,
            content_hash=content_hash,
        )
        session.add(fact)
        await session.flush()
        session.add(
            AtomicFactTag(
                atomic_fact_id=fact.id,
                node_id=node_id,
                source="llm",
                confidence=0.91,
                extractor_version=KB_EXTRACTOR_VERSION,
                manual_review=False,
            )
        )
        made_f += 1

    # concept_edges (similarity + one manual), deduped on (src,dst,kind)
    for ed in KB_EDGES:
        if ed["src_idx"] >= len(nodes) or ed["dst_idx"] >= len(nodes):
            continue
        src, dst = nodes[ed["src_idx"]], nodes[ed["dst_idx"]]
        if src == dst:
            continue
        existing = (
            await session.execute(
                select(ConceptEdge.id).where(
                    ConceptEdge.src_node_id == src,
                    ConceptEdge.dst_node_id == dst,
                    ConceptEdge.kind == ed["kind"],
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            continue
        session.add(
            ConceptEdge(src_node_id=src, dst_node_id=dst, kind=ed["kind"], score=ed["score"])
        )
        made_e += 1

    # notion_pages (one per node, UQ node_id), some synced + one pending (NULL)
    for nd in KB_NOTION:
        if nd["node_idx"] >= len(nodes):
            continue
        node_id = nodes[nd["node_idx"]]
        existing = (
            await session.execute(select(NotionPage.id).where(NotionPage.node_id == node_id))
        ).scalar_one_or_none()
        if existing is not None:
            continue
        hrs = nd["synced_hours_ago"]
        name = node_names.get(node_id)
        session.add(
            NotionPage(
                node_id=node_id,
                notion_page_id=f"dev-page-{node_id}",
                url=f"https://www.notion.so/dev-{node_id}",
                tags=[name] if name else [],
                last_synced_at=(now - timedelta(hours=hrs)) if hrs is not None else None,
            )
        )
        made_n += 1

    return (made_p, made_f, made_e, made_n)


async def wipe(session: AsyncSession) -> int:
    """Delete the dev namespace. Tags/attempts cascade off Question via FK, but
    attempt_notes have no cascade from Question — delete them explicitly first.
    Also clears dev-namespaced KB substrate (pdf_sources→facts→tags cascade,
    notion_pages) and all concept_edges (dev seed is their only writer here)."""
    q_ids = (
        await session.execute(select(Question.id).where(Question.qid.like(f"{QID_PREFIX}%")))
    ).scalars().all()
    if q_ids:
        a_ids = (
            await session.execute(select(Attempt.id).where(Attempt.question_id.in_(q_ids)))
        ).scalars().all()
        if a_ids:
            await session.execute(delete(AttemptNote).where(AttemptNote.attempt_id.in_(a_ids)))
        await session.execute(delete(Attempt).where(Attempt.question_id.in_(q_ids)))
        await session.execute(delete(QuestionTag).where(QuestionTag.question_id.in_(q_ids)))
        await session.execute(delete(Question).where(Question.id.in_(q_ids)))

    # KB substrate (dev-namespaced). Deleting pdf_sources cascades atomic_facts →
    # atomic_fact_tags via FK ondelete=CASCADE. concept_edges carry no namespace
    # column; the dev seed is their only writer here (similarity derivation is
    # unwired in dev), so clear them wholesale.
    await session.execute(delete(PdfSource).where(PdfSource.sha256.like(f"{QID_PREFIX}%")))
    await session.execute(delete(NotionPage).where(NotionPage.notion_page_id.like(f"{QID_PREFIX}%")))
    await session.execute(delete(ConceptEdge))
    return len(q_ids)


async def seed(session: AsyncSession, course_slug: str) -> None:
    rng = random.Random(42)  # deterministic
    course = (
        await session.execute(select(Course).where(Course.slug == course_slug))
    ).scalar_one_or_none()
    if course is None:
        raise SystemExit(
            f"course slug={course_slug!r} not found — create it + import the outline first "
            "(README §5)."
        )
    node_count = (
        await session.execute(
            select(func.count(OutlineNode.id)).where(OutlineNode.course_id == course.id)
        )
    ).scalar_one()
    if not node_count:
        raise SystemExit(
            f"course {course_slug!r} has no outline nodes — import the schema first "
            "(POST /courses/{id}/outline:import)."
        )

    pool = await _resolve_node_pool(session, course.id)
    now = datetime.now(timezone.utc)
    made_q = made_a = made_t = made_n = 0

    for qi, qd in enumerate(QUESTIONS):
        exists = (
            await session.execute(select(Question.id).where(Question.qid == qd["qid"]))
        ).scalar_one_or_none()
        if exists is not None:
            continue  # idempotent

        q = Question(
            source="uworld",
            qid=qd["qid"],
            stem_html=f"<p>{qd['stem']}</p>",
            stem_plain=qd["stem"],
            choices=_choices_json(qd["choices"]),
            correct_choice=qd["correct"],
            explanation_html=f"<p>{qd['explanation']}</p>",
            explanation_plain=qd["explanation"],
            uworld_aamc_tags=[qd["node_hint"]],
            needs_categorization=False,
        )
        session.add(q)
        await session.flush()  # need q.id
        made_q += 1

        # node tag(s): schema_map (confidence NULL) + one llm tag with confidence
        node_id = pool[qd["node_hint"]]
        session.add(QuestionTag(question_id=q.id, node_id=node_id, source="schema_map", confidence=None))
        made_t += 1
        # a low-confidence llm tag on every 3rd question → exercises needs-review
        if qi % 3 == 0:
            session.add(
                QuestionTag(
                    question_id=q.id, node_id=node_id, source="llm",
                    confidence=0.42, manual_review=True, extractor_version="dev",
                )
            )
            made_t += 1

        # 1–3 attempts, each in a different session/day (→ attempt history spans
        # dates). Within a session, attempts cluster in a ~50-min window that day
        # so the session's first→last wall-clock is realistic.
        n_attempts = rng.choice([1, 2, 3])
        for ai in range(n_attempts):
            test_id, days_ago = SESSIONS[(qi + ai) % len(SESSIONS)]
            day = (now - timedelta(days=days_ago)).replace(hour=9, minute=40, second=0, microsecond=0)
            attempted = day + timedelta(minutes=rng.randint(0, 50))
            # later attempts trend correct (learning); first attempt often the common-wrong
            correct = ai == n_attempts - 1 and rng.random() > 0.35
            picked = qd["correct"] if correct else qd["common_wrong"]
            flagged = (not correct) and rng.random() > 0.4
            attempt = Attempt(
                question_id=q.id,
                source="uworld",
                attempted_at=attempted,
                selected_choice=picked,
                is_correct=correct,
                time_seconds=rng.randint(45, 180),
                flagged=flagged,
                uworld_test_id=test_id,
            )
            session.add(attempt)
            await session.flush()
            made_a += 1
            if flagged:
                session.add(
                    AttemptNote(
                        attempt_id=attempt.id,
                        note_text="Mixed up the mechanism — review the discriminator.",
                        flag_for_review=True,
                        source="user",
                    )
                )
                made_n += 1

    made_p, made_kf, made_ke, made_np = await seed_kb(session, course)

    print(
        f"seeded course={course_slug}: +{made_q} questions, +{made_a} attempts, "
        f"+{made_t} tags, +{made_n} flagged notes "
        f"(skipped {len(QUESTIONS) - made_q} already-present)."
    )
    print(
        f"  KB substrate: +{made_p} pdfs, +{made_kf} facts(+tags), "
        f"+{made_ke} edges, +{made_np} notion pages."
    )


async def main() -> None:
    ap = argparse.ArgumentParser(description="Seed synthetic dev data.")
    ap.add_argument("--course", default="aamc", help="course slug (default: aamc)")
    ap.add_argument("--wipe", action="store_true", help="delete dev-namespace rows, then exit")
    args = ap.parse_args()

    async with AsyncSessionLocal() as session:
        async with session.begin():
            if args.wipe:
                n = await wipe(session)
                print(f"wiped {n} dev questions (+ their attempts/tags/notes).")
                return
            await seed(session, args.course)


if __name__ == "__main__":
    asyncio.run(main())
