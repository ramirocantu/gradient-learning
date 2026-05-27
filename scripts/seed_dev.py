"""Seed synthetic dev/test data straight into the DB — no OpenAI / Anki / Notion.

Populates `questions` + `attempts` + `question_tags(node_id)` + `attempt_notes`
so the client-facing reads (tutor captures/sessions/flagged, question-by-qid,
per-node mastery once exposed) have real rows, bypassing every external
pipeline and the legacy manual-tag route.

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
from app.models.attempt_note import AttemptNote
from app.models.captures import Attempt, Question, QuestionTag
from app.models.outline import Course, OutlineNode

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


async def wipe(session: AsyncSession) -> int:
    """Delete the dev namespace. Tags/attempts cascade off Question via FK, but
    attempt_notes have no cascade from Question — delete them explicitly first."""
    q_ids = (
        await session.execute(select(Question.id).where(Question.qid.like(f"{QID_PREFIX}%")))
    ).scalars().all()
    if not q_ids:
        return 0
    a_ids = (
        await session.execute(select(Attempt.id).where(Attempt.question_id.in_(q_ids)))
    ).scalars().all()
    if a_ids:
        await session.execute(delete(AttemptNote).where(AttemptNote.attempt_id.in_(a_ids)))
    await session.execute(delete(Attempt).where(Attempt.question_id.in_(q_ids)))
    await session.execute(delete(QuestionTag).where(QuestionTag.question_id.in_(q_ids)))
    await session.execute(delete(Question).where(Question.id.in_(q_ids)))
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

    print(
        f"seeded course={course_slug}: +{made_q} questions, +{made_a} attempts, "
        f"+{made_t} tags, +{made_n} flagged notes "
        f"(skipped {len(QUESTIONS) - made_q} already-present)."
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
