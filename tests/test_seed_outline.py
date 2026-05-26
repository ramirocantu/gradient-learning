from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.models.outline import ContentCategory, FoundationalConcept, Section, Topic


async def test_seed_runs_clean(seeded_report, test_engine):
    async with AsyncSession(test_engine) as session:
        sections = await session.scalar(select(func.count()).select_from(Section))
        fcs = await session.scalar(select(func.count()).select_from(FoundationalConcept))
        ccs = await session.scalar(select(func.count()).select_from(ContentCategory))
        topics = await session.scalar(select(func.count()).select_from(Topic))

    assert sections == 4
    assert fcs >= 10
    assert ccs >= 10
    assert topics > 100


async def test_seed_is_idempotent(seeded_report, test_engine):
    from scripts.seed_outline import seed

    async with AsyncSession(test_engine) as session:
        r2 = await seed(session)

    assert seeded_report.sections_upserted == r2.sections_upserted
    assert seeded_report.fcs_upserted == r2.fcs_upserted
    assert seeded_report.ccs_upserted == r2.ccs_upserted
    assert seeded_report.topics_upserted == r2.topics_upserted
    assert seeded_report.max_depth_observed == r2.max_depth_observed


async def test_content_category_lookup(db_session):
    result = await db_session.execute(select(ContentCategory).where(ContentCategory.code == "4A"))
    cc = result.scalar_one()
    assert cc.name


async def test_topic_subtree_via_recursive_cte(db_session):
    anchor_sel = (
        select(Topic.id, Topic.name)
        .join(ContentCategory, Topic.content_category_id == ContentCategory.id)
        .where(
            ContentCategory.code == "2A",
            Topic.name == "Composition of membranes",
        )
    )

    cte = anchor_sel.cte("subtree", recursive=True)

    t2 = aliased(Topic)
    recursive_sel = select(t2.id, t2.name).where(t2.parent_topic_id == cte.c.id)
    cte = cte.union_all(recursive_sel)

    result = await db_session.execute(select(cte.c.name))
    names = {row[0] for row in result}

    expected = {
        "Phospholipids (and phosphatids)",
        "Steroids",
        "Waxes",
        "Protein components",
        "Fluid mosaic model",
    }
    assert expected <= names


async def test_disciplines_round_trip(db_session):
    cc_result = await db_session.execute(
        select(ContentCategory).where(ContentCategory.code == "2A")
    )
    cc = cc_result.scalar_one()

    result = await db_session.execute(
        select(Topic).where(
            Topic.content_category_id == cc.id,
            Topic.name == "Lipid components",
        )
    )
    topic = result.scalar_one()
    assert set(topic.disciplines) == {"BIO", "BC", "OC"}


async def test_depth_is_correct(db_session):
    cc_result = await db_session.execute(
        select(ContentCategory).where(ContentCategory.code == "2A")
    )
    cc = cc_result.scalar_one()

    # depth=0: Plasma Membrane (direct CC child)
    pm_result = await db_session.execute(
        select(Topic).where(
            Topic.content_category_id == cc.id,
            Topic.name == "Plasma Membrane",
            Topic.parent_topic_id.is_(None),
        )
    )
    pm = pm_result.scalar_one()
    assert pm.depth == 0

    # depth=1: Composition of membranes
    com_result = await db_session.execute(
        select(Topic).where(
            Topic.content_category_id == cc.id,
            Topic.name == "Composition of membranes",
            Topic.parent_topic_id == pm.id,
        )
    )
    com = com_result.scalar_one()
    assert com.depth == 1

    # depth=2: Lipid components
    lc_result = await db_session.execute(
        select(Topic).where(
            Topic.content_category_id == cc.id,
            Topic.name == "Lipid components",
            Topic.parent_topic_id == com.id,
        )
    )
    lc = lc_result.scalar_one()
    assert lc.depth == 2

    # depth=3: Phospholipids (and phosphatids)
    ph_result = await db_session.execute(
        select(Topic).where(
            Topic.content_category_id == cc.id,
            Topic.name == "Phospholipids (and phosphatids)",
            Topic.parent_topic_id == lc.id,
        )
    )
    ph = ph_result.scalar_one()
    assert ph.depth == 3
