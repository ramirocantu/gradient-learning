"""System-status probes for the admin health endpoint (T39).

Backing service for ``GET /api/v1/admin/status`` (V-D1: read-only JSON on the
public seam). Lets a client show *real* connection health instead of the
"scheduled"/"unknown" guesses it could previously infer from
``/admin/jobs`` alone (desktop ``desktop/SPEC.md`` ¶T3).

Three external dependencies are probed with the cheapest reachability call
each, plus a per-job last-run rollup from ``task_runs``:

- **Anki**  — ``AnkiConnect.version()`` (read-only, V13).
- **OpenAI** — ``models.retrieve(model)`` (validates key + model presence).
- **Notion** — ``users.me()``. This is a *token-validity* probe: it returns
  the bot user identity, NOT wiki content, and persists nothing — so it
  honors V-N1 (⊥ read Notion content back, ⊥ keep a local copy). The only
  Notion state this repo stores remains the ``notion_pages`` pointer.

Every probe is total: it converts *any* failure (transport, auth, SDK
error) into ``reachable=False`` with the exception text in ``detail``. A
health read must never itself raise — a down dependency is the answer, not
an error. The injected clients (V16) keep the SDK boundary mockable in tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.task_run import TaskRun


@dataclass
class ServiceHealth:
    """Health of one external dependency.

    ``configured`` is False when the required env var is unset (no probe
    attempted); ``reachable`` is True only when the probe call succeeded.
    """

    configured: bool
    reachable: bool
    detail: str | None


async def probe_anki(client: Any) -> ServiceHealth:
    """Reachability via AnkiConnect ``version`` (read-only, V13). Anki is
    always considered configured — ``ANKICONNECT_URL`` has a default."""
    try:
        version = await client.version()
        return ServiceHealth(configured=True, reachable=True, detail=f"AnkiConnect v{version}")
    except Exception as exc:  # noqa: BLE001 — health read must not raise
        return ServiceHealth(configured=True, reachable=False, detail=str(exc))


async def probe_openai(client: Any, *, model: str, configured: bool) -> ServiceHealth:
    """Reachability via ``models.retrieve(model)`` — validates the key and
    that the configured model exists. V16: client is injected/mocked."""
    if not configured:
        return ServiceHealth(configured=False, reachable=False, detail="OPENAI_API_KEY unset")
    try:
        await client.models.retrieve(model)
        return ServiceHealth(configured=True, reachable=True, detail=None)
    except Exception as exc:  # noqa: BLE001 — health read must not raise
        return ServiceHealth(configured=True, reachable=False, detail=str(exc))


async def probe_notion(client: Any, *, configured: bool) -> ServiceHealth:
    """Token-validity probe via ``users.me()`` (bot identity, ⊥ content —
    V-N1). ``client`` is None when the token is unset."""
    if not configured or client is None:
        return ServiceHealth(configured=False, reachable=False, detail="NOTION_API_TOKEN unset")
    try:
        await client.users.me()
        return ServiceHealth(configured=True, reachable=True, detail=None)
    except Exception as exc:  # noqa: BLE001 — health read must not raise
        return ServiceHealth(configured=True, reachable=False, detail=str(exc))


async def job_last_runs(session: AsyncSession) -> dict[str, TaskRun]:
    """Latest ``TaskRun`` per ``job_name`` (Postgres ``DISTINCT ON``).

    ``/admin/jobs`` exposes only the *next* run time; this fills in the last
    *outcome* so a client can show "succeeded 5m ago" vs "failed". Keyed by
    ``job_name`` for O(1) merge against the scheduler snapshot.
    """
    rows = (
        (
            await session.execute(
                select(TaskRun)
                .order_by(TaskRun.job_name, TaskRun.started_at.desc())
                .distinct(TaskRun.job_name)
            )
        )
        .scalars()
        .all()
    )
    return {row.job_name: row for row in rows}


def _run_payload(run: TaskRun | None) -> dict[str, Any] | None:
    if run is None:
        return None
    return {
        "status": run.status.value,
        "started_at": run.started_at.isoformat(),
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "items_processed": run.items_processed,
        "error_text": run.error_text,
    }


def build_jobs_health(
    job_snapshots: list[dict[str, Any]], last_runs: dict[str, TaskRun]
) -> list[dict[str, Any]]:
    """Merge scheduler snapshot (``job_id`` + ``next_run_time``) with the
    last-run rollup. One row per scheduled job, in snapshot order."""
    merged: list[dict[str, Any]] = []
    for snap in job_snapshots:
        job_id = snap["job_id"]
        merged.append(
            {
                "job_id": job_id,
                "next_run_time": snap.get("next_run_time"),
                "last_run": _run_payload(last_runs.get(job_id)),
            }
        )
    return merged


async def collect_system_status(
    session: AsyncSession,
    *,
    anki_client: Any,
    openai_client: Any,
    notion_client: Any,
    job_snapshots: list[dict[str, Any]],
    openai_model: str,
    openai_configured: bool,
    notion_configured: bool,
) -> dict[str, Any]:
    """Assemble the full ``/admin/status`` payload. Single testable entry
    point — clients are injected so the SDK boundary is mockable (V16)."""
    anki = await probe_anki(anki_client)
    openai = await probe_openai(openai_client, model=openai_model, configured=openai_configured)
    notion = await probe_notion(notion_client, configured=notion_configured)
    last_runs = await job_last_runs(session)
    return {
        "anki": asdict(anki),
        "openai": asdict(openai),
        "notion": asdict(notion),
        "jobs": build_jobs_health(job_snapshots, last_runs),
    }
