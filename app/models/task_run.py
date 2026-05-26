import enum
from datetime import datetime

from sqlalchemy import NUMERIC, DateTime, Enum, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class TaskRunStatus(str, enum.Enum):
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class TaskRun(Base):
    __tablename__ = "task_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[TaskRunStatus] = mapped_column(
        Enum(TaskRunStatus, name="task_run_status"),
        nullable=False,
        default=TaskRunStatus.running,
    )
    items_processed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cost_usd: Mapped[float | None] = mapped_column(NUMERIC(10, 4), nullable=True)
    error_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (Index("ix_task_runs_job_started", "job_name", "started_at"),)
