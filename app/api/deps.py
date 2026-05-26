"""FastAPI dependencies: settings, DB session, auth."""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, settings
from app.database import AsyncSessionLocal


def get_settings() -> Settings:
    return settings


async def get_session() -> AsyncIterator[AsyncSession]:
    """Open a session; commit on clean exit, rollback on exception."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def verify_coach_token(
    x_coach_token: Annotated[str | None, Header()] = None,
    settings: Settings = Depends(get_settings),
) -> None:
    if x_coach_token is None or not secrets.compare_digest(x_coach_token, settings.COACH_TOKEN):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid coach token",
        )
