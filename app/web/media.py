from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import settings

router = APIRouter()


@router.get("/media/{file_path:path}")
async def serve(file_path: str) -> FileResponse:
    root = settings.MEDIA_ROOT.resolve()
    try:
        target = (root / file_path).resolve()
    except (OSError, ValueError):
        raise HTTPException(status_code=404, detail="not found")

    if not target.is_relative_to(root):
        raise HTTPException(status_code=404, detail="not found")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="not found")

    return FileResponse(
        target,
        headers={"Cache-Control": "public, max-age=86400"},
    )
