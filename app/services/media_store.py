"""Filesystem store for media bytes.

Pure helpers, no DB access. Idempotent on `content_hash`.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import mimetypes
import os
import tempfile
from pathlib import Path

from app.config import settings


def _extension_for(mime_type: str) -> str:
    ext = mimetypes.guess_extension(mime_type)
    if ext is None:
        return ".bin"
    if ext == ".jpe":
        return ".jpg"
    return ext


def media_path(content_hash: str, mime_type: str) -> Path:
    """Canonical absolute path: MEDIA_ROOT/{hash[:2]}/{hash}{ext}."""
    ext = _extension_for(mime_type)
    return (settings.MEDIA_ROOT / content_hash[:2] / f"{content_hash}{ext}").resolve()


def relative_media_path(content_hash: str, mime_type: str) -> str:
    """Relative path stored in media.local_path: {hash[:2]}/{hash}{ext}."""
    ext = _extension_for(mime_type)
    return f"{content_hash[:2]}/{content_hash}{ext}"


def _decode_bytes(bytes_b64: str) -> bytes:
    try:
        return base64.b64decode(bytes_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"invalid base64 payload: {exc}") from exc


def _write_atomic(target: Path, raw: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


async def write_media(content_hash: str, mime_type: str, bytes_b64: str) -> Path:
    """Write media bytes to canonical path. No-op if already present.

    Returns the absolute path written.
    """
    target = media_path(content_hash, mime_type)
    if target.exists():
        return target
    raw = _decode_bytes(bytes_b64)
    await asyncio.to_thread(_write_atomic, target, raw)
    return target
