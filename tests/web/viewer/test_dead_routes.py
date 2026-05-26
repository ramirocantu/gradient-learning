"""Regression guard: ensure R.3-deleted viewer routes stay deleted.

Each assertion is a cheap 404 check against the viewer sub-app. If a
future change accidentally re-introduces one of these handlers, the
matching test here flips and surfaces the regression.

Paths are prefixed with /viewer because R.2c's unified ``client``
fixture targets the merged ``app.main:app`` (viewer is mounted at
``/viewer``), not the viewer sub-app directly.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.asyncio


async def test_viewer_index_returns_404(client):
    r = await client.get("/viewer/")
    assert r.status_code == 404


async def test_stats_returns_404(client):
    r = await client.get("/viewer/stats")
    assert r.status_code == 404


async def test_media_gallery_returns_404(client):
    r = await client.get("/viewer/media-gallery")
    assert r.status_code == 404


async def test_quick_unmapped_taxonomy_returns_404(client):
    r = await client.get("/viewer/quick/unmapped-taxonomy")
    assert r.status_code == 404


async def test_quick_parse_warnings_returns_404(client):
    r = await client.get("/viewer/quick/parse-warnings")
    assert r.status_code == 404


async def test_quick_orphan_media_returns_404(client):
    r = await client.get("/viewer/quick/orphan-media")
    assert r.status_code == 404
