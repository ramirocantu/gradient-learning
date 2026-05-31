from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints


class DiscriminatorIn(BaseModel):
    """Persist-only write payload (V-M1): data fields only, no verdict /
    grade / heuristic. The host reasons; this seam persists."""

    question_id: int
    # strip first, then enforce min_length so whitespace-only is rejected (422)
    factor_text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    node_id: int | None = None


class DiscriminatorOut(BaseModel):
    id: int
    question_id: int
    factor_text: str
    node_id: int | None
    notion_block_id: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
