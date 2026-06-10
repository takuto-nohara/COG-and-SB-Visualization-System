"""Common helpers for timestamping and file naming."""

from __future__ import annotations

import uuid
from datetime import datetime


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def new_session_id(prefix: str = "session") -> str:
    return f"{prefix}-{uuid.uuid4()}"
