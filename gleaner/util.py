"""Small shared helpers: hashing, canonical JSON, and timestamps."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

Clock = Callable[[], str]


def utc_now() -> str:
    """Return the current UTC time as an ISO 8601 string with milliseconds."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    """Deterministic JSON used for exported records and hashes.

    Keys are sorted, separators are compact, and non-ASCII text is kept as is.
    """
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


MAX_AID = (1 << 63) - 1


def parse_aid(value: Any) -> int:
    """Parse a positive 64-bit video aid. Raises ValueError for anything else."""
    if isinstance(value, bool):
        raise ValueError(f"invalid aid: {value!r}")
    if isinstance(value, int):
        aid = value
    else:
        text = str(value).strip()
        if not text.isdigit():
            raise ValueError(f"invalid aid: {value!r}")
        aid = int(text)
    if aid <= 0 or aid > MAX_AID:
        raise ValueError(f"aid out of range: {value!r}")
    return aid
