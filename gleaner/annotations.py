"""Human annotation export and import through UTF-8 CSV.

Each exported row is one snapshot. Editable columns are the two labels,
``text_evidence``, ``rationale``, ``evidence_locator``, ``null_reason``, and
``annotator``. Context columns (title, desc, tags) are for reading only and
are never used as import input.

Import locates rows by ``aid`` + ``snapshot_id`` + ``base_revision`` and
appends revision ``base_revision + 1``. Invalid, duplicate, or stale rows are
skipped and reported; the other rows are applied.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .store import Store, StoreError
from .util import Clock, parse_aid, utc_now

EDITABLE = ["uses_voicebank", "is_vocal_work", "text_evidence", "rationale", "evidence_locator", "null_reason", "annotator"]
KEYS = ["aid", "snapshot_id", "base_revision", "input_sha256"]
CONTEXT = ["title", "desc", "tags"]
COLUMNS = KEYS + EDITABLE + CONTEXT
TEXT_EVIDENCE = ("sufficient", "insufficient", "uncertain")
LABEL_VALUES = {"true": 1, "false": 0, "null": None}


def _label_text(value: Optional[int]) -> str:
    return {1: "true", 0: "false", None: "null"}[value]


def latest_revisions(store: Store) -> Dict[str, object]:
    rows = store.conn.execute(
        "SELECT a.* FROM annotation a JOIN (SELECT snapshot_id, MAX(revision) AS r FROM annotation GROUP BY snapshot_id) m"
        " ON a.snapshot_id = m.snapshot_id AND a.revision = m.r"
    ).fetchall()
    return {r["snapshot_id"]: r for r in rows}


def export_annotations(store: Store, out: Path, all_snapshots: bool = False) -> int:
    """Write a CSV with one row per snapshot (latest snapshot per aid by default)."""
    if all_snapshots:
        snaps = store.conn.execute("SELECT * FROM snapshot ORDER BY aid, acquisition_id").fetchall()
    else:
        snaps = store.conn.execute(
            "SELECT s.* FROM snapshot s JOIN (SELECT aid, MAX(acquisition_id) AS a FROM snapshot GROUP BY aid) m"
            " ON s.aid = m.aid AND s.acquisition_id = m.a ORDER BY s.aid"
        ).fetchall()
    latest = latest_revisions(store)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for snap in snaps:
            data = json.loads(snap["input_text"])
            row = {
                "aid": snap["aid"],
                "snapshot_id": snap["snapshot_id"],
                "base_revision": 0,
                "input_sha256": snap["input_sha256"],
                "title": data["title"],
                "desc": data["desc"],
                "tags": json.dumps(data["tags"], ensure_ascii=False),
            }
            current = latest.get(snap["snapshot_id"])
            if current is not None:
                row.update({
                    "base_revision": current["revision"],
                    "uses_voicebank": _label_text(current["uses_voicebank"]),
                    "is_vocal_work": _label_text(current["is_vocal_work"]),
                    "text_evidence": current["text_evidence"],
                    "rationale": current["rationale"],
                    "evidence_locator": current["evidence_locator"],
                    "null_reason": current["null_reason"],
                    "annotator": current["annotator"],
                })
            writer.writerow(row)
    return len(snaps)


def _parse_row(line_no: int, row: Dict[str, str], default_annotator: Optional[str]) -> Tuple[Optional[dict], List[str]]:
    errors: List[str] = []
    where = f"line {line_no}"
    values = {k: (row.get(k) or "").strip() for k in KEYS + EDITABLE}
    if not any(values[k] for k in EDITABLE if k != "annotator") and values["base_revision"] in ("", "0"):
        return None, []  # untouched row
    try:
        aid = parse_aid(values["aid"])
    except ValueError as exc:
        return None, [f"{where}: {exc}"]
    try:
        base = int(values["base_revision"] or "0")
        if base < 0:
            raise ValueError
    except ValueError:
        return None, [f"{where}: base_revision must be a non-negative integer"]
    labels = {}
    for key in ("uses_voicebank", "is_vocal_work"):
        text = values[key].lower()
        if text not in LABEL_VALUES:
            errors.append(f"{where}: {key} must be true, false, or null (got {values[key]!r})")
        else:
            labels[key] = LABEL_VALUES[text]
    if values["text_evidence"] not in TEXT_EVIDENCE:
        errors.append(f"{where}: text_evidence must be one of {', '.join(TEXT_EVIDENCE)}")
    if not values["rationale"]:
        errors.append(f"{where}: rationale is required")
    if labels and None in labels.values() and not values["null_reason"]:
        errors.append(f"{where}: null_reason is required when a label is null")
    annotator = values["annotator"] or (default_annotator or "")
    if not annotator:
        errors.append(f"{where}: annotator is required (column or --annotator)")
    if errors:
        return None, errors
    return {
        "line": line_no, "aid": aid, "snapshot_id": values["snapshot_id"], "base_revision": base,
        "input_sha256": values["input_sha256"], **labels, "text_evidence": values["text_evidence"],
        "rationale": values["rationale"], "evidence_locator": values["evidence_locator"],
        "null_reason": values["null_reason"], "annotator": annotator,
    }, []


def _same_as(current, parsed: dict) -> bool:
    return all(current[k] == parsed[k] for k in (
        "uses_voicebank", "is_vocal_work", "text_evidence", "rationale", "evidence_locator", "null_reason", "annotator"))


def import_annotations(
    store: Store, path: Path, rules_version: str, default_annotator: Optional[str] = None, clock: Clock = utc_now,
) -> Dict[str, int]:
    """Append one revision per valid, changed row; report rejected rows.

    Invalid rows are skipped and listed in ``rejected`` with their line and
    reason. Valid rows in the same file are still applied.
    """
    if not rules_version:
        raise StoreError("rules_version is required")
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in KEYS + EDITABLE if c not in (reader.fieldnames or [])]
        if missing:
            raise StoreError(f"annotation CSV is missing columns: {', '.join(missing)}")
        rows = list(reader)

    rejected: List[str] = []
    parsed: List[dict] = []
    untouched = 0
    for line_no, row in enumerate(rows, start=2):
        item, row_errors = _parse_row(line_no, row, default_annotator)
        if row_errors:
            rejected.extend(row_errors)
        elif item is None:
            untouched += 1
        else:
            parsed.append(item)

    # A snapshot listed twice is ambiguous: reject every row for it.
    lines_by_snapshot: Dict[str, List[int]] = {}
    for item in parsed:
        lines_by_snapshot.setdefault(item["snapshot_id"], []).append(item["line"])

    latest = latest_revisions(store)
    to_insert = []
    unchanged = 0
    for item in parsed:
        where = f"line {item['line']}"
        lines = lines_by_snapshot[item["snapshot_id"]]
        if len(lines) > 1:
            rejected.append(f"{where}: duplicate rows for {item['snapshot_id']} (lines {', '.join(map(str, lines))})")
            continue
        snap = store.conn.execute("SELECT aid, input_sha256 FROM snapshot WHERE snapshot_id = ?", (item["snapshot_id"],)).fetchone()
        if snap is None:
            rejected.append(f"{where}: unknown snapshot_id {item['snapshot_id']!r}")
            continue
        if snap["aid"] != item["aid"]:
            rejected.append(f"{where}: snapshot {item['snapshot_id']} belongs to aid {snap['aid']}, not {item['aid']}")
            continue
        if item["input_sha256"] and item["input_sha256"] != snap["input_sha256"]:
            rejected.append(f"{where}: input_sha256 does not match snapshot {item['snapshot_id']}")
            continue
        current = latest.get(item["snapshot_id"])
        current_rev = current["revision"] if current is not None else 0
        if item["base_revision"] != current_rev:
            rejected.append(
                f"{where}: stale or duplicate revision for {item['snapshot_id']}: "
                f"base_revision {item['base_revision']}, current revision {current_rev}; export again and reapply"
            )
            continue
        if current is not None and _same_as(current, item):
            unchanged += 1
            continue
        to_insert.append((item, current_rev + 1))

    now = clock()
    batch = f"annotations:{now}:{path.name}"
    with store.transaction() as conn:
        for item, revision in to_insert:
            conn.execute(
                "INSERT INTO annotation (snapshot_id, revision, uses_voicebank, is_vocal_work, text_evidence, rationale,"
                " evidence_locator, null_reason, annotator, rules_version, annotated_at, import_batch)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (item["snapshot_id"], revision, item["uses_voicebank"], item["is_vocal_work"], item["text_evidence"],
                 item["rationale"], item["evidence_locator"], item["null_reason"], item["annotator"], rules_version, now, batch),
            )
    rejected.sort(key=lambda e: int(e.split(":", 1)[0].split()[1]))
    return {"rows": len(rows), "revisions_added": len(to_insert), "unchanged": unchanged, "untouched": untouched, "rejected": rejected}
