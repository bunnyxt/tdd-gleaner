"""Integrity checks, reproducible bundle export, and restore.

A bundle is a self-contained directory::

    manifest.json        versions, counts, dataset hash, and SHA-256 of every file
    gleaner.sqlite       consistent copy of the experiment database
    raw/...              every raw response body referenced by the database
    candidates.jsonl     candidates with all discoveries
    fetch_attempts.jsonl every attempt, including failures
    snapshots.jsonl      every snapshot with its exact model input string
    annotations.jsonl    every annotation revision
    dataset.jsonl        latest snapshot per aid with its latest annotation revision

The JSONL files are derived deterministically from the database, so a restore
can regenerate them from the restored database and compare hashes.
"""

import json
import shutil
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from . import __version__, bilibili
from .acquire import INPUT_SCHEMA_VERSION, build_input, input_sha256
from .store import DB_NAME, RAW_DIR, SCHEMA_VERSION, Store, StoreError, init_store, open_store
from .util import Clock, canonical_json, sha256_bytes, sha256_file, utc_now

BUNDLE_FORMAT = "gleaner-bundle/1"
MANIFEST = "manifest.json"
JSONL_FILES = ("candidates.jsonl", "fetch_attempts.jsonl", "snapshots.jsonl", "annotations.jsonl", "dataset.jsonl")


# -- integrity -----------------------------------------------------------


def verify_store(store: Store) -> List[str]:
    """Return a list of integrity problems (empty when the store is sound)."""
    problems: List[str] = []
    conn = store.conn
    check = conn.execute("PRAGMA integrity_check").fetchone()[0]
    if check != "ok":
        problems.append(f"sqlite integrity_check: {check}")
    attempts = {}
    for row in conn.execute("SELECT * FROM fetch_attempt ORDER BY id"):
        attempts[row["id"]] = row
        if row["raw_path"] is None:
            continue
        path = store.root / row["raw_path"]
        if not path.exists():
            problems.append(f"attempt {row['id']}: raw file missing ({row['raw_path']})")
            continue
        body = path.read_bytes()
        if sha256_bytes(body) != row["raw_sha256"] or len(body) != row["raw_size"]:
            problems.append(f"attempt {row['id']}: raw file hash mismatch")
            continue
        replay = bilibili.classify(row["endpoint"], row["aid"], bilibili.Response(row["http_status"], body))
        stored = json.loads(row["extracted"]) if row["extracted"] else None
        if replay.status != row["status"] or replay.extracted != stored:
            problems.append(f"attempt {row['id']}: replaying raw response gives {replay.status}, stored {row['status']}")
    for snap in conn.execute("SELECT * FROM snapshot ORDER BY snapshot_id"):
        sid = snap["snapshot_id"]
        view, tags = attempts.get(snap["view_attempt_id"]), attempts.get(snap["tags_attempt_id"])
        if view is None or tags is None or view["status"] != "ok" or tags["status"] != "ok":
            problems.append(f"{sid}: does not reference two ok attempts")
            continue
        if not (view["acquisition_id"] == tags["acquisition_id"] == snap["acquisition_id"]):
            problems.append(f"{sid}: attempts belong to a different acquisition")
            continue
        if view["aid"] != snap["aid"] or view["endpoint"] != "view" or tags["endpoint"] != "tags":
            problems.append(f"{sid}: attempt endpoints or aid do not match")
            continue
        v, t = json.loads(view["extracted"]), json.loads(tags["extracted"])
        text = build_input(v["title"], v["desc"], t["tags"])
        if text != snap["input_text"]:
            problems.append(f"{sid}: input text differs from the one rebuilt from its attempts")
        if input_sha256(snap["input_text"]) != snap["input_sha256"]:
            problems.append(f"{sid}: input_sha256 mismatch")
        if snap["input_schema_version"] != INPUT_SCHEMA_VERSION:
            problems.append(f"{sid}: unknown input schema {snap['input_schema_version']}")
    for row in conn.execute("SELECT snapshot_id, COUNT(*) AS n, MAX(revision) AS m FROM annotation GROUP BY snapshot_id"):
        if row["n"] != row["m"]:
            problems.append(f"{row['snapshot_id']}: annotation revisions are not contiguous")
    return problems


# -- JSONL views ---------------------------------------------------------


def _rows(conn: sqlite3.Connection, sql: str) -> List[dict]:
    return [dict(r) for r in conn.execute(sql)]


def _label(value: Optional[int]):
    return None if value is None else bool(value)


def jsonl_views(conn: sqlite3.Connection) -> Dict[str, List[dict]]:
    discoveries: Dict[int, List[dict]] = {}
    for d in _rows(conn, "SELECT * FROM discovery ORDER BY id"):
        discoveries.setdefault(d["aid"], []).append({
            "source": d["source"], "locator": d["locator"], "sample_group": d["sample_group"],
            "observed": json.loads(d["observed"]), "discovered_at": d["discovered_at"], "import_batch": d["import_batch"],
        })
    candidates = [
        {"aid": c["aid"], "first_seen_at": c["first_seen_at"], "discoveries": discoveries.get(c["aid"], [])}
        for c in _rows(conn, "SELECT * FROM candidate ORDER BY aid")
    ]
    attempts = []
    for a in _rows(conn, "SELECT * FROM fetch_attempt ORDER BY id"):
        a["extracted"] = json.loads(a["extracted"]) if a["extracted"] else None
        attempts.append(a)
    snapshots = _rows(conn, "SELECT * FROM snapshot ORDER BY aid, acquisition_id")
    annotations = []
    for a in _rows(conn, "SELECT * FROM annotation ORDER BY snapshot_id, revision"):
        a["uses_voicebank"], a["is_vocal_work"] = _label(a["uses_voicebank"]), _label(a["is_vocal_work"])
        annotations.append(a)
    latest_ann = {}
    for a in annotations:
        latest_ann[a["snapshot_id"]] = a
    latest_snap = {}
    for s in snapshots:
        latest_snap[s["aid"]] = s
    dataset = []
    for aid in sorted(latest_snap):
        s = latest_snap[aid]
        a = latest_ann.get(s["snapshot_id"])
        dataset.append({
            "aid": aid, "snapshot_id": s["snapshot_id"], "input_schema_version": s["input_schema_version"],
            "input_sha256": s["input_sha256"], "input": s["input_text"],
            "annotation_revision": a["revision"] if a else None,
            "rules_version": a["rules_version"] if a else None,
            "uses_voicebank": a["uses_voicebank"] if a else None,
            "is_vocal_work": a["is_vocal_work"] if a else None,
            "text_evidence": a["text_evidence"] if a else None,
        })
    return {
        "candidates.jsonl": candidates, "fetch_attempts.jsonl": attempts, "snapshots.jsonl": snapshots,
        "annotations.jsonl": annotations, "dataset.jsonl": dataset,
    }


def _jsonl_bytes(records: List[dict]) -> bytes:
    return "".join(canonical_json(r) + "\n" for r in records).encode("utf-8")


# -- export / restore ----------------------------------------------------


def export_bundle(store: Store, out: Path, clock: Clock = utc_now) -> Dict[str, object]:
    problems = verify_store(store)
    if problems:
        raise StoreError("refusing to export a store with integrity problems:\n  " + "\n  ".join(problems))
    if out.exists() and any(out.iterdir()):
        raise StoreError(f"bundle directory {out} is not empty")
    out.mkdir(parents=True, exist_ok=True)

    target = sqlite3.connect(str(out / DB_NAME))
    try:
        store.conn.backup(target)
    finally:
        target.close()

    raw_paths = sorted({r["raw_path"] for r in store.conn.execute("SELECT raw_path FROM fetch_attempt WHERE raw_path IS NOT NULL")})
    for rel in raw_paths:
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(store.root / rel, dest)

    views = jsonl_views(store.conn)
    for name in JSONL_FILES:
        (out / name).write_bytes(_jsonl_bytes(views[name]))

    files = {}
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        rel = path.relative_to(out).as_posix()
        files[rel] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    manifest = {
        "format": BUNDLE_FORMAT,
        "tool_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "input_schema_version": INPUT_SCHEMA_VERSION,
        "created_at": clock(),
        "counts": _counts(views, raw_paths),
        "dataset_sha256": files["dataset.jsonl"]["sha256"],
        "files": files,
    }
    data = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (out / MANIFEST).write_bytes(data)
    return {"bundle": str(out), "manifest_sha256": sha256_bytes(data), "dataset_sha256": manifest["dataset_sha256"], "counts": manifest["counts"]}


def _counts(views: Dict[str, List[dict]], raw_paths: List[str]) -> Dict[str, int]:
    counts = {name[: -len(".jsonl")]: len(views[name]) for name in JSONL_FILES}
    counts["raw_files"] = len(raw_paths)
    return counts


def verify_bundle(bundle: Path) -> List[str]:
    """Check every file listed in the manifest, and that nothing is unlisted."""
    manifest_path = bundle / MANIFEST
    if not manifest_path.exists():
        return [f"no {MANIFEST} in {bundle}"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems = []
    if manifest.get("format") != BUNDLE_FORMAT:
        problems.append(f"unknown bundle format {manifest.get('format')!r}")
    listed = manifest.get("files", {})
    present = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()} - {MANIFEST}
    for rel in sorted(present - set(listed)):
        problems.append(f"unlisted file: {rel}")
    for rel, info in sorted(listed.items()):
        path = bundle / rel
        if not path.exists():
            problems.append(f"missing file: {rel}")
        elif sha256_file(path) != info["sha256"]:
            problems.append(f"hash mismatch: {rel}")
    return problems


def restore_bundle(bundle: Path, data_dir: Path) -> Dict[str, object]:
    """Restore a bundle into an empty data directory and verify it end to end."""
    problems = verify_bundle(bundle)
    if problems:
        raise StoreError("bundle verification failed:\n  " + "\n  ".join(problems))
    if data_dir.exists() and any(data_dir.iterdir()):
        raise StoreError(f"restore target {data_dir} is not empty")
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bundle / DB_NAME, data_dir / DB_NAME)
    raw_src = bundle / RAW_DIR
    if raw_src.exists():
        shutil.copytree(raw_src, data_dir / RAW_DIR)
    else:
        (data_dir / RAW_DIR).mkdir()
    init_store(data_dir)  # validates the schema version of the restored database
    with open_store(data_dir) as store:
        problems = verify_store(store)
        views = jsonl_views(store.conn)
        for name in JSONL_FILES:
            if sha256_bytes(_jsonl_bytes(views[name])) != sha256_file(bundle / name):
                problems.append(f"{name} differs from the restored database")
    if problems:
        raise StoreError("restored data failed verification:\n  " + "\n  ".join(problems))
    manifest_bytes = (bundle / MANIFEST).read_bytes()
    return {"data_dir": str(data_dir), "manifest_sha256": sha256_bytes(manifest_bytes), "verified": True}
