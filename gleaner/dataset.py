"""Dataset manifests: a fixed, hashed selection of snapshots and annotation revisions.

A manifest is built from a verified bundle and a selection CSV. It pins, for
every selected aid, the snapshot, its exact model input, one annotation
revision, the sampling group, the family, and the split. Everything later
(model runs and reports) refers to the manifest by its SHA-256, so the same
manifest always means the same inputs and the same human labels.

Splits:

* ``dev`` -- used to tune questions and thresholds;
* ``acceptance`` -- held out until thresholds are frozen.

A row may force its split (``dev`` or ``acceptance``). Rows without one are
assigned by family: ``sha256(seed + ":" + family)`` is mapped to [0, 1) and
compared with the acceptance fraction (algorithm ``family-sha256-v1``). Every
member of a family therefore lands in the same split, and a family with
conflicting forced splits is an error.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from .bundle import verify_bundle
from .store import StoreError
from .util import Clock, canonical_json, parse_aid, sha256_bytes, utc_now

MANIFEST_FORMAT = "gleaner-dataset/1"
SPLIT_ALGORITHM = "family-sha256-v1"
SPLITS = ("dev", "acceptance")


def _read_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def family_fraction(seed: str, family: str) -> float:
    digest = sha256_bytes(f"{seed}:{family}".encode("utf-8"))
    return int(digest[:16], 16) / float(1 << 64)


def _read_selection(path: Path) -> List[dict]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "aid" not in reader.fieldnames:
            raise StoreError(f"{path}: selection CSV needs an 'aid' column")
        rows = []
        for line_no, row in enumerate(reader, start=2):
            try:
                aid = parse_aid(row.get("aid", ""))
            except ValueError as exc:
                raise StoreError(f"{path}:{line_no}: {exc}") from None
            split = (row.get("split") or "").strip()
            if split and split not in SPLITS:
                raise StoreError(f"{path}:{line_no}: split must be one of {SPLITS} or empty, got {split!r}")
            revision = (row.get("annotation_revision") or "").strip()
            rows.append({
                "line": line_no,
                "aid": aid,
                "snapshot_id": (row.get("snapshot_id") or "").strip() or None,
                "family": (row.get("family") or "").strip() or f"aid-{aid}",
                "split": split or None,
                "group": (row.get("group") or "").strip() or None,
                "annotation_revision": int(revision) if revision else None,
            })
    return rows


def build_manifest(
    bundle: Path,
    selection_csv: Path,
    dataset_version: str,
    seed: str,
    acceptance_fraction: float,
    popular_groups: List[str],
    clock: Clock = utc_now,
) -> dict:
    """Build a manifest dictionary. Raises StoreError on any inconsistency."""
    problems = verify_bundle(bundle)
    if problems:
        raise StoreError("bundle verification failed:\n  " + "\n  ".join(problems))
    if not 0.0 <= acceptance_fraction <= 1.0:
        raise StoreError("acceptance fraction must be between 0 and 1")
    bundle_manifest_bytes = (bundle / "manifest.json").read_bytes()
    bundle_manifest = json.loads(bundle_manifest_bytes)

    snapshots = {s["snapshot_id"]: s for s in _read_jsonl(bundle / "snapshots.jsonl")}
    annotations: Dict[str, Dict[int, dict]] = {}
    for a in _read_jsonl(bundle / "annotations.jsonl"):
        annotations.setdefault(a["snapshot_id"], {})[a["revision"]] = a
    latest = {d["aid"]: d for d in _read_jsonl(bundle / "dataset.jsonl")}
    discoveries = {c["aid"]: c["discoveries"] for c in _read_jsonl(bundle / "candidates.jsonl")}

    rows = _read_selection(selection_csv)
    seen_aids: Dict[int, int] = {}
    for row in rows:
        if row["aid"] in seen_aids:
            raise StoreError(f"{selection_csv}:{row['line']}: aid {row['aid']} already selected on line {seen_aids[row['aid']]}")
        seen_aids[row["aid"]] = row["line"]

    # Resolve splits per family.
    family_forced: Dict[str, set] = {}
    for row in rows:
        family_forced.setdefault(row["family"], set())
        if row["split"]:
            family_forced[row["family"]].add(row["split"])
    family_split: Dict[str, str] = {}
    for family, forced in family_forced.items():
        if len(forced) > 1:
            raise StoreError(f"family {family!r} is forced into both splits; a family must stay in one split")
        if forced:
            family_split[family] = forced.pop()
        else:
            family_split[family] = "acceptance" if family_fraction(seed, family) < acceptance_fraction else "dev"

    items = []
    for order, row in enumerate(rows, start=1):
        aid = row["aid"]
        group = row["group"]
        if group is None:
            groups = [d.get("sample_group") for d in discoveries.get(aid, []) if d.get("sample_group")]
            group = groups[0] if groups else ""
        item = {
            "order": order, "aid": aid, "family": row["family"], "split": family_split[row["family"]], "group": group,
            "split_forced": bool(row["split"]),
        }
        snapshot_id = row["snapshot_id"] or (latest.get(aid) or {}).get("snapshot_id")
        if snapshot_id is None:
            item.update({"status": "no_snapshot", "snapshot_id": None})
            items.append(item)
            continue
        snap = snapshots.get(snapshot_id)
        if snap is None or snap["aid"] != aid:
            raise StoreError(f"{selection_csv}:{row['line']}: snapshot {snapshot_id!r} not found for aid {aid}")
        revisions = annotations.get(snapshot_id, {})
        revision = row["annotation_revision"] if row["annotation_revision"] is not None else (max(revisions) if revisions else None)
        if revision is not None and revision not in revisions:
            raise StoreError(f"{selection_csv}:{row['line']}: annotation revision {revision} not found for {snapshot_id}")
        ann = revisions.get(revision) if revision is not None else None
        item.update({
            "status": "ok" if ann else "unannotated",
            "snapshot_id": snapshot_id,
            "input_schema_version": snap["input_schema_version"],
            "input_sha256": snap["input_sha256"],
            "input": snap["input_text"],
            "annotation_revision": revision,
            "rules_version": ann["rules_version"] if ann else None,
            "uses_voicebank": ann["uses_voicebank"] if ann else None,
            "is_vocal_work": ann["is_vocal_work"] if ann else None,
            "text_evidence": ann["text_evidence"] if ann else None,
        })
        if sha256_bytes(item["input"].encode("utf-8")) != item["input_sha256"]:
            raise StoreError(f"{snapshot_id}: input hash mismatch in bundle")
        items.append(item)

    manifest = {
        "format": MANIFEST_FORMAT,
        "dataset_version": dataset_version,
        "created_at": clock(),
        "source_bundle": {
            "manifest_sha256": sha256_bytes(bundle_manifest_bytes),
            "dataset_sha256": bundle_manifest.get("dataset_sha256"),
        },
        "split_algorithm": SPLIT_ALGORITHM,
        "seed": seed,
        "acceptance_fraction": acceptance_fraction,
        "popular_groups": sorted(set(popular_groups)),
        "selection_sha256": sha256_bytes(selection_csv.read_bytes()),
        "items": items,
        "items_sha256": sha256_bytes(canonical_json(items).encode("utf-8")),
    }
    validate_manifest(manifest)
    return manifest


def manifest_bytes(manifest: dict) -> bytes:
    return (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_manifest(manifest: dict, out: Path) -> str:
    if out.exists():
        raise StoreError(f"{out} already exists; manifests are never overwritten")
    data = manifest_bytes(manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return sha256_bytes(data)


def validate_manifest(manifest: dict) -> None:
    """Check structural invariants: unique aids, families inside one split, hashes."""
    if manifest.get("format") != MANIFEST_FORMAT:
        raise StoreError(f"unknown dataset manifest format {manifest.get('format')!r}")
    items = manifest["items"]
    if sha256_bytes(canonical_json(items).encode("utf-8")) != manifest["items_sha256"]:
        raise StoreError("dataset manifest items_sha256 does not match its items")
    aids, snapshots, family_split = set(), set(), {}
    for item in items:
        if item["aid"] in aids:
            raise StoreError(f"aid {item['aid']} appears twice in the manifest")
        aids.add(item["aid"])
        if item.get("snapshot_id"):
            if item["snapshot_id"] in snapshots:
                raise StoreError(f"snapshot {item['snapshot_id']} appears twice in the manifest")
            snapshots.add(item["snapshot_id"])
        if item["split"] not in SPLITS:
            raise StoreError(f"aid {item['aid']}: invalid split {item['split']!r}")
        previous = family_split.setdefault(item["family"], item["split"])
        if previous != item["split"]:
            raise StoreError(f"family {item['family']!r} appears in both splits")


def load_manifest(path: Path) -> tuple:
    """Return (manifest, sha256 of the file bytes)."""
    data = path.read_bytes()
    manifest = json.loads(data)
    validate_manifest(manifest)
    return manifest, sha256_bytes(data)


def model_items(manifest: dict, split: Optional[str] = None) -> List[dict]:
    """Items that have a snapshot, i.e. that can be sent to a model."""
    return [i for i in manifest["items"] if i.get("snapshot_id") and (split is None or i["split"] == split)]
