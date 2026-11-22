"""Offline evaluation: threshold selection, freezing, and reports.

Nothing here calls a model. Everything is computed from the attempts stored
in an evaluation directory and the human labels pinned in the run's dataset
manifest.

Decision rule (per provider): a video is predicted *included* when
``p_uses_voicebank >= t_uses_voicebank`` and ``p_is_vocal_work >= t_is_vocal_work``.

Rules that keep the numbers honest:

* A human ``null`` label is unknown, never negative. The joint label exists
  only when both flags are known.
* Inference failures are not negatives. Content metrics use successful
  inferences only; a separate *process* view counts failures as not included.
* Thresholds are selected on the ``dev`` split only. Reports on the
  ``acceptance`` split need a frozen threshold configuration, and a frozen
  name can never be rewritten.
"""

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import __version__
from .dataset import model_items
from .evalstore import DEFAULT_REPEAT, EvalStore
from .providers import FLAGS, OK
from .store import StoreError
from .util import Clock, canonical_json, sha256_bytes, utc_now

THRESHOLDS_FORMAT = "gleaner-thresholds/1"
REPORT_FORMAT = "gleaner-report/1"
SELECTION_RULE = "max-popular-recall-under-fpr-v1"
DEFAULT_GRID = [round(0.05 * i, 2) for i in range(1, 20)]
P_COLUMN = {"uses_voicebank": "p_uses_voicebank", "is_vocal_work": "p_is_vocal_work"}


# -- statistics ----------------------------------------------------------------


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> Optional[List[float]]:
    """Wilson score interval (95% by default); None when the denominator is 0."""
    if total == 0:
        return None
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [round(max(0.0, centre - half), 6), round(min(1.0, centre + half), 6)]


def ratio(successes: int, total: int) -> dict:
    return {
        "count": successes, "denominator": total,
        "value": round(successes / total, 6) if total else None,
        "wilson95": wilson(successes, total),
    }


def percentile(values: List[float], q: float) -> Optional[float]:
    """Nearest-rank percentile."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100.0 * len(ordered)))
    return round(ordered[rank - 1], 3)


def _parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


# -- decisions -------------------------------------------------------------------


def decisions(store: EvalStore, run_id: str, repeat_id: str = DEFAULT_REPEAT) -> Dict[str, Dict[str, dict]]:
    """Per provider and snapshot: the successful attempt, or the final failure."""
    result: Dict[str, Dict[str, dict]] = {}
    for a in store.attempts(run_id):
        if a["repeat_id"] != repeat_id:
            continue
        slot = result.setdefault(a["provider"], {}).setdefault(a["snapshot_id"], {"attempts": []})
        slot["attempts"].append(a)
    for per in result.values():
        for slot in per.values():
            ok = [a for a in slot["attempts"] if a["status"] == OK]
            final = ok[0] if ok else slot["attempts"][-1]
            slot["status"] = final["status"]
            slot["final"] = final
            slot["probabilities"] = {f: final[P_COLUMN[f]] for f in FLAGS} if ok else None
    return result


def _truth(item: dict) -> Dict[str, Optional[bool]]:
    uv, wk = item.get("uses_voicebank"), item.get("is_vocal_work")
    return {"uses_voicebank": uv, "is_vocal_work": wk, "joint": (uv and wk) if uv is not None and wk is not None else None}


def _predict(probs: Dict[str, float], thresholds: Dict[str, float]) -> Dict[str, bool]:
    pred = {f: probs[f] >= thresholds[f] for f in FLAGS}
    pred["joint"] = pred["uses_voicebank"] and pred["is_vocal_work"]
    return pred


def _confusion(pairs: List[Tuple[bool, bool]]) -> dict:
    tp = sum(1 for t, p in pairs if t and p)
    fp = sum(1 for t, p in pairs if not t and p)
    fn = sum(1 for t, p in pairs if t and not p)
    tn = sum(1 for t, p in pairs if not t and not p)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": tp + fp + fn + tn}


def _rates(cm: dict) -> dict:
    return {
        "recall": ratio(cm["tp"], cm["tp"] + cm["fn"]),
        "false_positive_rate": ratio(cm["fp"], cm["fp"] + cm["tn"]),
        "precision": ratio(cm["tp"], cm["tp"] + cm["fp"]),
    }


def _rows(items: List[dict], per_snapshot: Dict[str, dict], thresholds: Dict[str, float]) -> List[dict]:
    rows = []
    for item in items:
        slot = per_snapshot.get(item["snapshot_id"])
        probs = slot["probabilities"] if slot else None
        rows.append({
            "item": item, "truth": _truth(item),
            "inference": slot["status"] if slot else "not_attempted",
            "probabilities": probs,
            "pred": _predict(probs, thresholds) if probs else None,
            "slot": slot,
        })
    return rows


def _content_block(rows: List[dict]) -> dict:
    block = {}
    for key in ("uses_voicebank", "is_vocal_work", "joint"):
        pairs = [(r["truth"][key], r["pred"][key]) for r in rows if r["truth"][key] is not None and r["pred"] is not None]
        cm = _confusion(pairs)
        block[key] = {
            "confusion": cm, **_rates(cm),
            "human_unknown": sum(1 for r in rows if r["truth"][key] is None),
            "inference_failed_with_known_label": sum(1 for r in rows if r["truth"][key] is not None and r["pred"] is None),
        }
    return block


def _popular_block(rows: List[dict]) -> dict:
    positives = [r for r in rows if r["truth"]["joint"] is True]
    inferred = [r for r in positives if r["pred"] is not None]
    missed = [r for r in inferred if not r["pred"]["joint"]]
    failed = [r for r in positives if r["pred"] is None]
    return {
        "known_positives": len(positives),
        "missed": len(missed),
        "recall": ratio(len(inferred) - len(missed), len(inferred)),
        "inference_failed": len(failed),
        "process_recall": ratio(len(inferred) - len(missed), len(positives)),
        "human_unknown": sum(1 for r in rows if r["truth"]["joint"] is None),
    }


def _ref(r: dict) -> dict:
    item = r["item"]
    ref = {"aid": item["aid"], "snapshot_id": item["snapshot_id"], "group": item["group"], "split": item["split"],
           "family": item["family"], "text_evidence": item.get("text_evidence"),
           "truth": r["truth"], "inference": r["inference"]}
    if r["probabilities"]:
        ref["probabilities"] = r["probabilities"]
        ref["attempt_id"] = r["slot"]["final"]["id"]
    elif r["slot"]:
        ref["attempt_id"] = r["slot"]["final"]["id"]
        ref["error_class"] = r["slot"]["final"]["error_class"]
    return ref


def _errors(rows: List[dict]) -> dict:
    out = {"false_positive": [], "false_negative": [], "flag_errors": [], "inference_failures": []}
    for r in rows:
        if r["pred"] is None:
            out["inference_failures"].append(_ref(r))
            continue
        joint = r["truth"]["joint"]
        if joint is False and r["pred"]["joint"]:
            out["false_positive"].append(_ref(r))
        elif joint is True and not r["pred"]["joint"]:
            out["false_negative"].append(_ref(r))
        wrong = [f for f in FLAGS if r["truth"][f] is not None and r["truth"][f] != r["pred"][f]]
        if wrong:
            out["flag_errors"].append({**_ref(r), "wrong_flags": wrong})
    return out


# -- thresholds ------------------------------------------------------------------


def _check_run_identity(config: dict, run: dict) -> None:
    for key in ("run_id", "manifest_sha256", "questions_sha256", "settings_sha256"):
        if config.get(key) != run[key]:
            raise StoreError(f"threshold configuration does not match the run ({key})")


def select_thresholds(store: EvalStore, run_id: str, max_fpr: float, grid: Optional[List[float]] = None) -> dict:
    """Grid search on the dev split only. Returns a proposal and the full grid."""
    run = store.run(run_id)
    manifest = run["manifest"]
    grid = sorted(set(grid or DEFAULT_GRID))
    popular = set(manifest["popular_groups"])
    items = model_items(manifest, "dev")
    per_provider = decisions(store, run_id)
    proposal = {
        "format": THRESHOLDS_FORMAT, "run_id": run_id, "manifest_sha256": run["manifest_sha256"],
        "questions_sha256": run["questions_sha256"], "settings_sha256": run["settings_sha256"],
        "selected_on_split": "dev", "selection_rule": SELECTION_RULE, "max_false_positive_rate": max_fpr,
        "grid": grid, "providers": {}, "acceptance_criteria": None,
    }
    grids = {}
    for name, settings in sorted(run["providers"].items()):
        snap = per_provider.get(name, {})
        table = []
        for tv in grid:
            for tw in grid:
                thresholds = {"uses_voicebank": tv, "is_vocal_work": tw}
                rows = _rows(items, snap, thresholds)
                joint = _content_block(rows)["joint"]
                pop = _popular_block([r for r in rows if r["item"]["group"] in popular])
                table.append({
                    "t_uses_voicebank": tv, "t_is_vocal_work": tw, **{k: joint["confusion"][k] for k in ("tp", "fp", "fn", "tn")},
                    "false_positive_rate": joint["false_positive_rate"]["value"], "recall": joint["recall"]["value"],
                    "precision": joint["precision"]["value"], "popular_missed": pop["missed"], "popular_recall": pop["recall"]["value"],
                })
        eligible = [t for t in table if t["false_positive_rate"] is not None and t["false_positive_rate"] <= max_fpr]

        def key(t):
            return (-(t["popular_recall"] if t["popular_recall"] is not None else -1), t["false_positive_rate"],
                    -(t["recall"] if t["recall"] is not None else -1),
                    abs(t["t_uses_voicebank"] - 0.5) + abs(t["t_is_vocal_work"] - 0.5), t["t_uses_voicebank"], t["t_is_vocal_work"])

        best = min(eligible, key=key) if eligible else None
        proposal["providers"][name] = {
            "model": settings["model"],
            "t_uses_voicebank": best["t_uses_voicebank"] if best else None,
            "t_is_vocal_work": best["t_is_vocal_work"] if best else None,
            "dev_items_inferred": sum(1 for i in items if (snap.get(i["snapshot_id"]) or {}).get("status") == OK),
            "note": None if best else "no grid point meets the false-positive-rate limit on dev",
        }
        grids[name] = table
    return {"proposal": proposal, "grid": grids}


def load_thresholds(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("format") != THRESHOLDS_FORMAT:
        raise StoreError(f"{path}: not a threshold configuration")
    return config


def freeze_thresholds(store: EvalStore, name: str, config: dict, clock: Clock = utc_now) -> dict:
    """Store a threshold configuration under a name, once. Rewrites are refused."""
    run = store.run(config.get("run_id", ""))
    _check_run_identity(config, run)
    if config.get("selected_on_split") != "dev":
        raise StoreError("only thresholds selected on the dev split can be frozen")
    for provider, values in config.get("providers", {}).items():
        if provider not in run["providers"]:
            raise StoreError(f"provider {provider!r} is not part of the run")
        for flag in FLAGS:
            t = values.get(f"t_{flag}")
            if not isinstance(t, (int, float)) or isinstance(t, bool) or not 0.0 <= t <= 1.0:
                raise StoreError(f"{provider}: t_{flag} must be a number in [0, 1] before freezing")
    content = canonical_json(config)
    digest = sha256_bytes(content.encode("utf-8"))
    existing = store.conn.execute("SELECT * FROM frozen_thresholds WHERE name = ?", (name,)).fetchone()
    if existing is not None:
        if existing["sha256"] != digest:
            raise StoreError(f"frozen thresholds {name!r} already exist with different content; frozen configurations cannot be rewritten")
        return {"name": name, "sha256": digest, "frozen_at": existing["frozen_at"], "already_frozen": True}
    frozen_at = clock()
    with store.transaction() as conn:
        conn.execute("INSERT INTO frozen_thresholds VALUES (?,?,?,?,?)", (name, run["run_id"], content, digest, frozen_at))
    return {"name": name, "sha256": digest, "frozen_at": frozen_at, "already_frozen": False}


def frozen(store: EvalStore, name: str) -> dict:
    row = store.conn.execute("SELECT * FROM frozen_thresholds WHERE name = ?", (name,)).fetchone()
    if row is None:
        raise StoreError(f"no frozen thresholds named {name!r}")
    if sha256_bytes(row["content"].encode("utf-8")) != row["sha256"]:
        raise StoreError(f"frozen thresholds {name!r} fail their hash check")
    return {"config": json.loads(row["content"]), "name": name, "sha256": row["sha256"], "frozen_at": row["frozen_at"]}


# -- report ----------------------------------------------------------------------


def _usage_block(rows: List[dict], attempts: List[dict]) -> dict:
    sent = [a for a in attempts if a["sent"]]
    ok = [a for a in sent if a["status"] == OK]
    usage_cost = sum(a["cost_usd"] for a in sent if a["cost_usd"] is not None)
    estimated_cost = sum(a["estimated_cost_usd"] for a in sent if a["cost_usd"] is None)
    statuses: Dict[str, int] = {}
    for a in attempts:
        statuses[a["status"]] = statuses.get(a["status"], 0) + 1
    end_to_end = []
    for r in rows:
        slot = r["slot"]
        if slot and r["probabilities"]:
            first, last = slot["attempts"][0], slot["final"]
            end_to_end.append((_parse_time(last["finished_at"]) - _parse_time(first["started_at"])).total_seconds() * 1000.0)
    items_ok = sum(1 for r in rows if r["probabilities"])
    total_cost = usage_cost + estimated_cost
    return {
        "attempts_sent": len(sent),
        "attempts_not_sent": len(attempts) - len(sent),
        "statuses": statuses,
        "attempt_failure_rate": ratio(len(sent) - len(ok), len(sent)),
        "retries": sum(1 for a in sent if a["retry_of"] is not None),
        "input_tokens_reported": sum(a["input_tokens"] or 0 for a in sent),
        "cost_usd_from_usage": round(usage_cost, 9),
        "cost_usd_estimated_without_usage": round(estimated_cost, 9),
        "cost_usd_total": round(total_cost, 9),
        "cost_usd_per_1000_successful_items": round(total_cost / items_ok * 1000, 6) if items_ok else None,
        "latency_ms_successful_attempts": {"n": len(ok), "p50": percentile([a["latency_ms"] for a in ok], 50), "p95": percentile([a["latency_ms"] for a in ok], 95)},
        "end_to_end_ms_including_retries": {"n": len(end_to_end), "p50": percentile(end_to_end, 50), "p95": percentile(end_to_end, 95)},
        "returned_models": sorted({a["returned_model"] for a in ok if a["returned_model"]}),
        "server_compute_unknown": sum(a["server_compute_unknown"] for a in sent),
    }


def _stability(store: EvalStore, run_id: str, provider: str, items: List[dict], main: Dict[str, dict], thresholds: Dict[str, float]) -> List[dict]:
    repeats = sorted({a["repeat_id"] for a in store.attempts(run_id) if a["provider"] == provider and a["repeat_id"] != DEFAULT_REPEAT})
    out = []
    ids = {i["snapshot_id"] for i in items}
    for repeat_id in repeats:
        other = decisions(store, run_id, repeat_id).get(provider, {})
        pairs = [(main[s]["probabilities"], other[s]["probabilities"]) for s in sorted(ids)
                 if s in main and s in other and main[s]["probabilities"] and other[s]["probabilities"]]
        diffs = {f: [abs(a[f] - b[f]) for a, b in pairs] for f in FLAGS}
        flips = sum(1 for a, b in pairs if _predict(a, thresholds)["joint"] != _predict(b, thresholds)["joint"])
        out.append({
            "repeat_id": repeat_id, "paired_items": len(pairs), "joint_decision_flips": flips,
            "max_abs_diff": {f: round(max(d), 6) if d else None for f, d in diffs.items()},
            "mean_abs_diff": {f: round(sum(d) / len(d), 6) if d else None for f, d in diffs.items()},
        })
    return out


def build_report(store: EvalStore, run_id: str, split: str, thresholds_config: dict, frozen_info: Optional[dict] = None) -> dict:
    """Compute the full report for one split (``dev``, ``acceptance`` or ``all``)."""
    run = store.run(run_id)
    _check_run_identity(thresholds_config, run)
    if split in ("acceptance", "all") and frozen_info is None:
        raise StoreError("reports that include the acceptance split need frozen thresholds (--frozen NAME)")
    manifest = run["manifest"]
    scope = None if split == "all" else split
    selected = [i for i in manifest["items"] if scope is None or i["split"] == scope]
    items = model_items(manifest, scope)
    popular = set(manifest["popular_groups"])
    attempts_all = store.attempts(run_id)
    item_ids = {i["snapshot_id"] for i in items}
    per_provider = decisions(store, run_id)

    denominators = {
        "selected": len(selected),
        "acquisition_failed": sum(1 for i in selected if not i.get("snapshot_id")),
        "with_snapshot": len(items),
        "unannotated": sum(1 for i in items if i["status"] == "unannotated"),
        "human_known": {k: sum(1 for i in items if _truth(i)[k] is not None) for k in ("uses_voicebank", "is_vocal_work", "joint")},
        "human_unknown": {k: sum(1 for i in items if _truth(i)[k] is None) for k in ("uses_voicebank", "is_vocal_work", "joint")},
    }
    providers = {}
    for name in sorted(run["providers"]):
        values = (thresholds_config.get("providers") or {}).get(name)
        if not values or values.get("t_uses_voicebank") is None or values.get("t_is_vocal_work") is None:
            providers[name] = {"thresholds": None, "note": "no thresholds for this provider"}
            continue
        thresholds = {f: values[f"t_{f}"] for f in FLAGS}
        snap = per_provider.get(name, {})
        rows = _rows(items, snap, thresholds)
        groups = {}
        group_names = sorted({i["group"] for i in items})
        for g in group_names:
            g_rows = [r for r in rows if r["item"]["group"] == g]
            groups[f"group:{g}"] = {"items": len(g_rows), "joint": _content_block(g_rows)["joint"]}
        for evidence in ("insufficient", "uncertain"):
            e_rows = [r for r in rows if r["item"].get("text_evidence") == evidence]
            groups[f"text_evidence:{evidence}"] = {"items": len(e_rows), "joint": _content_block(e_rows)["joint"]}
        if split == "all":
            for s in ("dev", "acceptance"):
                s_rows = [r for r in rows if r["item"]["split"] == s]
                groups[f"split:{s}"] = {"items": len(s_rows), "joint": _content_block(s_rows)["joint"]}
        inference = {}
        for r in rows:
            inference[r["inference"]] = inference.get(r["inference"], 0) + 1
        provider_attempts = [a for a in attempts_all if a["provider"] == name and a["snapshot_id"] in item_ids and a["repeat_id"] == DEFAULT_REPEAT]
        providers[name] = {
            "model_requested": run["providers"][name]["model"],
            "thresholds": thresholds,
            "inference": {"ok": inference.get(OK, 0), "failed": sum(v for k, v in inference.items() if k != OK), "by_status": inference},
            "content": _content_block(rows),
            "popular": _popular_block([r for r in rows if r["item"]["group"] in popular]),
            "groups": groups,
            "usage": _usage_block(rows, provider_attempts),
            "repeat_stability": _stability(store, run_id, name, items, snap, thresholds),
            "errors": _errors(rows),
        }
    body = {
        "format": REPORT_FORMAT,
        "tool_version": __version__,
        "run": {k: run[k] for k in ("run_id", "code_revision", "dataset_version", "manifest_sha256", "question_set_id", "questions_sha256", "settings_sha256")},
        "split": split,
        "popular_groups": sorted(popular),
        "thresholds": {"sha256": sha256_bytes(canonical_json(thresholds_config).encode("utf-8")),
                       "frozen": frozen_info and {k: frozen_info[k] for k in ("name", "sha256", "frozen_at")},
                       "selection_rule": thresholds_config.get("selection_rule"),
                       "acceptance_criteria": thresholds_config.get("acceptance_criteria")},
        "denominators": denominators,
        "providers": providers,
        "limitations": [
            "Probabilities are not calibrated guarantees; equal values from two providers do not mean equal confidence.",
            "Costs use list prices for reported input tokens; cached-input discounts and cache-write charges are not modelled. Attempts without usage fall back to the conservative pre-call estimate.",
            "Latency figures are exploratory and depend on the machine, network and time of the run.",
            "Results describe this sample only; they are not a site-wide recall estimate, and zero misses in a small sample does not guarantee zero misses later.",
        ],
    }
    return {**body, "report_sha256": sha256_bytes(canonical_json(body).encode("utf-8"))}


# -- markdown --------------------------------------------------------------------


def _fmt_ratio(r: dict) -> str:
    if r["value"] is None:
        return f"N/A (0/{r['denominator']})"
    lo, hi = r["wilson95"]
    return f"{r['count']}/{r['denominator']} = {r['value']:.3f} [{lo:.3f}, {hi:.3f}]"


def render_markdown(report: dict) -> str:
    run = report["run"]
    lines = [
        f"# Evaluation report: {run['run_id']} ({report['split']})", "",
        f"- Dataset `{run['dataset_version']}`, manifest `{run['manifest_sha256'][:16]}`",
        f"- Questions `{run['question_set_id']}` `{run['questions_sha256'][:16]}`, code `{run['code_revision']}`",
        f"- Thresholds `{report['thresholds']['sha256'][:16]}`" + (f", frozen as `{report['thresholds']['frozen']['name']}`" if report["thresholds"]["frozen"] else ", not frozen"),
        f"- Report `{report['report_sha256'][:16]}`", "",
        "Intervals are 95% Wilson intervals. Human `null` labels are unknown, not negative.", "",
        "## Denominators", "",
    ]
    d = report["denominators"]
    lines += [
        f"- Selected {d['selected']}, acquisition failed {d['acquisition_failed']}, with snapshot {d['with_snapshot']}, unannotated {d['unannotated']}",
        "- Human known / unknown: " + ", ".join(f"{k} {d['human_known'][k]}/{d['human_unknown'][k]}" for k in ("uses_voicebank", "is_vocal_work", "joint")),
        "",
    ]
    for name, p in report["providers"].items():
        lines += [f"## {name}", ""]
        if p.get("thresholds") is None:
            lines += [p.get("note", ""), ""]
            continue
        t = p["thresholds"]
        lines += [f"Model `{p['model_requested']}`, returned {', '.join(p['usage']['returned_models']) or 'none'}; thresholds voicebank {t['uses_voicebank']}, work {t['is_vocal_work']}.", "",
                  f"Inference ok {p['inference']['ok']}, failed {p['inference']['failed']} {json.dumps(p['inference']['by_status'], sort_keys=True)}", "",
                  "| Label | TP | FP | FN | TN | Recall | False-positive rate | Precision | Unknown | Failed |", "|---|---|---|---|---|---|---|---|---|---|"]
        for key in ("uses_voicebank", "is_vocal_work", "joint"):
            c = p["content"][key]
            cm = c["confusion"]
            lines.append(f"| {key} | {cm['tp']} | {cm['fp']} | {cm['fn']} | {cm['tn']} | {_fmt_ratio(c['recall'])} | {_fmt_ratio(c['false_positive_rate'])} | {_fmt_ratio(c['precision'])} | {c['human_unknown']} | {c['inference_failed_with_known_label']} |")
        pop = p["popular"]
        lines += ["", f"Popular ({', '.join(report['popular_groups']) or 'none'}): known positives {pop['known_positives']}, missed {pop['missed']}, recall {_fmt_ratio(pop['recall'])}; "
                  f"inference failed {pop['inference_failed']}, process recall (failures count as not included) {_fmt_ratio(pop['process_recall'])}.", "",
                  "| Group | Items | TP | FP | FN | TN | Recall | False-positive rate |", "|---|---|---|---|---|---|---|---|"]
        for g, block in p["groups"].items():
            cm = block["joint"]["confusion"]
            lines.append(f"| {g} | {block['items']} | {cm['tp']} | {cm['fp']} | {cm['fn']} | {cm['tn']} | {_fmt_ratio(block['joint']['recall'])} | {_fmt_ratio(block['joint']['false_positive_rate'])} |")
        u = p["usage"]
        lines += ["", f"Usage: {u['attempts_sent']} attempts sent ({u['retries']} retries, {u['attempts_not_sent']} not sent), failure rate {_fmt_ratio(u['attempt_failure_rate'])}, "
                  f"{u['input_tokens_reported']} input tokens, cost ${u['cost_usd_total']:.6f} (usage ${u['cost_usd_from_usage']:.6f}, estimated ${u['cost_usd_estimated_without_usage']:.6f}), "
                  f"per 1000 items {u['cost_usd_per_1000_successful_items']}.",
                  f"Latency (exploratory): successful attempts n={u['latency_ms_successful_attempts']['n']} p50={u['latency_ms_successful_attempts']['p50']} ms p95={u['latency_ms_successful_attempts']['p95']} ms; "
                  f"end to end with retries n={u['end_to_end_ms_including_retries']['n']} p50={u['end_to_end_ms_including_retries']['p50']} ms p95={u['end_to_end_ms_including_retries']['p95']} ms.", ""]
        for s in p["repeat_stability"]:
            lines.append(f"Repeat `{s['repeat_id']}`: {s['paired_items']} paired, {s['joint_decision_flips']} joint decision flips, max |Δp| {s['max_abs_diff']}.")
        errors = p["errors"]
        lines += ["", "Errors:", ""]
        for kind in ("false_negative", "false_positive", "flag_errors", "inference_failures"):
            refs = errors[kind]
            lines.append(f"- {kind}: {len(refs)}" + ("" if not refs else " — " + ", ".join(f"{r['aid']} ({r['group']})" for r in refs)))
        lines.append("")
    lines += ["## Limitations", ""] + [f"- {x}" for x in report["limitations"]]
    return "\n".join(lines) + "\n"


def write_report(report: dict, out: Path, clock: Clock = utc_now) -> dict:
    if out.exists() and any(out.iterdir()):
        raise StoreError(f"report directory {out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    stamped = {**report, "generated_at": clock()}
    (out / "report.json").write_text(json.dumps(stamped, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
    return {"out": str(out), "report_sha256": report["report_sha256"]}
