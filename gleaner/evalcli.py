"""Command-line entry points for dataset manifests and model evaluation."""

import json
import os
from pathlib import Path

from .dataset import build_manifest, load_manifest, write_manifest
from .evalstore import (DEFAULT_REPEAT, EVAL_DIR_ENV, Budget, Runner, ensure_run, eval_status, load_credentials,
                        load_questions, open_eval_store, provider_settings)
from .evaluate import build_report, freeze_thresholds, frozen, load_thresholds, select_thresholds, write_report
from .providers import PROVIDERS, http_transport, synthetic_transport
from .store import StoreError, resolve_data_dir
from .util import parse_aid

COMMANDS = {"dataset-manifest", "eval-run", "eval-status", "eval-select", "eval-freeze", "eval-report"}
CREDENTIALS_ENV = "GLEANER_CREDENTIALS"


def _print(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _outside_git(path: Path, allow_git: bool) -> Path:
    resolve_data_dir(str(path), allow_git=allow_git)
    return path


def _eval_dir(args) -> Path:
    raw = args.eval_dir or os.environ.get(EVAL_DIR_ENV)
    if not raw:
        raise StoreError(f"no evaluation directory given: pass --eval-dir or set {EVAL_DIR_ENV}")
    return resolve_data_dir(raw, allow_git=args.allow_git_dir)


def _key_values(values, cast=str):
    out = {}
    for value in values or []:
        if "=" not in value:
            raise StoreError(f"expected PROVIDER=VALUE, got {value!r}")
        k, v = value.split("=", 1)
        out[k.strip()] = cast(v.strip())
    return out


def add_parsers(sub) -> None:
    p = sub.add_parser("dataset-manifest", help="build a hashed dataset manifest from a bundle and a selection CSV")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True, help="CSV with aid and optional family, split, group, snapshot_id, annotation_revision")
    p.add_argument("--dataset-version", required=True)
    p.add_argument("--seed", required=True, help="seed for the family split")
    p.add_argument("--acceptance-fraction", type=float, required=True, help="share of unforced families assigned to acceptance")
    p.add_argument("--popular-group", action="append", default=[], help="sample group counted as popular (repeatable)")
    p.add_argument("--out", type=Path, required=True)

    def eval_dir(q):
        q.add_argument("--eval-dir", help=f"evaluation directory (or set {EVAL_DIR_ENV}); required")

    p = sub.add_parser("eval-run", help="send model requests for a run, resuming earlier work")
    eval_dir(p)
    p.add_argument("--run-id", required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--questions", type=Path, required=True)
    p.add_argument("--provider", action="append", required=True, choices=sorted(PROVIDERS), help="provider to call (repeatable)")
    p.add_argument("--model", action="append", help="PROVIDER=MODEL; fixed when the run is created")
    p.add_argument("--price", action="append", help="PROVIDER=USD per million input tokens, for estimates")
    p.add_argument("--max-calls", type=int, required=True, help="cap on sent attempts across the whole evaluation directory")
    p.add_argument("--max-calls-per-provider", type=int, required=True)
    p.add_argument("--max-cost-usd", type=float, required=True)
    p.add_argument("--max-estimated-tokens", type=int, default=60000, help="do not send requests estimated above this many input tokens")
    p.add_argument("--split", choices=("dev", "acceptance"))
    p.add_argument("--aid", action="append")
    p.add_argument("--limit", type=int)
    p.add_argument("--repeat-id", default=DEFAULT_REPEAT, help="use a new id for deliberate repeat measurements")
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--delay", type=float, default=0.5)
    p.add_argument("--retry-delay", type=float, default=2.0)
    p.add_argument("--credentials", help=f"private JSON file of API keys outside Git (or set {CREDENTIALS_ENV}); environment variables win")
    p.add_argument("--synthetic", action="store_true", help="answer offline with meaningless deterministic probabilities")

    p = sub.add_parser("eval-status", help="summarise runs, attempts, spend, and frozen thresholds")
    eval_dir(p)

    p = sub.add_parser("eval-select", help="grid-search thresholds on the dev split")
    eval_dir(p)
    p.add_argument("--run-id", required=True)
    p.add_argument("--max-false-positive-rate", type=float, required=True)
    p.add_argument("--grid", help="comma-separated threshold values (default 0.05..0.95 step 0.05)")
    p.add_argument("--out", type=Path, required=True, help="directory for thresholds.json and grid.json")

    p = sub.add_parser("eval-freeze", help="freeze a threshold configuration under a name")
    eval_dir(p)
    p.add_argument("--name", required=True)
    p.add_argument("--thresholds", type=Path, required=True)

    p = sub.add_parser("eval-report", help="write the offline evaluation report")
    eval_dir(p)
    p.add_argument("--run-id", required=True)
    p.add_argument("--split", choices=("dev", "acceptance", "all"), required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--thresholds", type=Path, help="threshold file (dev split only)")
    group.add_argument("--frozen", help="name of frozen thresholds")
    p.add_argument("--out", type=Path, required=True)


def run(args) -> int:
    allow = args.allow_git_dir
    if args.command == "dataset-manifest":
        _outside_git(args.out.parent.resolve(), allow)
        manifest = build_manifest(args.bundle, args.selection, args.dataset_version, args.seed, args.acceptance_fraction, args.popular_group)
        digest = write_manifest(manifest, args.out)
        splits = {}
        for item in manifest["items"]:
            splits[item["split"]] = splits.get(item["split"], 0) + 1
        _print({"manifest": str(args.out), "manifest_sha256": digest, "items": len(manifest["items"]), "splits": splits,
                "statuses": {s: sum(1 for i in manifest["items"] if i["status"] == s) for s in ("ok", "unannotated", "no_snapshot")}})
        return 0

    root = _eval_dir(args)
    with open_eval_store(root) as store:
        if args.command == "eval-status":
            _print(eval_status(store))
        elif args.command == "eval-run":
            manifest, manifest_sha = load_manifest(args.manifest)
            question_set = load_questions(args.questions)
            existing = store.conn.execute("SELECT providers_text FROM run WHERE run_id = ?", (args.run_id,)).fetchone()
            models, prices = _key_values(args.model), _key_values(args.price, float)
            if existing is not None:
                # Settings are fixed when a run is created; flags may only repeat them.
                settings = json.loads(existing["providers_text"])
                for name, model in models.items():
                    if name in settings and settings[name]["model"] != model:
                        raise StoreError(f"run {args.run_id!r} uses model {settings[name]['model']!r} for {name}; use a new run id")
                for name, price in prices.items():
                    if name in settings and settings[name]["input_price_usd_per_mtok"] != price:
                        raise StoreError(f"run {args.run_id!r} uses a different price for {name}; use a new run id")
            else:
                settings = provider_settings(args.provider, models, prices)
            run_rec = ensure_run(store, args.run_id, manifest, manifest_sha, question_set, settings)
            if args.synthetic:
                transports = {n: synthetic_transport(n) for n in args.provider}
                credentials = {}
            else:
                cred_path = args.credentials or os.environ.get(CREDENTIALS_ENV)
                path = None
                if cred_path:
                    path = Path(cred_path).expanduser().resolve()
                    resolve_data_dir(str(path.parent))  # the private file must live outside Git work trees
                credentials = load_credentials(args.provider, path)
                transports = {n: http_transport for n in args.provider}
            budget = Budget(args.max_calls, args.max_calls_per_provider, args.max_cost_usd)
            runner = Runner(store, run_rec, transports, credentials, budget, timeout=args.timeout, delay=args.delay,
                            retry_delay=args.retry_delay, max_estimated_tokens=args.max_estimated_tokens)
            aids = [parse_aid(a) for a in args.aid] if args.aid else None
            report = runner.execute(args.provider, split=args.split, aids=aids, limit=args.limit, repeat_id=args.repeat_id)
            _print({**report.as_dict(), "spent": budget.spent(store)})
        elif args.command == "eval-select":
            out = _outside_git(args.out.resolve(), allow)
            if out.exists() and any(out.iterdir()):
                raise StoreError(f"{out} is not empty")
            grid = [float(x) for x in args.grid.split(",")] if args.grid else None
            result = select_thresholds(store, args.run_id, args.max_false_positive_rate, grid)
            out.mkdir(parents=True, exist_ok=True)
            (out / "thresholds.json").write_text(json.dumps(result["proposal"], ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            (out / "grid.json").write_text(json.dumps(result["grid"], ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            _print({"thresholds": str(out / "thresholds.json"), "providers": result["proposal"]["providers"]})
        elif args.command == "eval-freeze":
            _print(freeze_thresholds(store, args.name, load_thresholds(args.thresholds)))
        elif args.command == "eval-report":
            out = _outside_git(args.out.resolve(), allow)
            if args.frozen:
                info = frozen(store, args.frozen)
                report = build_report(store, args.run_id, args.split, info["config"], info)
            else:
                report = build_report(store, args.run_id, args.split, load_thresholds(args.thresholds))
            _print(write_report(report, out))
    return 0
