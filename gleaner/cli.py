"""Command-line interface: ``python -m gleaner --data-dir DIR <command>``."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__, bilibili
from .acquire import Fetcher, candidate_aids, candidates_without_snapshot, import_candidates, status_summary
from .annotations import export_annotations, import_annotations
from .bundle import export_bundle, restore_bundle, verify_bundle, verify_store
from .store import StoreError, init_store, open_store, resolve_data_dir
from .util import parse_aid
from . import evalcli


ENDPOINTS_CONFIG_ENV = "GLEANER_ENDPOINTS_CONFIG"


def _print(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _aids(values: Optional[List[str]]) -> Optional[List[int]]:
    return [parse_aid(v) for v in values] if values else None


def _transport(args):
    if args.fixtures:
        return bilibili.fixture_transport(Path(args.fixtures))
    config_path = args.endpoints_config or os.environ.get(ENDPOINTS_CONFIG_ENV)
    config = None
    if config_path:
        path = Path(config_path).expanduser().resolve()
        resolve_data_dir(str(path.parent))  # the private config must live outside Git work trees
        config = bilibili.load_endpoints_config(path)
    return bilibili.http_transport(config, timeout=args.timeout)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gleaner", description="Experiment store and annotation tools.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--data-dir", help="experiment data directory (or set GLEANER_DATA_DIR); required")
    parser.add_argument("--allow-git-dir", action="store_true", help="allow data directories and exports inside a Git work tree")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create an empty data directory")

    p = sub.add_parser("import-candidates", help="import an aid list from CSV")
    p.add_argument("csv", type=Path)
    p.add_argument("--source", help="source name for rows without a 'source' column")

    for name, help_text in (("fetch", "acquire candidates that have no snapshot"), ("refresh", "start a new acquisition for given aids")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--aid", action="append", required=(name == "refresh"), help="limit to this aid (repeatable)")
        p.add_argument("--max-requests", type=int, required=True, help="hard cap on endpoint requests in this run")
        p.add_argument("--delay", type=float, default=1.0, help="seconds between requests (default 1.0)")
        p.add_argument("--max-attempts", type=int, default=2, help="attempts per endpoint per acquisition (default 2)")
        p.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout in seconds")
        p.add_argument("--endpoints-config", help=f"private workers and User-Agent config (or set {ENDPOINTS_CONFIG_ENV})")
        p.add_argument("--fixtures", help="serve responses from <dir>/<endpoint>/<aid>.json instead of the network")
        if name == "fetch":
            p.add_argument("--retry", action="store_true", help="also retry acquisitions with retryable failures")

    p = sub.add_parser("status", help="summarise candidates, attempts, snapshots, and annotations")
    p.add_argument("--max-attempts", type=int, default=2, help="attempts per endpoint used to judge retryable vs failed (default 2)")
    sub.add_parser("verify", help="check database integrity, raw hashes, and snapshot inputs")

    p = sub.add_parser("export-annotations", help="write an annotation CSV")
    p.add_argument("csv", type=Path)
    p.add_argument("--all-snapshots", action="store_true", help="include earlier snapshots, not only the latest per aid")

    p = sub.add_parser("import-annotations", help="append annotation revisions from a CSV")
    p.add_argument("csv", type=Path)
    p.add_argument("--rules-version", required=True, help="version of the annotation rules used")
    p.add_argument("--annotator", help="annotator for rows without an 'annotator' value")

    p = sub.add_parser("export-bundle", help="export a verifiable data bundle")
    p.add_argument("out", type=Path)

    p = sub.add_parser("restore-bundle", help="restore a bundle into an empty data directory and verify it")
    p.add_argument("bundle", type=Path)

    p = sub.add_parser("verify-bundle", help="check a bundle's manifest hashes without restoring it")
    p.add_argument("bundle", type=Path)

    evalcli.add_parsers(sub)
    return parser


def run(args) -> int:
    if args.command == "verify-bundle":
        problems = verify_bundle(args.bundle)
        _print({"bundle": str(args.bundle), "problems": problems})
        return 1 if problems else 0
    if args.command in evalcli.COMMANDS:
        return evalcli.run(args)

    data_dir = resolve_data_dir(args.data_dir, allow_git=args.allow_git_dir)

    if args.command == "init":
        init_store(data_dir)
        _print({"data_dir": str(data_dir), "initialized": True})
        return 0
    if args.command == "restore-bundle":
        _print(restore_bundle(args.bundle, data_dir))
        return 0

    with open_store(data_dir) as store:
        if args.command == "import-candidates":
            _print(import_candidates(store, args.csv, args.source))
        elif args.command in ("fetch", "refresh"):
            fetcher = Fetcher(store, _transport(args), max_requests=args.max_requests, delay=args.delay, max_attempts=args.max_attempts)
            if args.command == "fetch":
                report = fetcher.fetch(candidates_without_snapshot(store, _aids(args.aid)), retry=args.retry)
            else:
                report = fetcher.refresh(candidate_aids(store, _aids(args.aid)))
            _print(report.as_dict())
        elif args.command == "status":
            _print(status_summary(store, args.max_attempts))
        elif args.command == "verify":
            problems = verify_store(store)
            _print({"problems": problems})
            return 1 if problems else 0
        elif args.command == "export-annotations":
            resolve_data_dir(str(args.csv.parent), allow_git=args.allow_git_dir)
            _print({"rows": export_annotations(store, args.csv, args.all_snapshots), "csv": str(args.csv)})
        elif args.command == "import-annotations":
            result = import_annotations(store, args.csv, args.rules_version, args.annotator)
            _print(result)
            return 1 if result["rejected"] else 0
        elif args.command == "export-bundle":
            resolve_data_dir(str(args.out), allow_git=args.allow_git_dir)
            _print(export_bundle(store, args.out))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (StoreError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
