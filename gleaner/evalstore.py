"""Evaluation store and runner: durable model-run records with a budget stop.

An evaluation directory holds one SQLite database (``eval.sqlite``) with:

* ``run`` -- one row per run: the dataset manifest (full text and SHA-256),
  code revision, question set (full text and SHA-256), and per-provider
  settings (requested model, URL, price used for estimates);
* ``attempt`` -- one row per outbound request (and per request that was not
  sent because of an input limit): request body and non-secret headers,
  response status, headers and body, parsed probabilities, returned model,
  usage, timing, estimated and usage-based cost, errors, and retry link;
* ``frozen_thresholds`` -- named threshold configurations, frozen once.

All three tables are append-only. Credentials are read from the environment
or from a private file outside Git and are never written anywhere.

The budget counts every sent attempt in the directory, across all runs, so a
stage budget is not reset by starting a new run.
"""

import json
import os
import sqlite3
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

import fcntl

from . import __version__
from .dataset import model_items
from .providers import FLAGS, OK, PROVIDERS, RETRYABLE, STOP_PROVIDER, HttpResult, Parsed, Transport
from .store import StoreError
from .util import Clock, canonical_json, sha256_bytes, utc_now

EVAL_DB = "eval.sqlite"
EVAL_SCHEMA_VERSION = 1
EVAL_DIR_ENV = "GLEANER_EVAL_DIR"
MAX_ATTEMPTS = 2  # the first request plus at most one retry
DEFAULT_REPEAT = "main"

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE run (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    tool_version TEXT NOT NULL,
    code_revision TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    manifest_text TEXT NOT NULL,
    question_set_id TEXT NOT NULL,
    questions_text TEXT NOT NULL,
    questions_sha256 TEXT NOT NULL,
    providers_text TEXT NOT NULL,
    settings_sha256 TEXT NOT NULL
);

CREATE TABLE attempt (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES run(run_id),
    provider TEXT NOT NULL,
    request_model TEXT NOT NULL,
    aid INTEGER NOT NULL,
    snapshot_id TEXT NOT NULL,
    input_sha256 TEXT NOT NULL,
    repeat_id TEXT NOT NULL,
    attempt_no INTEGER NOT NULL,
    retry_of INTEGER REFERENCES attempt(id),
    code_revision TEXT NOT NULL,
    sent INTEGER NOT NULL CHECK (sent IN (0, 1)),
    url TEXT NOT NULL,
    request_headers TEXT NOT NULL,
    request_body TEXT NOT NULL,
    request_sha256 TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    latency_ms REAL,
    http_status INTEGER,
    response_headers TEXT,
    response_body BLOB,
    status TEXT NOT NULL,
    error_class TEXT,
    error_detail TEXT,
    returned_model TEXT,
    p_uses_voicebank REAL,
    p_is_vocal_work REAL,
    usage TEXT,
    input_tokens INTEGER,
    estimated_input_tokens INTEGER NOT NULL,
    estimated_cost_usd REAL NOT NULL,
    cost_usd REAL,
    server_compute_unknown INTEGER NOT NULL DEFAULT 0,
    UNIQUE (run_id, provider, snapshot_id, repeat_id, attempt_no)
);

CREATE TABLE frozen_thresholds (
    name TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES run(run_id),
    content TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    frozen_at TEXT NOT NULL
);
"""

APPEND_ONLY = ("run", "attempt", "frozen_thresholds")


def _triggers() -> str:
    return "\n".join(
        f"CREATE TRIGGER {t}_no_{a.lower()} BEFORE {a} ON {t} BEGIN SELECT RAISE(ABORT, '{t} is append-only'); END;"
        for t in APPEND_ONLY for a in ("UPDATE", "DELETE")
    )


class EvalStore:
    def __init__(self, root: Path, conn: sqlite3.Connection):
        self.root = root
        self.conn = conn

    @contextmanager
    def transaction(self):
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self.conn
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")

    def run(self, run_id: str) -> dict:
        row = self.conn.execute("SELECT * FROM run WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise StoreError(f"no run {run_id!r} in {self.root}")
        run = dict(row)
        run["manifest"] = json.loads(run["manifest_text"])
        run["questions"] = json.loads(run["questions_text"])
        run["providers"] = json.loads(run["providers_text"])
        return run

    def attempts(self, run_id: str) -> List[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM attempt WHERE run_id = ? ORDER BY id", (run_id,))]


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = DELETE")
    return conn


@contextmanager
def open_eval_store(root: Path) -> Iterator[EvalStore]:
    """Open (creating if needed) an evaluation directory with an exclusive lock."""
    root.mkdir(parents=True, exist_ok=True)
    lock = open(root / ".lock", "a+")
    try:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise StoreError(f"another process is using {root}") from None
        db = root / EVAL_DB
        fresh = not db.exists()
        conn = _connect(db)
        try:
            if fresh:
                conn.executescript("BEGIN;" + SCHEMA + _triggers() + "COMMIT;")
                conn.execute("INSERT INTO meta VALUES ('schema_version', ?)", (str(EVAL_SCHEMA_VERSION),))
            row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
            if row is None or int(row["value"]) != EVAL_SCHEMA_VERSION:
                raise StoreError(f"unsupported evaluation schema in {db}")
            yield EvalStore(root, conn)
        finally:
            conn.close()
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


# -- versions --------------------------------------------------------------


def code_revision() -> str:
    """Git commit of the running code, with ``-dirty`` for local changes."""
    here = Path(__file__).resolve().parent
    try:
        head = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        if head.returncode != 0:
            return f"unknown+{__version__}"
        dirty = subprocess.run(["git", "-C", str(here), "status", "--porcelain", "--", "."], capture_output=True, text=True, timeout=5)
        return head.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return f"unknown+{__version__}"


def load_questions(path: Path) -> dict:
    """A question set: ``{"question_set_id": ..., "questions": {flag: text}}``."""
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = data.get("questions")
    if not isinstance(data.get("question_set_id"), str) or not data["question_set_id"]:
        raise StoreError(f"{path}: question_set_id is required")
    if not isinstance(questions, dict) or set(questions) != set(FLAGS):
        raise StoreError(f"{path}: questions must have exactly {list(FLAGS)}")
    for name, text in questions.items():
        if not isinstance(text, str) or not text.strip():
            raise StoreError(f"{path}: question {name!r} must be non-empty text")
    return {"question_set_id": data["question_set_id"], "questions": {f: questions[f] for f in FLAGS}}


def questions_sha256(question_set: dict) -> str:
    return sha256_bytes(canonical_json(question_set).encode("utf-8"))


def provider_settings(names: List[str], models: Dict[str, str], prices: Dict[str, float]) -> Dict[str, dict]:
    settings = {}
    for name in names:
        if name not in PROVIDERS:
            raise StoreError(f"unknown provider {name!r}; known: {sorted(PROVIDERS)}")
        p = PROVIDERS[name]
        settings[name] = {
            "model": models.get(name) or p.default_model,
            "url": p.url,
            "input_price_usd_per_mtok": prices.get(name, p.default_input_price),
        }
    return settings


def ensure_run(store: EvalStore, run_id: str, manifest: dict, manifest_sha: str, question_set: dict,
               providers: Dict[str, dict], clock: Clock = utc_now, revision: Optional[str] = None) -> dict:
    """Create the run, or check that an existing run has the same identity."""
    q_sha = questions_sha256(question_set)
    settings_sha = sha256_bytes(canonical_json(providers).encode("utf-8"))
    row = store.conn.execute("SELECT * FROM run WHERE run_id = ?", (run_id,)).fetchone()
    if row is not None:
        mismatches = []
        if row["manifest_sha256"] != manifest_sha:
            mismatches.append("dataset manifest")
        if row["questions_sha256"] != q_sha:
            mismatches.append("question set")
        if row["settings_sha256"] != settings_sha:
            mismatches.append("provider settings (models, URLs or prices)")
        if mismatches:
            raise StoreError(f"run {run_id!r} already exists with a different {', '.join(mismatches)}; use a new run id")
        return store.run(run_id)
    with store.transaction() as conn:
        conn.execute(
            "INSERT INTO run VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, clock(), __version__, revision or code_revision(), manifest["dataset_version"], manifest_sha,
             canonical_json(manifest), question_set["question_set_id"], canonical_json(question_set), q_sha,
             canonical_json(providers), settings_sha),
        )
    return store.run(run_id)


# -- credentials -------------------------------------------------------------


def load_credentials(names: List[str], config_path: Optional[Path]) -> Dict[str, str]:
    """API keys from ``<PROVIDER>_API_KEY`` variables or a private JSON file.

    The file maps provider name to key: ``{"jev": "...", "openai": "..."}``.
    Environment variables win. Keys stay in memory only.
    """
    from_file: Dict[str, str] = {}
    if config_path is not None:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        from_file = {k: v for k, v in data.items() if isinstance(v, str)}
    keys, missing = {}, []
    for name in names:
        key = os.environ.get(PROVIDERS[name].credential_env) or from_file.get(name)
        if not key:
            missing.append(f"{name} ({PROVIDERS[name].credential_env})")
        else:
            keys[name] = key
    if missing:
        raise StoreError("missing credentials for: " + ", ".join(missing))
    return keys


def _scrub(text: Optional[str], secrets: List[str]) -> Optional[str]:
    if text is None:
        return None
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text


# -- budget ------------------------------------------------------------------


@dataclass
class Budget:
    max_calls: int
    max_calls_per_provider: int
    max_cost_usd: float

    def spent(self, store: EvalStore) -> dict:
        rows = store.conn.execute(
            "SELECT provider, COUNT(*) AS calls, SUM(COALESCE(cost_usd, estimated_cost_usd)) AS cost "
            "FROM attempt WHERE sent = 1 GROUP BY provider"
        ).fetchall()
        per = {r["provider"]: r["calls"] for r in rows}
        return {"calls": sum(per.values()), "per_provider": per, "cost_usd": sum((r["cost"] or 0.0) for r in rows)}

    def refusal(self, store: EvalStore, provider: str, next_cost: float) -> Optional[str]:
        """Why the next call must not be sent, or None if it fits."""
        spent = self.spent(store)
        if spent["calls"] + 1 > self.max_calls:
            return "budget:max_calls"
        if spent["per_provider"].get(provider, 0) + 1 > self.max_calls_per_provider:
            return "budget:max_calls_per_provider"
        if spent["cost_usd"] + next_cost > self.max_cost_usd:
            return "budget:max_cost_usd"
        return None


def estimate_tokens(body: bytes, overhead: int = 64) -> int:
    """Conservative input-token estimate: one token per UTF-8 byte plus overhead."""
    return len(body) + overhead


# -- runner --------------------------------------------------------------------


@dataclass
class RunReport:
    run_id: str
    repeat_id: str
    items: int = 0
    sent: int = 0
    reused: int = 0
    skipped_failed: int = 0
    not_sent: int = 0
    statuses: Dict[str, Dict[str, int]] = field(default_factory=dict)
    stopped: Dict[str, str] = field(default_factory=dict)
    stop_reason: Optional[str] = None

    def count(self, provider: str, status: str) -> None:
        self.statuses.setdefault(provider, {})
        self.statuses[provider][status] = self.statuses[provider].get(status, 0) + 1

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class Runner:
    def __init__(self, store: EvalStore, run: dict, transports: Dict[str, Transport], credentials: Dict[str, str],
                 budget: Budget, timeout: float = 30.0, delay: float = 0.5, retry_delay: float = 2.0,
                 max_estimated_tokens: int = 60000, clock: Clock = utc_now,
                 sleep: Callable[[float], None] = time.sleep, monotonic: Callable[[], float] = time.monotonic,
                 revision: Optional[str] = None):
        self.store, self.run_rec = store, run
        self.transports, self.credentials, self.budget = transports, credentials, budget
        self.timeout, self.delay, self.retry_delay = timeout, delay, retry_delay
        self.max_estimated_tokens = max_estimated_tokens
        self.clock, self.sleep, self.monotonic = clock, sleep, monotonic
        self.revision = revision or code_revision()
        self.secrets = [v for v in credentials.values() if v]

    def _previous(self, provider: str, snapshot_id: str, repeat_id: str) -> List[dict]:
        return [dict(r) for r in self.store.conn.execute(
            "SELECT * FROM attempt WHERE run_id = ? AND provider = ? AND snapshot_id = ? AND repeat_id = ? ORDER BY attempt_no",
            (self.run_rec["run_id"], provider, snapshot_id, repeat_id))]

    def execute(self, providers: List[str], split: Optional[str] = None, aids: Optional[List[int]] = None,
                limit: Optional[int] = None, repeat_id: str = DEFAULT_REPEAT) -> RunReport:
        for name in providers:
            if name not in self.run_rec["providers"]:
                raise StoreError(f"provider {name!r} is not part of run {self.run_rec['run_id']!r}")
        items = model_items(self.run_rec["manifest"], split)
        if aids:
            wanted = set(aids)
            items = [i for i in items if i["aid"] in wanted]
        if limit is not None:
            items = items[:limit]
        report = RunReport(self.run_rec["run_id"], repeat_id, items=len(items))
        first_request = True
        for index, item in enumerate(items):
            order = providers if index % 2 == 0 else list(reversed(providers))
            for name in order:
                if name in report.stopped:
                    continue
                previous = self._previous(name, item["snapshot_id"], repeat_id)
                if any(a["status"] == OK for a in previous):
                    report.reused += 1
                    continue
                if previous and (previous[-1]["status"] not in RETRYABLE or len(previous) >= MAX_ATTEMPTS):
                    report.skipped_failed += 1
                    continue
                while len(previous) < MAX_ATTEMPTS:
                    if not first_request:
                        self.sleep(self.retry_delay if previous else self.delay)
                    attempt = self._attempt(name, item, repeat_id, previous, report)
                    if attempt is None:
                        break
                    first_request = False
                    previous.append(attempt)
                    report.count(name, attempt["status"])
                    if attempt["status"] in STOP_PROVIDER:
                        report.stopped[name] = attempt["status"]
                        break
                    if attempt["status"] not in RETRYABLE:
                        break
            if len(report.stopped) == len(providers):
                break
        reasons = sorted(set(report.stopped.values()))
        report.stop_reason = ",".join(reasons) if reasons else None
        return report

    def _attempt(self, name: str, item: dict, repeat_id: str, previous: List[dict], report: RunReport) -> Optional[dict]:
        provider = PROVIDERS[name]
        settings = self.run_rec["providers"][name]
        request = provider.build_request(settings["model"], item["input"], self.run_rec["questions"]["questions"])
        body = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        est_tokens = estimate_tokens(body)
        est_cost = est_tokens * settings["input_price_usd_per_mtok"] / 1e6
        recorded_headers = provider.headers(None)
        base = {
            "run_id": self.run_rec["run_id"], "provider": name, "request_model": settings["model"], "aid": item["aid"],
            "snapshot_id": item["snapshot_id"], "input_sha256": item["input_sha256"], "repeat_id": repeat_id,
            "attempt_no": len(previous) + 1, "retry_of": previous[-1]["id"] if previous else None,
            "code_revision": self.revision, "url": settings["url"], "request_headers": canonical_json(recorded_headers),
            "request_body": body.decode("utf-8"), "request_sha256": sha256_bytes(body),
            "estimated_input_tokens": est_tokens, "estimated_cost_usd": est_cost,
        }
        if est_tokens > self.max_estimated_tokens:
            now = self.clock()
            report.not_sent += 1
            return self._insert({**base, "sent": 0, "started_at": now, "finished_at": now, "status": "input_too_large",
                                 "error_class": "estimated_input_limit", "error_detail": f"estimated {est_tokens} > {self.max_estimated_tokens}"})
        refusal = self.budget.refusal(self.store, name, est_cost)
        if refusal:
            report.stopped[name] = refusal
            return None
        started_at, t0 = self.clock(), self.monotonic()
        result: HttpResult = self.transports[name](settings["url"], provider.headers(self.credentials.get(name)), body, self.timeout)
        latency_ms = (self.monotonic() - t0) * 1000.0
        finished_at = self.clock()
        parsed: Parsed = provider.parse(result, list(FLAGS))
        report.sent += 1
        cost = None
        if parsed.input_tokens is not None:
            cost = parsed.input_tokens * settings["input_price_usd_per_mtok"] / 1e6
        response_body = result.body
        if response_body is not None and any(s.encode("utf-8") in response_body for s in self.secrets):
            for secret in self.secrets:
                response_body = response_body.replace(secret.encode("utf-8"), b"[REDACTED]")
        return self._insert({
            **base, "sent": 1, "started_at": started_at, "finished_at": finished_at, "latency_ms": latency_ms,
            "http_status": result.status, "response_headers": canonical_json(result.headers), "response_body": response_body,
            "status": parsed.status, "error_class": parsed.error_class, "error_detail": _scrub(parsed.error_detail, self.secrets),
            "returned_model": parsed.returned_model,
            "p_uses_voicebank": parsed.probabilities.get("uses_voicebank"), "p_is_vocal_work": parsed.probabilities.get("is_vocal_work"),
            "usage": canonical_json(parsed.usage) if parsed.usage is not None else None, "input_tokens": parsed.input_tokens,
            "cost_usd": cost, "server_compute_unknown": 1 if parsed.status == "timeout" else 0,
        })

    def _insert(self, record: dict) -> dict:
        columns = sorted(record)
        with self.store.transaction() as conn:
            cur = conn.execute(
                f"INSERT INTO attempt ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})",
                [record[c] for c in columns],
            )
            record = dict(conn.execute("SELECT * FROM attempt WHERE id = ?", (cur.lastrowid,)).fetchone())
        return record


def eval_status(store: EvalStore) -> dict:
    runs = [dict(r) for r in store.conn.execute("SELECT run_id, created_at, dataset_version, manifest_sha256, question_set_id, questions_sha256, code_revision FROM run ORDER BY created_at")]
    for run in runs:
        rows = store.conn.execute(
            "SELECT provider, repeat_id, status, COUNT(*) AS n FROM attempt WHERE run_id = ? GROUP BY provider, repeat_id, status ORDER BY provider, repeat_id, status",
            (run["run_id"],)).fetchall()
        run["attempts"] = [dict(r) for r in rows]
    spent = Budget(0, 0, 0.0).spent(store)
    return {"runs": runs, "spent": spent, "frozen_thresholds": [dict(r) for r in store.conn.execute("SELECT name, run_id, sha256, frozen_at FROM frozen_thresholds ORDER BY frozen_at")]}
