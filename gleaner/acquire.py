"""Candidate import, endpoint acquisition, and immutable input snapshots."""

import csv
import json
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from . import bilibili
from .store import Store, StoreError
from .util import Clock, canonical_json, parse_aid, sha256_bytes, utc_now

INPUT_SCHEMA_VERSION = "v1"
ENDPOINT_ORDER = ("view", "tags")


# -- model input ---------------------------------------------------------


def build_input(title: str, desc: str, tags: Sequence[str]) -> str:
    """Build the v1 model input string.

    Field order is fixed (title, desc, tags), separators are compact, and
    non-ASCII text is not escaped. Values are used exactly as returned: no
    cleaning, truncation, or tag de-duplication.
    """
    return json.dumps({"title": title, "desc": desc, "tags": list(tags)}, ensure_ascii=False, separators=(",", ":"))


def input_sha256(input_text: str) -> str:
    return sha256_bytes(input_text.encode("utf-8"))


# -- candidates ----------------------------------------------------------


def import_candidates(store: Store, path: Path, default_source: Optional[str], clock: Clock = utc_now) -> Dict[str, int]:
    """Import an aid list from CSV.

    Required column: ``aid``. Optional columns: ``source`` (or ``--source``),
    ``locator`` (URL or query), ``sample_group``, and any ``observed_*``
    columns, which are stored as sampling-time observations and never reach
    the model. Candidates are de-duplicated by aid; every distinct discovery
    row is kept. Re-importing an identical row is a no-op.
    """
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if rows and "aid" not in rows[0]:
        raise StoreError("candidate CSV needs an 'aid' column")
    now = clock()
    batch = f"import:{now}:{path.name}"
    parsed = []
    for line_no, row in enumerate(rows, start=2):
        try:
            aid = parse_aid(row.get("aid", ""))
        except ValueError as exc:
            raise StoreError(f"{path.name} line {line_no}: {exc}") from None
        source = (row.get("source") or default_source or "").strip()
        if not source:
            raise StoreError(f"{path.name} line {line_no}: no source (add a 'source' column or pass --source)")
        observed = {k[len("observed_"):]: v for k, v in row.items() if k and k.startswith("observed_") and v not in (None, "")}
        parsed.append((aid, source, (row.get("locator") or "").strip(), (row.get("sample_group") or "").strip(), canonical_json(observed)))

    counts = {"rows": len(parsed), "new_candidates": 0, "new_discoveries": 0}
    with store.transaction() as conn:
        for aid, source, locator, group, observed in parsed:
            cur = conn.execute("INSERT OR IGNORE INTO candidate (aid, first_seen_at) VALUES (?, ?)", (aid, now))
            counts["new_candidates"] += cur.rowcount
            cur = conn.execute(
                "INSERT OR IGNORE INTO discovery (aid, source, locator, sample_group, observed, discovered_at, import_batch)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (aid, source, locator, group, observed, now, batch),
            )
            counts["new_discoveries"] += cur.rowcount
    return counts


# -- acquisition state ---------------------------------------------------


def acquisition_state(store: Store, acquisition_id: int, max_attempts: int) -> Dict[str, object]:
    """Summarise an acquisition: per-endpoint latest attempt and overall state.

    States: ``complete`` (snapshot exists), ``retryable`` (no terminal failure
    and some endpoint still has attempts left), ``failed`` (terminal failure
    or attempts exhausted), ``incomplete`` (endpoints not attempted yet).
    """
    conn = store.conn
    snap = conn.execute("SELECT snapshot_id FROM snapshot WHERE acquisition_id = ?", (acquisition_id,)).fetchone()
    endpoints = {}
    for endpoint in ENDPOINT_ORDER:
        row = conn.execute(
            "SELECT * FROM fetch_attempt WHERE acquisition_id = ? AND endpoint = ? ORDER BY attempt_no DESC LIMIT 1",
            (acquisition_id, endpoint),
        ).fetchone()
        endpoints[endpoint] = row
    if snap is not None:
        state = "complete"
    elif any(r is not None and r["status"] in ("unavailable", "invalid_response") for r in endpoints.values()):
        state = "failed"
    elif any(r is None for r in endpoints.values()):
        state = "incomplete"
    elif all(r["status"] == "ok" or r["attempt_no"] >= max_attempts for r in endpoints.values()):
        state = "failed"
    else:
        state = "retryable"
    return {"state": state, "endpoints": endpoints, "snapshot_id": snap["snapshot_id"] if snap else None}


def _latest_acquisition(store: Store, aid: int):
    return store.conn.execute("SELECT * FROM acquisition WHERE aid = ? ORDER BY id DESC LIMIT 1", (aid,)).fetchone()


class FetchReport:
    def __init__(self) -> None:
        self.requests = 0
        self.snapshots: List[str] = []
        self.outcomes: List[Dict[str, object]] = []
        self.skipped: List[Dict[str, object]] = []
        self.budget_exhausted = False

    def as_dict(self) -> Dict[str, object]:
        return {
            "requests": self.requests,
            "snapshots_created": self.snapshots,
            "attempts": self.outcomes,
            "skipped": self.skipped,
            "budget_exhausted": self.budget_exhausted,
        }


class Fetcher:
    """Runs endpoint attempts under an explicit request budget."""

    def __init__(
        self,
        store: Store,
        transport: bilibili.Transport,
        max_requests: int,
        delay: float = 1.0,
        max_attempts: int = 2,
        clock: Clock = utc_now,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_requests < 0:
            raise StoreError("max_requests must be >= 0")
        if max_attempts < 1:
            raise StoreError("max_attempts must be >= 1")
        self.store = store
        self.transport = transport
        self.max_requests = max_requests
        self.delay = delay
        self.max_attempts = max_attempts
        self.clock = clock
        self.sleep = sleep
        self.report = FetchReport()

    # A request is only made when the budget allows it.
    def _request(self, endpoint: str, aid: int) -> Optional[tuple]:
        if self.report.requests >= self.max_requests:
            self.report.budget_exhausted = True
            return None
        if self.report.requests > 0 and self.delay > 0:
            self.sleep(self.delay)
        url = bilibili.endpoint_url(endpoint, aid)
        started = self.clock()
        response = self.transport(url, aid)
        finished = self.clock()
        self.report.requests += 1
        return url, started, finished, response

    def _attempt(self, acquisition_id: int, aid: int, endpoint: str) -> Optional[str]:
        previous = self.store.conn.execute(
            "SELECT id, attempt_no FROM fetch_attempt WHERE acquisition_id = ? AND endpoint = ? ORDER BY attempt_no DESC LIMIT 1",
            (acquisition_id, endpoint),
        ).fetchone()
        result = self._request(endpoint, aid)
        if result is None:
            return None
        url, started, finished, response = result
        classified = bilibili.classify(endpoint, aid, response)
        raw = self.store.put_raw(response.body) if response.body is not None else (None, None, None)
        attempt_no = previous["attempt_no"] + 1 if previous else 1
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO fetch_attempt (acquisition_id, aid, endpoint, attempt_no, retry_of, url, started_at, finished_at,"
                " status, error_class, error_detail, http_status, api_code, raw_path, raw_sha256, raw_size, extracted)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    acquisition_id, aid, endpoint, attempt_no, previous["id"] if previous else None, url, started, finished,
                    classified.status, classified.error_class, classified.error_detail, response.http_status,
                    classified.api_code, raw[0], raw[1], raw[2],
                    canonical_json(classified.extracted) if classified.extracted is not None else None,
                ),
            )
        self.report.outcomes.append({
            "aid": aid, "acquisition_id": acquisition_id, "endpoint": endpoint, "attempt_no": attempt_no,
            "status": classified.status, "error_class": classified.error_class,
            "http_status": response.http_status, "api_code": classified.api_code,
        })
        return classified.status

    def _new_acquisition(self, aid: int, reason: str) -> int:
        with self.store.transaction() as conn:
            cur = conn.execute("INSERT INTO acquisition (aid, reason, created_at) VALUES (?, ?, ?)", (aid, reason, self.clock()))
            return int(cur.lastrowid)

    def run_acquisition(self, aid: int, acquisition_id: int) -> str:
        """Attempt every endpoint that still needs one, then try to snapshot."""
        for endpoint in ENDPOINT_ORDER:
            state = acquisition_state(self.store, acquisition_id, self.max_attempts)
            if state["state"] in ("complete", "failed"):
                break
            latest = state["endpoints"][endpoint]
            if latest is not None and (latest["status"] == "ok" or latest["attempt_no"] >= self.max_attempts):
                continue
            if self._attempt(acquisition_id, aid, endpoint) is None:
                break
        snapshot_id = make_snapshot(self.store, acquisition_id, self.clock)
        if snapshot_id:
            self.report.snapshots.append(snapshot_id)
        return acquisition_state(self.store, acquisition_id, self.max_attempts)["state"]

    def fetch(self, aids: Iterable[int], retry: bool) -> FetchReport:
        """Acquire candidates that have no snapshot yet.

        A candidate with no acquisition gets a new ``initial`` acquisition.
        One whose latest acquisition is ``retryable`` or ``incomplete`` is
        continued only when ``retry`` is set (``incomplete`` always resumes).
        Failed acquisitions are left alone; use :meth:`refresh` to start over.
        """
        for aid in aids:
            if self.report.budget_exhausted:
                break
            latest = _latest_acquisition(self.store, aid)
            if latest is None:
                self.run_acquisition(aid, self._new_acquisition(aid, "initial"))
                continue
            state = acquisition_state(self.store, latest["id"], self.max_attempts)["state"]
            if state == "incomplete" or (state == "retryable" and retry):
                self.run_acquisition(aid, latest["id"])
            else:
                self.report.skipped.append({"aid": aid, "acquisition_id": latest["id"], "state": state})
        return self.report

    def refresh(self, aids: Iterable[int]) -> FetchReport:
        """Start a new ``refresh`` acquisition. Earlier snapshots are kept."""
        for aid in aids:
            if self.report.requests >= self.max_requests:
                self.report.budget_exhausted = True
                break
            self.run_acquisition(aid, self._new_acquisition(aid, "refresh"))
        return self.report


def make_snapshot(store: Store, acquisition_id: int, clock: Clock = utc_now) -> Optional[str]:
    """Create the snapshot for an acquisition once view and tags are both ok."""
    conn = store.conn
    if conn.execute("SELECT 1 FROM snapshot WHERE acquisition_id = ?", (acquisition_id,)).fetchone():
        return None
    ok = {}
    for endpoint in ENDPOINT_ORDER:
        row = conn.execute(
            "SELECT * FROM fetch_attempt WHERE acquisition_id = ? AND endpoint = ? AND status = 'ok' ORDER BY attempt_no DESC LIMIT 1",
            (acquisition_id, endpoint),
        ).fetchone()
        if row is None:
            return None
        ok[endpoint] = row
    view = json.loads(ok["view"]["extracted"])
    tags = json.loads(ok["tags"]["extracted"])
    text = build_input(view["title"], view["desc"], tags["tags"])
    aid = ok["view"]["aid"]
    snapshot_id = f"snap-{aid}-{acquisition_id}"
    with store.transaction() as tx:
        tx.execute(
            "INSERT INTO snapshot (snapshot_id, aid, acquisition_id, view_attempt_id, tags_attempt_id,"
            " input_schema_version, input_text, input_sha256, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (snapshot_id, aid, acquisition_id, ok["view"]["id"], ok["tags"]["id"], INPUT_SCHEMA_VERSION, text, input_sha256(text), clock()),
        )
    return snapshot_id


def candidate_aids(store: Store, only: Optional[Sequence[int]] = None) -> List[int]:
    rows = store.conn.execute("SELECT aid FROM candidate ORDER BY first_seen_at, aid").fetchall()
    aids = [r["aid"] for r in rows]
    if only:
        known = set(aids)
        missing = [a for a in only if a not in known]
        if missing:
            raise StoreError(f"not imported as candidates: {', '.join(map(str, missing))}")
        wanted = set(only)
        aids = [a for a in aids if a in wanted]
    return aids


def candidates_without_snapshot(store: Store, only: Optional[Sequence[int]] = None) -> List[int]:
    with_snapshot = {r["aid"] for r in store.conn.execute("SELECT DISTINCT aid FROM snapshot")}
    return [a for a in candidate_aids(store, only) if a not in with_snapshot]


def status_summary(store: Store, max_attempts: int = 2) -> Dict[str, object]:
    conn = store.conn
    per_aid = []
    counts: Dict[str, int] = {}
    for aid in candidate_aids(store):
        snaps = conn.execute("SELECT COUNT(*) AS n FROM snapshot WHERE aid = ?", (aid,)).fetchone()["n"]
        latest = _latest_acquisition(store, aid)
        state = "not_acquired" if latest is None else acquisition_state(store, latest["id"], max_attempts)["state"]
        counts[state] = counts.get(state, 0) + 1
        per_aid.append({"aid": aid, "latest_acquisition_state": state, "snapshots": snaps})
    attempts = {r["status"]: r["n"] for r in conn.execute("SELECT status, COUNT(*) AS n FROM fetch_attempt GROUP BY status")}
    return {
        "candidates": len(per_aid),
        "discoveries": conn.execute("SELECT COUNT(*) AS n FROM discovery").fetchone()["n"],
        "snapshots": conn.execute("SELECT COUNT(*) AS n FROM snapshot").fetchone()["n"],
        "annotation_revisions": conn.execute("SELECT COUNT(*) AS n FROM annotation").fetchone()["n"],
        "endpoint_requests": sum(attempts.values()),
        "attempts_by_status": attempts,
        "latest_acquisition_states": counts,
        "per_aid": per_aid,
    }
