"""The experiment store: a local, single-writer SQLite database plus raw files.

Layout of a data directory::

    <data-dir>/
      gleaner.sqlite      experiment database
      raw/ab/<sha256>     raw response bodies, content addressed
      .lock               single-writer lock file

Acquisition evidence, snapshots, discoveries, and annotation revisions are
append-only: SQLite triggers reject UPDATE and DELETE on those tables.
"""

import fcntl
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from .util import sha256_bytes

DB_NAME = "gleaner.sqlite"
RAW_DIR = "raw"
LOCK_NAME = ".lock"
SCHEMA_VERSION = 1
DATA_DIR_ENV = "GLEANER_DATA_DIR"


class StoreError(Exception):
    """A user-facing error about the data directory or its contents."""


SCHEMA = """
CREATE TABLE meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE candidate (
    aid INTEGER PRIMARY KEY,
    first_seen_at TEXT NOT NULL
);

CREATE TABLE discovery (
    id INTEGER PRIMARY KEY,
    aid INTEGER NOT NULL REFERENCES candidate(aid),
    source TEXT NOT NULL,
    locator TEXT NOT NULL DEFAULT '',
    sample_group TEXT NOT NULL DEFAULT '',
    observed TEXT NOT NULL DEFAULT '{}',
    discovered_at TEXT NOT NULL,
    import_batch TEXT NOT NULL,
    UNIQUE (aid, source, locator, sample_group, observed)
);

CREATE TABLE acquisition (
    id INTEGER PRIMARY KEY,
    aid INTEGER NOT NULL REFERENCES candidate(aid),
    reason TEXT NOT NULL CHECK (reason IN ('initial', 'refresh')),
    created_at TEXT NOT NULL
);

CREATE TABLE fetch_attempt (
    id INTEGER PRIMARY KEY,
    acquisition_id INTEGER NOT NULL REFERENCES acquisition(id),
    aid INTEGER NOT NULL,
    endpoint TEXT NOT NULL CHECK (endpoint IN ('view', 'tags')),
    attempt_no INTEGER NOT NULL,
    retry_of INTEGER REFERENCES fetch_attempt(id),
    url TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('ok', 'retryable_error', 'unavailable', 'invalid_response')),
    error_class TEXT,
    error_detail TEXT,
    http_status INTEGER,
    api_code INTEGER,
    raw_path TEXT,
    raw_sha256 TEXT,
    raw_size INTEGER,
    extracted TEXT,
    UNIQUE (acquisition_id, endpoint, attempt_no)
);

CREATE TABLE snapshot (
    snapshot_id TEXT PRIMARY KEY,
    aid INTEGER NOT NULL REFERENCES candidate(aid),
    acquisition_id INTEGER NOT NULL UNIQUE REFERENCES acquisition(id),
    view_attempt_id INTEGER NOT NULL REFERENCES fetch_attempt(id),
    tags_attempt_id INTEGER NOT NULL REFERENCES fetch_attempt(id),
    input_schema_version TEXT NOT NULL,
    input_text TEXT NOT NULL,
    input_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE annotation (
    snapshot_id TEXT NOT NULL REFERENCES snapshot(snapshot_id),
    revision INTEGER NOT NULL CHECK (revision >= 1),
    uses_voicebank INTEGER CHECK (uses_voicebank IN (0, 1)),
    is_vocal_work INTEGER CHECK (is_vocal_work IN (0, 1)),
    text_evidence TEXT NOT NULL CHECK (text_evidence IN ('sufficient', 'insufficient', 'uncertain')),
    rationale TEXT NOT NULL,
    evidence_locator TEXT NOT NULL DEFAULT '',
    null_reason TEXT NOT NULL DEFAULT '',
    annotator TEXT NOT NULL,
    rules_version TEXT NOT NULL,
    annotated_at TEXT NOT NULL,
    import_batch TEXT NOT NULL,
    PRIMARY KEY (snapshot_id, revision)
);
"""

APPEND_ONLY_TABLES = ("discovery", "acquisition", "fetch_attempt", "snapshot", "annotation")


def _append_only_triggers() -> str:
    statements = []
    for table in APPEND_ONLY_TABLES:
        for action in ("UPDATE", "DELETE"):
            statements.append(
                f"CREATE TRIGGER {table}_no_{action.lower()} BEFORE {action} ON {table} "
                f"BEGIN SELECT RAISE(ABORT, '{table} is append-only'); END;"
            )
    return "\n".join(statements)


def _inside_git_work_tree(path: Path) -> Optional[Path]:
    for parent in [path, *path.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def resolve_data_dir(value: Optional[str], allow_git: bool = False) -> Path:
    """Resolve the data directory from an explicit value or the environment.

    There is deliberately no default location. Directories inside a Git work
    tree are refused unless ``allow_git`` is set, so that experiment data does
    not end up in a repository by accident.
    """
    raw = value or os.environ.get(DATA_DIR_ENV)
    if not raw:
        raise StoreError(f"no data directory given: pass --data-dir or set {DATA_DIR_ENV}")
    path = Path(raw).expanduser().resolve()
    if not allow_git:
        repo = _inside_git_work_tree(path)
        if repo is not None:
            raise StoreError(
                f"refusing to write experiment data inside a Git work tree ({repo}); "
                "choose a directory outside version control"
            )
    return path


class Store:
    """An open experiment store. Use :func:`open_store` to create one."""

    def __init__(self, root: Path, conn: sqlite3.Connection):
        self.root = root
        self.conn = conn

    # -- raw files -----------------------------------------------------

    def put_raw(self, body: bytes) -> tuple:
        """Store a raw body by content hash. Returns (relative path, sha256, size)."""
        digest = sha256_bytes(body)
        rel = f"{RAW_DIR}/{digest[:2]}/{digest}"
        target = self.root / rel
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".tmp")
            with open(tmp, "wb") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, target)
        return rel, digest, len(body)

    def read_raw(self, rel: str) -> bytes:
        return (self.root / rel).read_bytes()

    # -- transactions --------------------------------------------------

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self.conn
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")

    def close(self) -> None:
        self.conn.close()


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = DELETE")
    return conn


def init_store(root: Path) -> None:
    """Create a new, empty data directory (or confirm an existing one)."""
    root.mkdir(parents=True, exist_ok=True)
    db_path = root / DB_NAME
    if db_path.exists():
        with open_store(root):
            return
    conn = _connect(db_path)
    try:
        conn.executescript("BEGIN;" + SCHEMA + _append_only_triggers() + "COMMIT;")
        conn.execute("INSERT INTO meta (key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
    finally:
        conn.close()
    (root / RAW_DIR).mkdir(exist_ok=True)


@contextmanager
def open_store(root: Path) -> Iterator[Store]:
    """Open an existing store, holding an exclusive single-writer lock."""
    db_path = root / DB_NAME
    if not db_path.exists():
        raise StoreError(f"no experiment database in {root}; run 'init' first")
    lock_handle = open(root / LOCK_NAME, "a+")
    try:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise StoreError(f"another process is using {root}") from None
        conn = _connect(db_path)
        try:
            row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
            if row is None or int(row["value"]) != SCHEMA_VERSION:
                raise StoreError(f"unsupported schema version in {db_path}")
            yield Store(root, conn)
        finally:
            conn.close()
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()
