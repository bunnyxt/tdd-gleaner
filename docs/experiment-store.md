# Experiment store

The `gleaner` command keeps the evaluation data reproducible: it records candidate videos, acquires their title, description, and tags, freezes model inputs as immutable snapshots, and round-trips human annotations. It is an offline research tool, not the inclusion service.

It needs Python 3.9 or later and has no third-party dependencies.

## Data directory

Every command except `verify-bundle` needs a data directory, given with `--data-dir` or the `GLEANER_DATA_DIR` environment variable. There is no default. The tool refuses to write a data directory, an annotation CSV, or a bundle inside a Git work tree, so experiment data cannot end up in a repository by accident; `--allow-git-dir` overrides the check.

```
<data-dir>/
  gleaner.sqlite      experiment database (single writer)
  raw/ab/<sha256>     raw response bodies, content addressed
  .lock               lock file held by the running command
```

Only one command can use a data directory at a time. Discoveries, acquisitions, fetch attempts, snapshots, and annotation revisions are append-only; database triggers reject updates and deletions.

## Workflow

```bash
gleaner --data-dir DIR init
gleaner --data-dir DIR import-candidates candidates.csv [--source NAME]
gleaner --data-dir DIR fetch --max-requests 20 [--retry] [--aid AID ...]
gleaner --data-dir DIR refresh --aid AID --max-requests 2
gleaner --data-dir DIR status
gleaner --data-dir DIR export-annotations labels.csv
gleaner --data-dir DIR import-annotations labels.csv --rules-version RULES [--annotator NAME]
gleaner --data-dir DIR verify
gleaner --data-dir DIR export-bundle BUNDLE_DIR
gleaner verify-bundle BUNDLE_DIR
gleaner --data-dir NEW_DIR restore-bundle BUNDLE_DIR
```

Run it as `python -m gleaner` from a checkout, or install the package to get the `gleaner` command.

### Candidates

The candidate CSV needs an `aid` column. Optional columns are `source`, `locator` (a URL or query), `sample_group`, and any number of `observed_*` columns for statistics seen at sampling time. Candidates are de-duplicated by aid, and every distinct discovery row is kept, so the same video found by two sources has one candidate and two discoveries. Importing an identical row again changes nothing. Discovery data is never part of the model input.

### Acquisition

Each acquisition requests the `view` endpoint and then the `tags` endpoint. Every request is stored as a fetch attempt with its start and end times, URL, HTTP status, business code, status, error class, and the raw body (if any). No cookies or credentials are sent or stored.

| Status | Meaning |
| --- | --- |
| `ok` | The response passed validation. |
| `retryable_error` | Transport error, non-200 HTTP status (including 403 and 412), or a business code that is not known to be terminal. |
| `unavailable` | A business code known to mean the video cannot be viewed. |
| `invalid_response` | Not JSON, a missing field, a wrong type, or a returned aid that differs from the requested aid. |

None of these failure statuses says anything about the content labels.

A snapshot is created only when both endpoints have an `ok` attempt in the same acquisition. A successful empty description `""` or empty tag list `[]` is valid. A terminal `view` failure skips the `tags` request.

- `fetch` acquires candidates that have no snapshot. Acquisitions interrupted by the request budget resume automatically. With `--retry`, acquisitions with retryable failures request only the failed endpoints again, up to `--max-attempts` per endpoint (default 2). Each retry records the attempt it retries.
- `refresh` starts a new acquisition for the given aids. A successful refresh creates a new snapshot. Earlier snapshots and their annotations are kept, and a failed refresh never borrows fields from an earlier acquisition.
- `--max-requests` is a hard cap on endpoint requests for the run, and `--delay` sets the pause between requests (default 1 second). `status` reports the total number of requests recorded.

### Model input, version `v1`

The model input is one UTF-8 JSON string with exactly `title`, `desc`, and `tags`, in that order, with compact separators and no escaping of non-ASCII text:

```json
{"title":"【示例声库】一首原创歌曲","desc":"本例为虚构数据。","tags":["虚拟歌手","原创曲","原创曲"]}
```

Values are used as returned: no cleaning, truncation, or tag de-duplication. The snapshot stores this string and its SHA-256. Snapshot IDs have the form `snap-<aid>-<acquisition>`.

### Annotation

`export-annotations` writes a UTF-8 CSV with one row per snapshot (the latest snapshot per aid; `--all-snapshots` for all of them).

| Column | Editable | Values |
| --- | --- | --- |
| `aid`, `snapshot_id`, `base_revision`, `input_sha256` | no | Row identity. Keep them as exported. |
| `uses_voicebank`, `is_vocal_work` | yes | `true`, `false`, or `null` |
| `text_evidence` | yes | `sufficient`, `insufficient`, or `uncertain` |
| `rationale` | yes | Short reason; required |
| `evidence_locator` | yes | For example a video timestamp |
| `null_reason` | yes | Required when a label is `null` |
| `annotator` | yes | Or pass `--annotator` |
| `title`, `desc`, `tags` | no | For reading only; ignored on import |

`import-annotations` applies each valid row and skips the rest, listing every skipped row with its line number and reason; the command exits with status 1 when any row was skipped. A row is skipped if it has an invalid value, does not match its snapshot's aid or input hash, or has a `base_revision` that is not the current revision. The last check catches both stale exports and re-imports of an already applied file. If one snapshot appears on several rows, all of them are skipped. Each applied row appends revision `base_revision + 1` with the given `--rules-version`; unchanged and untouched rows are left alone. Snapshots are never modified.

### Bundles

`export-bundle` writes a self-contained directory:

| File | Content |
| --- | --- |
| `manifest.json` | Format, tool and schema versions, counts, dataset hash, and the SHA-256 of every file |
| `gleaner.sqlite` | Consistent copy of the database |
| `raw/` | Every raw body referenced by an attempt |
| `candidates.jsonl` | Candidates with their discoveries |
| `fetch_attempts.jsonl` | Every attempt, failures included |
| `snapshots.jsonl` | Every snapshot with its exact input string and hash |
| `annotations.jsonl` | Every annotation revision |
| `dataset.jsonl` | Latest snapshot per aid with its latest annotation revision |

`verify-bundle` checks the manifest hashes and looks for unlisted files. `restore-bundle` does the same, copies the bundle into an empty data directory, and runs `verify` on the result. `verify` checks database integrity, raw file hashes, and that every snapshot's input string matches its stored hash.

## Synthetic example

`examples/synthetic/` contains fictional candidates and responses. They cover two sources for one aid, a complete snapshot with a duplicate tag, a valid empty description and tag list, a missing tags response, an unavailable video, an aid mismatch, and a missing field. Run the walkthrough offline with `--fixtures`, which serves `<dir>/<endpoint>/<aid>.json` instead of using the network:

```bash
export GLEANER_DATA_DIR="$(mktemp -d)/data"
python -m gleaner init
python -m gleaner import-candidates examples/synthetic/candidates.csv
python -m gleaner fetch --max-requests 20 --delay 0 --fixtures examples/synthetic/responses
python -m gleaner status
```

## Tests

```bash
python -m unittest discover -s tests -t .
```
