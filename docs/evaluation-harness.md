# Evaluation harness

The evaluation commands turn a data bundle from the [experiment store](experiment-store.md) into a fixed dataset manifest, send each snapshot to Jev and OpenAI Decisions, keep every request and response, and compute reports offline. Like the store, they need Python 3.9 or later and nothing else.

**Status: tested with synthetic responses only.** No real model request has been made with this code. The provider request and response formats follow the public documentation as read in November 2026, and real-world quality, cost and latency are unknown. Re-check the formats (see [Provider formats](#provider-formats)) before the first real run.

## Workflow

```bash
gleaner dataset-manifest --bundle BUNDLE --selection selection.csv \
    --dataset-version NAME --seed SEED --acceptance-fraction 0.6 \
    --popular-group popular --out manifest.json
gleaner eval-run --eval-dir EVAL --run-id RUN --manifest manifest.json \
    --questions questions.json --provider jev --provider openai \
    --max-calls 1000 --max-calls-per-provider 500 --max-cost-usd 1 [--split dev]
gleaner eval-status --eval-dir EVAL
gleaner eval-select --eval-dir EVAL --run-id RUN --max-false-positive-rate 0.1 --out SELECTION_DIR
gleaner eval-freeze --eval-dir EVAL --name NAME --thresholds SELECTION_DIR/thresholds.json
gleaner eval-report --eval-dir EVAL --run-id RUN --split dev --thresholds SELECTION_DIR/thresholds.json --out REPORT_DIR
gleaner eval-report --eval-dir EVAL --run-id RUN --split acceptance --frozen NAME --out REPORT_DIR
```

The evaluation directory is given with `--eval-dir` or `GLEANER_EVAL_DIR`. There is no default, and the commands refuse to write the evaluation directory, manifests, selections or reports inside a Git work tree unless `--allow-git-dir` is given.

## Dataset manifest

`dataset-manifest` reads a verified bundle and a selection CSV, and writes one JSON file that pins everything a run and a report depend on. An existing manifest file is never overwritten.

| Selection column | Required | Meaning |
| --- | --- | --- |
| `aid` | yes | Video to include. Each aid may appear once. |
| `family` | no | Group of near-duplicates (re-uploads, reposts). Defaults to `aid-<aid>`. A family always stays in one split. |
| `split` | no | `dev` or `acceptance` to force the split, for example to keep pilot samples in `dev`. Empty means assigned by seed. |
| `group` | no | Sampling group for reports (for example `popular`, `ordinary`, `boundary`). Defaults to the first discovery's `sample_group`. |
| `snapshot_id` | no | A specific snapshot; defaults to the latest one in the bundle. |
| `annotation_revision` | no | A specific annotation revision; defaults to the latest one. |

Unforced families are assigned with algorithm `family-sha256-v1`: the first 64 bits of `sha256(seed + ":" + family)`, read as a fraction of 2^64, go to `acceptance` when below `--acceptance-fraction`, otherwise to `dev`. A family with members forced into both splits is an error.

Each item records the snapshot, its exact model input and hash, the annotation revision, both labels (`null` stays `null`), text evidence, group, family and split. Selected aids without a snapshot stay in the manifest with status `no_snapshot`, so acquisition failures remain visible in the denominators. The manifest also records the source bundle's hashes, the selection file hash, the seed, the popular groups, and `items_sha256`; loading a manifest whose items do not match that hash fails. Runs refer to the manifest by the SHA-256 of the file.

## Questions

A question set is a JSON file with an id and one instruction per flag:

```json
{"question_set_id": "v1", "questions": {"uses_voicebank": "...", "is_vocal_work": "..."}}
```

Both providers get the same text. [`examples/questions.example.json`](../examples/questions.example.json) shows the format; its wording is illustrative and has not been evaluated.

## Runs

A run is identified by `--run-id` and fixed on creation: manifest hash and full text, question set and its hash, code revision (`git` commit, with `-dirty` for local changes), and per-provider settings (requested model, URL, and the price used for estimates). Running the same id again with a different manifest, question set, model or price fails; use a new run id instead. Defaults are `jev-1.14.0` and `gpt-6-luna`; override with `--model jev=...`.

For every item, each provider gets one request containing the snapshot string and both questions. Providers alternate order between items. Requests use the standard library only, so there are no hidden SDK retries.

Every outbound request is one stored attempt, with:

- the request body and non-secret headers, the URL, and the requested model;
- HTTP status, selected response headers, and the raw response body;
- status, error class and detail, the model the provider returned, both probabilities, usage, input tokens;
- start and end times, latency, the pre-call estimate, the usage-based cost, and the retry link;
- the code revision that sent it.

### Statuses

| Status | Meaning | Next |
| --- | --- | --- |
| `ok` | Both answers found by name, each a finite number in [0, 1] | reused on resume |
| `invalid_response` | Not JSON, duplicate keys or answers, missing or extra answers, wrong type, value out of range | terminal |
| `refusal` | The provider refused a question | terminal |
| `input_too_large` | HTTP 413, a context-length 400, or a request estimated above `--max-estimated-tokens` (then not sent) | terminal |
| `rate_limited`, `server_error`, `timeout`, `transport_error` | HTTP 429, HTTP 5xx, timeout, network failure | retried once |
| `auth_error`, `billing_error`, `config_error` | HTTP 401/403, 402, other 400/404/415/422 | stops that provider for the rest of the command |
| `http_error` | Any other status | terminal |

None of these statuses is a negative label. A timeout is marked `server_compute_unknown`, because the provider may still have processed and billed it.

### Resume, retries and repeats

Each item gets at most two attempts per provider and repeat id: the first and one retry, counted across interrupted and resumed commands. Running a command again skips items with an `ok` attempt and items whose failures are final, and retries only retryable failures that have not used their retry. For a deliberate stability measurement, run again with a new `--repeat-id` (for example on ten items with `--limit 10`); repeats are recorded and reported separately from the default `main`.

### Budget

`--max-calls`, `--max-calls-per-provider` and `--max-cost-usd` are required. They count every sent attempt in the evaluation directory, across all runs, including failures, retries, timeouts and repeats. Before each request, the harness estimates its cost conservatively (one token per UTF-8 byte of the request body plus 64, at the configured price) and does not send it if any limit would be exceeded; that provider then stops with `budget:...`. Spend uses usage-based cost when the response reports input tokens, and the estimate otherwise. Prices default to the documented base input prices (Jev $0.042 and OpenAI $0.10 per million input tokens; output not charged); override with `--price jev=...`. Cached-input discounts and cache-write charges are not modelled.

### Credentials

Keys are read from `TYPESAFE_API_KEY` and `OPENAI_API_KEY`, or from a private JSON file (`{"jev": "...", "openai": "..."}`) given with `--credentials` or `GLEANER_CREDENTIALS`. The file must be outside any Git work tree, and environment variables take precedence. Keys are only held in memory: the `Authorization` header is not stored, and any key that appears in a response body or error detail is replaced with `[REDACTED]`. Missing keys stop the command before any request.

`--synthetic` replaces both providers with an offline stand-in that returns deterministic, meaningless probabilities; it needs no keys and is meant for trying the pipeline.

## Thresholds

The current decision rule is `p_uses_voicebank >= t_uses_voicebank AND p_is_vocal_work >= t_is_vocal_work`, with separate thresholds per provider. Probabilities are not treated as calibrated or comparable across providers.

`eval-select` searches a grid (default 0.05 to 0.95 in steps of 0.05, or `--grid`) on the **dev split only**, and writes `grid.json` with every point and `thresholds.json` with a proposal. The rule `max-popular-recall-under-fpr-v1` keeps points whose joint false-positive rate on dev is at most `--max-false-positive-rate`, then prefers the highest popular recall, the lowest false-positive rate, the highest overall recall, and thresholds closest to 0.5. When no point qualifies (for example when dev has no known negatives), the proposal's thresholds are `null`.

`thresholds.json` can be edited before freezing, for example to add `acceptance_criteria` or to set thresholds by hand (then set `selection_rule` to say so). `eval-freeze` stores it under a name once. Freezing the same content again is a no-op, freezing different content under an existing name fails, and the table is append-only. Only configurations marked `selected_on_split: dev` that match the run's manifest, questions and settings can be frozen. Freeze thresholds and acceptance criteria before opening any acceptance report.

## Reports

`eval-report` writes `report.json` and `report.md`. Reports on `dev` can use a threshold file; reports on `acceptance` or `all` need `--frozen`. The JSON includes a `report_sha256` over everything except `generated_at`, so the same run, manifest, questions, code and thresholds give the same hash.

For each provider the report has:

- **Denominators**: selected, acquisition failed, with snapshot, unannotated, human known and unknown per flag and joint, inference ok and failed by status.
- **Confusion matrices** for `uses_voicebank`, `is_vocal_work`, and the joint decision, using items with a known label and a successful inference. The joint label exists only when both human labels are known; known single flags still count in their own matrix.
- **Popular groups** (from the manifest): known positives, misses, recall, and a *process recall* in which inference failures count as not included.
- **Rates**: recall, false-positive rate FP/(FP+TN) and precision TP/(TP+FP), each with count, denominator and 95% Wilson interval; `N/A` when the denominator is zero.
- **Groups**: every sampling group, `text_evidence` insufficient and uncertain, and both splits for `all`.
- **Usage**: attempts sent and not sent, retries, statuses, failure rate, reported input tokens, usage-based and estimated cost, cost per 1,000 successful items, returned model ids, latency p50/p95 of successful attempts and end-to-end p50/p95 including retries (nearest rank, with sample sizes).
- **Repeat stability**: for each extra repeat id, paired items, joint decision flips and probability differences.
- **Errors**: every false negative, false positive, single-flag error and inference failure, with aid, snapshot, group, labels, probabilities and attempt id.

Results describe the sample only. They are not a site-wide recall estimate, and zero misses in a small sample does not mean zero misses later.

## Provider formats

Re-check these against current documentation before a real run, and update `gleaner/providers.py` if they changed.

| | Jev (TypeSafe System One) | OpenAI Decisions |
| --- | --- | --- |
| Request | `POST https://api.typesafe.ai/v1/systemone` | `POST https://api.openai.com/v1/decisions` |
| Auth | `Authorization: Bearer <key>` | `Authorization: Bearer <key>` |
| Input | `state`: the snapshot string | `input`: the snapshot string |
| Questions | object keyed by flag: `{"type": "noul", "instructions": ...}` | array: `{"type": "predicate", "name": flag, "instructions": ...}` |
| Answer | `answers.<flag>.noul` | `answers[]` item with `name` and `probability`; `{"type": "refusal"}` per question |
| Model | `model` in the response (may differ from an alias) | `model` in the response |
| Usage | `usage.input_tokens` (billed), `usage.output_tokens` (free) | `usage.input_tokens` including cached tokens |

Things not covered by the documentation and worth checking with a first authenticated smoke test: error body formats, request id headers, and whether rate-limit responses include `Retry-After`.

## Synthetic walkthrough

```bash
export GLEANER_DATA_DIR="$(mktemp -d)/data" GLEANER_EVAL_DIR="$(mktemp -d)/eval"
WORK="$(mktemp -d)"
python -m gleaner init
python -m gleaner import-candidates examples/synthetic/candidates.csv
python -m gleaner fetch --max-requests 20 --delay 0 --fixtures examples/synthetic/responses
python -m gleaner export-bundle "$WORK/bundle"
python -m gleaner dataset-manifest --bundle "$WORK/bundle" --selection examples/synthetic/selection.csv \
    --dataset-version synthetic-1 --seed example --acceptance-fraction 0.5 --popular-group popular --out "$WORK/manifest.json"
python -m gleaner eval-run --run-id demo --manifest "$WORK/manifest.json" --questions examples/questions.example.json \
    --provider jev --provider openai --max-calls 20 --max-calls-per-provider 10 --max-cost-usd 0.01 --synthetic --delay 0
python -m gleaner eval-status
```

The synthetic snapshots have no annotations, so the report shows every label as unknown; the walkthrough only shows the records and the flow.
