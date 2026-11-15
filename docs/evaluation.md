# Dataset and evaluation plan

## First round

Start with 50 manually labeled pilot samples to check whether the criteria are usable, then expand to approximately 200-300 samples. These are starting sizes, not guarantees of statistical accuracy.

Cover three groups:

- Popular eligible works and popular negative examples, emphasizing content that must not be missed.
- Ordinary candidates drawn from multiple sources, including videos outside the existing collection.
- Edge cases such as human covers, obscure or new voicebanks, comic dialogue, tuned speech, news, and fully AI-generated music.

Select candidates and assign human labels independently of the models under evaluation. Agreement between models is not a substitute for human ground truth.

## Samples and versions

Each real sample should retain at least its aid, a title/description/tags snapshot, acquisition time, two human labels, a short rationale, source, and evaluation group. Unresolved human labels may remain null.

Models receive only the three text fields from the snapshot. Sources, groups, and human judgments are evaluation context, not model input. The final request metadata structure will be designed separately.

The [synthetic example](../examples/sample.json) illustrates the evaluation envelope; only its `input` object goes to the model. It does not represent a real video and must not count toward accuracy. Real samples also require an aid and acquisition time.

Freeze dataset versions and input snapshots. Refetching information must not silently overwrite historical evaluation inputs. Keep snapshots of the same aid in the same dataset split to prevent leakage between tuning and final evaluation.

Use pilot data to improve criteria, prompts, and thresholds. Reserve a separate set of manually verified samples for final evaluation, without repeatedly tuning against it. Choose split proportions after reviewing the available samples.

## Comparison and records

The first round compares Jev with OpenAI Decisions using identical input content, with prompts and thresholds selected separately for each model. These are candidates; no evaluation results are available yet.

Retain the following for each run:

- Dataset version, exact model input and prompt, requested model identifier, and any version information returned by the service.
- Raw responses, available probabilities or labels, thresholds, final flags, and inclusion decisions.
- Usage, actual charges or explicitly marked cost estimates, latency, errors, and retries.

Report performance for each flag and for the final inclusion decision. Focus on missed eligible works and incorrectly included ineligible works. Show sample counts and errors separately for popular, ordinary, and edge-case groups. Report acquisition failures, inference failures, and unresolved annotations separately; do not treat them as negative examples or silently omit them.

Report cost per 10,000 items and latency distributions, identifying the measurement sample size. Results on deliberately selected difficult examples do not represent platform-wide accuracy. Discovery misses, checker rejections, and persistence failures are distinct problems.

## Data storage

Keep real datasets, acquisition snapshots, and complete run records in the ignored `data/`, `runs/`, and `artifacts/` directories, or in separate durable storage, with appropriate backups. Git ignore rules do not provide backups or prevent forced additions.

The repository maintains criteria, format documentation, and synthetic examples. Publishing real data requires a separate review. Run records must support later reconstruction of decisions rather than relying on short-lived service logs.
