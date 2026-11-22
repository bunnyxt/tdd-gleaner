<img src="https://gw.alipayobjects.com/zos/antfincdn/R8sN%24GNdh6/language.svg" width="18"> English | [简体中文](./README_zh-CN.md)

# tdd-gleaner

**Gathering virtual singer creations from the community.**

tdd-gleaner is the video discovery, inclusion, and recheck project for [TianDian Daily](https://tdd.bunnyxt.com) (TDD). Fetchers find candidate videos, the checker evaluates content eligibility, and the inclusion workflow manages tracking eligibility.

## Current stage

The project is in its evaluation phase. It currently contains criteria, an evaluation plan, synthetic examples, and an offline experiment store for building the evaluation dataset. Fetchers, model integrations, and the inclusion service are not implemented yet.

The first milestone is to manually label 50 pilot samples and compare Jev with OpenAI Decisions using the same titles, descriptions, and tags. Record classification errors, cost, latency, and failures, then expand the evaluation dataset.

## Experiment store

The `gleaner` command (Python 3.9+, standard library only) imports candidate aids, acquires title, description, and tags into immutable input snapshots, exports and imports human annotations as CSV, and writes verifiable data bundles. Experiment data lives in a data directory you choose explicitly, outside version control.

```bash
python -m gleaner --data-dir /path/outside/the/repo init
python -m unittest discover -s tests -t .
```

See the [experiment store guide](docs/experiment-store.md) for the full workflow and an offline walkthrough with synthetic data.

## Documentation

- [Inclusion and annotation criteria (Chinese)](docs/criteria.md)
- [Dataset and evaluation plan (Chinese)](docs/evaluation.md)
- [Experiment store](docs/experiment-store.md)
- [Synthetic sample format](examples/sample.json) and [synthetic acquisition fixtures](examples/synthetic/)

## System boundaries

- Discovery sources are extensible. Finding the same video again should not trigger inference again by default, but every request must remain traceable.
- The checker supports both initial inclusion and rechecks. Content labels, tracking eligibility, and collection frequency are separate concerns.
- Persistent administrator overrides for inclusion or exclusion take precedence over automated decisions.
- Ongoing collection for newly included videos requires frequency controls and filtering by tracking status.

Interfaces, request metadata, storage, and deployment remain to be designed. The evaluation sample format is not the final service interface.
