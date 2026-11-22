import contextlib
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from gleaner.acquire import import_candidates
from gleaner.annotations import export_annotations, import_annotations
from gleaner.bundle import export_bundle
from gleaner.cli import main
from gleaner.dataset import build_manifest, load_manifest, manifest_bytes, validate_manifest, write_manifest
from gleaner.evalstore import Budget, Runner, ensure_run, load_credentials, open_eval_store, provider_settings
from gleaner.evaluate import build_report, decisions, freeze_thresholds, frozen, render_markdown, select_thresholds, wilson
from gleaner.providers import FLAGS, PROVIDERS, HttpResult
from gleaner.store import StoreError, init_store, open_store
from gleaner.util import canonical_json, sha256_bytes

from tests.test_workflow import Clock, ScriptedTransport as BiliTransport, ok as bili_ok, tags_body, view_body

QUESTIONS = {"question_set_id": "qs-test", "questions": {"uses_voicebank": "Q1?", "is_vocal_work": "Q2?"}}
SECRET = "sk-test-SECRET-value-123"


def jev_body(pv, pw, model="jev-1.14.0", tokens=100, extra=None):
    answers = {"uses_voicebank": {"type": "noul", "noul": pv}, "is_vocal_work": {"type": "noul", "noul": pw}}
    answers.update(extra or {})
    return json.dumps({"model": model, "answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 3}}).encode()


def openai_body(pv, pw, model="gpt-6-luna", tokens=100, answers=None):
    answers = answers if answers is not None else [
        {"type": "predicate", "name": "uses_voicebank", "probability": pv},
        {"type": "predicate", "name": "is_vocal_work", "probability": pw},
    ]
    return json.dumps({"model": model, "answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 0, "total_tokens": tokens}}).encode()


def ok_for(provider, pv, pw):
    return HttpResult(200, jev_body(pv, pw) if provider == "jev" else openai_body(pv, pw))


class ModelTransport:
    """Scripted provider responses keyed by input string; records every request."""

    def __init__(self, provider, default=None):
        self.provider = provider
        self.queue = {}
        self.default = default
        self.calls = []

    def add(self, input_text, *results):
        self.queue.setdefault(input_text, []).extend(results)

    def __call__(self, url, headers, body, timeout):
        request = json.loads(body)
        text = request["state"] if self.provider == "jev" else request["input"]
        self.calls.append({"url": url, "headers": dict(headers), "request": request, "text": text})
        items = self.queue.get(text)
        if items:
            return items.pop(0)
        if self.default is not None:
            return self.default(text)
        return HttpResult(None, error="transport", error_detail="no scripted response")


def make_item(order, aid, split, group, uv, wk, family=None, text_evidence="sufficient", snapshot=True):
    item = {"order": order, "aid": aid, "family": family or f"aid-{aid}", "split": split, "group": group, "split_forced": False}
    if not snapshot:
        item.update({"status": "no_snapshot", "snapshot_id": None})
        return item
    text = json.dumps({"title": f"t{aid}", "desc": "", "tags": []}, ensure_ascii=False, separators=(",", ":"))
    item.update({
        "status": "ok", "snapshot_id": f"snap-{aid}-1", "input_schema_version": "v1", "input": text,
        "input_sha256": sha256_bytes(text.encode()), "annotation_revision": 1, "rules_version": "r1",
        "uses_voicebank": uv, "is_vocal_work": wk, "text_evidence": text_evidence,
    })
    return item


def make_manifest(items, popular=("hot",)):
    m = {
        "format": "gleaner-dataset/1", "dataset_version": "test-v1", "created_at": "2026-01-01T00:00:00.000Z",
        "source_bundle": {"manifest_sha256": "x", "dataset_sha256": "y"}, "split_algorithm": "family-sha256-v1",
        "seed": "s", "acceptance_fraction": 0.5, "popular_groups": list(popular), "selection_sha256": "z",
        "items": items, "items_sha256": sha256_bytes(canonical_json(items).encode()),
    }
    validate_manifest(m)
    return m, sha256_bytes(manifest_bytes(m))


class EvalCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self._ctx = open_eval_store(self.root / "eval")
        self.store = self._ctx.__enter__()
        self.clock = Clock()

    def tearDown(self):
        self._ctx.__exit__(None, None, None)
        self.tmp.cleanup()

    def make_run(self, manifest, sha, run_id="run-1", providers=("jev", "openai")):
        settings = provider_settings(list(providers), {}, {})
        return ensure_run(self.store, run_id, manifest, sha, QUESTIONS, settings, clock=self.clock, revision="test-rev")

    def runner(self, run, transports, budget=None, **kw):
        budget = budget or Budget(1000, 500, 1.0)
        creds = {n: SECRET for n in transports}
        return Runner(self.store, run, transports, creds, budget, delay=0, retry_delay=0, clock=self.clock,
                      sleep=lambda s: None, revision="test-rev", **kw)


# -- adapters -------------------------------------------------------------------


class AdapterTests(unittest.TestCase):
    names = list(FLAGS)

    def test_both_build_one_request_with_two_questions_over_the_same_string(self):
        text = '{"title":"标题","desc":"","tags":["a"]}'
        jev = PROVIDERS["jev"].build_request("jev-1.14.0", text, QUESTIONS["questions"])
        oai = PROVIDERS["openai"].build_request("gpt-6-luna", text, QUESTIONS["questions"])
        self.assertEqual(jev["state"], text)
        self.assertEqual(oai["input"], text)
        self.assertEqual(jev["questions"], {"uses_voicebank": {"type": "noul", "instructions": "Q1?"}, "is_vocal_work": {"type": "noul", "instructions": "Q2?"}})
        self.assertEqual([q["name"] for q in oai["questions"]], self.names)
        self.assertTrue(all(q["type"] == "predicate" for q in oai["questions"]))

    def parse(self, provider, result):
        return PROVIDERS[provider].parse(result, self.names)

    def test_valid_responses(self):
        p = self.parse("jev", HttpResult(200, jev_body(0.9, 0.2)))
        self.assertEqual((p.status, p.probabilities, p.returned_model, p.input_tokens), ("ok", {"uses_voicebank": 0.9, "is_vocal_work": 0.2}, "jev-1.14.0", 100))
        # Answers are matched by name, not position.
        reordered = [{"type": "predicate", "name": "is_vocal_work", "probability": 0.0}, {"type": "predicate", "name": "uses_voicebank", "probability": 1}]
        p = self.parse("openai", HttpResult(200, openai_body(0, 0, answers=reordered)))
        self.assertEqual(p.probabilities, {"uses_voicebank": 1.0, "is_vocal_work": 0.0})

    def test_invalid_answers_are_errors_not_false(self):
        cases = {
            "jev": [
                (b'{"model":"m","answers":{"uses_voicebank":{"type":"noul","noul":0.1}},"usage":{"input_tokens":1}}', "missing_answer"),
                (b'{"model":"m","answers":{"uses_voicebank":{"type":"noul","noul":0.1},"uses_voicebank":{"type":"noul","noul":0.2},"is_vocal_work":{"type":"noul","noul":0.1}},"usage":{"input_tokens":1}}', "duplicate_key"),
                (jev_body(1.5, 0.1), "invalid_probability"),
                (jev_body(True, 0.1), "invalid_probability"),
                (jev_body("0.5", 0.1), "invalid_probability"),
                (b'{"model":"m","answers":{"uses_voicebank":{"type":"noul","noul":NaN},"is_vocal_work":{"type":"noul","noul":0.1}}}', "parse_error"),
                (jev_body(0.1, 0.1, extra={"other": {"type": "noul", "noul": 0.5}}), "unexpected_answer"),
                (b"not json", "parse_error"),
            ],
            "openai": [
                (openai_body(0, 0, answers=[{"type": "predicate", "name": "uses_voicebank", "probability": 0.2}]), "missing_answer"),
                (openai_body(0, 0, answers=[{"type": "predicate", "name": "uses_voicebank", "probability": 0.2}] * 2 + [{"type": "predicate", "name": "is_vocal_work", "probability": 0.2}]), "duplicate_answer"),
                (openai_body(-0.1, 0.5), "invalid_probability"),
                (openai_body(None, 0.5), "invalid_probability"),
            ],
        }
        for provider, items in cases.items():
            for body, error_class in items:
                with self.subTest(provider=provider, error_class=error_class):
                    p = self.parse(provider, HttpResult(200, body))
                    self.assertEqual(p.status, "invalid_response")
                    self.assertEqual(p.error_class, error_class)
                    self.assertEqual(p.probabilities, {})

    def test_refusal(self):
        answers = [{"type": "refusal", "name": "uses_voicebank"}, {"type": "predicate", "name": "is_vocal_work", "probability": 0.9}]
        p = self.parse("openai", HttpResult(200, openai_body(0, 0, answers=answers)))
        self.assertEqual((p.status, p.error_detail, p.probabilities), ("refusal", "uses_voicebank", {}))

    def test_http_and_transport_failures(self):
        cases = [
            (HttpResult(401, b'{"detail":"bad key"}'), "auth_error"),
            (HttpResult(403, b""), "auth_error"),
            (HttpResult(402, b""), "billing_error"),
            (HttpResult(422, b'{"detail":[]}'), "config_error"),
            (HttpResult(400, b'{"detail":"max_tokens_exceeded"}'), "input_too_large"),
            (HttpResult(413, b""), "input_too_large"),
            (HttpResult(429, b""), "rate_limited"),
            (HttpResult(529, b""), "server_error"),
            (HttpResult(500, b""), "server_error"),
            (HttpResult(None, error="timeout"), "timeout"),
            (HttpResult(None, error="transport", error_detail="ConnectionRefusedError"), "transport_error"),
        ]
        for provider in PROVIDERS:
            for result, status in cases:
                with self.subTest(provider=provider, status=status):
                    self.assertEqual(self.parse(provider, result).status, status)


# -- runner and budget --------------------------------------------------------------


class RunnerTests(EvalCase):
    def setUp(self):
        super().setUp()
        items = [make_item(i, 900 + i, "dev", "hot", True, True) for i in range(1, 5)]
        self.manifest, self.sha = make_manifest(items)
        self.run = self.make_run(self.manifest, self.sha)
        self.texts = [i["input"] for i in items]

    def transports(self, jev_default=None, openai_default=None):
        return {
            "jev": ModelTransport("jev", jev_default or (lambda t: ok_for("jev", 0.9, 0.9))),
            "openai": ModelTransport("openai", openai_default or (lambda t: ok_for("openai", 0.8, 0.8))),
        }

    def test_both_providers_receive_the_identical_snapshot_string(self):
        t = self.transports()
        self.runner(self.run, t).execute(["jev", "openai"])
        self.assertEqual([c["text"] for c in t["jev"].calls], self.texts)
        self.assertEqual([c["text"] for c in t["openai"].calls], self.texts)

    def test_resume_reuses_successes_and_repeat_is_separate(self):
        t = self.transports()
        first = self.runner(self.run, t).execute(["jev", "openai"], limit=2)
        self.assertEqual(first.sent, 4)
        second = self.runner(self.run, t).execute(["jev", "openai"])
        self.assertEqual((second.sent, second.reused), (4, 4))
        third = self.runner(self.run, t).execute(["jev", "openai"])
        self.assertEqual((third.sent, third.reused), (0, 8))
        repeat = self.runner(self.run, t).execute(["jev"], limit=2, repeat_id="rep-1")
        self.assertEqual(repeat.sent, 2)
        self.assertEqual(len(t["jev"].calls), 6)
        self.assertEqual(len(decisions(self.store, "run-1", "rep-1")["jev"]), 2)
        self.assertEqual(len(decisions(self.store, "run-1")["jev"]), 4)

    def test_one_retry_at_most_including_across_resumes(self):
        t = self.transports(jev_default=lambda text: HttpResult(500, b"oops"))
        report = self.runner(self.run, t).execute(["jev"], limit=1)
        self.assertEqual(report.statuses["jev"], {"server_error": 2})
        attempts = self.store.attempts("run-1")
        self.assertEqual([a["attempt_no"] for a in attempts], [1, 2])
        self.assertEqual(attempts[1]["retry_of"], attempts[0]["id"])
        again = self.runner(self.run, t).execute(["jev"], limit=1)
        self.assertEqual((again.sent, again.skipped_failed), (0, 1))
        self.assertEqual(len(t["jev"].calls), 2)

    def test_retry_then_success_and_terminal_errors_are_not_retried(self):
        t = self.transports()
        t["jev"].add(self.texts[0], HttpResult(None, error="timeout"))
        t["jev"].add(self.texts[1], HttpResult(200, b"garbage"))
        self.runner(self.run, t).execute(["jev"], limit=2)
        attempts = self.store.attempts("run-1")
        self.assertEqual([a["status"] for a in attempts], ["timeout", "ok", "invalid_response"])
        self.assertEqual(attempts[0]["server_compute_unknown"], 1)
        d = decisions(self.store, "run-1")["jev"]
        self.assertEqual(d[self.manifest["items"][1]["snapshot_id"]]["probabilities"], None)

    def test_auth_error_stops_only_that_provider(self):
        t = self.transports(openai_default=lambda text: HttpResult(401, b'{"error":"invalid key"}'))
        report = self.runner(self.run, t).execute(["jev", "openai"])
        self.assertEqual(report.stopped, {"openai": "auth_error"})
        self.assertEqual(len(t["openai"].calls), 1)
        self.assertEqual(len(t["jev"].calls), 4)

    def test_call_budget_counts_failures_and_stops_before_sending(self):
        t = self.transports(jev_default=lambda text: HttpResult(429, b""))
        report = self.runner(self.run, t, Budget(3, 500, 1.0)).execute(["jev"])
        self.assertEqual(len(t["jev"].calls), 3)
        self.assertEqual(report.stopped, {"jev": "budget:max_calls"})
        # The budget spans runs in the same directory.
        run2 = self.make_run(self.manifest, self.sha, run_id="run-2")
        report = self.runner(run2, self.transports(), Budget(3, 500, 1.0)).execute(["openai"])
        self.assertEqual((report.sent, report.stopped), (0, {"openai": "budget:max_calls"}))

    def test_per_provider_and_cost_caps(self):
        t = self.transports()
        report = self.runner(self.run, t, Budget(100, 2, 1.0)).execute(["jev", "openai"])
        self.assertEqual(len(t["jev"].calls), 2)
        self.assertEqual(len(t["openai"].calls), 2)
        self.assertEqual(report.stopped, {"jev": "budget:max_calls_per_provider", "openai": "budget:max_calls_per_provider"})

        run2 = self.make_run(self.manifest, self.sha, run_id="run-2")
        t2 = self.transports()
        report = self.runner(run2, t2, Budget(100, 100, 0.0)).execute(["jev"])
        self.assertEqual((len(t2["jev"].calls), report.stopped), (0, {"jev": "budget:max_cost_usd"}))

    def test_oversized_input_is_recorded_but_not_sent(self):
        t = self.transports()
        report = self.runner(self.run, t, max_estimated_tokens=10).execute(["jev"], limit=1)
        self.assertEqual((report.sent, report.not_sent, len(t["jev"].calls)), (0, 1, 0))
        attempt = self.store.attempts("run-1")[0]
        self.assertEqual((attempt["sent"], attempt["status"]), (0, "input_too_large"))

    def test_record_contents_and_no_credentials_anywhere(self):
        t = self.transports(openai_default=lambda text: HttpResult(401, json.dumps({"error": f"bad key {SECRET}"}).encode()))
        self.runner(self.run, t).execute(["jev", "openai"], limit=1)
        self.assertEqual(t["jev"].calls[0]["headers"]["Authorization"], f"Bearer {SECRET}")
        a = [x for x in self.store.attempts("run-1") if x["provider"] == "jev"][0]
        self.assertNotIn("Authorization", json.loads(a["request_headers"]))
        self.assertEqual(json.loads(a["request_body"])["state"], self.texts[0])
        self.assertEqual((a["request_model"], a["returned_model"], a["input_tokens"]), ("jev-1.14.0", "jev-1.14.0", 100))
        self.assertAlmostEqual(a["cost_usd"], 100 * 0.042 / 1e6)
        self.assertIsNotNone(a["latency_ms"])
        self.assertEqual(a["code_revision"], "test-rev")
        run = self.store.run("run-1")
        self.assertEqual(run["questions"], QUESTIONS)
        self.assertEqual(run["manifest_sha256"], self.sha)
        self.store.conn.execute("PRAGMA wal_checkpoint")
        raw = (self.root / "eval" / "eval.sqlite").read_bytes()
        self.assertNotIn(SECRET.encode(), raw)

    def test_run_identity_is_fixed(self):
        changed = {"question_set_id": "qs-test", "questions": {"uses_voicebank": "Q1 changed?", "is_vocal_work": "Q2?"}}
        with self.assertRaisesRegex(StoreError, "question set"):
            ensure_run(self.store, "run-1", self.manifest, self.sha, changed, self.run["providers"])
        with self.assertRaisesRegex(StoreError, "provider settings"):
            ensure_run(self.store, "run-1", self.manifest, self.sha, QUESTIONS, provider_settings(["jev", "openai"], {"jev": "jev-latest"}, {}))

    def test_records_are_append_only(self):
        self.runner(self.run, self.transports()).execute(["jev"], limit=1)
        with self.assertRaises(sqlite3.DatabaseError):
            self.store.conn.execute("UPDATE attempt SET status = 'ok'")
        with self.assertRaises(sqlite3.DatabaseError):
            self.store.conn.execute("DELETE FROM run")

    def test_credentials_come_from_env_or_file(self):
        path = self.root / "keys.json"
        path.write_text(json.dumps({"jev": "file-key"}))
        import os
        from unittest import mock
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "env-key", "TYPESAFE_API_KEY": ""}):
            self.assertEqual(load_credentials(["jev", "openai"], path), {"jev": "file-key", "openai": "env-key"})
            with self.assertRaisesRegex(StoreError, "jev"):
                load_credentials(["jev"], None)


# -- dataset manifest -------------------------------------------------------------------


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        data = self.root / "data"
        init_store(data)
        clock = Clock()
        with open_store(data) as store:
            rows = ["aid,source,sample_group"] + [f"{aid},src,{'hot' if aid % 2 else 'boundary'}" for aid in range(101, 111)]
            (self.root / "c.csv").write_text("\n".join(rows) + "\n")
            import_candidates(store, self.root / "c.csv", None, clock=clock)
            from gleaner.acquire import Fetcher, candidates_without_snapshot
            t = BiliTransport()
            for aid in range(101, 110):  # 110 has no snapshot
                t.add("view", aid, bili_ok(view_body(aid, title=f"t{aid}")))
                t.add("tags", aid, bili_ok(tags_body(["x"])))
            Fetcher(store, t, max_requests=100, delay=0, clock=clock, sleep=lambda s: None).fetch(candidates_without_snapshot(store), retry=False)
            export_annotations(store, self.root / "a.csv")
            import csv
            with open(self.root / "a.csv", newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                header, rows = reader.fieldnames, list(reader)
            for row in rows:
                unknown = row["aid"] == "101"
                row.update({"uses_voicebank": "true", "is_vocal_work": "null" if unknown else "true", "text_evidence": "sufficient",
                            "rationale": "r", "null_reason": "unclear" if unknown else "", "annotator": "a"})
            with open(self.root / "a.csv", "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=header)
                writer.writeheader()
                writer.writerows(rows)
            import_annotations(store, self.root / "a.csv", "rules-1", clock=clock)
            export_bundle(store, self.root / "bundle", clock=clock)

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, rows, seed="seed-1", fraction=0.5):
        path = self.root / "sel.csv"
        path.write_text("\n".join(rows) + "\n")
        return build_manifest(self.root / "bundle", path, "ds-1", seed, fraction, ["hot"], clock=lambda: "2026-01-01T00:00:00.000Z")

    def test_forced_dev_families_and_determinism(self):
        rows = ["aid,family,split"] + [f"{a},,dev" for a in (101, 102, 103)] + ["104,fam-a,", "105,fam-a,", "106,,", "107,,", "108,,", "109,,", "110,,"]
        m1, m2 = self.build(rows), self.build(rows)
        self.assertEqual(m1["items_sha256"], m2["items_sha256"])
        by_aid = {i["aid"]: i for i in m1["items"]}
        self.assertTrue(all(by_aid[a]["split"] == "dev" for a in (101, 102, 103)))
        self.assertEqual(by_aid[104]["split"], by_aid[105]["split"])
        self.assertEqual(by_aid[110]["status"], "no_snapshot")
        self.assertEqual(by_aid[101]["is_vocal_work"], None)
        self.assertEqual(by_aid[101]["annotation_revision"], 1)
        self.assertEqual((by_aid[101]["group"], by_aid[102]["group"]), ("hot", "boundary"))
        # Unforced families follow the seeded hash, so the assignment is reproducible from the seed.
        from gleaner.dataset import family_fraction
        for aid in (106, 107, 108, 109):
            expected = "acceptance" if family_fraction("seed-1", f"aid-{aid}") < 0.5 else "dev"
            self.assertEqual(by_aid[aid]["split"], expected)
        all_acc = self.build(["aid,split"] + ["101,dev", "106,"], fraction=1.0)
        self.assertEqual([i["split"] for i in all_acc["items"]], ["dev", "acceptance"])

    def test_rejections(self):
        with self.assertRaisesRegex(StoreError, "already selected"):
            self.build(["aid", "101", "101"])
        with self.assertRaisesRegex(StoreError, "both splits"):
            self.build(["aid,family,split", "101,f,dev", "102,f,acceptance"])
        with self.assertRaisesRegex(StoreError, "revision 5 not found"):
            self.build(["aid,annotation_revision", "101,5"])

    def test_written_manifest_is_never_overwritten_and_tampering_is_detected(self):
        m = self.build(["aid", "101", "102"])
        out = self.root / "m.json"
        digest = write_manifest(m, out)
        loaded, sha = load_manifest(out)
        self.assertEqual(sha, digest)
        with self.assertRaisesRegex(StoreError, "never overwritten"):
            write_manifest(m, out)
        tampered = json.loads(out.read_text())
        tampered["items"][0]["uses_voicebank"] = False
        out2 = self.root / "m2.json"
        out2.write_text(json.dumps(tampered))
        with self.assertRaisesRegex(StoreError, "items_sha256"):
            load_manifest(out2)
        dup = make_item(1, 1, "dev", "hot", True, True, family="f")
        dup2 = make_item(2, 2, "acceptance", "hot", True, True, family="f")
        with self.assertRaisesRegex(StoreError, "both splits"):
            make_manifest([dup, dup2])


# -- offline evaluation --------------------------------------------------------------------


class EvaluationTests(EvalCase):
    """A synthetic dataset with a hand-computed confusion matrix.

    Dev items (thresholds 0.5/0.5), joint truth vs prediction:
      1 hot  T/T  p=.9/.9  -> TP
      2 hot  T/T  p=.9/.2  -> FN (popular miss; voicebank right, work wrong)
      3 hot  F/T  p=.9/.9  -> FP (voicebank wrong)
      4 ord  F/F  p=.1/.1  -> TN
      5 ord  T/null        -> no joint label; voicebank TP
      6 hot  T/T  inference fails (invalid) -> not a negative
      7 bnd  F/T  p=.2/.9  -> TN, text evidence insufficient
    Acceptance items 8-9 would push thresholds if they were used for selection.
    """

    def setUp(self):
        super().setUp()
        spec = [
            (1, "dev", "hot", True, True, (0.9, 0.9)),
            (2, "dev", "hot", True, True, (0.9, 0.2)),
            (3, "dev", "hot", False, True, (0.9, 0.9)),
            (4, "dev", "ordinary", False, False, (0.1, 0.1)),
            (5, "dev", "ordinary", True, None, (0.9, 0.9)),
            (6, "dev", "hot", True, True, "invalid"),
            (7, "dev", "boundary", False, True, (0.2, 0.9)),
            (8, "acceptance", "hot", True, True, (0.06, 0.06)),
            (9, "acceptance", "hot", False, False, (0.04, 0.04)),
        ]
        items = [make_item(i, 1000 + i, split, group, uv, wk, text_evidence="insufficient" if i == 7 else "sufficient")
                 for i, split, group, uv, wk, _ in spec]
        items.append(make_item(10, 1010, "dev", "hot", True, True, snapshot=False))
        self.manifest, self.sha = make_manifest(items)
        self.run = self.make_run(self.manifest, self.sha)
        transports = {}
        for provider in ("jev", "openai"):
            t = ModelTransport(provider)
            for (i, *_rest, probs), item in zip(spec, items):
                if probs == "invalid":
                    t.add(item["input"], HttpResult(200, b"{}"))
                else:
                    t.add(item["input"], ok_for(provider, *probs))
            transports[provider] = t
        self.runner(self.run, transports).execute(["jev", "openai"])
        self.thresholds = {
            "format": "gleaner-thresholds/1", "run_id": "run-1", "manifest_sha256": self.sha,
            "questions_sha256": self.run["questions_sha256"], "settings_sha256": self.run["settings_sha256"],
            "selected_on_split": "dev", "selection_rule": "manual", "providers": {
                p: {"model": "m", "t_uses_voicebank": 0.5, "t_is_vocal_work": 0.5} for p in ("jev", "openai")},
            "acceptance_criteria": {"max_false_positive_rate": 0.2},
        }

    def test_known_confusion_matrices(self):
        report = build_report(self.store, "run-1", "dev", self.thresholds)
        d = report["denominators"]
        self.assertEqual((d["selected"], d["acquisition_failed"], d["with_snapshot"]), (8, 1, 7))
        self.assertEqual(d["human_unknown"], {"uses_voicebank": 0, "is_vocal_work": 1, "joint": 1})
        p = report["providers"]["jev"]
        self.assertEqual(p["inference"]["ok"], 6)
        self.assertEqual(p["inference"]["by_status"], {"ok": 6, "invalid_response": 1})
        self.assertEqual(p["content"]["joint"]["confusion"], {"tp": 1, "fp": 1, "fn": 1, "tn": 2, "n": 5})
        self.assertEqual(p["content"]["joint"]["human_unknown"], 1)
        self.assertEqual(p["content"]["joint"]["inference_failed_with_known_label"], 1)
        self.assertEqual(p["content"]["uses_voicebank"]["confusion"], {"tp": 3, "fp": 1, "fn": 0, "tn": 2, "n": 6})
        self.assertEqual(p["content"]["is_vocal_work"]["confusion"], {"tp": 3, "fp": 0, "fn": 1, "tn": 1, "n": 5})
        self.assertEqual(p["content"]["is_vocal_work"]["false_positive_rate"]["value"], 0.0)
        pop = p["popular"]
        self.assertEqual((pop["known_positives"], pop["missed"], pop["inference_failed"]), (3, 1, 1))
        self.assertEqual((pop["recall"]["count"], pop["recall"]["denominator"]), (1, 2))
        self.assertEqual((pop["process_recall"]["count"], pop["process_recall"]["denominator"]), (1, 3))
        self.assertEqual(p["content"]["joint"]["false_positive_rate"]["denominator"], 3)
        self.assertEqual(p["content"]["joint"]["precision"]["value"], 0.5)
        # Groups with an empty denominator report N/A.
        boundary = p["groups"]["group:boundary"]["joint"]
        self.assertEqual((boundary["recall"]["value"], boundary["recall"]["wilson95"]), (None, None))
        self.assertEqual(p["groups"]["text_evidence:insufficient"]["items"], 1)
        errors = p["errors"]
        self.assertEqual([r["aid"] for r in errors["false_negative"]], [1002])
        self.assertEqual([r["aid"] for r in errors["false_positive"]], [1003])
        self.assertEqual([r["aid"] for r in errors["inference_failures"]], [1006])
        self.assertTrue(all("attempt_id" in r for r in errors["false_negative"] + errors["inference_failures"]))
        self.assertEqual(p["usage"]["attempts_sent"], 7)
        self.assertEqual(p["usage"]["latency_ms_successful_attempts"]["n"], 6)
        self.assertIn("N/A", render_markdown(report))

    def test_report_is_reproducible(self):
        a = build_report(self.store, "run-1", "dev", self.thresholds)
        b = build_report(self.store, "run-1", "dev", self.thresholds)
        self.assertEqual(a["report_sha256"], b["report_sha256"])
        changed = json.loads(json.dumps(self.thresholds))
        changed["providers"]["jev"]["t_is_vocal_work"] = 0.1
        c = build_report(self.store, "run-1", "dev", changed)
        self.assertNotEqual(a["report_sha256"], c["report_sha256"])
        self.assertEqual(c["providers"]["jev"]["content"]["joint"]["confusion"]["fn"], 0)

    def test_thresholds_from_another_run_are_rejected(self):
        other = dict(self.thresholds, manifest_sha256="0" * 64)
        with self.assertRaisesRegex(StoreError, "manifest_sha256"):
            build_report(self.store, "run-1", "dev", other)

    def test_selection_uses_dev_only(self):
        result = select_thresholds(self.store, "run-1", max_fpr=0.5, grid=[0.05, 0.1, 0.3, 0.5, 0.95])
        jev = result["proposal"]["providers"]["jev"]
        # Popular recall 1.0 on dev needs t_work <= 0.2; t_voicebank 0.3 and 0.5 tie on FPR (1/3),
        # and the tie-break prefers thresholds closer to 0.5. Acceptance item 8 (p=.06) would
        # favour 0.05, so a selection of 0.05 would mean acceptance data leaked in.
        self.assertEqual(result["proposal"]["selected_on_split"], "dev")
        point = [r for r in result["grid"]["jev"] if (r["t_uses_voicebank"], r["t_is_vocal_work"]) == (jev["t_uses_voicebank"], jev["t_is_vocal_work"])][0]
        self.assertLessEqual(point["false_positive_rate"], 0.5)
        self.assertEqual(point["popular_recall"], 1.0)
        self.assertEqual((jev["t_uses_voicebank"], jev["t_is_vocal_work"]), (0.5, 0.1))
        self.assertEqual(jev["dev_items_inferred"], 6)
        strict = select_thresholds(self.store, "run-1", max_fpr=0.0, grid=[0.05, 0.1, 0.3, 0.5, 0.95])
        self.assertIsNotNone(strict["proposal"]["providers"]["jev"]["t_uses_voicebank"])

    def test_freeze_once_and_acceptance_needs_frozen(self):
        with self.assertRaisesRegex(StoreError, "frozen"):
            build_report(self.store, "run-1", "acceptance", self.thresholds)
        info = freeze_thresholds(self.store, "final", self.thresholds, clock=self.clock)
        self.assertFalse(info["already_frozen"])
        self.assertTrue(freeze_thresholds(self.store, "final", self.thresholds)["already_frozen"])
        changed = json.loads(json.dumps(self.thresholds))
        changed["providers"]["jev"]["t_uses_voicebank"] = 0.05
        with self.assertRaisesRegex(StoreError, "cannot be rewritten"):
            freeze_thresholds(self.store, "final", changed)
        with self.assertRaises(sqlite3.DatabaseError):
            self.store.conn.execute("UPDATE frozen_thresholds SET content = '{}'")
        not_dev = dict(self.thresholds, selected_on_split="acceptance")
        with self.assertRaisesRegex(StoreError, "dev split"):
            freeze_thresholds(self.store, "other", not_dev)
        f = frozen(self.store, "final")
        report = build_report(self.store, "run-1", "acceptance", f["config"], f)
        p = report["providers"]["jev"]
        self.assertEqual(p["content"]["joint"]["confusion"], {"tp": 0, "fp": 0, "fn": 1, "tn": 1, "n": 2})
        self.assertEqual(report["thresholds"]["frozen"]["name"], "final")
        self.assertEqual(report["thresholds"]["acceptance_criteria"], {"max_false_positive_rate": 0.2})
        all_report = build_report(self.store, "run-1", "all", f["config"], f)
        self.assertEqual(all_report["providers"]["jev"]["groups"]["split:acceptance"]["items"], 2)

    def test_repeat_stability(self):
        t = ModelTransport("jev")
        for item in self.manifest["items"][:2]:
            t.add(item["input"], ok_for("jev", 0.9, 0.6))
        self.runner(self.run, {"jev": t}).execute(["jev"], limit=2, repeat_id="rep-1")
        s = build_report(self.store, "run-1", "dev", self.thresholds)["providers"]["jev"]["repeat_stability"]
        self.assertEqual(s[0]["repeat_id"], "rep-1")
        self.assertEqual(s[0]["paired_items"], 2)
        self.assertEqual(s[0]["joint_decision_flips"], 1)  # item 2: .2 -> .6 crosses 0.5
        self.assertAlmostEqual(s[0]["max_abs_diff"]["is_vocal_work"], 0.4)

    def test_wilson(self):
        self.assertIsNone(wilson(0, 0))
        self.assertEqual(wilson(5, 10), [0.236593, 0.763407])
        self.assertEqual(wilson(0, 5)[0], 0.0)


# -- command line --------------------------------------------------------------------


class CliTests(unittest.TestCase):
    def test_synthetic_end_to_end(self):
        examples = Path(__file__).resolve().parent.parent / "examples"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = io.StringIO()

            def run(*argv):
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()) as err:
                    code = main(list(argv))
                self.assertEqual(code, 0, err.getvalue())

            data = str(root / "data")
            run("--data-dir", data, "init")
            run("--data-dir", data, "import-candidates", str(examples / "synthetic" / "candidates.csv"))
            run("--data-dir", data, "fetch", "--max-requests", "20", "--delay", "0", "--fixtures", str(examples / "synthetic" / "responses"))
            run("--data-dir", data, "export-bundle", str(root / "bundle"))
            run("dataset-manifest", "--bundle", str(root / "bundle"), "--selection", str(examples / "synthetic" / "selection.csv"),
                "--dataset-version", "synthetic-1", "--seed", "example", "--acceptance-fraction", "0.5", "--popular-group", "popular", "--out", str(root / "manifest.json"))
            ev = ["--eval-dir", str(root / "eval")]
            common = ["eval-run", *ev, "--run-id", "demo", "--manifest", str(root / "manifest.json"), "--questions", str(examples / "questions.example.json"),
                      "--provider", "jev", "--provider", "openai", "--max-calls", "20", "--max-calls-per-provider", "10", "--max-cost-usd", "0.01", "--synthetic", "--delay", "0"]
            run(*common)
            run(*common)
            i = common.index("openai")
            subset = common[:i - 1] + common[i + 1:]
            run(*subset, "--model", "jev=jev-1.14.0")  # resuming with a subset of providers is allowed
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(main([*common, "--model", "jev=jev-latest"]), 2)
            self.assertIn("use a new run id", err.getvalue())
            run("eval-status", *ev)
            status = json.loads(out.getvalue()[out.getvalue().rindex('{\n  "frozen_thresholds"'):])
            self.assertEqual(status["spent"]["calls"], 4)
            run("eval-select", *ev, "--run-id", "demo", "--max-false-positive-rate", "1", "--out", str(root / "sel"))
            # The synthetic data has no labels, so no grid point qualifies; set dev thresholds by hand.
            proposal = json.loads((root / "sel" / "thresholds.json").read_text())
            self.assertIsNone(proposal["providers"]["jev"]["t_uses_voicebank"])
            for values in proposal["providers"].values():
                values.update({"t_uses_voicebank": 0.5, "t_is_vocal_work": 0.5})
            proposal["selection_rule"] = "manual"
            (root / "sel" / "thresholds.json").write_text(json.dumps(proposal))
            run("eval-freeze", *ev, "--name", "demo-final", "--thresholds", str(root / "sel" / "thresholds.json"))
            run("eval-report", *ev, "--run-id", "demo", "--split", "all", "--frozen", "demo-final", "--out", str(root / "report"))
            report = json.loads((root / "report" / "report.json").read_text())
            self.assertEqual(report["run"]["run_id"], "demo")
            self.assertTrue((root / "report" / "report.md").read_text().startswith("# Evaluation report"))

    def test_missing_credentials_fail_before_any_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            items = [make_item(1, 1, "dev", "hot", True, True)]
            m, _ = make_manifest(items)
            (root / "m.json").write_bytes(manifest_bytes(m))
            (root / "q.json").write_text(json.dumps(QUESTIONS))
            import os
            from unittest import mock
            err = io.StringIO()
            with mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": "", "OPENAI_API_KEY": ""}), contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                code = main(["eval-run", "--eval-dir", str(root / "eval"), "--run-id", "r", "--manifest", str(root / "m.json"), "--questions", str(root / "q.json"),
                             "--provider", "jev", "--max-calls", "1", "--max-calls-per-provider", "1", "--max-cost-usd", "0.01"])
            self.assertEqual(code, 2)
            self.assertIn("missing credentials", err.getvalue())
            with open_eval_store(root / "eval") as store:
                self.assertEqual(store.conn.execute("SELECT COUNT(*) FROM attempt").fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()
