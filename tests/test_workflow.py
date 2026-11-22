import contextlib
import csv
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from gleaner import bilibili
from gleaner.acquire import Fetcher, build_input, candidates_without_snapshot, import_candidates, input_sha256, status_summary
from gleaner.annotations import export_annotations, import_annotations
from gleaner.bundle import export_bundle, restore_bundle, verify_bundle, verify_store
from gleaner.cli import main
from gleaner.store import StoreError, init_store, open_store, resolve_data_dir

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "synthetic"


def view_body(aid, title="标题", desc="简介", **extra):
    data = {"aid": aid, "title": title, "desc": desc, **extra}
    return json.dumps({"code": 0, "message": "0", "ttl": 1, "data": data}, ensure_ascii=False).encode()


def tags_body(names):
    data = [{"tag_id": i, "tag_name": n} for i, n in enumerate(names)]
    return json.dumps({"code": 0, "message": "0", "ttl": 1, "data": data}, ensure_ascii=False).encode()


class ScriptedTransport:
    """Returns queued responses per (endpoint, aid); records every call."""

    def __init__(self):
        self.queue = {}
        self.calls = []

    def add(self, endpoint, aid, *responses):
        self.queue.setdefault((endpoint, aid), []).extend(responses)

    def __call__(self, url, aid):
        endpoint = "view" if "/view" in url else "tags"
        self.calls.append((endpoint, aid))
        items = self.queue.get((endpoint, aid))
        if not items:
            return bilibili.Response(None, None, "no scripted response")
        return items.pop(0)


def ok(body):
    return bilibili.Response(200, body)


class Clock:
    def __init__(self):
        self.n = 0

    def __call__(self):
        self.n += 1
        return f"2026-01-01T00:00:{self.n:02d}.000Z" if self.n < 60 else f"2026-01-01T00:{self.n // 60:02d}:{self.n % 60:02d}.000Z"


class StoreTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        init_store(self.data)
        self._ctx = open_store(self.data)
        self.store = self._ctx.__enter__()
        self.clock = Clock()

    def tearDown(self):
        self._ctx.__exit__(None, None, None)
        self.tmp.cleanup()

    def write_csv(self, name, header, rows):
        path = self.root / name
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def import_aids(self, *aids, source="test"):
        path = self.write_csv("aids.csv", ["aid", "source"], [[a, source] for a in aids])
        return import_candidates(self.store, path, None, self.clock)

    def fetcher(self, transport, max_requests=100, **kw):
        return Fetcher(self.store, transport, max_requests=max_requests, delay=0, clock=self.clock, **kw)

    def snapshot_ids(self, aid):
        return [r["snapshot_id"] for r in self.store.conn.execute("SELECT snapshot_id FROM snapshot WHERE aid = ? ORDER BY acquisition_id", (aid,))]

    def attempts(self, aid):
        return [dict(r) for r in self.store.conn.execute("SELECT * FROM fetch_attempt WHERE aid = ? ORDER BY id", (aid,))]


class DataDirTests(unittest.TestCase):
    def test_requires_explicit_data_dir(self):
        with self.assertRaises(StoreError):
            resolve_data_dir(None)

    def test_refuses_git_work_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ".git").mkdir()
            with self.assertRaises(StoreError):
                resolve_data_dir(str(Path(tmp) / "data"))
            self.assertEqual(resolve_data_dir(str(Path(tmp) / "data"), allow_git=True), (Path(tmp) / "data").resolve())

    def test_cli_refuses_exports_into_git_work_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "repo" / ".git").mkdir(parents=True)
            init_store(tmp / "data")
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--data-dir", str(tmp / "data"), "export-bundle", str(tmp / "repo" / "b")]), 2)
                self.assertEqual(main(["--data-dir", str(tmp / "data"), "export-annotations", str(tmp / "repo" / "l.csv")]), 2)
                self.assertEqual(main(["--data-dir", str(tmp / "repo" / "d"), "init"]), 2)
            self.assertFalse((tmp / "repo" / "b").exists())
            self.assertFalse((tmp / "repo" / "d").exists())

    def test_single_writer_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            init_store(Path(tmp))
            with open_store(Path(tmp)):
                with self.assertRaises(StoreError):
                    with open_store(Path(tmp)):
                        pass


class CandidateTests(StoreTestCase):
    def test_same_aid_from_several_sources_is_one_candidate(self):
        path = self.write_csv("c.csv", ["aid", "source", "locator", "observed_view"], [
            [101, "ranking", "week-1", "500"],
            [101, "search", "query=a", ""],
            [102, "search", "query=a", ""],
        ])
        counts = import_candidates(self.store, path, None, self.clock)
        self.assertEqual(counts, {"rows": 3, "new_candidates": 2, "new_discoveries": 3})
        again = import_candidates(self.store, path, None, self.clock)
        self.assertEqual(again["new_candidates"], 0)
        self.assertEqual(again["new_discoveries"], 0)
        rows = self.store.conn.execute("SELECT source, observed FROM discovery WHERE aid = 101 ORDER BY id").fetchall()
        self.assertEqual([r["source"] for r in rows], ["ranking", "search"])
        self.assertEqual(json.loads(rows[0]["observed"]), {"view": "500"})

    def test_rejects_invalid_aid_and_missing_source(self):
        with self.assertRaises(StoreError):
            import_candidates(self.store, self.write_csv("a.csv", ["aid", "source"], [["abc", "x"]]), None)
        with self.assertRaises(StoreError):
            import_candidates(self.store, self.write_csv("b.csv", ["aid"], [["5"]]), None)
        self.assertEqual(import_candidates(self.store, self.write_csv("c.csv", ["aid"], [["5"]]), "manual")["new_candidates"], 1)

    def test_discoveries_are_append_only(self):
        self.import_aids(7)
        with self.assertRaises(sqlite3.DatabaseError):
            self.store.conn.execute("DELETE FROM discovery")


class SnapshotTests(StoreTestCase):
    def test_complete_fetch_creates_fixed_input_and_hash(self):
        self.import_aids(1)
        t = ScriptedTransport()
        t.add("view", 1, ok(view_body(1, title="【示例】歌", desc='含 "引号"\n换行', owner={"name": "不进模型"})))
        t.add("tags", 1, ok(tags_body(["b", "a", "b"])))
        report = self.fetcher(t).fetch([1], retry=False)
        self.assertEqual(report.snapshots, ["snap-1-1"])
        snap = self.store.conn.execute("SELECT * FROM snapshot").fetchone()
        expected = '{"title":"【示例】歌","desc":"含 \\"引号\\"\\n换行","tags":["b","a","b"]}'
        self.assertEqual(snap["input_text"], expected)
        self.assertEqual(snap["input_sha256"], input_sha256(expected))
        self.assertEqual(list(json.loads(snap["input_text"]).keys()), ["title", "desc", "tags"])
        self.assertEqual(snap["input_schema_version"], "v1")
        self.assertEqual(verify_store(self.store), [])

    def test_valid_empty_desc_and_tags(self):
        self.import_aids(2)
        t = ScriptedTransport()
        t.add("view", 2, ok(view_body(2, desc="")))
        t.add("tags", 2, ok(tags_body([])))
        self.fetcher(t).fetch([2], retry=False)
        snap = self.store.conn.execute("SELECT input_text FROM snapshot").fetchone()
        self.assertEqual(json.loads(snap["input_text"]), {"title": "标题", "desc": "", "tags": []})

    def test_missing_field_wrong_type_and_null_tags_are_invalid(self):
        cases = {
            3: (ok(json.dumps({"code": 0, "data": {"aid": 3, "title": "t"}}).encode()), "missing_field"),
            4: (ok(json.dumps({"code": 0, "data": {"aid": 4, "title": 5, "desc": ""}}).encode()), "type_error"),
            5: (ok(b"<html>not json</html>"), "not_json"),
        }
        self.import_aids(*cases)
        t = ScriptedTransport()
        for aid, (resp, _) in cases.items():
            t.add("view", aid, resp)
        self.fetcher(t).fetch(list(cases), retry=False)
        for aid, (_, error_class) in cases.items():
            attempts = self.attempts(aid)
            self.assertEqual([(a["endpoint"], a["status"], a["error_class"]) for a in attempts], [("view", "invalid_response", error_class)])
            self.assertEqual(self.snapshot_ids(aid), [])
        self.assertNotIn(("tags", 3), t.calls)  # terminal view failure saves the tags request
        self.import_aids(6)
        t.add("view", 6, ok(view_body(6)))
        t.add("tags", 6, ok(json.dumps({"code": 0, "data": None}).encode()))
        self.fetcher(t).fetch([6], retry=False)
        self.assertEqual(self.attempts(6)[1]["error_class"], "missing_field")
        self.assertEqual(self.snapshot_ids(6), [])

    def test_aid_mismatch(self):
        self.import_aids(8)
        t = ScriptedTransport()
        t.add("view", 8, ok(view_body(9)))
        self.fetcher(t).fetch([8], retry=False)
        attempt = self.attempts(8)[0]
        self.assertEqual((attempt["status"], attempt["error_class"]), ("invalid_response", "aid_mismatch"))
        self.assertIsNotNone(attempt["raw_sha256"])
        self.assertEqual(status_summary(self.store)["per_aid"][0]["latest_acquisition_state"], "failed")

    def test_unavailable_versus_retryable(self):
        self.import_aids(10, 11, 12)
        t = ScriptedTransport()
        t.add("view", 10, ok(json.dumps({"code": -404, "message": "x"}).encode()))
        t.add("view", 11, bilibili.Response(403, b'{"code":-403}'))
        t.add("tags", 11, ok(tags_body(["x"])))
        t.add("view", 12, ok(json.dumps({"code": -412, "message": "blocked"}).encode()))
        t.add("tags", 12, bilibili.Response(None, None, "timeout"))
        self.fetcher(t).fetch([10, 11, 12], retry=False)
        self.assertEqual([(a["status"], a["api_code"]) for a in self.attempts(10)], [("unavailable", -404)])
        self.assertEqual([(a["endpoint"], a["status"], a["http_status"]) for a in self.attempts(11)],
                         [("view", "retryable_error", 403), ("tags", "ok", 200)])
        a12 = self.attempts(12)
        self.assertEqual([(a["status"], a["error_class"], a["api_code"]) for a in a12],
                         [("retryable_error", "api_code", -412), ("retryable_error", "transport", None)])
        self.assertIsNone(a12[1]["raw_path"])

    def test_partial_failure_is_kept_and_retry_completes_same_acquisition(self):
        self.import_aids(20)
        t = ScriptedTransport()
        t.add("view", 20, ok(view_body(20)))
        t.add("tags", 20, bilibili.Response(500, b"oops"))
        self.fetcher(t).fetch([20], retry=False)
        self.assertEqual(self.snapshot_ids(20), [])
        self.assertEqual(status_summary(self.store)["per_aid"][0]["latest_acquisition_state"], "retryable")
        # Without --retry nothing is requested.
        report = self.fetcher(t).fetch(candidates_without_snapshot(self.store), retry=False)
        self.assertEqual(report.requests, 0)
        self.assertEqual(report.skipped[0]["state"], "retryable")
        # With --retry only the failed endpoint is requested again.
        t.add("tags", 20, ok(tags_body(["x"])))
        report = self.fetcher(t).fetch(candidates_without_snapshot(self.store), retry=True)
        self.assertEqual(report.requests, 1)
        self.assertEqual(self.snapshot_ids(20), ["snap-20-1"])
        tags_attempts = [a for a in self.attempts(20) if a["endpoint"] == "tags"]
        self.assertEqual([(a["attempt_no"], a["status"]) for a in tags_attempts], [(1, "retryable_error"), (2, "ok")])
        self.assertEqual(tags_attempts[1]["retry_of"], tags_attempts[0]["id"])

    def test_attempts_are_capped_per_acquisition(self):
        self.import_aids(21)
        t = ScriptedTransport()
        t.add("view", 21, bilibili.Response(503, b""), bilibili.Response(503, b""))
        t.add("tags", 21, ok(tags_body([])))
        self.fetcher(t).fetch([21], retry=False)
        self.fetcher(t).fetch([21], retry=True)
        report = self.fetcher(t).fetch([21], retry=True)
        self.assertEqual(report.requests, 0)
        self.assertEqual(report.skipped[0]["state"], "failed")

    def test_explicit_refresh_creates_new_snapshot_and_keeps_old(self):
        self.import_aids(30)
        t = ScriptedTransport()
        t.add("view", 30, ok(view_body(30, title="旧")), ok(view_body(30, title="新")))
        t.add("tags", 30, ok(tags_body(["a"])), ok(tags_body(["a"])))
        self.fetcher(t).fetch([30], retry=False)
        old = self.store.conn.execute("SELECT * FROM snapshot").fetchone()
        self.assertEqual(self.fetcher(t).fetch(candidates_without_snapshot(self.store), retry=False).requests, 0)
        self.fetcher(t).refresh([30])
        snaps = self.store.conn.execute("SELECT * FROM snapshot ORDER BY acquisition_id").fetchall()
        self.assertEqual([s["snapshot_id"] for s in snaps], ["snap-30-1", "snap-30-2"])
        self.assertEqual(dict(snaps[0]), dict(old))
        self.assertEqual(json.loads(snaps[1]["input_text"])["title"], "新")
        reasons = [r["reason"] for r in self.store.conn.execute("SELECT reason FROM acquisition ORDER BY id")]
        self.assertEqual(reasons, ["initial", "refresh"])

    def test_failed_refresh_does_not_reuse_older_fields(self):
        self.import_aids(31)
        t = ScriptedTransport()
        t.add("view", 31, ok(view_body(31)), ok(view_body(31, title="新")))
        t.add("tags", 31, ok(tags_body(["a"])), bilibili.Response(None, None, "timeout"))
        self.fetcher(t).fetch([31], retry=False)
        self.fetcher(t).refresh([31])
        self.assertEqual(self.snapshot_ids(31), ["snap-31-1"])

    def test_snapshots_and_attempts_are_immutable(self):
        self.import_aids(40)
        t = ScriptedTransport()
        t.add("view", 40, ok(view_body(40)))
        t.add("tags", 40, ok(tags_body([])))
        self.fetcher(t).fetch([40], retry=False)
        for sql in ("UPDATE snapshot SET input_text = 'x'", "DELETE FROM snapshot", "UPDATE fetch_attempt SET status = 'ok'", "DELETE FROM acquisition"):
            with self.assertRaises(sqlite3.DatabaseError, msg=sql):
                self.store.conn.execute(sql)

    def test_request_budget_is_a_hard_cap_and_resumes(self):
        self.import_aids(50, 51)
        t = ScriptedTransport()
        for aid in (50, 51):
            t.add("view", aid, ok(view_body(aid)))
            t.add("tags", aid, ok(tags_body([])))
        report = self.fetcher(t, max_requests=3).fetch([50, 51], retry=False)
        self.assertEqual((report.requests, report.budget_exhausted), (3, True))
        self.assertEqual(len(t.calls), 3)
        self.assertEqual(status_summary(self.store)["latest_acquisition_states"], {"complete": 1, "incomplete": 1})
        report = self.fetcher(t, max_requests=3).fetch(candidates_without_snapshot(self.store), retry=False)
        self.assertEqual(report.requests, 1)
        self.assertEqual(self.snapshot_ids(51), ["snap-51-2"])

    def test_verify_detects_tampered_raw_file(self):
        self.import_aids(60)
        t = ScriptedTransport()
        t.add("view", 60, ok(view_body(60)))
        t.add("tags", 60, ok(tags_body([])))
        self.fetcher(t).fetch([60], retry=False)
        rel = self.attempts(60)[0]["raw_path"]
        (self.data / rel).write_bytes(view_body(60, title="改"))
        self.assertTrue(any("hash mismatch" in p for p in verify_store(self.store)))


class AnnotationTests(StoreTestCase):
    def setUp(self):
        super().setUp()
        self.import_aids(1, 2)
        t = ScriptedTransport()
        for aid in (1, 2):
            t.add("view", aid, ok(view_body(aid, title=f"标题{aid}")))
            t.add("tags", aid, ok(tags_body(["x"])))
        self.fetcher(t).fetch([1, 2], retry=False)
        self.snapshot_before = [dict(r) for r in self.store.conn.execute("SELECT * FROM snapshot ORDER BY snapshot_id")]

    def export(self, name="labels.csv"):
        path = self.root / name
        export_annotations(self.store, path)
        with open(path, newline="", encoding="utf-8-sig") as handle:
            return path, list(csv.DictReader(handle))

    def write_rows(self, path, rows):
        with open(path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def fill(self, row, **values):
        base = {"uses_voicebank": "true", "is_vocal_work": "true", "text_evidence": "sufficient",
                "rationale": "示例依据", "evidence_locator": "00:30", "null_reason": "", "annotator": "tester"}
        row.update(base)
        row.update(values)
        return row

    def revisions(self, sid):
        return [dict(r) for r in self.store.conn.execute("SELECT * FROM annotation WHERE snapshot_id = ? ORDER BY revision", (sid,))]

    def test_round_trip_with_null_and_history(self):
        path, rows = self.export()
        self.assertEqual([r["base_revision"] for r in rows], ["0", "0"])
        self.assertEqual(rows[0]["title"], "标题1")
        self.fill(rows[0])
        self.fill(rows[1], uses_voicebank="null", text_evidence="uncertain", null_reason="混合演唱，待规则确认")
        self.write_rows(path, rows)
        result = import_annotations(self.store, path, "rules-1", clock=self.clock)
        self.assertEqual(result["revisions_added"], 2)
        rev = self.revisions("snap-2-2")[0]
        self.assertEqual((rev["uses_voicebank"], rev["is_vocal_work"], rev["rules_version"]), (None, 1, "rules-1"))

        path2, rows2 = self.export("labels2.csv")
        self.assertEqual(rows2[1]["uses_voicebank"], "null")
        self.assertEqual(rows2[0]["base_revision"], "1")
        rows2[0]["is_vocal_work"] = "false"
        rows2[0]["rationale"] = "复查后修改"
        self.write_rows(path2, rows2)
        result = import_annotations(self.store, path2, "rules-2", clock=self.clock)
        self.assertEqual((result["revisions_added"], result["unchanged"]), (1, 1))
        history = self.revisions("snap-1-1")
        self.assertEqual([(h["revision"], h["is_vocal_work"]) for h in history], [(1, 1), (2, 0)])
        after = [dict(r) for r in self.store.conn.execute("SELECT * FROM snapshot ORDER BY snapshot_id")]
        self.assertEqual(after, self.snapshot_before)

    def test_reimporting_same_file_is_rejected_as_stale(self):
        path, rows = self.export()
        self.fill(rows[0])
        self.write_rows(path, rows)
        import_annotations(self.store, path, "rules-1")
        result = import_annotations(self.store, path, "rules-1")
        self.assertEqual(result["revisions_added"], 0)
        self.assertIn("stale or duplicate revision", result["rejected"][0])
        self.assertEqual(len(self.revisions("snap-1-1")), 1)

    def test_invalid_rows_are_skipped_and_valid_rows_applied(self):
        path, rows = self.export()
        self.fill(rows[0])
        self.fill(rows[1], uses_voicebank="maybe")
        self.write_rows(path, rows)
        result = import_annotations(self.store, path, "rules-1")
        self.assertEqual(result["revisions_added"], 1)
        self.assertEqual(len(result["rejected"]), 1)
        self.assertTrue(result["rejected"][0].startswith("line 3:"))
        self.assertEqual(len(self.revisions("snap-1-1")), 1)
        self.assertEqual(self.revisions("snap-2-2"), [])

    def test_validation_errors(self):
        path, rows = self.export()
        bad = [
            (dict(self.fill(dict(rows[0]), uses_voicebank="null")), "null_reason is required"),
            (dict(self.fill(dict(rows[0]), text_evidence="enough")), "text_evidence"),
            (dict(self.fill(dict(rows[0]), rationale="")), "rationale is required"),
            (dict(self.fill(dict(rows[0]), annotator="")), "annotator is required"),
            (dict(self.fill(dict(rows[0]), input_sha256="0" * 64)), "input_sha256 does not match"),
            (dict(self.fill(dict(rows[0]), aid="2")), "belongs to aid"),
            (dict(self.fill(dict(rows[0]), snapshot_id="snap-404")), "unknown snapshot_id"),
            (dict(self.fill(dict(rows[0]), base_revision="3")), "stale or duplicate"),
        ]
        for row, message in bad:
            self.write_rows(path, [row])
            result = import_annotations(self.store, path, "rules-1")
            self.assertEqual(result["revisions_added"], 0, message)
            self.assertIn(message, "\n".join(result["rejected"]))
        self.write_rows(path, [self.fill(dict(rows[0])), self.fill(dict(rows[0], rationale="另一个"))])
        result = import_annotations(self.store, path, "rules-1")
        self.assertEqual(result["revisions_added"], 0)
        self.assertEqual(len(result["rejected"]), 2)
        self.assertIn("duplicate rows", result["rejected"][0])
        self.assertEqual(self.store.conn.execute("SELECT COUNT(*) FROM annotation").fetchone()[0], 0)

    def test_context_columns_are_not_inputs(self):
        path, rows = self.export()
        self.fill(rows[0], title="被篡改的标题", tags='["fake"]')
        self.write_rows(path, rows)
        import_annotations(self.store, path, "rules-1")
        after = [dict(r) for r in self.store.conn.execute("SELECT * FROM snapshot ORDER BY snapshot_id")]
        self.assertEqual(after, self.snapshot_before)

    def test_untouched_rows_are_skipped_and_default_annotator(self):
        path, rows = self.export()
        self.fill(rows[0], annotator="")
        self.write_rows(path, rows)
        result = import_annotations(self.store, path, "rules-1", default_annotator="cli-user")
        self.assertEqual((result["revisions_added"], result["untouched"]), (1, 1))
        self.assertEqual(self.revisions("snap-1-1")[0]["annotator"], "cli-user")


class BundleTests(StoreTestCase):
    def populate(self):
        self.import_aids(1, 2, 3)
        t = ScriptedTransport()
        t.add("view", 1, ok(view_body(1)))
        t.add("tags", 1, ok(tags_body(["声库"])))
        t.add("view", 2, bilibili.Response(412, b"blocked"))
        t.add("tags", 2, bilibili.Response(None, None, "timeout"))
        t.add("view", 3, ok(json.dumps({"code": -404}).encode()))
        self.fetcher(t).fetch([1, 2, 3], retry=False)
        path = self.root / "labels.csv"
        export_annotations(self.store, path)
        with open(path, newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        rows[0].update({"uses_voicebank": "true", "is_vocal_work": "null", "text_evidence": "insufficient",
                        "rationale": "r", "null_reason": "合集", "annotator": "a"})
        with open(path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        import_annotations(self.store, path, "rules-1", clock=self.clock)

    def test_export_restore_and_verify(self):
        self.populate()
        bundle = self.root / "bundle"
        result = export_bundle(self.store, bundle, clock=self.clock)
        self.assertEqual(result["counts"], {"candidates": 3, "fetch_attempts": 5, "snapshots": 1, "annotations": 1, "dataset": 1, "raw_files": 4})
        self.assertEqual(verify_bundle(bundle), [])
        attempts = [json.loads(l) for l in (bundle / "fetch_attempts.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(sorted(a["status"] for a in attempts), ["ok", "ok", "retryable_error", "retryable_error", "unavailable"])
        self.assertEqual({a["status"] for a in attempts}, {"ok", "retryable_error", "unavailable"})
        dataset = [json.loads(l) for l in (bundle / "dataset.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(dataset[0]["annotation_revision"], 1)
        self.assertIsNone(dataset[0]["is_vocal_work"])
        self.assertEqual(dataset[0]["input_sha256"], input_sha256(dataset[0]["input"]))

        restored = self.root / "restored"
        out = restore_bundle(bundle, restored)
        self.assertTrue(out["verified"])
        with open_store(restored) as store:
            self.assertEqual(verify_store(store), [])
            self.assertEqual(status_summary(store)["endpoint_requests"], 5)
        with self.assertRaises(StoreError):
            restore_bundle(bundle, restored)  # target not empty

    def test_tampered_bundle_is_rejected(self):
        self.populate()
        bundle = self.root / "bundle"
        export_bundle(self.store, bundle)
        path = bundle / "dataset.jsonl"
        path.write_text(path.read_text(encoding="utf-8").replace('"annotation_revision":1', '"annotation_revision":2'), encoding="utf-8")
        self.assertIn("hash mismatch: dataset.jsonl", verify_bundle(bundle))
        with self.assertRaises(StoreError):
            restore_bundle(bundle, self.root / "r2")
        (bundle / "extra.txt").write_text("x")
        self.assertIn("unlisted file: extra.txt", verify_bundle(bundle))

    def test_refuses_non_empty_output(self):
        self.populate()
        out = self.root / "busy"
        out.mkdir()
        (out / "x").write_text("x")
        with self.assertRaises(StoreError):
            export_bundle(self.store, out)


class SyntheticExampleTests(unittest.TestCase):
    """Runs the documented CLI walkthrough against the synthetic fixtures."""

    def test_cli_walkthrough(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = str(tmp / "data")

            def cli(*args, data_dir=data):
                argv = ["--data-dir", data_dir, *args] if data_dir else list(args)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    return main(argv)

            self.assertEqual(cli("init"), 0)
            self.assertEqual(cli("import-candidates", str(EXAMPLES / "candidates.csv")), 0)
            fixtures = str(EXAMPLES / "responses")
            self.assertEqual(cli("fetch", "--max-requests", "20", "--delay", "0", "--fixtures", fixtures), 0)
            with open_store(Path(data)) as store:
                summary = status_summary(store)
                self.assertEqual(summary["candidates"], 6)
                self.assertEqual(summary["discoveries"], 7)
                self.assertEqual(summary["snapshots"], 2)
                states = {p["aid"]: p["latest_acquisition_state"] for p in summary["per_aid"]}
                self.assertEqual(states, {900000001: "complete", 900000002: "complete", 900000003: "retryable",
                                          900000004: "failed", 900000005: "failed", 900000006: "failed"})
                snap = store.conn.execute("SELECT input_text FROM snapshot WHERE aid = 900000002").fetchone()
                self.assertEqual(json.loads(snap["input_text"]), {"title": "真人翻唱示例", "desc": "", "tags": []})
            labels = tmp / "labels.csv"
            self.assertEqual(cli("export-annotations", str(labels)), 0)
            with open(labels, newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            rows[0].update({"uses_voicebank": "true", "is_vocal_work": "true", "text_evidence": "sufficient", "rationale": "虚构样例"})
            rows[1].update({"uses_voicebank": "false", "is_vocal_work": "true", "text_evidence": "insufficient", "rationale": "虚构样例"})
            with open(labels, "w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            self.assertEqual(cli("import-annotations", str(labels), "--rules-version", "example", "--annotator", "demo"), 0)
            self.assertEqual(cli("import-annotations", str(labels), "--rules-version", "example", "--annotator", "demo"), 1)
            self.assertEqual(cli("verify"), 0)
            self.assertEqual(cli("export-bundle", str(tmp / "bundle")), 0)
            self.assertEqual(cli("verify-bundle", str(tmp / "bundle"), data_dir=None), 0)
            self.assertEqual(cli("restore-bundle", str(tmp / "bundle"), data_dir=str(tmp / "restored")), 0)


class InputTests(unittest.TestCase):
    def test_build_input_is_compact_and_unescaped(self):
        self.assertEqual(build_input("歌", "", ["a"]), '{"title":"歌","desc":"","tags":["a"]}')


if __name__ == "__main__":
    unittest.main()
