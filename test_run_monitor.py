"""Offline tests for the monitoring and failure control every stage shares.

The point of these is the second half of the contract: a failing item must not take the
loop with it, must print completely, and must end up in the ledger and in failures.json;
a healthy run must stay quiet per call and dense per item.
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import dental_eval as ev
import dental_pipeline as dp
import location_adapter as la
import report_writer as rw
import run_monitor as mon

DENTVLM = hasattr(dp, "extract_answer")
NO = "no" if DENTVLM else "B"


def protocol(**kwargs):
    """The cheapest protocol on either branch: whole-image presence, no counts, no crops."""
    base = {"location": "rationale"} if DENTVLM else {"presence_level": "overall", "counting": False}
    return dp.Protocol(**dict(base, **kwargs))


def reply(text):
    return {"text": text, "finish_reason": "stop", "truncated": False,
            "prompt_tokens": 10, "completion_tokens": 2, "latency_seconds": 0.0}


class FlakyRunner:
    """Answers every question, except on the images named in `fail`, where the request raises."""

    def __init__(self, fail=()):
        self.fail, self.asked = set(fail), []

    def settings(self):
        return {"model": "fake"}

    def ask(self, image, question):
        stem = "crop" if isinstance(image, bytes) else Path(image).stem
        self.asked.append(stem)
        if stem in self.fail:
            raise RuntimeError(f"Error code: 500 - upstream refused {stem}")
        return reply("0" if question.startswith("How many") else ("No" if DENTVLM else "B"))


def _images(root, names):
    images = {}
    for name in names:
        path = root / f"{name}.png"
        path.write_bytes(b"not a real image, and no image decoder is needed: " + name.encode())
        images[name] = path
    return images


class FormattingTests(unittest.TestCase):
    def test_durations_and_counts_stay_short(self):
        self.assertEqual(mon.human_time(8.44), "8.4s")
        self.assertEqual(mon.human_time(192), "3m12s")
        self.assertEqual(mon.human_time(3900), "1h05m")
        self.assertEqual(mon.human_time(None), "-")
        self.assertEqual([mon.human_count(v) for v in (0, 812, 12400, 2_500_000)], ["0", "812", "12.4k", "2.5M"])

    def test_clip_is_one_line_and_says_what_is_missing(self):
        clipped = mon.clip("a\nb   c" + "x" * 300, 10)
        self.assertEqual(len(clipped.split("...")[0]), 10)
        self.assertIn("(+295 chars)", clipped)
        self.assertNotIn("\n", clipped)
        self.assertEqual(mon.clip("short", 10), "short")


class CallLogTests(unittest.TestCase):
    def _log(self, policy, replies):
        log = mon.CallLog("analyzer", policy)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            for item in replies:
                log.live(item)
        return log, out.getvalue()

    def test_healthy_calls_are_counted_not_listed(self):
        log, printed = self._log("auto", [reply("A") for _ in range(5)])
        self.assertEqual(printed.count("ANALYZER CALL"), 1)  # the first call only
        self.assertEqual(log.calls, 5)
        self.assertIn("calls=5", log.line())

    def test_notable_calls_always_print(self):
        truncated = {**reply("A"), "truncated": True, "finish_reason": "length"}
        log, printed = self._log("auto", [reply("A"), reply("A"), truncated, {**reply(""), "latency_seconds": 999.0}])
        self.assertEqual(printed.count("ANALYZER CALL"), 3)
        self.assertIn("TRUNCATED", printed)
        self.assertIn("EMPTY", printed)
        self.assertIn("SLOW", printed)
        self.assertEqual((log.truncated, log.empty, log.slow), (1, 1, 1))

    def test_each_and_off(self):
        _, each = self._log("each", [reply("A") for _ in range(4)])
        self.assertEqual(each.count("ANALYZER CALL"), 4)
        _, off = self._log("off", [reply("A"), {**reply("A"), "truncated": True}])
        self.assertEqual(off, "")
        with self.assertRaises(ValueError):
            mon.CallLog("analyzer", "loud")

    def test_cache_hits_are_counted_separately(self):
        log = mon.CallLog("analyzer", "off")
        log.cached(reply("A"))
        log.live(reply("A"))
        self.assertEqual((log.requests, log.calls, log.cache_hits), (2, 1, 1))
        self.assertIn("cache=1", log.line())
        self.assertNotIn("calls=", log.line(counts=False))


class LedgerAndGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_failure_prints_completely_and_is_recorded(self):
        ledger = mon.Ledger("sweep")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            with mon.guard("umfih/img1", ledger) as step:
                raise RuntimeError("Error code: 500 - {'error': 'the whole provider message'}")
        printed = out.getvalue()
        self.assertFalse(step.ok)
        self.assertIsInstance(step.error, RuntimeError)
        self.assertIn("[FAILED] umfih/img1", printed)
        self.assertIn("the whole provider message", printed)   # never clipped
        self.assertIn("TRACEBACK (full):", printed)
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger.entries[0]["reason"], "RuntimeError")

    def test_healthy_step_is_untouched(self):
        ledger = mon.Ledger()
        with mon.guard("scope", ledger) as step:
            value = 2 + 2
        self.assertTrue(step.ok)
        self.assertIsNone(step.error)
        self.assertEqual(value, 4)
        self.assertFalse(ledger)

    def test_interrupts_and_named_types_still_stop_the_run(self):
        ledger = mon.Ledger()
        with self.assertRaises(KeyboardInterrupt), contextlib.redirect_stdout(io.StringIO()):
            with mon.guard("scope", ledger):
                raise KeyboardInterrupt
        with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
            with mon.guard("scope", ledger, reraise=(ValueError,)):
                raise ValueError("configuration, not a bad image")
        self.assertFalse(ledger)

    def test_report_groups_reasons_and_saves_tracebacks(self):
        ledger = mon.Ledger("sweep")
        with contextlib.redirect_stdout(io.StringIO()):
            for image in ("a", "b"):
                with mon.guard(f"run/{image}", ledger):
                    raise RuntimeError("boom")
            with mon.guard("run/c", ledger):
                raise OSError("disk")
            ledger.note("run/d", "no results yet", path="somewhere")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            ledger.report(path=self.root / "failures.json")
        printed = out.getvalue()
        self.assertIn("failures=4", printed)
        self.assertIn("RuntimeError x2: run/a, run/b", printed)
        self.assertIn("no results yet x1: run/d", printed)
        saved = json.loads((self.root / "failures.json").read_text())
        self.assertEqual(len(saved["failures"]), 4)
        self.assertIn("Traceback", saved["failures"][0]["traceback"])

    def test_nothing_is_saved_when_nothing_failed(self):
        ledger = mon.Ledger("clean")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            ledger.report(path=self.root / "failures.json")
        self.assertIn("failures=0", out.getvalue())
        self.assertFalse((self.root / "failures.json").exists())


class ProgressTests(unittest.TestCase):
    def test_counters_and_consecutive_failures(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            progress = mon.Progress(5, label="umfih_test", unit="image")
            progress.item("img1", "present=2 absent=12 unclear=0", calls=14, cache=None)
            progress.skip("img2")
            self.assertEqual(progress.failure("img3"), 1)
            self.assertEqual(progress.failure("img4"), 2)
            progress.item("img5", "", calls=14)
            self.assertEqual(progress.consecutive_failures, 0)
            summary = progress.done()
        printed = out.getvalue()
        self.assertIn("[1/5] img1", printed)
        self.assertIn("present=2 absent=12 unclear=0", printed)
        self.assertIn("calls=14", printed)
        self.assertNotIn("cache=", printed)          # a None counter is left out
        self.assertIn("done=2 resumed=1 failed=2", printed)
        self.assertEqual(summary["done"], 2)
        self.assertEqual(summary["failed"], 2)
        self.assertEqual(summary["calls"], 28)




class AdapterAndReportFailureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.gt = {}
        for name in ("img1", "img2", "img3"):
            path = self.root / f"{name}.png"
            path.write_bytes(b"bytes " + name.encode())
            self.gt[name] = {"path": str(path), "annotated": set(ev.CONDITIONS),
                             "boxes": [{"condition": ev.CONDITIONS[0], "xc": 0.3, "yc": 0.3, "w": 0.1, "h": 0.1}]}

    def tearDown(self):
        self.temp.cleanup()

    def test_adapt_dataset_steps_over_a_failing_image(self):
        class Adapter:
            kind, name = "fake", "fake-adapter"

            def settings(self):
                return {"kind": "fake"}

            def adapt(self, image_path, boxes, image_id=None, drawn_dir=None):
                if image_id == "img2":
                    raise RuntimeError("Error code: 429 - adapter rate limited")
                return [{"regions": None, "units": None, "teeth": [], "source": None, "raw": "",
                         "attempts": [], "fallback_reason": None} for _ in boxes]

        ledger = mon.Ledger("session")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            adapted = la.adapt_dataset(Adapter(), self.gt, self.root / "truth", ledger=ledger)
        self.assertEqual(sorted(adapted), ["img1", "img3"])
        self.assertEqual([e["scope"] for e in ledger.entries], ["truth/img2"])
        self.assertIn("adapter rate limited", out.getvalue())
        self.assertTrue((self.root / "truth" / "failures.json").is_file())

    def test_report_dataset_steps_over_a_failing_image(self):
        results = {}
        with contextlib.redirect_stdout(io.StringIO()):
            run_dir = dp.run_dataset(FlakyRunner(), _images(self.root, ["img1", "img2", "img3"]),
                                     self.root / "run", protocol=protocol())
            results = dp.load_results(run_dir)

        class Writer:
            run_name = "fake-english"

            def settings(self):
                return {"kind": "report", "model": "fake"}

            def public(self):
                return {"model": "fake"}

            def write(self, result, analyzer=None):
                image_id = Path(result["image"]).stem
                if image_id == "img2":
                    raise RuntimeError("Error code: 401 - report key rejected")
                return {"image_id": image_id, "image": result["image"], "verified": True, "problems": [],
                        "attempts": [{"completion_tokens": 5}], "markdown": "# report", "report": {},
                        "structured": {}, "language": "English"}

        ledger = mon.Ledger("session")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            written = rw.report_dataset(Writer(), results, self.root / "reports", ledger=ledger)
        self.assertEqual(sorted(written), ["img1", "img3"])
        self.assertEqual([e["scope"] for e in ledger.entries], ["reports/img2"])
        self.assertIn("report key rejected", out.getvalue())


class TruthReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "images").mkdir()
        (self.root / "labels").mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def test_missing_labels_and_orphan_labels_are_reported(self):
        for name in ("a", "b"):
            (self.root / "images" / f"{name}.png").write_bytes(b"x")
        (self.root / "labels" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
        (self.root / "labels" / "ghost.txt").write_text("0 0.5 0.5 0.1 0.1\n")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            gt = ev.load_yolo(self.root / "images", self.root / "labels")
        printed = out.getvalue()
        self.assertIn("images with no label file", printed)
        self.assertIn("label files with no image", printed)
        self.assertIn("ghost", printed)
        self.assertEqual(sorted(gt), ["a", "b"])

    def test_truth_report_is_one_dense_line_plus_problems(self):
        (self.root / "images" / "a.png").write_bytes(b"x")
        (self.root / "labels" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n0 0.2 0.2 0.1 0.1\n")
        with contextlib.redirect_stdout(io.StringIO()):
            gt = ev.load_yolo(self.root / "images", self.root / "labels")
        gt["gone"] = {"path": str(self.root / "images" / "gone.png"), "boxes": [], "annotated": set(ev.CONDITIONS)}
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            summary = ev.truth_report(gt, "umfih_test")
        printed = out.getvalue()
        self.assertIn("[TRUTH] umfih_test", printed)
        self.assertIn("images=2", printed)
        self.assertIn("boxes per finding:", printed)
        self.assertIn("image files missing", printed)
        self.assertEqual(summary["boxes"], 2)
        self.assertEqual(summary["missing_image_files"], 1)


if __name__ == "__main__":
    unittest.main()
