"""Synthetic evidence-export checks; no training and no network access."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import build_suite_report as report
import export_suite_evidence as exporter
import test_suite_report as report_tests


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = report_tests.ReportTests("test_macro_is_not_pooled_and_MAPE_denominators_are_distinct")
        self.fixture.setUp()
        self.folder = self.fixture.folder
        self.config, self.manifest = self.fixture.store_task()
        (self.folder / "model.pth").write_bytes(b"synthetic model identity; exporter does not load weights")
        (self.folder / "logging.txt").write_text(
            "[Train] epoch:1, lr:0.001, total loss:0.1\n[Valid] epoch:1, MSE: 0.03\n"
            "[Test] MSE: 0.01, MAE: 0.02, MAPE: 0.03, RMSE: 0.1\n"
            "[Train] epoch:2, lr:0.001, total loss:0.09\n[Valid] epoch:2, MSE: 0.02\n"
            "[Train] epoch:3, lr:0.0009, total loss:0.08\n[Valid] epoch:3, MSE: 0.025\n", encoding="utf-8")
        self.manifest.update(best_epoch=2, epochs_completed=3)
        self.manifest["artifact_sha256"].update({name: report.digest(self.folder / name)
                                                for name in ["model.pth", "logging.txt", "metrics.json"]})
        report.write_json(self.folder / "manifest.json", self.manifest)
        self.source = self.folder / "report"
        process = subprocess.run([sys.executable, "-X", "utf8", str(Path(report.__file__)),
                                  "--suite", str(self.folder), "--output", str(self.source), "--expected-repeats", "1"],
                                 text=True, capture_output=True)
        if process.returncode:
            raise AssertionError(process.stderr)
        self.options = argparse.Namespace(report=self.source, output=self.folder / "export", max_json_mb=.03,
                                          main_curve_points=2, other_curve_points=2)

    def tearDown(self):
        self.fixture.tearDown()

    def parts(self, summary, kind):
        result = []
        for item in summary["files"]:
            if item["kind"] == kind:
                result.extend(report.read_json(self.options.output / item["path"]))
        return result

    def test_export_has_exact_recoverable_documents_histories_and_fixed_curves(self):
        result = exporter.export(self.options)
        self.assertEqual(result["counts"]["tasks_audited_complete"], 1)
        self.assertEqual(result["counts"]["curve_prediction_battery_series"], 2)
        nodes = dict(self.parts(result, "json_nodes"))
        snapshot = self.parts(result, "tasks")[0]
        restored = exporter.resolve(snapshot["documents"]["config.json"]["value"], nodes)
        self.assertEqual(restored, self.config)
        self.assertEqual(report.stable_hash(restored), snapshot["documents"]["config.json"]["canonical_value_sha256"])
        history = self.parts(result, "training_histories")[0]
        self.assertEqual(history["best_epoch"], 2)
        self.assertEqual(len(history["valid"]), 3)
        self.assertEqual(len(history["train"]), 3)
        self.assertEqual(history["author_test"][0][0], 1)
        truth = self.parts(result, "curve_truth")
        self.assertEqual(next(item for item in truth if item["battery"] == "b.csv")["relative_retained_point_indices"], [0, 2])
        self.assertFalse(result["prespecified_curve_policy"]["selection_uses_test_error"])
        self.assertTrue(all(item["bytes"] <= result["max_json_file_bytes"] for item in result["files"]
                            if item["path"].endswith(".json")))
        for item in result["files"]:
            self.assertEqual(report.digest(self.options.output / item["path"]), item["sha256"])

    def test_model_hash_corruption_blocks_export(self):
        (self.folder / "model.pth").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            exporter.export(self.options)

    def test_best_epoch_mismatch_blocks_export(self):
        self.manifest["best_epoch"] = 1
        report.write_json(self.folder / "manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            exporter.export(self.options)

    def test_best_epoch_mismatch_in_audited_manifest_blocks_log_selection(self):
        self.manifest["best_epoch"] = 1
        with self.assertRaisesRegex(ValueError, "Best epoch"):
            exporter.log_evidence(self.folder, self.manifest, {})

    def test_other_repeat_never_replaces_missing_repeat1_curve(self):
        task = dict(self.config["task"], repeat=2)
        curves, truths = exporter.selected_curves(self.folder, task, self.fixture.splits)
        self.assertEqual((curves, truths), ([], []))

    def test_json_partition_cap_and_roundtrip(self):
        destination = self.folder / "parts"
        destination.mkdir()
        writer = exporter.PartWriter(destination, 1000)
        original = [{"id": i, "value": "x" * 250} for i in range(20)]
        writer.parts("test", original)
        restored = []
        for item in writer.files:
            self.assertLessEqual(item["bytes"], 1000)
            restored.extend(report.read_json(destination / item["path"]))
        self.assertEqual(original, restored)
        self.assertGreater(len(writer.files), 1)

    def test_rounded_baseline_ties_preserve_recorded_best_epoch(self):
        text = "".join("[Train] epoch:{0:03d}, data loss:0.01\n[Valid] epoch:{0:03d}, data loss:{1}\n".format(epoch, loss)
                       for epoch, loss in [(1, "0.001001"), (2, "0.001000"), (3, "0.001000")])
        (self.folder / "logging.txt").write_text(text, encoding="utf-8")
        manifest = {"best_epoch": 3}
        evidence = exporter.log_evidence(self.folder, manifest, {})
        self.assertEqual(evidence["best_epoch"], 3)
        self.assertEqual(evidence["minimum_logged_validation"][0], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
