"""Synthetic evidence checks for independent reporting; no model training."""

import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

import numpy as np

import build_suite_report as report


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pinn4soh_report_")
        self.folder = Path(self.temporary.name)
        self.splits = [{"role": "train_validation", "paths": ["train.csv"]},
                       {"role": "test", "paths": ["a.csv", "b.csv"], "batteries": [
                           {"path": "a.csv", "n_pairs": 1}, {"path": "b.csv", "n_pairs": 3}]}]
        self.truth, self.prediction = np.array([.5, 1., 1., 1.]), np.array([.4, 1., 1., 1.])
        np.save(self.folder / "true_label.npy", self.truth)
        np.save(self.folder / "pred_label.npy", self.prediction)

    def tearDown(self):
        self.temporary.cleanup()

    def computed(self):
        return report.recompute(self.folder / "true_label.npy", self.folder / "pred_label.npy", self.splits)

    def store_extended(self):
        summaries, batteries = self.computed()
        mapping = {(row["aggregation"], row["convention"]): row for row in summaries}
        metrics = {"pooled_standard": mapping["pooled", "standard"],
                   "battery_macro_standard": mapping["battery_macro", "standard"],
                   "battery_macro_original": mapping["battery_macro", "reversed"]}
        reverse = {row["battery"]: row for row in batteries if row["convention"] == "reversed"}
        stored = [{**row, **{name + "_original": reverse[row["battery"]][name] for name in report.METRICS}}
                  for row in batteries if row["convention"] == "standard"]
        report.write_csv(self.folder / "metrics_per_battery.csv", stored)
        report.write_json(self.folder / "metrics.json", metrics)
        return summaries, metrics

    def store_task(self):
        self.store_extended()
        task = {"task_id": "full/test/XJTU/2C/PINN/Experiment01", "family": "full", "dataset": "XJTU",
                "batch": "2C", "model": "PINN", "protocol_id": "author_current", "loss_variant": "published_label_monotonicity",
                "repeat": 1, "seed": 11, "epochs": 200}
        config = {"task": task, "parameters": {"epochs": 200}, "source_fingerprint": "source-sha"}
        report.write_json(self.folder / "config.json", config)
        report.write_json(self.folder / "splits.json", self.splits)
        manifest = {"status": "complete", "task": task, "checkpoint_verification": {"passed": True},
                    "artifact_sha256": {name: report.digest(self.folder / name) for name in [
                        "config.json", "splits.json", "true_label.npy", "pred_label.npy"]}}
        report.write_json(self.folder / "manifest.json", manifest)
        return config, manifest

    def test_macro_is_not_pooled_and_MAPE_denominators_are_distinct(self):
        rows, batteries = self.computed()
        mapping = {(row["aggregation"], row["convention"]): row for row in rows}
        self.assertAlmostEqual(mapping["battery_macro", "standard"]["MAPE"], .1)
        self.assertAlmostEqual(mapping["pooled", "standard"]["MAPE"], .05)
        self.assertAlmostEqual(mapping["battery_macro", "reversed"]["MAPE"], .125)
        self.assertEqual([row["n_test"] for row in batteries[:2]], [1, 3])

    def test_inconsistent_and_reordered_boundaries_fail(self):
        self.splits[-1]["batteries"][0]["n_pairs"] = 2
        with self.assertRaisesRegex(ValueError, "counts"):
            self.computed()
        self.splits[-1]["batteries"][0]["n_pairs"] = 1
        self.splits[-1]["paths"].reverse()
        with self.assertRaisesRegex(ValueError, "order"):
            self.computed()

    def test_additional_monotonicity_diagnostics_use_within_battery_retained_points(self):
        np.save(self.folder / "true_label.npy", np.array([1., .9, .95, .8]))
        np.save(self.folder / "pred_label.npy", np.array([1., 1.1, .9, 1.]))
        self.splits[-1] = {"role": "test", "paths": ["a.csv"],
                           "batteries": [{"path": "a.csv", "n_pairs": 4}]}
        summaries, batteries = self.computed()
        row = batteries[0]
        self.assertEqual(row["n_adjacent_retained_points"], 3)
        self.assertAlmostEqual(row["prediction_rise_fraction"], 2 / 3)
        self.assertAlmostEqual(row["prediction_total_positive_increment"], .2)
        self.assertAlmostEqual(row["truth_rise_fraction"], 1 / 3)
        self.assertAlmostEqual(row["truth_total_positive_increment"], .05)
        self.assertEqual(row["increment_opposite_sign_fraction"], 1.)
        self.assertIn("not necessarily consecutive", row["diagnostic_scope"])
        self.assertEqual(summaries[0]["n_adjacent_retained_points"], 3)

    def test_extended_metadata_adapter_and_missing_pooled_reversed(self):
        computed, metrics = self.store_extended()
        summaries, batteries = report.read_metadata_metrics(self.folder, metrics, self.splits)
        self.assertAlmostEqual(summaries[1]["MAPE"], computed[1]["MAPE"])
        self.assertIsNone(summaries[2]["MAPE"])
        self.assertAlmostEqual(summaries[3]["MAPE"], .125)
        self.assertEqual(len(batteries), 4)

    def test_transfer_adapter_including_source_only(self):
        summaries, batteries = self.computed()
        mapping = {(row["aggregation"], row["convention"]): row for row in summaries}
        current = [dict(row, MAPE_author_reversed=.25 if row["battery"] == "a.csv" else 0.)
                   for row in batteries if row["convention"] == "standard"]
        for prefix in ["", "source_only_"]:
            report.write_csv(self.folder / (prefix + "metrics_per_battery.csv"), current)
        metrics = {"metrics": mapping["pooled", "standard"],
                   "metrics_battery_macro": mapping["battery_macro", "standard"],
                   "source_only_metrics": mapping["pooled", "standard"],
                   "source_only_battery_macro": mapping["battery_macro", "standard"]}
        for prefix in ["", "source_only_"]:
            rows, _ = report.read_metadata_metrics(self.folder, metrics, self.splits, prefix)
            self.assertAlmostEqual(rows[3]["MAPE"], .125)

    def test_full_family_normalization_and_smoke_exclusion(self):
        config, manifest = self.store_task()
        status, rows, _, _ = report.inspect_task(self.folder)
        self.assertTrue(status["included"])
        self.assertEqual(rows[0]["family"], "regular")
        config["smoke"] = True
        report.write_json(self.folder / "config.json", config)
        status, rows, _, _ = report.inspect_task(self.folder)
        self.assertFalse(status["included"])
        self.assertEqual(rows, [])

    def test_hash_corruption_and_metric_corruption_fail(self):
        self.store_task()
        with (self.folder / "pred_label.npy").open("ab") as stream:
            stream.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "artifact hash"):
            report.inspect_task(self.folder)
        np.save(self.folder / "pred_label.npy", self.prediction)
        metrics = report.read_json(self.folder / "metrics.json")
        metrics["pooled_standard"]["MAPE"] = 3.
        report.write_json(self.folder / "metrics.json", metrics)
        with self.assertRaisesRegex(ValueError, "disagrees"):
            report.inspect_task(self.folder)

    def test_protocol_separation_repeat_SD_and_duplicate_detection(self):
        config, manifest = self.store_task()
        _, rows, _, _ = report.inspect_task(self.folder)
        first = rows[0]
        second = dict(first, repeat=2, seed=12, MAPE=first["MAPE"] + .02)
        summaries = report.summarize_rows([first, second], 2)
        self.assertTrue(summaries[0]["complete_repeat_set"])
        self.assertAlmostEqual(summaries[0]["MAPE_sd_ddof0"], .01)
        self.assertAlmostEqual(summaries[0]["MAPE_sd_ddof1"], math_sqrt_two() * .01)
        self.assertFalse(report.summarize_rows([first, first], 2)[0]["complete_repeat_set"])
        config["task"]["loss_variant"] = "paper_prediction_monotonicity"
        report.write_json(self.folder / "config.json", config)
        manifest["artifact_sha256"]["config.json"] = report.digest(self.folder / "config.json")
        report.write_json(self.folder / "manifest.json", manifest)
        _, other, _, _ = report.inspect_task(self.folder)
        self.assertNotEqual(first["comparison_group"], other[0]["comparison_group"])
        self.assertEqual(len(report.summarize_rows([first, other[0]], 2)), 2)

    def test_shared_baseline_comparison_has_explicit_scope(self):
        self.store_task()
        _, rows, _, _ = report.inspect_task(self.folder, 1)
        primary = next(row for row in rows if row["aggregation"] == "battery_macro" and row["convention"] == "standard")
        baseline = dict(primary, comparison_group="mlp-group", model="MLP", loss_variant="author_current")
        summaries = report.summarize_rows([primary, baseline], 1)
        effects = report.comparison_effects(summaries, [primary, baseline], [])
        matched = [row for row in effects if row["kind"] == "PINN_vs_MLP"]
        self.assertEqual(len(matched), 3)
        self.assertTrue(all(row["direction"] == "tie" and row["shared_baseline"] for row in matched))
        self.assertTrue(any(row["status"] == "missing_or_ambiguous_comparator" for row in effects))

    def test_transfer_comparisons_require_fixed_G_and_target_paths(self):
        self.store_task()
        _, rows, _, proof = report.inspect_task(self.folder, 1)
        primary = next(row for row in rows if row["aggregation"] == "battery_macro" and row["convention"] == "standard")
        transfer = dict(primary, family="transfer", source="TJU", source_batch=2,
                        source_mode="supplied_author_checkpoint", sample_size=1)
        before = dict(transfer, evaluation="source_only", RMSE=transfer["RMSE"] + .1)
        target = dict(primary, task_id="target", comparison_group="target-only", family="small_sample", sample_size=1)
        proof["frozen_dynamical_F"] = {"unchanged": True, "before_sha256": "same", "after_sha256": "same"}
        proof["solution_u_updated"] = True
        target_proof = copy.deepcopy(proof)
        target_proof["task_id"] = "target"
        data = [transfer, before, target]
        effects = report.comparison_effects(report.summarize_rows(data, 1), data, [proof, target_proof])
        paired = next(row for row in effects if row["kind"] == "fine_tune_vs_source_only")
        self.assertTrue(paired["frozen_G_and_updated_F_verified"])
        self.assertEqual(paired["paired_repeats_improved"], 1)
        scratch = next(row for row in effects if row["kind"] == "fine_tune_vs_target_only_n1")
        self.assertEqual(scratch["status"], "descriptive_comparison_available")
        target_proof["splits"][0]["paths"] = ["other.csv"]
        effects = report.comparison_effects(report.summarize_rows(data, 1), data, [proof, target_proof])
        scratch = next(row for row in effects if row["kind"] == "fine_tune_vs_target_only_n1")
        self.assertEqual(scratch["status"], "unmatched_target_training_battery_paths")

    def test_cli_produces_report_and_all_coverage_without_training(self):
        self.store_task()
        output = self.folder / "report"
        process = subprocess.run([sys.executable, "-X", "utf8", str(Path(report.__file__)),
                                  "--suite", str(self.folder), "--output", str(output), "--expected-repeats", "1"],
                                 text=True, capture_output=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        result = report.read_json(output / "suite_report.json")
        self.assertEqual(result["task_status_counts"], {"included": 1})
        self.assertEqual(len(result["summaries"]), 4)
        self.assertEqual(len(result["claim_assessments"]), 21)
        self.assertEqual(len(result["coverage"]), 25)
        with (output / "per_battery_summaries.csv").open(encoding="utf-8-sig", newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 4)


def math_sqrt_two():
    return 2 ** .5


if __name__ == "__main__":
    unittest.main(verbosity=2)
