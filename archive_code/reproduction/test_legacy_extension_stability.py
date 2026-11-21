"""Synthetic semantic and layout tests; no actual extension conclusion."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import summarize_legacy_extension_stability as report


def write_csv(path, rows):
    report.write_csv(path, rows)


def fixture(root):
    comparison, extension = root / "comparison", root / "extension"
    comparison.mkdir()
    extension.mkdir()
    tasks, matching, metrics, effects = [], [], [], []
    regular = [("PINN", "published_label_monotonicity", "published_code"),
               ("PINN", "paper_prediction_monotonicity", "paper_equation"),
               ("MLP", "author_current", "shared_baseline"), ("CNN", "author_current", "shared_baseline")]
    for model, loss, lane in regular:
        for repeat in range(1, 11):
            group = "full/%s/MIT/all/%s" % (lane, model)
            task = {"family": "full", "dataset": "MIT", "batch": "all", "model": model, "loss_variant": loss}
            item = {"logical_group": group, "repeat": repeat, "seed": 1000 + repeat, "legacy_task_id": "legacy/" + group + "/%02d" % repeat,
                "modern_task_id": group + "/%02d" % repeat, "legacy_task": task}
            tasks.append(item)
            matching.append({**{key: item[key] for key in ["logical_group", "repeat", "seed"]}, "status": "matched"})
            for metric in report.METRICS:
                # Published PINN wins9/10 in legacy but one large error makes
                # its legacy mean effect positive. Modern effect always<0.
                lerror = (.009 if repeat < 10 else .12) if lane == "published_code" else .008 if model == "PINN" else .01 if model == "MLP" else .012
                rerror = .008 if model == "PINN" else .01 if model == "MLP" else .012
                scale = .02 if metric == "MSE" else 1
                left, right = lerror * scale, rerror * scale
                metrics.append({**{key: item[key] for key in ["logical_group", "repeat", "seed", "legacy_task_id", "modern_task_id"]},
                    "evaluation": "final", "metric": metric, "comparable": True, "legacy": left, "modern": right, "paired_difference": left - right})
    transfer_group = "transfer/paper_equation/reproduced_checkpoint/HUST_-1-to-MIT_-1"
    for repeat in range(1, 11):
        item = {"logical_group": transfer_group, "repeat": repeat, "seed": 1000 + repeat,
                "legacy_task_id": "legacy/" + transfer_group + "/%02d" % repeat, "modern_task_id": transfer_group + "/%02d" % repeat,
                "legacy_task": {"family": "transfer", "dataset": "MIT", "batch": "all", "model": "PINN", "loss_variant": "paper_prediction_monotonicity"}}
        tasks.append(item)
        matching.append({**{key: item[key] for key in ["logical_group", "repeat", "seed"]}, "status": "matched"})
        for metric in report.METRICS:
            for evaluation in ["final", "source_only"]:
                left = (.008 if repeat < 6 else .012) if evaluation == "final" else .01
                right = .009 if evaluation == "final" else .01
                metrics.append({**{key: item[key] for key in ["logical_group", "repeat", "seed", "legacy_task_id", "modern_task_id"]},
                    "evaluation": evaluation, "metric": metric, "comparable": True, "legacy": left, "modern": right, "paired_difference": left - right})
    lookup = {(x["logical_group"], x["repeat"], x["evaluation"], x["metric"]): x for x in metrics}
    for item in tasks:
        task, group, repeat = item["legacy_task"], item["logical_group"], item["repeat"]
        comparators = [("fine_tune-minus-source_only", group, "source_only")] if task["family"] == "transfer" else [
            ("PINN-minus-" + model, "full/shared_baseline/MIT/all/" + model, "final") for model in ["MLP", "CNN"]] if task["model"] == "PINN" else []
        for evaluation, other_group, other_eval in comparators:
            for metric in report.METRICS:
                final, other = lookup[group, repeat, "final", metric], lookup[other_group, repeat, other_eval, metric]
                left, right = final["legacy"] - other["legacy"], final["modern"] - other["modern"]
                effects.append({"logical_group": group, "repeat": repeat, "seed": item["seed"], "evaluation": evaluation,
                    "metric": metric, "legacy": left, "modern": right, "comparable": True,
                    "empirical_direction_changed": report.direction(left) != report.direction(right)})
    summaries, _ = report.summarize_effects(effects)
    upstream = [{"logical_group": x["logical_group"], "evaluation": x["evaluation"], "metric": x["metric"],
        "complete_ten_repeat_set": True, "repeat_ids": list(range(1, 11)), **{stack: {key: x[stack][key] for key in ["n", "mean", "sd_ddof1"]} for stack in report.STACKS}} for x in summaries]
    selection = {"regular_groups": [{"dataset": "MIT", "batch": "all"}], "reason": "SYNTHETIC FIXTURE ONLY"}
    manifest = {"tasks": tasks, "selection": selection}
    report.write_json(extension / "extension_analysis_manifest.json", manifest)
    report.write_json(extension / "extension_plan.json", {"tasks": [], "synthetic_fixture": True})
    design = {"analysis_manifest_sha256": report.digest(extension / "extension_analysis_manifest.json"),
              "plan_sha256": report.digest(extension / "extension_plan.json"), "analysis_task_count": len(tasks)}
    report.write_json(extension / "extension_design.json", design)
    report.write_json(comparison / "summary.json", {"format": "pinn4soh-selected-legacy-extension-comparison-v1",
        "assessment": "complete_selected_ten_repeat_sensitivity", "expected_repeats": 10, "expected_tasks": len(tasks),
        "task_counts": {"matched": len(tasks)}, "selection": selection, "analysis_manifest_sha256": design["analysis_manifest_sha256"],
        "method_effect_summaries": upstream, "synthetic_fixture": True})
    write_csv(comparison / "task_matching.csv", matching)
    write_csv(comparison / "per_repeat_metric_differences.csv", metrics)
    write_csv(comparison / "per_repeat_method_effects.csv", effects)
    return comparison, extension


class DirectionStatistics(unittest.TestCase):
    def test_mean_direction_is_not_seed_majority(self):
        summary = report.direction_summary([-1] * 9 + [100])
        self.assertEqual(summary["direction_counts"], {"lower": 9, "exact_zero": 0, "higher": 1})
        self.assertEqual(summary["direction_of_mean"], "higher")
        self.assertEqual(summary["strict_majority_direction"], "lower")
        self.assertEqual(summary["number_same_direction_as_mean"], 1)

    def test_exact_near_zero_is_not_equivalence(self):
        summary = report.direction_summary([-1e-20, 0, 1e-20])
        self.assertEqual(summary["direction_counts"], {"lower": 1, "exact_zero": 1, "higher": 1})
        self.assertEqual(summary["smallest_nonzero_absolute_effect"], 1e-20)

    def test_joint_pairs_and_postscreen_repeat1_separation(self):
        rows = [{"logical_group": "group", "evaluation": "effect", "metric": "MAE", "repeat": r, "seed": 10 + r,
                 "legacy": -1 if r < 8 else 1, "modern": 1 if r == 1 else -1} for r in range(1, 11)]
        summary, pairs = report.summarize_effects(rows)
        self.assertEqual(summary[0]["paired_direction_changes"], 4)
        self.assertEqual(summary[0]["new_repeat2_to10"]["paired_direction_changes"], 3)
        self.assertEqual(summary[0]["paired_joint_direction_counts"], {"higher__lower": 3, "lower__higher": 1, "lower__lower": 6})
        self.assertTrue(pairs[0]["screened_repeat1"])
        self.assertFalse(pairs[1]["screened_repeat1"])

    def test_rank_of_mean_and_mean_rank_differ(self):
        per_seed = [{"A": .01, "B": .02}] * 9 + [{"A": 1, "B": .02}]
        ranks = [report.ranks(row) for row in per_seed]
        self.assertLess(sum(x["A"] for x in ranks), sum(x["B"] for x in ranks))
        self.assertGreater(report.ranks({"A": .109, "B": .02})["A"], report.ranks({"A": .109, "B": .02})["B"])
        self.assertEqual(report.ranks({"A": 1, "B": 1, "C": 2}), {"A": 1.5, "B": 1.5, "C": 3.0})

    def test_partial_duplicate_and_seed_duplicate_rejected(self):
        rows = [{"logical_group": "g", "evaluation": "e", "metric": "MAE", "repeat": r, "seed": 10 + r, "legacy": 1, "modern": 2} for r in range(1, 11)]
        for invalid in [rows[:-1], rows + [rows[0]], [dict(row, seed=10) for row in rows]]:
            with self.assertRaises(ValueError):
                report.summarize_effects(invalid)


class InputAudit(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pinn_stability_fixture_")
        self.root = Path(self.temp.name)
        self.comparison, self.extension = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_fixture_report_keeps_seed_and_mean_separate(self):
        result = report.report(self.comparison, self.extension, self.root / "report", no_plots=True)
        self.assertTrue(result["synthetic_fixture"])
        self.assertEqual(result["effect_configurations"], 5)
        published = next(x for x in result["summaries"] if "published_code" in x["logical_group"] and x["evaluation"] == "PINN-minus-MLP" and x["metric"] == "MAE")
        self.assertEqual(published["legacy"]["direction_of_mean"], "higher")
        self.assertEqual(published["legacy"]["direction_counts"]["lower"], 9)
        self.assertTrue((self.root / "report/paired_seed_effect_directions.csv").is_file())
        self.assertEqual(len(result["method_mean_ranks"]), 32)

    def test_declared_effect_tamper_rejected(self):
        path = self.comparison / "per_repeat_method_effects.csv"
        rows = report.read_csv(path)
        rows[0]["legacy"] = ".333"
        path.unlink()
        write_csv(path, rows)
        with self.assertRaisesRegex(ValueError, "Declared method effect"):
            report.audit_inputs(self.comparison, self.extension)

    def test_missing_metric_rejected(self):
        path = self.comparison / "per_repeat_metric_differences.csv"
        rows = report.read_csv(path)
        path.unlink()
        write_csv(path, rows[:-1])
        with self.assertRaisesRegex(ValueError, "Metric row set"):
            report.audit_inputs(self.comparison, self.extension)

    def test_upstream_mean_tamper_rejected(self):
        path = self.comparison / "summary.json"
        value = report.read(path)
        value["method_effect_summaries"][0]["legacy"]["mean"] += .2
        report.write_json(path, value)
        with self.assertRaisesRegex(ValueError, "summary disagrees"):
            report.audit_inputs(self.comparison, self.extension)

    def test_incomplete_comparison_and_support_sha_rejected(self):
        path = self.comparison / "summary.json"
        value = report.read(path)
        value["task_counts"] = {"matched": 49, "pending": 1}
        report.write_json(path, value)
        with self.assertRaisesRegex(ValueError, "accounting"):
            report.audit_inputs(self.comparison, self.extension)


if __name__ == "__main__":
    unittest.main()
