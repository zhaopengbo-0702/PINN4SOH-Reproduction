"""Synthetic array-level checks for the matched runtime comparison, no training."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import build_suite_report as report
import compare_legacy_control as compare


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "outputs/full_reproduction/legacy_control_support/legacy_control_plan.json"
ORIGINAL = ROOT / "outputs/full_reproduction/plans/stage1_plan.json"


class RuntimeComparison(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pinn-runtime-comparison-")
        self.root = Path(self.temp.name)
        self.legacy, self.current = self.root / "legacy", self.root / "current"
        self.legacy.mkdir()
        self.current.mkdir()
        self.tasks = json.loads(PLAN.read_text(encoding="utf-8"))["tasks"]
        self.originals = {row["task_id"]:row for row in json.loads(ORIGINAL.read_text(encoding="utf-8"))["tasks"]}

    def tearDown(self):
        self.temp.cleanup()

    def store(self, task, legacy, error, test_paths=None):
        folder = (self.legacy if legacy else self.current) / "tasks" / task["task_id"]
        folder.mkdir(parents=True)
        truth = np.array([.5, 1., 1., 1.])
        prediction = truth - error
        np.save(folder / "true_label.npy", truth)
        np.save(folder / "pred_label.npy", prediction)
        test_paths = test_paths or ["a.csv", "b.csv"]
        splits = [{"role":"train_validation", "paths":["train.csv"], "nominal_capacity":2},
            {"role":"test", "paths":test_paths, "nominal_capacity":2,
             "batteries":[{"path":test_paths[0], "n_pairs":1},{"path":test_paths[1], "n_pairs":3}]}]
        report.write_json(folder / "splits.json", splits)
        summaries, batteries = report.recompute(folder / "true_label.npy", folder / "pred_label.npy", splits)
        by_key = {(row["aggregation"],row["convention"]):row for row in summaries}
        report.write_json(folder / "metrics.json", {"pooled_standard":by_key["pooled","standard"],
            "battery_macro_standard":by_key["battery_macro","standard"],"battery_macro_original":by_key["battery_macro","reversed"]})
        reversed_rows = {row["battery"]:row for row in batteries if row["convention"]=="reversed"}
        rows = [{**row, **{metric+"_original":reversed_rows[row["battery"]][metric] for metric in report.METRICS}}
                for row in batteries if row["convention"]=="standard"]
        report.write_csv(folder / "metrics_per_battery.csv", rows)
        config = {"task":task, "parameters":{"epochs":200,"lr":.001}, "source_fingerprint":"same-source",
            "environment":{"python":"3.7.10" if legacy else "3.10", "torch":"1.7.1" if legacy else "2.7"}}
        report.write_json(folder / "config.json", config)
        (folder / "model.pth").write_bytes(b"synthetic checksum-only fixture, no pickle execution")
        (folder / "logging.txt").write_text("[Train] epoch:1\n[Valid] epoch:1, MSE: 0.2\n[Train] epoch:2\n[Valid] epoch:2, MSE: 0.1\n",encoding="utf-8")
        report.write_json(folder / "manifest.json", {"status":"complete","task":task,
            "checkpoint_verification":{"passed":True}, "synthetic_fixture":True,
            "artifact_sha256":{name:report.digest(folder/name) for name in
             ["config.json","splits.json","true_label.npy","pred_label.npy","model.pth"]}})
        return folder

    def run_comparison(self):
        return compare.compare(PLAN,self.legacy,[self.current],self.root/"output")

    def test_missing_results_are_pending_not_zero(self):
        summary = self.run_comparison()
        self.assertEqual(summary["task_counts"],{"pending":40})
        self.assertEqual(summary["assessment"],"pending_or_unmatched")

    def test_all_signed_differences_and_direction_changes_are_preserved(self):
        for task in self.tasks[:4]:
            is_pinn = task["model"]=="PINN"
            self.store(task,True,.04 if is_pinn else .01)
            self.store(self.originals[task["original_task_id"]],False,.01 if is_pinn else .04)
        summary = self.run_comparison()
        self.assertEqual(summary["task_counts"],{"matched":4,"pending":36})
        self.assertEqual(summary["method_direction_changes"],16)
        with (self.root/"output/all_metric_differences.csv").open(encoding="utf-8-sig") as stream:
            rows = list(__import__('csv').DictReader(stream))
        self.assertEqual(len(rows),16)
        self.assertTrue(any(float(row["legacy_minus_current"])>0 for row in rows))
        self.assertTrue(any(float(row["legacy_minus_current"])<0 for row in rows))

    def test_test_battery_mismatch_is_retained_as_unmatched(self):
        task=self.tasks[0]
        self.store(task,True,.02)
        self.store(self.originals[task["original_task_id"]],False,.01,["other.csv","b.csv"])
        summary=self.run_comparison()
        self.assertEqual(summary["task_counts"],{"unmatched":1,"pending":39})
        import csv
        with (self.root/"output/all_metric_differences.csv").open(encoding="utf-8-sig") as stream:
            rows=list(csv.DictReader(stream))
        self.assertTrue(all(row["legacy_minus_current"]=="" for row in rows))
        self.assertTrue(all(row["descriptive_unmatched_difference"] for row in rows))

    def test_array_tampering_is_audit_failure(self):
        task=self.tasks[0]
        folder=self.store(task,True,.02)
        self.store(self.originals[task["original_task_id"]],False,.01)
        np.save(folder/"pred_label.npy",np.zeros(4))
        summary=self.run_comparison()
        self.assertEqual(summary["task_counts"],{"audit_failed":1,"pending":39})

    def test_missing_frozen_G_or_source_proof_prevents_transfer_pairing(self):
        base={"family":"transfer","dataset":"HUST","model":"PINN"}
        reasons=compare.match_reasons(base,base,{}, {})
        self.assertIn("legacy_frozen_G_unverified",reasons)
        self.assertIn("current_F_update_unverified",reasons)
        self.assertIn("legacy_prespecified_reproduced_source_unverified",reasons)


if __name__=="__main__":
    unittest.main()
