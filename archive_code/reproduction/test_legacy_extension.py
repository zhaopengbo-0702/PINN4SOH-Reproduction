"""Selection/DAG/seed/reuse and ten-repeat array audit checks; no training."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import build_suite_report as audit
import compare_legacy_extension as extension_compare
from launch_legacy_extension import check_probe

import prepare_legacy_extension as prepare
from compare_legacy_extension import aggregate


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/full_reproduction"
CONTROL = prepare.read(BASE / "legacy_control_support/legacy_control_plan.json" if BASE.is_dir() else ROOT / "plans/legacy_control_plan.json")
STAGE1 = prepare.read(BASE / "plans/stage1_plan.json" if BASE.is_dir() else ROOT / "plans/stage1_plan.json")
STAGE2 = prepare.read(BASE / "plans/stage2_reproduced_transfer_plan.json" if BASE.is_dir() else ROOT / "plans/stage2_reproduced_transfer_plan.json")
CLOUD = "/workspace/pinn4soh_legacy_selected_extension_TEST_ONLY"


def selection(groups=None, transfers=None):
    return {"format": prepare.SELECTION_FORMAT, "reason": "SYNTHETIC test selection; no experiment launch",
            "regular_groups": groups or [], "transfer_repeat1_task_ids": transfers or [],
            "evidence_refs": ["SYNTHETIC test fixture only"]}


def transfer_task(source="TJU", target="XJTU"):
    return next(task for task in CONTROL["tasks"] if task["family"] == "transfer" and
                task["source"] == source and task["target"] == target)


class ExtensionPlanTests(unittest.TestCase):
    def test_regular_group_selects_four_methods_and_nine_new_repeats(self):
        plan, analysis = prepare.build(selection(groups=[{"dataset": "XJTU", "batch": "2C"}]), CONTROL, STAGE1, STAGE2, CLOUD)
        self.assertEqual(len(plan["tasks"]), 36)
        self.assertEqual(len(analysis), 40)
        self.assertEqual(sum(row["storage"] == "immutable_original_repeat1" for row in analysis), 4)
        for row in analysis:
            self.assertEqual(row["seed"], row["modern_task"]["seed"])
            self.assertEqual(row["repeat"], row["modern_task"]["repeat"])
            if row["repeat"] == 1:
                self.assertTrue(row["legacy_folder"].startswith(CONTROL["expected_suite_output"] + "/"))
            else:
                self.assertTrue(row["legacy_folder"].startswith(CLOUD + "/"))

    def test_transfer_only_adds_matching_source_pinn_closure(self):
        chosen = transfer_task()
        plan, analysis = prepare.build(selection(transfers=[chosen["task_id"]]), CONTROL, STAGE1, STAGE2, CLOUD)
        self.assertEqual(len(plan["tasks"]), 18)
        self.assertEqual(len(analysis), 20)
        lookup = {task["task_id"]: task for task in plan["tasks"]}
        for task in plan["tasks"]:
            if task["family"] != "transfer":
                continue
            source = lookup[task["depends_on"][0]]
            self.assertEqual(source["repeat"], task["repeat"])
            self.assertEqual(source["loss_variant"], task["loss_variant"])
            self.assertEqual(source["dataset"], task["source"])
            self.assertEqual(task["source_selection"], prepare.SOURCE_POLICY)
            self.assertEqual(task["source_checkpoint"], task["source_run"] + "/model.pth")
            self.assertEqual(task["source_manifest"], task["source_run"] + "/manifest.json")
            self.assertTrue(task["source_run"].startswith(CLOUD + "/tasks/"))
        # Source and target seed offsets are intentionally dataset-specific.
        source = next(task for task in plan["tasks"] if task["family"] == "full")
        target = next(task for task in plan["tasks"] if task["family"] == "transfer")
        self.assertNotEqual(source["seed"], target["seed"])

    def test_all_bounded_catalog_is_360_new_not2120(self):
        groups = sorted({(task["dataset"], str(task["batch"])) for task in CONTROL["tasks"] if task["family"] == "full"})
        transfers = [task["task_id"] for task in CONTROL["tasks"] if task["family"] == "transfer"]
        plan, analysis = prepare.build(selection([{"dataset": d, "batch": b} for d, b in groups], transfers), CONTROL, STAGE1, STAGE2, CLOUD)
        self.assertEqual((len(plan["tasks"]), len(analysis)), (360, 400))
        self.assertTrue(all(task["repeat"] > 1 for task in plan["tasks"]))

    def test_duplicate_or_offcatalog_selection_is_rejected(self):
        invalid = [selection(), selection(groups=[{"dataset": "XJTU", "batch": "3C"}]),
                   selection(transfers=["transfer/not-in-frozen-catalog"]),
                   selection(transfers=[transfer_task()["task_id"]] * 2),
                   selection(groups=[{"dataset": "XJTU", "batch": "2C", "all": True}])]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                prepare.build(value, CONTROL, STAGE1, STAGE2, CLOUD)

    def test_unknown_broad_selector_rejected(self):
        value = selection(groups=[{"dataset": "XJTU", "batch": "2C"}])
        value["all_2120"] = True
        with self.assertRaises(ValueError):
            prepare.build(value, CONTROL, STAGE1, STAGE2, CLOUD)

    def test_source_loss_or_repeat_corruption_rejected(self):
        chosen = transfer_task()
        for field, value in [("loss_variant", "wrong_loss"), ("repeat", 7)]:
            stage1 = deepcopy(STAGE1)
            source_id = chosen["original_source_run"].split("/tasks/", 1)[1].replace("Experiment01", "Experiment02")
            next(task for task in stage1["tasks"] if task["task_id"] == source_id)[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                prepare.build(selection(transfers=[chosen["task_id"]]), CONTROL, stage1, STAGE2, CLOUD)

    def test_seed_duplication_rejected(self):
        stage1 = deepcopy(STAGE1)
        rows = [task for task in stage1["tasks"] if task["task_id"].startswith("full/published_code/XJTU/2C/PINN/")]
        rows[1]["seed"] = rows[0]["seed"]
        with self.assertRaises(ValueError):
            prepare.build(selection(groups=[{"dataset": "XJTU", "batch": "2C"}]), CONTROL, stage1, STAGE2, CLOUD)

    def test_same_old_suite_or_nested_path_is_rejected(self):
        for cloud in [CONTROL["expected_suite_output"], CONTROL["expected_suite_output"] + "/nested", "/workspace/../wrong"]:
            with self.assertRaises(ValueError):
                prepare.build(selection(groups=[{"dataset": "XJTU", "batch": "2C"}]), CONTROL, STAGE1, STAGE2, cloud)

    def test_coordinator_validates_generated_topological_plan(self):
        try:
            import run_full_suite
        except ImportError:
            self.skipTest("Coordinator belongs to immutable runtime support; local source-tree check only")
        plan, _ = prepare.build(selection(transfers=[transfer_task()["task_id"]]), CONTROL, STAGE1, STAGE2, CLOUD)
        ordered = run_full_suite.prepare_plan(plan, 2)
        position = {task["task_id"]: i for i, task in enumerate(ordered)}
        for task in ordered:
            for dependency in task.get("depends_on", []):
                self.assertLess(position[dependency], position[task["task_id"]])

    def test_ten_repeat_aggregation_rejects_partial_or_duplicate_sets(self):
        rows = [{"logical_group": "group", "evaluation": "final", "metric": "RMSE", "repeat": r,
                 "seed": 100 + r, "comparable": True, "legacy": r / 100, "modern": r / 200} for r in range(1, 11)]
        full = aggregate(rows)[0]
        self.assertTrue(full["complete_ten_repeat_set"])
        self.assertAlmostEqual(full["legacy"]["mean"], .055)
        self.assertFalse(aggregate(rows[:-1])[0]["complete_ten_repeat_set"])
        corrupt = deepcopy(rows)
        corrupt[-1]["repeat"] = 1
        self.assertFalse(aggregate(corrupt)[0]["complete_ten_repeat_set"])
        corrupt = deepcopy(rows)
        corrupt[-1]["comparable"] = False
        self.assertFalse(aggregate(corrupt)[0]["complete_ten_repeat_set"])


def synthetic_frozen(plan, modern=False):
    environment = {"python": "3.10 synthetic" if modern else "3.7.10 synthetic", "torch": "1.13.1 synthetic" if modern else "1.7.1 synthetic", "numpy": "1.20.3", "pandas": "1.3.5",
                   "scipy": "1.7.3", "sklearn": "0.24.2", "cuda": "11.0", "cuda_available": True, "gpu": "synthetic GPU"}
    return {"plan_sha256": prepare.stable(plan), "environment_signatures": {prepare.stable(environment): environment},
            "source_fingerprint": "synthetic source", "data_directory_order_sha256": prepare.stable(["synthetic order"]),
            "implementation_sha256": {"/frozen/" + name: name + " synthetic checksum" for name in
                ["run_full_suite.py", "run_extended.py", "run_reproduction.py", "run_transfer.py"]},
            "threads": 2, "smoke": False, "allow_cpu": False}


def synthetic_store(suite, task, frozen, error, source=None):
    """Persist real array/CSV/manifest schemas; checkpoint is never unpickled."""
    folder = suite / "tasks" / task["task_id"]
    folder.mkdir(parents=True)
    truth = np.array([.5, .8, .9, 1.])
    splits = [{"role": "train_validation", "paths": ["train.csv"], "nominal_capacity": 2},
              {"role": "test", "paths": ["a.csv", "b.csv"], "nominal_capacity": 2,
               "batteries": [{"path": "a.csv", "n_pairs": 1}, {"path": "b.csv", "n_pairs": 3}]}]
    audit.write_json(folder / "splits.json", splits)
    metrics = {}
    for prefix in (["", "source_only_"] if task["family"] == "transfer" else [""]):
        np.save(folder / (prefix + "true_label.npy"), truth)
        np.save(folder / (prefix + "pred_label.npy"), truth - error * (2 if prefix else 1))
        summaries, rows = audit.recompute(folder / (prefix + "true_label.npy"), folder / (prefix + "pred_label.npy"), splits)
        by_key = {(row["aggregation"], row["convention"]): row for row in summaries}
        if task["family"] == "transfer":
            metrics[prefix + "metrics"] = by_key["pooled", "standard"]
            metrics["source_only_battery_macro" if prefix else "metrics_battery_macro"] = by_key["battery_macro", "standard"]
        else:
            metrics.update(pooled_standard=by_key["pooled", "standard"], battery_macro_standard=by_key["battery_macro", "standard"],
                           battery_macro_original=by_key["battery_macro", "reversed"])
        reverse = {row["battery"]: row for row in rows if row["convention"] == "reversed"}
        audit.write_csv(folder / (prefix + "metrics_per_battery.csv"), [{**row, **{name + "_original": reverse[row["battery"]][name]
            for name in audit.METRICS}} for row in rows if row["convention"] == "standard"])
    audit.write_json(folder / "metrics.json", metrics)
    config = {"task": prepare.effective_task(task), "parameters": {"epochs": 200, "lr": .001},
              "environment": prepare.runtime_signature(frozen), "source_fingerprint": frozen["source_fingerprint"],
              "data_directory_order_sha256": frozen["data_directory_order_sha256"], "source_directory_order": ["synthetic order"],
              "checkpoint_snapshot_deepcopy": True}
    lineage, inputs = None, {}
    if source is not None:
        source_task, source_folder = source
        lineage = {"source_mode": task["source_mode"], "path": task["source_checkpoint"], "source_run": task["source_run"],
                   "source_dataset": task["source"], "source_batch": task["source_batch"], "training_seed": source_task["seed"],
                   "training_history_verified": True, "source_selection": prepare.SOURCE_POLICY,
                   "source_paths": prepare.read(source_folder / "splits.json"), "source_best_validation_mse": .1}
        for field, name in [("sha256", "model.pth"), ("source_config_sha256", "config.json"),
                            ("source_splits_sha256", "splits.json"), ("source_manifest_sha256", "manifest.json")]:
            lineage[field] = prepare.digest(source_folder / name)
            inputs[task["source_run"] + "/" + name] = lineage[field]
        config["source_checkpoint_lineage"] = lineage
    audit.write_json(folder / "config.json", config)
    log = "[Train] epoch:1\n[Valid] epoch:1, MSE: 0.2\n[Train] epoch:2\n[Valid] epoch:2, MSE: 0.1\n"
    if task["family"] == "transfer":
        parameters = dict(task.get("overrides", {}))
        parameters.update({key: task.get({"source_data": "source", "target_data": "target"}.get(key, key)) for key in
            ["adaptation_lr", "adaptation_epochs", "source_data", "source_batch", "target_data", "target_batch"]})
        parameters.update(pretrain_model=task["source_checkpoint"], epochs=200, lr=.001)
        audit.write_json(folder / "parameters.json", parameters)
        log += "".join("synthetic - CRITICAL - \t%s:%s\n" % (key, value) for key, value in parameters.items())
    (folder / "logging.txt").write_text(log, encoding="utf-8")
    (folder / "model.pth").write_bytes(b"synthetic checksum-only checkpoint, no pickle execution")
    manifest = {"status": "complete", "task": task, "effective_task": prepare.effective_task(task), "task_id": task["task_id"],
                "task_sha256": prepare.stable(task), "source_fingerprint": frozen["source_fingerprint"],
                "implementation_sha256": prepare.implementation(task, frozen), "checkpoint_verification": {"passed": True},
                "artifact_sha256": {path.name: prepare.digest(path) for path in folder.iterdir()}}
    if lineage:
        manifest.update(source_checkpoint_lineage=lineage, frozen_dynamical_F={"unchanged": True, "before_sha256": "same", "after_sha256": "same"}, solution_u_updated=True)
    audit.write_json(folder / "manifest.json", manifest)
    record = {"status": "complete", "task_sha256": prepare.stable(task), "worker_manifest_sha256": prepare.digest(folder / "manifest.json"),
              "verified_file_sha256": {path.name: prepare.digest(path) for path in folder.iterdir()}, "input_sha256": inputs}
    return folder, record


class ExtensionArrayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pinn_extension_arrays_")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, transfer=False):
        plan, analysis = prepare.build(selection(transfers=[transfer_task()["task_id"]]) if transfer else
            selection(groups=[{"dataset": "XJTU", "batch": "2C"}]), CONTROL, STAGE1, STAGE2, CLOUD)
        self.old, self.new, self.modern, self.support = [self.root / name for name in ["old", "new", "modern", "support"]]
        self.support.mkdir()
        self.analysis = analysis
        states = {self.old: {"status": "complete", "frozen": synthetic_frozen(CONTROL),
                              "tasks": {task["task_id"]: {"status": "complete"} for task in CONTROL["tasks"]}},
                  self.new: {"status": "complete", "frozen": synthetic_frozen(plan), "tasks": {}},
                  self.modern: {"status": "complete", "frozen": synthetic_frozen({"tasks": [row["modern_task"] for row in analysis]}, modern=True), "tasks": {}}}
        # Source regular tasks precede target transfer tasks in sorted group order.
        for item in analysis:
            for modern in [False, True]:
                task = item["modern_task" if modern else "legacy_task"]
                suite = self.modern if modern else self.old if item["repeat"] == 1 else self.new
                source = None
                if task["family"] == "transfer":
                    source_id = task["source_run"].split("/tasks/", 1)[1]
                    source_item = next(row for row in analysis if row["modern_task_id" if modern else "legacy_task_id"] == source_id)
                    source = (source_item["modern_task" if modern else "legacy_task"], suite / "tasks" / source_id)
                folder, record = synthetic_store(suite, task, states[suite]["frozen"], .01 if modern else .02, source)
                states[suite]["tasks"][task["task_id"]] = record
        for suite, state in states.items():
            audit.write_json(suite / "manifest.json", state)
        prepare.write(self.support / "extension_plan.json", plan)
        chosen_selection = selection(transfers=[transfer_task()["task_id"]]) if transfer else selection(groups=[{"dataset": "XJTU", "batch": "2C"}])
        self.manifest = {"format": prepare.FORMAT, "tasks": analysis, "selection": chosen_selection,
                         "reuse_evidence": prepare.verify_reused(analysis, self.old)}
        prepare.write(self.support / "extension_analysis_manifest.json", self.manifest)
        prepare.write(self.support / "extension_design.json", {"plan_sha256": prepare.digest(self.support / "extension_plan.json"),
            "analysis_manifest_sha256": prepare.digest(self.support / "extension_analysis_manifest.json"),
            "new_task_count": len(plan["tasks"]), "analysis_task_count": len(analysis),
            "reused_repeat1_count": sum(row["repeat"] == 1 for row in analysis)})
        return states

    def compare(self):
        # Synthetic checksums are not the approved real modern code; only this
        # external fixed-code gate is bypassed, never any task/file/array audit.
        from unittest.mock import patch
        with patch.object(extension_compare, "verify_modern_frozen"):
            return extension_compare.compare(self.support, self.old, self.new, [self.modern], self.root / "report")

    def test_saved_arrays_composite_ten_repeats_complete(self):
        self.fixture()
        result = self.compare()
        self.assertEqual(result["task_counts"], {"matched": 40})
        self.assertEqual(result["assessment"], "complete_selected_ten_repeat_sensitivity")
        self.assertEqual(len(result["summaries"]), 16)
        self.assertTrue(all(row["complete_ten_repeat_set"] for row in result["summaries"]))
        self.assertEqual(len(result["method_effect_summaries"]), 16)

    def test_transfer_arrays_and_both_source_lineages_complete(self):
        self.fixture(transfer=True)
        result = self.compare()
        self.assertEqual(result["task_counts"], {"matched": 20})
        self.assertTrue(all(row["complete_ten_repeat_set"] for row in result["summaries"]))
        self.assertEqual(len(result["method_effect_summaries"]), 4)

    def test_reuse_hashes_reject_changed_prediction(self):
        self.fixture()
        item = next(row for row in self.analysis if row["repeat"] == 1)
        (self.old / "tasks" / item["legacy_task_id"] / "pred_label.npy").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Completion-time artifact changed"):
            prepare.verify_reused(self.analysis, self.old)

    def test_reuse_evidence_survives_byte_exact_relocation(self):
        self.fixture()
        import shutil
        relocated = self.root / "relocated"
        shutil.copytree(self.old, relocated)
        self.assertEqual(prepare.verify_reused(self.analysis, relocated), self.manifest["reuse_evidence"])

    def test_pending_arrays_never_enter_ten_repeat_summary(self):
        self.fixture()
        item = next(row for row in self.analysis if row["repeat"] == 10)
        (self.new / "tasks" / item["legacy_task_id"] / "manifest.json").unlink()
        result = self.compare()
        self.assertEqual(result["task_counts"], {"matched": 39, "pending": 1})
        self.assertTrue(any(not row["complete_ten_repeat_set"] for row in result["summaries"]))

    def test_modern_completion_hash_tampering_is_audit_failed(self):
        self.fixture()
        item = self.analysis[-1]
        (self.modern / "tasks" / item["modern_task_id"] / "logging.txt").write_text("changed", encoding="utf-8")
        self.assertEqual(self.compare()["task_counts"], {"matched": 39, "audit_failed": 1})

    def test_self_consistent_modern_plan_identity_tampering_still_rejected(self):
        states = self.fixture()
        item = next(row for row in self.analysis if row["repeat"] == 10 and row["modern_task"]["model"] == "PINN")
        folder = self.modern / "tasks" / item["modern_task_id"]
        config = prepare.read(folder / "config.json")
        config["task"]["overrides"]["alpha"] = 99
        audit.write_json(folder / "config.json", config)
        record = states[self.modern]["tasks"][item["modern_task_id"]]
        record["verified_file_sha256"]["config.json"] = prepare.digest(folder / "config.json")
        audit.write_json(self.modern / "manifest.json", states[self.modern])
        self.assertEqual(self.compare()["task_counts"], {"matched": 39, "audit_failed": 1})

    def test_mismatched_test_splits_are_unmatched(self):
        states = self.fixture()
        item = self.analysis[-1]
        folder = self.modern / "tasks" / item["modern_task_id"]
        import shutil
        shutil.rmtree(folder)
        folder, record = synthetic_store(self.modern, item["modern_task"], states[self.modern]["frozen"], .01)
        splits = prepare.read(folder / "splits.json")
        splits[0]["paths"] = ["different-train.csv"]
        audit.write_json(folder / "splits.json", splits)
        worker = prepare.read(folder / "manifest.json")
        worker["artifact_sha256"]["splits.json"] = prepare.digest(folder / "splits.json")
        audit.write_json(folder / "manifest.json", worker)
        record["worker_manifest_sha256"] = prepare.digest(folder / "manifest.json")
        record["verified_file_sha256"] = {path.name: prepare.digest(path) for path in folder.iterdir()}
        states[self.modern]["tasks"][item["modern_task_id"]] = record
        audit.write_json(self.modern / "manifest.json", states[self.modern])
        result = self.compare()
        self.assertEqual(result["task_counts"], {"matched": 39, "unmatched": 1})
        self.assertTrue(any(not row["complete_ten_repeat_set"] for row in result["summaries"]))

    def test_same_runtime_cannot_be_labeled_a_two_stack_comparison(self):
        states = self.fixture()
        states[self.modern]["frozen"]["environment_signatures"] = deepcopy(states[self.old]["frozen"]["environment_signatures"])
        for item in self.analysis:
            folder = self.modern / "tasks" / item["modern_task_id"]
            config, worker = prepare.read(folder / "config.json"), prepare.read(folder / "manifest.json")
            config["environment"] = prepare.runtime_signature(states[self.modern]["frozen"])
            audit.write_json(folder / "config.json", config)
            worker["artifact_sha256"]["config.json"] = prepare.digest(folder / "config.json")
            audit.write_json(folder / "manifest.json", worker)
            record = states[self.modern]["tasks"][item["modern_task_id"]]
            record["worker_manifest_sha256"] = prepare.digest(folder / "manifest.json")
            record["verified_file_sha256"] = {path.name: prepare.digest(path) for path in folder.iterdir()}
        audit.write_json(self.modern / "manifest.json", states[self.modern])
        result = self.compare()
        self.assertEqual(result["task_counts"], {"unmatched": 40})
        self.assertFalse(any(row["complete_ten_repeat_set"] for row in result["summaries"]))

    def test_reuse_requires_complete_forty_screen(self):
        states = self.fixture()
        states[self.old]["tasks"][next(iter(states[self.old]["tasks"]))]["status"] = "running"
        audit.write_json(self.old / "manifest.json", states[self.old])
        with self.assertRaisesRegex(ValueError, "not complete"):
            prepare.verify_reused(self.analysis, self.old)

    def test_multiple_or_changed_legacy_runtime_cohort_rejected(self):
        states = self.fixture()
        new = deepcopy(states[self.new]["frozen"])
        new["environment_signatures"]["second"] = next(iter(new["environment_signatures"].values()))
        with self.assertRaises(ValueError):
            prepare.same_legacy_cohort(states[self.old]["frozen"], new)
        new = deepcopy(states[self.new]["frozen"])
        new["source_fingerprint"] = "changed"
        with self.assertRaises(ValueError):
            prepare.same_legacy_cohort(states[self.old]["frozen"], new)

    def test_probe_mapping_and_assigned_gpu(self):
        frozen = synthetic_frozen({})
        environment = prepare.runtime_signature(frozen)
        current = {key: value for key, value in environment.items() if key in {"python", "cuda", "cuda_available", "gpu"}}
        current.update(status="passed", assigned_visibility="0", versions={key: value for key, value in environment.items() if key not in current})
        check_probe(current, frozen)
        current["assigned_visibility"] = "1"
        with self.assertRaises(ValueError):
            check_probe(current, frozen)

    def test_approved_modern_code_gate_rejects_other_runner(self):
        frozen = synthetic_frozen({})
        frozen["implementation_sha256"] = {"/approved/" + name: checksum for name, checksum in prepare.MODERN_CODE_SHA.items()}
        prepare.verify_modern_frozen(frozen)
        frozen["implementation_sha256"]["/approved/run_transfer.py"] = "changed"
        with self.assertRaises(ValueError):
            prepare.verify_modern_frozen(frozen)

    def test_selected_collection_includes_all_ten_modern_counterparts(self):
        self.fixture(transfer=True)
        self.compare()
        import collect_legacy_extension_results as collector
        import restore_backup_json as restore
        from unittest.mock import patch
        screen = self.root / "screen"
        screen.mkdir()
        audit.write_json(screen / "summary.json", {"task_counts": {"matched": 40}})
        design = prepare.read(self.support / "extension_design.json")
        design["inputs"] = {"screening_report": {"sha256": prepare.digest(screen / "summary.json")}}
        audit.write_json(self.support / "extension_design.json", design)
        with patch.object(collector, "verify_modern_frozen"):
            evidence = collector.collect(self.root, self.support, self.old, self.new, [self.modern],
                self.root / "report", screen / "summary.json", self.root / "backup", chunk_mib=1)
        self.assertEqual(evidence["selected_modern_counterparts"], 20)
        self.assertEqual(evidence["new_tasks"], 18)
        self.assertTrue(evidence["all_archived_member_hashes_verified"])
        names = {row["path"] for row in evidence["inventory"]}
        for item in self.analysis:
            self.assertIn("modern/tasks/" + item["modern_task_id"] + "/model.pth", names)
        restored = restore.restore(self.root / "backup", self.root / "restored")
        self.assertTrue(restored[0]["sha256_verified"])

    def test_source_lineage_tamper_is_rejected_even_with_rehashed_target(self):
        states = self.fixture(transfer=True)
        item = next(row for row in self.analysis if row["legacy_task"]["family"] == "transfer" and row["repeat"] == 10)
        folder = self.new / "tasks" / item["legacy_task_id"]
        config, worker = prepare.read(folder / "config.json"), prepare.read(folder / "manifest.json")
        config["source_checkpoint_lineage"]["source_manifest_sha256"] = "wrong source"
        worker["source_checkpoint_lineage"] = config["source_checkpoint_lineage"]
        audit.write_json(folder / "config.json", config)
        worker["artifact_sha256"]["config.json"] = prepare.digest(folder / "config.json")
        audit.write_json(folder / "manifest.json", worker)
        record = states[self.new]["tasks"][item["legacy_task_id"]]
        record["verified_file_sha256"] = {path.name: prepare.digest(path) for path in folder.iterdir()}
        record["worker_manifest_sha256"] = prepare.digest(folder / "manifest.json")
        audit.write_json(self.new / "manifest.json", states[self.new])
        result = self.compare()
        self.assertEqual(result["task_counts"], {"matched": 19, "audit_failed": 1})

    def test_screen_report_is_bound_to_all_original_forty_artifacts(self):
        states = self.fixture()
        old_state = states[self.old]
        for task in CONTROL["tasks"]:
            if (self.old / "tasks" / task["task_id"] / "manifest.json").exists():
                continue
            source = None
            if task["family"] == "transfer":
                source_id = task["source_run"].split("/tasks/", 1)[1]
                source_task = next(row for row in CONTROL["tasks"] if row["task_id"] == source_id)
                source = (source_task, self.old / "tasks" / source_id)
            folder, record = synthetic_store(self.old, task, old_state["frozen"], .02, source)
            old_state["tasks"][task["task_id"]] = record
        audit.write_json(self.old / "manifest.json", old_state)
        screen_dir = self.root / "screen"
        screen_dir.mkdir()
        rows = [{"stack": "legacy", "task_id": task["task_id"],
                 "config": prepare.read(self.old / "tasks" / task["task_id"] / "config.json"),
                 "artifact_sha256": prepare.read(self.old / "tasks" / task["task_id"] / "manifest.json")["artifact_sha256"]}
                for task in CONTROL["tasks"]]
        evidence_path = screen_dir / "comparison_evidence.json"
        audit.write_json(evidence_path, rows)
        screen = {"artifacts": {"comparison_evidence.json": prepare.digest(evidence_path)}}
        prepare.verify_screen_binding(screen, screen_dir / "summary.json", self.old, CONTROL)
        rows[-1]["config"]["task"]["seed"] += 1
        audit.write_json(evidence_path, rows)
        with self.assertRaisesRegex(ValueError, "Screen report artifact changed"):
            prepare.verify_screen_binding(screen, screen_dir / "summary.json", self.old, CONTROL)


if __name__ == "__main__":
    unittest.main()
