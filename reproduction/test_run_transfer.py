import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import run_transfer as runner


REPO = Path(__file__).resolve().parents[1] / "PINN4SOH"


class TransferProtocolTests(unittest.TestCase):
    def test_native_and_paper_mappings_cover_all_directions(self):
        native = runner.transfer_configurations("script_native19")
        paper = runner.transfer_configurations("paper_fixed12")
        directions = {(source, target) for source in runner.DATASETS for target in runner.DATASETS if source != target}
        self.assertEqual(len(native), 19)
        self.assertEqual({(item["source"], item["target"]) for item in native}, directions)
        self.assertEqual(len(paper), 12)
        self.assertEqual({(item["source"], item["target"]) for item in paper}, directions)
        for item in paper:
            for role in ["source", "target"]:
                if item[role] == "XJTU":
                    self.assertEqual(item[role + "_batch"], 0)
                if item[role] == "TJU":
                    self.assertEqual(item[role + "_batch"], 2)
        conflict = next(item for item in paper if item["source"] == "XJTU" and item["target"] == "TJU")
        self.assertTrue(conflict["paper_source_batch_override"])

    def test_task_identifiers_and_raw_hash_are_stable(self):
        tasks = runner.build_transfer_tasks()
        self.assertEqual(len(tasks), 190)
        self.assertEqual(len({task["task_id"] for task in tasks}), 190)
        self.assertEqual(tasks, runner.build_transfer_tasks())
        expected = hashlib.sha256(json.dumps(tasks[0], sort_keys=True).encode()).hexdigest()
        self.assertEqual(runner.stable_hash(tasks[0]), expected)
        with self.assertRaisesRegex(ValueError, "Explicit reproduced source"):
            runner.build_transfer_tasks(source_mode="reproduced_checkpoint")

    def test_split_overlap_is_rejected(self):
        task = runner.build_transfer_tasks()[0]
        split = [{"role": "train_validation", "paths": ["data/x.csv"]},
                 {"role": "test", "paths": ["data/x.csv"]}]
        with self.assertRaisesRegex(ValueError, "leakage"):
            runner.validate_splits(split, task, REPO)

    def test_source_selection_uses_validation_instead_of_test_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = {}
            for repeat, valid, test in [(1, .01, 100), (2, .02, .00001)]:
                key = f"XJTU/2C/Experiment{repeat:02d}"
                folder = root / key
                folder.mkdir(parents=True)
                (folder / "logging.txt").write_text(f"[Valid] epoch:1, MSE: {valid}\n[Test] MSE: {test}\n")
                runs[key] = {"dataset": "XJTU", "batch": "2C", "relative_folder": key, "status": "complete"}
            (root / "manifest.json").write_text(json.dumps({"runs": runs}))
            selected = runner.select_reproduced_source(root, "XJTU", 0)
            self.assertTrue(selected["source_run"].endswith("Experiment01"))
            self.assertEqual(selected["source_selection_candidates"], 2)

    def test_published_label_penalty_has_no_model_gradient(self):
        import torch
        u1 = torch.tensor([[.9]], requires_grad=True)
        u2 = torch.tensor([[1.0]], requires_grad=True)
        y1, y2 = torch.tensor([[1.0]]), torch.tensor([[.9]])
        published = torch.relu(y2 - y1).sum()
        paper = torch.relu(u2 - u1).sum()
        self.assertFalse(published.requires_grad)
        gradients = torch.autograd.grad(paper, (u1, u2))
        self.assertEqual([float(value) for value in gradients], [-1.0, 1.0])


class ActualAuthorTransferSmokeTests(unittest.TestCase):
    def test_frozen_source_best_checkpoint_resume_and_artifact_tamper(self):
        task = runner.build_transfer_tasks(repeats=1)[0]
        watched = [REPO / "Model/Model.py", REPO / "main_adaptation - fine-tuning.py",
                   REPO / task["source_checkpoint"]]
        hashes = {path: runner.common.digest(path) for path in watched}
        with tempfile.TemporaryDirectory(prefix="pinn_transfer_test_") as temporary:
            output = Path(temporary) / "run"
            result = runner.run_task(task, REPO, output, allow_cpu=True, smoke=True)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["epochs_completed"], 2)
            self.assertEqual(result["task_sha256"], runner.stable_hash(task))
            self.assertEqual(result["source_checkpoint_lineage"]["training_seed"], None)
            self.assertTrue(result["frozen_dynamical_F"]["unchanged"])
            self.assertTrue(result["frozen_dynamical_F"]["eval_mode"])
            self.assertTrue(result["solution_u_updated"])
            self.assertTrue(result["checkpoint_verification"]["passed"])
            self.assertEqual(result["sample_counts"]["test"], result["metrics"]["n_test"])
            resumed = runner.run_task(task, REPO, output, resume=True, allow_cpu=True, smoke=True)
            self.assertEqual(resumed["artifact_sha256"], result["artifact_sha256"])
            with (output / "pred_label.npy").open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "artifact.*changed"):
                runner.run_task(task, REPO, output, resume=True, allow_cpu=True, smoke=True)
        self.assertEqual({path: runner.common.digest(path) for path in watched}, hashes)

    def test_explicit_battery_split_and_paper_loss_variant(self):
        task = runner.build_transfer_tasks(repeats=1)[0]
        task.update(task_id="test/explicit-batteries", validation_protocol="explicit_battery_split",
                    loss_variant="paper_prediction_monotonicity")
        with tempfile.TemporaryDirectory(prefix="pinn_transfer_split_") as temporary:
            root = Path(temporary)
            split = [{"role": role, "paths": [f"data/XJTU data/2C_battery-{number}.csv"], "nominal_capacity": 2.0}
                     for role, number in [("train", 1), ("valid", 2), ("test", 4)]]
            (root / "split.json").write_text(json.dumps(split))
            task["split_json"] = str(root / "split.json")
            result = runner.run_task(task, REPO, root / "run", allow_cpu=True, smoke=True)
            self.assertTrue(result["split_audit"]["reference_split_checked"])
            self.assertTrue(result["frozen_dynamical_F"]["unchanged"])
            records = runner.read_json(root / "run/splits.json")
            self.assertEqual([item["role"] for item in records], ["train", "valid", "test"])
            self.assertEqual([item["paths"] for item in records], [item["paths"] for item in split])

    def test_source_pretraining_with_published_loss_variant(self):
        task = dict(task_id="test/pretraining", family="transfer_pretrain", dataset="XJTU", source="XJTU",
                    source_batch=0, batch="2C", model="PINN", repeat=1, seed=12345,
                    protocol_id="published_code", loss_variant="published_label_monotonicity",
                    epochs=200, threads=1)
        with tempfile.TemporaryDirectory(prefix="pinn_pretrain_test_") as temporary:
            output = Path(temporary) / "run"
            result = runner.run_task(task, REPO, output, allow_cpu=True, smoke=True)
            self.assertEqual(result["status"], "complete")
            self.assertIsNone(result["source_checkpoint_lineage"])
            self.assertTrue(result["checkpoint_verification"]["passed"])
            self.assertEqual(runner.read_json(output / "config.json")["physical_loss_parameter_gradient"], "identically_zero")


if __name__ == "__main__":
    unittest.main()
