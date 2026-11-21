"""Focused exact-identity and recorded-run replay checks, without training."""

import argparse
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import audit_validation_overlap as audit


class ValidationOverlapTests(unittest.TestCase):
    def test_endpoint_identity_includes_battery(self):
        train = [("A", 0, 1), ("A", 2, 3), ("B", 0, 1)]
        valid = [("A", 1, 2), ("B", 1, 2), ("C", 1, 2)]
        result = audit.endpoint_counts(train, valid, [("T", 0, 1)])
        self.assertEqual(result["valid"]["x1_in_train_x1"], 0)
        self.assertEqual(result["valid"]["x1_in_train_x2"], 2)
        self.assertEqual(result["valid"]["x1_in_train_either_endpoint_fraction"], 2 / 3)
        self.assertEqual(result["test"]["x1_in_train_either_endpoint"], 0)

    def test_extra_identity_array_preserves_original_split_and_rng(self):
        import numpy as np
        import torch
        from sklearn.model_selection import train_test_split
        arrays = [torch.arange(41 * 3).reshape(41, 3).float() + offset for offset in range(4)]
        before = audit.rng_snapshot(np, torch)
        original = train_test_split(*arrays, test_size=.2, random_state=420)
        traced = train_test_split(*arrays, np.arange(41), test_size=.2, random_state=420)
        self.assertEqual(before, audit.rng_snapshot(np, torch))
        for expected, observed in zip(original, traced[:-2]):
            self.assertTrue(torch.equal(expected, observed))
        self.assertEqual(set(traced[-2]) | set(traced[-1]), set(range(41)))
        self.assertFalse(set(traced[-2]) & set(traced[-1]))

    def test_real_n1_recorded_metadata_and_tensor_hash_replay(self):
        repo = Path(__file__).resolve().parents[1] / "PINN4SOH"
        if not (repo / "main_XJTU.py").is_file():
            self.skipTest("Local author repository not present")
        sys.path.insert(0, str(repo))
        options = argparse.Namespace(repo=repo, dataset="XJTU", batch="2C", seed=20261109,
                                     normalization_method="min-max")
        previous = Path.cwd()
        try:
            os.chdir(repo)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                traced = audit.replay(options, 1, {}, None)
                original = audit.replay(options, 1, {}, None, instrument=False)
        finally:
            os.chdir(previous)
        self.assertEqual(traced["tensor_sha256"], original["tensor_sha256"])
        self.assertTrue(traced["global_rng_unchanged_by_loading"])
        with tempfile.TemporaryDirectory(prefix="pinn4soh_overlap_replay_") as temporary:
            root = Path(temporary)
            recorded, output = root / "recorded_loader_fixture", root / "audit"
            recorded.mkdir()
            config = {"parameters": traced["parameters"], "seed": options.seed,
                      "task": {"dataset": "XJTU", "model": "PINN", "sample_size": 1}}
            splits = [{"role": role, "paths": frame["paths"], "nominal_capacity": frame["nominal_capacity"]}
                      for role, frame in zip(["train_validation", "test"], traced["loader_trace"])]
            inputs = {battery["path"]: battery["sha256"] for frame in traced["loader_trace"] for battery in frame["batteries"]}
            audit.write_json(recorded / "config.json", config)
            audit.write_json(recorded / "splits.json", splits)
            audit.write_json(recorded / "data_identity.json", {"tensor_sha256": original["tensor_sha256"], "input_file_sha256": inputs})
            command = [sys.executable, "-W", "ignore::FutureWarning", str(Path(audit.__file__).resolve()),
                       "--repo", str(repo), "--run", str(recorded), "--output", str(output)]
            response = subprocess.run(command, text=True, capture_output=True, timeout=60)
            self.assertEqual(response.returncode, 0, response.stderr)
            result = audit.read_json(output / "summary.json")["rows"][0]
            self.assertTrue(result["original_run_tensor_hash_match"])
            self.assertTrue(result["instrumentation_preserved_unmodified_loader_tensors"])
            self.assertEqual(result["recorded_input_hash_checks"], 3)
            self.assertEqual(result["valid"]["x1_in_train_x2"], 57)
            self.assertEqual(result["valid"]["n_evaluated_x1"], 71)


if __name__ == "__main__":
    unittest.main(verbosity=2)
