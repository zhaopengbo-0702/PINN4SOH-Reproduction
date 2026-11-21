"""CPU checks of real author models on tiny synthetic battery CSVs; two epochs only."""

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    runner = load("extended", ROOT / "run_extended.py")
    helpers = load("helpers", ROOT / "run_reproduction.py")
    import numpy as np
    truth = np.array([0.95, 0.85, 1.0, 0.9, 0.8])
    prediction = truth + np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    splits = [{"role": "test", "batteries": [{"path": "one", "n_pairs": 2}, {"path": "two", "n_pairs": 3}]}]
    result = runner.evaluate_predictions(truth, prediction, splits, "XJTU", helpers)
    expected = (helpers.calculate_metrics(truth[:2], prediction[:2])["MAPE"] +
                helpers.calculate_metrics(truth[2:], prediction[2:])["MAPE"]) / 2
    assert abs(result["battery_macro_standard"]["MAPE"] - expected) < 1e-12
    assert result["battery_macro_original"]["MAPE"] != result["battery_macro_standard"]["MAPE"]
    assert result["legacy_author_analysis"]["dropped_sample_indices"] == [1]
    print("PASS exact battery-macro weighting, both MAPE denominators, and legacy dropped boundary sample", flush=True)
    repo = ROOT.parent / "PINN4SOH"
    original = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in repo.rglob("*.py")}
    with tempfile.TemporaryDirectory(prefix="pinn4soh_extended_test_") as temporary:
        directory = Path(temporary)
        batteries = []
        for battery in range(4):
            path = directory / f"battery{battery}.csv"
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow([f"feature{i}" for i in range(16)] + ["capacity"])
                for cycle in range(40):
                    writer.writerow([math.sin((cycle + battery + 1) * (i + 1) / 50) + cycle / 100
                                     for i in range(16)] + [2 * (1 - cycle * 0.001)])
            batteries.append(str(path))
        split_path = directory / "split.json"
        split_path.write_text(json.dumps([
            {"role": "train", "paths": batteries[:2], "nominal_capacity": 2},
            {"role": "valid", "paths": batteries[2:3], "nominal_capacity": 2},
            {"role": "test", "paths": batteries[3:], "nominal_capacity": 2}]))
        fingerprints = []
        for model in ["PINN", "MLP", "CNN"]:
            task = {"task_id": "synthetic_" + model, "family": "small_sample", "dataset": "XJTU",
                    "batch": "2C", "model": model, "repeat": 1, "seed": 20261109, "epochs": 200,
                    "threads": 2, "protocol_id": "paper_note3", "sample_size": 1,
                    "validation_protocol": "explicit_battery_split", "split_json": str(split_path)}
            task_path, output = directory / (model + ".json"), directory / model
            task_path.write_text(json.dumps(task))
            command = [sys.executable, "-W", "ignore::FutureWarning", str(ROOT / "run_extended.py"),
                       "--task-json", str(task_path), "--repo", str(repo), "--output", str(output),
                       "--smoke", "--allow-cpu", "--resume"]
            process = subprocess.run(command, capture_output=True, text=True)
            if process.returncode:
                error = (output / "error.txt").read_text() if (output / "error.txt").exists() else process.stderr
                raise AssertionError(model + " " + error)
            state = json.loads((output / "manifest.json").read_text())
            assert state["status"] == "complete" and state["epochs_completed"] == 2
            assert state["task_sha256"] == runner.config_hash(task)
            assert json.loads((output / "verification_checkpoint.json").read_text())["passed"]
            config = json.loads((output / "config.json").read_text())
            assert config["parameters"]["batch_size"] == 256
            if model == "PINN":
                assert config["parameters"]["alpha"] == 0.7 and config["parameters"]["beta"] == 20
            fingerprints.append(json.loads((output / "data_identity.json").read_text())["tensor_sha256"])
            before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.rglob("*") if p.is_file()}
            rerun = subprocess.run(command, capture_output=True, text=True)
            assert rerun.returncode == 0 and "SKIP" in rerun.stdout, rerun.stderr
            assert all((p.read_bytes(), p.stat().st_mtime_ns) == values for p, values in before.items())
            print("PASS " + model + ": two CPU epochs, explicit disjoint batteries, checkpoint reload, legacy metrics and immutable resume", flush=True)
        assert fingerprints[0] == fingerprints[1] == fingerprints[2]
        print("PASS all three models use identical paired train/valid/test tensors", flush=True)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == value for p, value in original.items())
    print("PASS author Python source unchanged; all temporary smoke outputs removed", flush=True)


if __name__ == "__main__":
    main()
