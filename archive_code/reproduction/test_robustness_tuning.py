"""Small real-model fixture verification of robustness search lanes and gates."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from search_replacement_grid import read, write, sha
from search_robustness_tuning import plan_search


def main():
    import numpy as np
    root = Path(__file__).resolve().parents[1]
    original = read(root / "reproduction/replacement_search_example.json")
    records = []
    with tempfile.TemporaryDirectory(prefix="robustness_fixture_") as temporary:
        directory = Path(temporary)
        paths = []
        for battery in range(3):
            path = directory / ("battery%d.csv" % battery)
            x = np.arange(40)
            data = np.stack([np.sin((x + battery + 1) * (i + 1) / 50) + x / 100 for i in range(16)], axis=1)
            np.savetxt(path, np.column_stack([data, 2 * (1 - x * .001)]), delimiter=",", comments="",
                       header=",".join(["feature%d" % i for i in range(16)] + ["capacity"]))
            paths.append(str(path))
        base = dict(label="SYNTHETIC_FUNCTIONAL_ONLY", dataset="MIT", model="PINN", loss_variant="paper_prediction_monotonicity",
                    seeds=[20261111], splits={role: dict(paths=[name], nominal_capacity=2) for role, name in zip(("train", "valid", "test"), paths)},
                    fixed=dict(original["fixed"], epochs=2, warmup_epochs=1, early_stop=1, threads=1, batch_size=32),
                    structure=dict(u_layers=3, u_hidden=60, F_layers_num=3, F_hidden_dim=60, alpha=.5, beta=.1),
                    candidates=[dict(lr=.001, lr_F=.0001, batch_size=32, normalization_method="min-max", alpha=.1, beta=0),
                                dict(lr=.003, lr_F=.0005, batch_size=64, normalization_method="z-score", alpha=.5, beta=.1)],
                    protocol_evidence=["Synthetic battery IDs are disjoint; functional smoke only"])
        bad = copy.deepcopy(base)
        bad["splits"]["test"] = bad["splits"]["train"]
        try:
            plan_search(bad, root / "PINN4SOH")
            raise AssertionError("Overlap accepted")
        except ValueError:
            pass
        bad = copy.deepcopy(base)
        bad["loss_variant"] = "published_label_monotonicity"
        try:
            plan_search(bad, root / "PINN4SOH")
            raise AssertionError("Zero-gradient beta tuning accepted")
        except ValueError:
            pass
        print("PASS battery overlap and inactive label-only beta tuning gates", flush=True)
        for model in ("PINN", "MLP", "CNN"):
            config = copy.deepcopy(base)
            config["model"] = model
            if model != "PINN":
                config["loss_variant"] = "supervised_pair_mse"
                config["candidates"] = [{k: v for k, v in c.items() if k not in {"alpha", "beta", "lr_F"}} for c in config["candidates"]]
            spec, output = directory / (model + ".json"), directory / model
            write(spec, config)
            command = [sys.executable, str(root / "reproduction/search_robustness_tuning.py"), "--spec", str(spec), "--repo", str(root / "PINN4SOH"),
                       "--output", str(output), "--device", "cpu", "--execute", "--allow-cpu", "--evaluate-test"]
            process = subprocess.run(command, capture_output=True, text=True)
            if process.returncode:
                print(process.stderr)
                for path in output.rglob("console.log"):
                    print(path.read_text(encoding="utf-8"))
            assert process.returncode == 0
            state, selection = read(output / "status.json"), read(output / "selection.json")
            assert state["status"] == "complete" and state["completed_runs"] == 2 and not selection["test_metrics_used"]
            for path in output.glob("candidate_*/seed_*/worker.json"):
                worker = read(path)
                assert set(worker["splits"]) == {"train", "valid"} and paths[2] not in json.dumps(worker)
            before = {str(path): (sha(path), path.stat().st_mtime_ns) for path in output.rglob("*") if path.name in {"result.json", "selection.json"}}
            resumed = subprocess.run(command + ["--resume"], capture_output=True, text=True)
            assert resumed.returncode == 0, resumed.stderr
            assert all((sha(path), Path(path).stat().st_mtime_ns) == value for path, value in before.items())
            records.append(dict(model=model, status=state["status"], candidate_runs=2, selected=selection["selected"]["candidate_id"]))
            print("PASS " + model + " two candidates, normalization/batch/lr changes, frozen validation selection, test isolation, immutable resume", flush=True)
        # Same candidate preprocessing must be identical across all three methods.
        for candidate in ("candidate_0001", "candidate_0002"):
            data = [read(directory / model / candidate / "seed_20261111/data_identity.json") for model in ("PINN", "MLP", "CNN")]
            assert data[0] == data[1] == data[2]
        print("PASS per-candidate PINN/MLP/CNN paired tensors identical", flush=True)
    print(json.dumps(dict(status="passed", records=records)))
    print("ALL_ROBUSTNESS_TUNING_TESTS_PASSED")


if __name__ == "__main__":
    main()
