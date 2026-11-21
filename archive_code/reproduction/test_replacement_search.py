"""Selection isolation, failure gates and real-model CPU smoke for replacement search."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("replacement_search", ROOT / "search_replacement_grid.py")
search = importlib.util.module_from_spec(spec)
spec.loader.exec_module(search)


def main():
    repo = ROOT.parent / "PINN4SOH"
    template = search.read(ROOT / "replacement_search_example.json")
    candidates = [dict(candidate_id="candidate_0001", parameters={}), dict(candidate_id="candidate_0002", parameters={})]
    rows = [dict(candidate_id=c["candidate_id"], seed=s, validation_mse=v, test_mse=t)
            for c, v, t in [(candidates[0], 1.0, 9000), (candidates[1], 2.0, 0)] for s in [11, 22]]
    chosen = search.select(rows, candidates, [11, 22])
    assert chosen["selected"]["candidate_id"] == "candidate_0001" and not chosen["test_metrics_used"]
    changed = copy.deepcopy(rows)
    for row in changed:
        row["test_mse"] = -row["test_mse"]
    assert search.select(changed, candidates, [11, 22]) == chosen
    try:
        search.select(rows[:-1], candidates, [11, 22])
        raise AssertionError("Incomplete search accepted")
    except ValueError:
        pass
    invalid = copy.deepcopy(template)
    invalid["splits"]["test"] = invalid["splits"]["valid"]
    try:
        search.build_plan(invalid, repo, 128)
        raise AssertionError("Overlapping battery IDs accepted")
    except ValueError:
        pass
    print("PASS test-metric inversion has no effect; missing pairs and overlapping battery IDs rejected")
    with tempfile.TemporaryDirectory(prefix="replacement_grid_fixture_") as tmp:
        root = Path(tmp)
        paths = []
        import numpy as np
        for battery in range(3):
            path = root / ("battery_%d.csv" % battery)
            x = np.arange(40)
            features = np.stack([np.sin((x + battery + 1) * (i + 1) / 50) + x / 100 for i in range(16)], axis=1)
            np.savetxt(path, np.column_stack([features, 2 * (1 - x * .001)]), delimiter=",",
                       header=",".join(["feature%d" % i for i in range(16)] + ["capacity"]), comments="")
            paths.append(str(path))
        config = copy.deepcopy(template)
        config.update(label="SYNTHETIC_FUNCTIONAL_SMOKE_ONLY", seeds=[11, 22])
        config["splits"] = {role: dict(paths=[name], nominal_capacity=2)
                            for role, name in zip(("train", "valid", "test"), paths)}
        config["fixed"].update(epochs=2, warmup_epochs=1, early_stop=1, threads=1, batch_size=64)
        config["grid"] = dict(u_layers=[2], u_hidden=[8, 12], F_layers_num=[2], F_hidden_dim=[8], alpha=[0.35, .7], beta=[20])
        task_path, output = root / "spec.json", root / "search"
        search.write(task_path, config)
        command = [sys.executable, str(ROOT / "search_replacement_grid.py"), "--spec", str(task_path),
                   "--repo", str(repo), "--output", str(output), "--device", "cpu"]
        dry = subprocess.run(command, text=True, capture_output=True)
        assert dry.returncode == 0 and search.read(output / "plan.json")["candidate_runs"] == 8, dry.stderr
        assert not (output / "status.json").exists()
        run = subprocess.run(command + ["--execute", "--allow-cpu", "--evaluate-test"], text=True, capture_output=True)
        if run.returncode:
            print(run.stderr)
            for path in output.rglob("console.log"):
                print(str(path), path.read_text(encoding="utf-8"))
        assert run.returncode == 0
        state = search.read(output / "status.json")
        assert state["status"] == "complete" and state["completed_runs"] == 8
        files = list(output.glob("candidate_*/seed_*/result.json"))
        assert len(files) == 8
        for path in files:
            result, worker = search.read(path), search.read(path.with_name("worker.json"))
            assert result["test_loaded"] is False and set(worker["splits"]) == {"train", "valid"}
            assert paths[2] not in json.dumps(worker)
        tests = list(output.glob("selected_test/seed_*/result.json"))
        assert len(tests) == 2 and all(not search.read(path)["test_used_for_selection"] for path in tests)
        preserve = {str(path): (search.sha(path), path.stat().st_mtime_ns) for path in files + tests + [output / "selection.json"]}
        resumed = subprocess.run(command + ["--execute", "--allow-cpu", "--evaluate-test", "--resume"], text=True, capture_output=True)
        assert resumed.returncode == 0, resumed.stderr
        assert all((search.sha(path), Path(path).stat().st_mtime_ns) == observed for path, observed in preserve.items())
        target = files[0].with_name("valid_prediction.npy")
        with target.open("ab") as stream:
            stream.write(b"tamper")
        refused = subprocess.run(command + ["--execute", "--allow-cpu", "--resume"], text=True, capture_output=True)
        assert refused.returncode != 0 and "Completed artifact changed" in refused.stderr
        print("PASS 8 two-epoch real PINN workers; train/validation-only payloads; 2 selected-test workers; immutable completed resume; SHA tamper rejected")
    print("ALL_REPLACEMENT_SEARCH_TESTS_PASSED")


if __name__ == "__main__":
    main()
