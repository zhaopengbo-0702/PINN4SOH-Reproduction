"""Check the final evidence contract with synthetic files and stdlib only."""

import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

SCRIPT = Path(__file__).resolve().with_name("export_final_evidence.py")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"))


def write_csv(path, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    path.write_bytes(stream.getvalue().encode("utf-8-sig"))


def snapshot(root):
    return {path.relative_to(root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()}


def main():
    spec = importlib.util.spec_from_file_location("exporter", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix="pinn4soh_final_export_test_") as temporary:
        root = Path(temporary)
        repo, results, restored = root / "repo", root / "results", root / "restored"
        (repo / "data").mkdir(parents=True)
        results.mkdir()
        (repo / "model.py").write_bytes(b"# synthetic source\r\n")
        (repo / "__init__.py").write_bytes(b"")
        (repo / "data/test.csv").write_bytes(b"value\n1\n")
        hashes = {path.relative_to(repo).as_posix(): module.hash_file(path)["sha256"]
                  for path in sorted(repo.rglob("*")) if path.is_file()}
        fingerprint = module.sha256(json.dumps(hashes, sort_keys=True).encode())
        order = {Path(path).relative_to(repo).as_posix(): {"directories": dirs, "files": files}
                 for path, dirs, files in os.walk(repo / "data")}
        write_json(results / "source_inventory.json", {"fingerprint": fingerprint, "sha256": hashes,
                                                       "data_directory_order": order})
        write_json(results / "environment.json", {"threads": 2, "cuda_available": True, "gpu": "Synthetic GPU"})
        manifest, rows, summaries = {"runs": {}, "note": "中文编码校验"}, [], []
        for dataset, batches in module.BATCHES.items():
            for batch in batches:
                group = []
                for repeat in range(1, 11):
                    key = f"{dataset}/{batch}/Experiment{repeat:02d}"
                    folder = results / key
                    folder.mkdir(parents=True)
                    seed = 20261109 + list(module.BATCHES).index(dataset) * 10000 + batches.index(batch) * 100 + repeat - 1
                    config = {"parameters": {"epochs": 200}, "seed": seed, "threads": 2,
                              "smoke": False, "checkpoint_snapshot_deepcopy": True,
                              "source_fingerprint": fingerprint}
                    write_json(folder / "config.json", config)
                    for name in ["model.pth", "pred_label.npy", "true_label.npy"]:
                        (folder / name).write_bytes(("synthetic " + key + name).encode())
                    metrics = {"MAE": 0.01, "MAPE": 0.02, "MSE": 0.0004, "RMSE": 0.02,
                               "MAE_pp": 1.0, "MAPE_pct": 2.0, "RMSE_pp": 2.0, "n_test": 5}
                    record = {"dataset": dataset, "batch": batch, "repeat": repeat, "seed": seed,
                              "status": "complete", "relative_folder": key,
                              "config_hash": module.sha256(json.dumps(config, sort_keys=True).encode()),
                              "epochs_completed": 100, "metrics": metrics, "sample_counts": {"test": 5},
                              "training_seconds": 100.0 + repeat}
                    manifest["runs"][key] = record
                    row = {"run": key, "dataset": dataset, "batch": batch, "repeat": repeat, "seed": seed,
                           "n_test": 5, "training_seconds": record["training_seconds"],
                           **{metric: metrics[metric] for metric in module.METRICS}}
                    rows.append(row)
                    group.append(row)
                    if key in module.REPRESENTATIVES:
                        report = {flag: True for flag in ["passed", "source_fingerprint_verified", "all_arrays_finite",
                                                          "labels_bitwise_equal", "predictions_allclose"]}
                        report.update(run=str(folder.resolve()), n_test=5, atol=1e-6, rtol=1e-5,
                                      checkpoint_sha256=module.hash_file(folder / "model.pth")["sha256"],
                                      saved_prediction_sha256=module.hash_file(folder / "pred_label.npy")["sha256"],
                                      saved_metrics=metrics, predictions_bitwise_equal=False,
                                      max_absolute_prediction_difference=1e-7)
                        write_json(folder / "verification_checkpoint.json", report)
                summary = {"dataset": dataset, "batch": batch, "n_runs": 10}
                for metric in [*module.METRICS, "training_seconds"]:
                    values = [row[metric] for row in group]
                    summary[metric + "_mean"] = statistics.mean(values)
                    summary[metric + "_std"] = statistics.stdev(values)
                summaries.append(summary)
        write_json(results / "manifest.json", manifest)
        write_csv(results / "metrics_per_run.csv", rows)
        write_csv(results / "metrics_summary.csv", summaries)
        before = snapshot(results)
        output = root / "evidence.json"
        command = [sys.executable, "-S", str(SCRIPT), "--results", str(results), "--repo", str(repo),
                   "--output-file", str(output)]
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert snapshot(results) == before
        bundle = json.loads(output.read_text(encoding="utf-8"))
        assert bundle["audit"]["status_counts"] == {"complete": 110}
        assert len(bundle["audit"]["group_counts"]) == 11 and len(bundle["files"]) == 9
        for relative, text in bundle["files"].items():
            destination = restored / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(text.encode("utf-8"))
            assert destination.read_bytes() == (results / relative).read_bytes()
            assert hashlib.sha256(destination.read_bytes()).hexdigest() == bundle["sha256"][relative]
        assert bundle["files"]["metrics_per_run.csv"].startswith("\ufeff")
        print("PASS stdlib-only final audit: 110 complete, 11 groups, four verified checkpoints")
        print("PASS nine original files restore byte-for-byte, including UTF-8 BOM and CRLF; results untouched")
        manifest["runs"]["HUST/all/Experiment10"]["status"] = "running"
        write_json(results / "manifest.json", manifest)
        failure_output = root / "must_not_exist.json"
        failed = subprocess.run(command[:-1] + [str(failure_output)], capture_output=True, text=True)
        assert failed.returncode != 0 and not failure_output.exists()
        print("PASS incomplete final state is rejected without exporting a misleading result")


if __name__ == "__main__":
    main()
