"""Audit all 110 completed experiments and export exact UTF-8 evidence, using only stdlib."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import sys
import uuid

BATCHES = {"XJTU": ["2C", "3C", "R2.5", "R3", "RW", "satellite"],
           "MIT": ["all"], "HUST": ["all"], "TJU": ["0", "1", "2"]}
METRICS = ["MAE", "MAPE", "MSE", "RMSE", "MAE_pp", "MAPE_pct", "RMSE_pp"]
ROOT_FILES = ["metrics_per_run.csv", "metrics_summary.csv", "manifest.json",
              "environment.json", "source_inventory.json"]
REPRESENTATIVES = ["XJTU/2C/Experiment01", "MIT/all/Experiment01",
                   "HUST/all/Experiment01", "TJU/0/Experiment01"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def hash_file(path, nonempty=False):
    before = path.stat()
    require(not nonempty or before.st_size > 0, "Empty artifact: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            "File changed while hashing: " + str(path))
    return {"sha256": digest.hexdigest(), "bytes": after.st_size,
            "mtime_ns": after.st_mtime_ns}


def inside(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def json_text(text):
    return json.loads(text.lstrip("\ufeff"))


def csv_rows(text):
    return list(csv.DictReader(io.StringIO(text.lstrip("\ufeff"), newline="")))


def number(value, label):
    result = float(value)
    require(math.isfinite(result) and result >= 0, "Invalid numeric value: " + label)
    return result


def close(actual, expected, label):
    require(math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-12),
            f"Numeric mismatch for {label}: {actual!r} != {expected!r}")


def audit(results, repo):
    evidence_paths = ROOT_FILES + [key + "/verification_checkpoint.json" for key in REPRESENTATIVES]
    raw = {relative: (results / relative).read_bytes() for relative in evidence_paths}
    texts = {relative: data.decode("utf-8") for relative, data in raw.items()}
    manifest = json_text(texts["manifest.json"])
    inventory = json_text(texts["source_inventory.json"])
    environment = json_text(texts["environment.json"])
    expected = {f"{dataset}/{batch}/Experiment{repeat:02d}": (dataset, batch, repeat)
                for dataset, batches in BATCHES.items() for batch in batches for repeat in range(1, 11)}
    runs = manifest["runs"]
    require(set(runs) == set(expected), "Manifest must contain exactly the expected 110 experiment keys")
    statuses = Counter(record.get("status") for record in runs.values())
    require(statuses == {"complete": 110}, "Final export requires 110 complete and zero other statuses")
    require(environment.get("threads") == 2, "Unexpected CPU thread setting")
    require(environment.get("cuda_available") is True, "Final training environment did not report CUDA")
    stored_hashes = inventory["sha256"]
    internal_fingerprint = sha256(json.dumps(stored_hashes, sort_keys=True).encode())
    require(internal_fingerprint == inventory["fingerprint"], "Source inventory fingerprint is inconsistent")
    actual_hashes = {}
    for path in sorted(repo.rglob("*")):
        if path.is_file() and (path.suffix == ".py" or path.name in {"README.md", "fix.md"}
                               or (path.suffix == ".csv" and "data" in path.relative_to(repo).parts)):
            actual_hashes[path.relative_to(repo).as_posix()] = hash_file(path)["sha256"]
    require(actual_hashes == stored_hashes, "Current repository/data files do not match the training inventory")
    directory_order = {Path(root).relative_to(repo).as_posix(): {"directories": dirs, "files": files}
                       for root, dirs, files in os.walk(repo / "data")}
    require(directory_order == inventory["data_directory_order"], "Data enumeration order changed")

    artifacts = {}
    for key, (dataset, batch, repeat) in expected.items():
        record = runs[key]
        require((record["dataset"], str(record["batch"]), record["repeat"]) == (dataset, batch, repeat),
                "Manifest task identity mismatch: " + key)
        require(record["relative_folder"] == key, "Manifest output path mismatch: " + key)
        seed = 20261109 + list(BATCHES).index(dataset) * 10000 + BATCHES[dataset].index(batch) * 100 + repeat - 1
        require(record["seed"] == seed, "Manifest seed mismatch: " + key)
        folder = results / key
        config_data = (folder / "config.json").read_bytes()
        config = json_text(config_data.decode("utf-8"))
        require(sha256(json.dumps(config, sort_keys=True).encode()) == record["config_hash"],
                "Configuration hash mismatch: " + key)
        require(config["source_fingerprint"] == internal_fingerprint, "Source fingerprint mismatch: " + key)
        require(config["seed"] == seed and config["threads"] == 2 and config["smoke"] is False
                and config.get("checkpoint_snapshot_deepcopy") is True,
                "Unexpected training configuration: " + key)
        require(config["parameters"]["epochs"] == 200, "Not a formal 200-epoch-cap run: " + key)
        require(0 < int(record["epochs_completed"]) <= 200, "Invalid completed epoch count: " + key)
        require(record["metrics"]["n_test"] == record["sample_counts"]["test"] > 0,
                "Test sample count mismatch: " + key)
        artifacts[key] = {name: hash_file(folder / name, nonempty=True)
                          for name in ["model.pth", "pred_label.npy", "true_label.npy"]}
        artifacts[key]["config.json"] = {"sha256": sha256(config_data), "bytes": len(config_data)}

    rows = csv_rows(texts["metrics_per_run.csv"])
    require(len(rows) == 110 and Counter(row["run"] for row in rows) == Counter(expected.keys()),
            "Per-run CSV must contain exactly the same 110 unique runs")
    groups = defaultdict(list)
    for row in rows:
        key = row["run"]
        dataset, batch, repeat = expected[key]
        record = runs[key]
        require((row["dataset"], row["batch"], int(row["repeat"])) == (dataset, batch, repeat),
                "Per-run CSV task mismatch: " + key)
        require(int(row["seed"]) == record["seed"] and int(row["n_test"]) == record["metrics"]["n_test"],
                "Per-run CSV seed/sample count mismatch: " + key)
        for metric in METRICS:
            close(number(row[metric], key + "/" + metric), record["metrics"][metric], key + "/" + metric)
        close(number(row["training_seconds"], key + "/time"), record["training_seconds"], key + "/time")
        groups[(dataset, batch)].append(row)
    expected_groups = {(dataset, batch) for dataset, batches in BATCHES.items() for batch in batches}
    require(set(groups) == expected_groups and all(len(items) == 10 for items in groups.values()),
            "Expected 11 dataset/batch groups with 10 runs each")
    summaries = csv_rows(texts["metrics_summary.csv"])
    require(len(summaries) == 11 and Counter((row["dataset"], row["batch"]) for row in summaries)
            == Counter(expected_groups), "Summary CSV must have exactly 11 unique groups")
    for row in summaries:
        group = (row["dataset"], row["batch"])
        require(int(row["n_runs"]) == 10, "Summary repetition count is not 10")
        for metric in [*METRICS, "training_seconds"]:
            values = [float(item[metric]) for item in groups[group]]
            close(row[metric + "_mean"], statistics.mean(values), str(group) + "/" + metric + " mean")
            close(row[metric + "_std"], statistics.stdev(values), str(group) + "/" + metric + " sample std")

    verifications = {}
    for key in REPRESENTATIVES:
        relative = key + "/verification_checkpoint.json"
        report = json_text(texts[relative])
        for flag in ["passed", "source_fingerprint_verified", "all_arrays_finite",
                     "labels_bitwise_equal", "predictions_allclose"]:
            require(report.get(flag) is True, "Checkpoint verification failed: " + key + "/" + flag)
        require(Path(report["run"]).resolve() == (results / key).resolve(), "Verification run path mismatch: " + key)
        require(report["checkpoint_sha256"] == artifacts[key]["model.pth"]["sha256"],
                "Stale checkpoint verification: " + key)
        require(report["saved_prediction_sha256"] == artifacts[key]["pred_label.npy"]["sha256"],
                "Stale prediction verification: " + key)
        require(report["n_test"] == runs[key]["metrics"]["n_test"], "Verification sample mismatch: " + key)
        require(0 <= float(report["atol"]) <= 1e-6 and 0 <= float(report["rtol"]) <= 1e-5,
                "Verification tolerance is too loose: " + key)
        for metric in METRICS:
            close(report["saved_metrics"][metric], runs[key]["metrics"][metric], key + "/verified " + metric)
        verifications[key] = {"passed": True, "predictions_bitwise_equal": report["predictions_bitwise_equal"],
                              "max_absolute_prediction_difference": number(report["max_absolute_prediction_difference"], key)}
    for relative, before in raw.items():
        require((results / relative).read_bytes() == before, "Evidence changed during audit: " + relative)
    for key, entries in artifacts.items():
        for name, entry in entries.items():
            if "mtime_ns" in entry:
                stat = (results / key / name).stat()
                require((stat.st_size, stat.st_mtime_ns) == (entry["bytes"], entry["mtime_ns"]),
                        "Artifact changed during audit: " + key + "/" + name)
    return {"format": "pinn4soh-final-evidence-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "results_root": str(results), "repo_root": str(repo),
            "audit": {"passed": True, "expected_runs": 110, "status_counts": dict(statuses),
                      "group_counts": {dataset + "/" + batch: len(items)
                                       for (dataset, batch), items in groups.items()},
                      "summary_mean_and_sample_std_verified": True,
                      "source_fingerprint_verified": internal_fingerprint,
                      "source_files_verified": len(stored_hashes), "data_directory_order_verified": True,
                      "representative_checkpoint_verifications": verifications,
                      "training_seed_base": 20261109, "threads_per_run": 2,
                      "scope": "Current bug-fixed code; pooled-test-sample metrics; no claim of exact paper-number reproduction"},
            "files": texts, "sha256": {relative: sha256(data) for relative, data in raw.items()},
            "artifacts": artifacts,
            "restore_note": "Encode each files[relative_path] with UTF-8, without newline conversion; sha256 verifies exact original bytes. Artifacts are hashed, not embedded."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("/workspace/pinn4soh_main_20261109"))
    parser.add_argument("--repo", type=Path, default=Path("/workspace/PINN4SOH-main"))
    parser.add_argument("--output-file", type=Path, default=Path("/workspace/pinn4soh_final_evidence.json"))
    options = parser.parse_args()
    results, repo, output = options.results.resolve(), options.repo.resolve(), options.output_file.resolve()
    require(not inside(output, results) and not inside(output, repo), "Export file must be outside results and repository")
    bundle = audit(results, repo)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp." + uuid.uuid4().hex)
    temporary.write_bytes(json.dumps(bundle, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"))
    temporary.replace(output)
    print(json.dumps({"passed": True, "complete": 110, "groups": 11, "checkpoint_verifications": 4,
                      "export_file": str(output), "bytes": output.stat().st_size,
                      "sha256": hash_file(output)["sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"passed": False, "error": repr(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
