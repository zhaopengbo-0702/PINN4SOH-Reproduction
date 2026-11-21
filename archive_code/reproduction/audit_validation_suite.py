"""Audit the requested 43 memberships using recorded cloud directory order."""

import argparse
from collections import Counter
import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import warnings

import audit_validation_overlap as audit


@contextlib.contextmanager
def recorded_directory_order(repo, inventory):
    original = os.listdir

    def listdir(path="."):
        resolved = Path(path).resolve()
        try:
            relative = resolved.relative_to(repo).as_posix()
        except ValueError:
            return original(path)
        if not relative.startswith("data/"):
            return original(path)
        record = inventory["data_directory_order"].get(relative)
        if record is None:
            raise ValueError("No recorded directory order for " + relative)
        if record["files"] and record["directories"]:
            raise ValueError("Mixed file/directory enumeration cannot be reconstructed unambiguously: " + relative)
        expected = record["directories"] + record["files"]
        if set(original(path)) != set(expected):
            raise ValueError("Current directory members differ from recorded cloud inventory: " + relative)
        return list(expected)

    os.listdir = listdir
    try:
        yield
    finally:
        os.listdir = original


def evidence_inputs(path):
    evidence = audit.read_json(path)
    if evidence.get("format") != "pinn4soh-final-evidence-v1" or not evidence["audit"]["passed"]:
        raise ValueError("Expected the verified final-evidence package")
    for name in ["source_inventory.json", "manifest.json"]:
        if hashlib.sha256(evidence["files"][name].encode()).hexdigest() != evidence["sha256"][name]:
            raise ValueError("Embedded evidence hash mismatch: " + name)
    return evidence, json.loads(evidence["files"]["source_inventory.json"]), json.loads(evidence["files"]["manifest.json"])


def cases(manifest):
    groups = [("XJTU", batch) for batch in ["2C", "3C", "R2.5", "R3", "RW", "satellite"]]
    groups += [("MIT", "all"), ("HUST", "all"), *[("TJU", str(index)) for index in range(3)]]
    result = []
    for dataset, batch in groups:
        reference = dataset + "/" + batch + "/Experiment01"
        old = manifest["runs"][reference]
        sizes = [None]
        if dataset in {"XJTU", "HUST"}:
            sizes += [1, 2, 3, 4]
        elif dataset == "MIT" or dataset == "TJU" and batch == "2":
            sizes += [1, 2]
        for size in sizes:
            result.append({"dataset": dataset, "batch": batch, "sample_size": size,
                           "seed": old["seed"], "reference_run": reference,
                           "reference_full_sample_counts": old["sample_counts"]})
    if len(result) != 43:
        raise AssertionError("Unexpected case count")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group", action="append", help="Optional exact DATASET/BATCH subset")
    options = parser.parse_args()
    options.repo, options.output, options.evidence = options.repo.resolve(), options.output.resolve(), options.evidence.resolve()
    if options.output == options.repo or options.repo in options.output.parents or options.output in options.repo.parents:
        raise ValueError("Diagnostic output must be separate from the author repository")
    if options.output.exists() and any(options.output.iterdir()):
        raise ValueError("Output must be new or empty")
    evidence, inventory, manifest = evidence_inputs(options.evidence)
    selected = [item for item in cases(manifest) if not options.group or item["dataset"] + "/" + item["batch"] in options.group]
    if not selected:
        raise ValueError("No audit cases selected")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import pandas as pd
    import sklearn
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(options.repo))
    for name in ["dataloader/dataloader.py", "Model/Model.py", *["main_" + name + ".py" for name in ["XJTU", "MIT", "HUST", "TJU"]]]:
        if audit.digest(options.repo / name) != inventory["sha256"][name]:
            raise ValueError("Author source differs from recorded cloud source: " + name)
    options.output.mkdir(parents=True)
    rows, checked, started, previous = [], set(), time.monotonic(), Path.cwd()
    try:
        os.chdir(options.repo)
        with recorded_directory_order(options.repo, inventory), warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            for case in selected:
                args = argparse.Namespace(repo=options.repo, dataset=case["dataset"], batch=case["batch"],
                                          normalization_method="min-max", seed=case["seed"])
                result = audit.replay(args, case["sample_size"], {}, None)
                unmodified = audit.replay(args, case["sample_size"], {}, None, instrument=False)
                if result["tensor_sha256"] != unmodified["tensor_sha256"]:
                    raise ValueError("Instrumentation changed tensors")
                # One full case per group also checks model-initialization/global-RNG independence.
                perturbation_verified = None
                if case["sample_size"] is None:
                    perturbed = audit.replay(args, None, {}, None, perturb_rng=True)
                    perturbation_verified = result["tensor_sha256"] == perturbed["tensor_sha256"] and result["provenance"] == perturbed["provenance"]
                    if not perturbation_verified:
                        raise ValueError("Global RNG/model initialization changed membership")
                    if {key: result["counts"]["n_" + key + "_pairs"] for key in ["train", "valid", "test"]} != case["reference_full_sample_counts"]:
                        raise ValueError("Recorded cloud sample counts disagree: " + case["reference_run"])
                for frame in result["loader_trace"]:
                    for battery in frame["batteries"]:
                        if battery["sha256"] != inventory["sha256"][battery["battery_id"]]:
                            raise ValueError("CSV differs from recorded cloud source: " + battery["battery_id"])
                        checked.add(battery["battery_id"])
                name = case["dataset"] + "/" + case["batch"] + "/" + ("nall" if case["sample_size"] is None else "n" + str(case["sample_size"]))
                destination = options.output / name
                destination.mkdir(parents=True)
                audit.write_json(destination / "trace.json", {key: value for key, value in result.items() if key != "provenance"})
                with (destination / "cycle_provenance.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["role", "dataset_position", "battery_id", "x1_original_row_0based", "x2_original_row_0based"])
                    for role, pairs in result["provenance"].items():
                        writer.writerows((role, index, *pair) for index, pair in enumerate(pairs))
                counts = result["counts"]
                row = {**case, "case": name, **{key: counts[key] for key in ["n_train_pairs", "n_valid_pairs", "n_test_pairs", "train_unique_either_endpoints"]},
                       "valid_x1_in_train_x1": counts["valid"]["x1_in_train_x1"],
                       "valid_x1_in_train_x2": counts["valid"]["x1_in_train_x2"],
                       "valid_x1_in_train_either": counts["valid"]["x1_in_train_either_endpoint"],
                       "valid_x1_in_train_either_fraction": counts["valid"]["x1_in_train_either_endpoint_fraction"],
                       "test_x1_in_train_either": counts["test"]["x1_in_train_either_endpoint"],
                       "global_rng_unchanged_by_loading": result["global_rng_unchanged_by_loading"],
                       "full_group_membership_invariant_to_model_rng": perturbation_verified,
                       "instrumentation_preserved_loader_tensors": True,
                       "split_random_states": sorted({split["random_state"] for frame in result["loader_trace"] for split in frame["splits"]})}
                if row["split_random_states"] != [420] or not row["global_rng_unchanged_by_loading"]:
                    raise ValueError("Author random split protocol changed")
                rows.append(row)
                audit.write_json(options.output / "progress.json", {"completed": len(rows), "expected": len(selected), "last_case": name})
                print("COMPLETE " + name + " " + str(row["valid_x1_in_train_either"]) + "/" + str(row["n_valid_pairs"]), flush=True)
    finally:
        os.chdir(previous)
    report = {"status": "complete", "case_count": len(rows), "required_full_scope_count": 43,
              "scope": "11 full groups + 28 XJTU-six-batch/HUST n1..4 + MIT and TJU third-batch n1/n2",
              "evidence_path": str(options.evidence), "evidence_sha256": audit.digest(options.evidence),
              "recorded_repo_root": evidence["repo_root"], "mapped_repo_root": str(options.repo),
              "source_fingerprint": inventory["fingerprint"], "csv_files_hash_checked": len(checked),
              "directory_order_source": "Exact recorded cloud data_directory_order replay; no local enumeration order used",
              "original_run_tensor_hashes_available": False,
              "full_sample_counts_match_recorded_cloud_runs": True,
              "iterations": "One case per membership configuration; fixed sklearn random_state=420, ten model repeats do not alter membership",
              "interpretation": "PINN uses y1 and y2; MLP/CNN use y1 only. Direct validation label reuse is not independent-test battery leakage or a quantified performance bias.",
              "audit_script_sha256": audit.digest(Path(audit.__file__)), "suite_script_sha256": audit.digest(__file__),
              "environment": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
              "elapsed_seconds": time.monotonic() - started, "training_performed": False, "rows": rows}
    audit.write_json(options.output / "summary.json", report)
    with (options.output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()} for row in rows)
    audit.write_json(options.output / "artifact_sha256.json", {str(path.relative_to(options.output)): audit.digest(path)
                     for path in options.output.rglob("*") if path.is_file() and path.name != "artifact_sha256.json"})
    print(json.dumps({"status": "complete", "cases": len(rows), "csv_files_checked": len(checked), "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
