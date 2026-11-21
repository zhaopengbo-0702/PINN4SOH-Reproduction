"""Read-only local audit of the original current-repository MIT/HUST runs.

Does not import the training code, unpickle checkpoints, or write into inputs.
"""
import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metrics(y, p):
    # Double precision arithmetic is intentional; saved float32 arrays remain unchanged.
    y, p = np.asarray(y, dtype=np.float64).ravel(), np.asarray(p, dtype=np.float64).ravel()
    if y.shape != p.shape or not y.size or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Invalid arrays")
    e = p - y
    mse = float(np.mean(e * e))
    return {"MAE": float(np.mean(np.abs(e))),
            "MAPE": float(np.mean(np.abs(e) / np.maximum(np.abs(y), np.finfo(np.float64).eps))),
            "MSE": mse, "RMSE": float(np.sqrt(mse))}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cleaned_labels(path, nominal):
    # Independent translation of DF.read_one_csv/delete_3_sigma/load_one_battery.
    frame = pd.read_csv(path)
    frame.insert(frame.shape[1] - 1, "cycle index", np.arange(frame.shape[0]))
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    keep = np.ones(len(frame), dtype=bool)
    for name in frame.columns:
        values = frame[name]
        keep &= ~((values < values.mean() - 3 * values.std()) |
                  (values > values.mean() + 3 * values.std())).to_numpy()
    frame = frame.loc[keep].reset_index(drop=True)
    return (frame.iloc[:-1, -1].to_numpy() / nominal).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--recovery-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new audit output directory")
    for input_path in [args.results, args.repo, args.evidence, args.recovery_inventory]:
        if args.output.resolve() == input_path.resolve() or input_path.resolve() in args.output.resolve().parents:
            raise ValueError("Output must not be inside any input")
    manifest = read(args.results / "manifest.json")
    inventory = read(args.results / "source_inventory.json")
    evidence = read(args.evidence)
    recovery = {r["path"]: r for r in read(args.recovery_inventory)["records"]}
    source_checks, recovery_checks, original_checks = [], [], []
    for name, expected in inventory["sha256"].items():
        actual = sha(args.repo / name)
        if actual != expected:
            raise ValueError("Source mismatch: " + name)
        source_checks.append({"path": name, "sha256": actual})
    fingerprint = hashlib.sha256(json.dumps(inventory["sha256"], sort_keys=True).encode()).hexdigest()
    if fingerprint != inventory["fingerprint"]:
        raise ValueError("Inventory fingerprint mismatch")
    for name, expected in evidence["sha256"].items():
        path = args.results / name
        if path.is_file():
            actual = sha(path)
            if actual != expected:
                raise ValueError("Original evidence metadata mismatch: " + name)
            original_checks.append({"path": name, "sha256": actual})
    loss = (args.repo / "Model/Model.py").read_text(encoding="utf-8")
    if "self.relu(torch.mul(u2-u1,y1-y2)).sum()" not in loss:
        raise ValueError("Current-repository loss expression not present")
    rows, cells, per_run, saved_differences = [], [], [], []
    all_configs = {}
    for dataset, alpha, beta, lr_f, seed_start in [("MIT", .5, .01, .001, 20271109),
                                                  ("HUST", .6, .1, .0005, 20281109)]:
        expected_keys = [f"{dataset}/all/Experiment{i:02d}" for i in range(1, 11)]
        selected = [k for k, v in manifest["runs"].items() if v["dataset"] == dataset]
        if set(selected) != set(expected_keys):
            raise ValueError("Incomplete current-default matrix: " + dataset)
        for repeat, key in enumerate(expected_keys, 1):
            run = manifest["runs"][key]
            if run["status"] != "complete":
                raise ValueError("Run incomplete: " + key)
            folder = args.results / run["relative_folder"]
            for file in sorted(folder.iterdir()):
                if not file.is_file():
                    continue
                relative = args.results.name + "/" + file.relative_to(args.results).as_posix()
                rec = recovery[relative]
                actual = sha(file)
                if actual != rec["sha256"] or file.stat().st_size != rec["bytes"]:
                    raise ValueError("Recovered artifact changed: " + relative)
                recovery_checks.append({"path": relative, "sha256": actual, "bytes": file.stat().st_size})
            for name, record in evidence["artifacts"][key].items():
                file = folder / name
                if sha(file) != record["sha256"] or file.stat().st_size != record["bytes"]:
                    raise ValueError("Original completed-run artifact changed: " + key + "/" + name)
                original_checks.append({"path": key + "/" + name, "sha256": record["sha256"]})
            config = read(folder / "config.json")
            c_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
            if c_hash != run["config_hash"] or config["source_fingerprint"] != fingerprint:
                raise ValueError("Config binding mismatch: " + key)
            params = config["parameters"]
            required = {"alpha": alpha, "beta": beta, "epochs": 200, "early_stop": 20,
                        "warmup_epochs": 30, "batch_size": 512, "warmup_lr": .002,
                        "lr": .01, "final_lr": .0002, "lr_F": lr_f,
                        "F_layers_num": 3, "F_hidden_dim": 60,
                        "normalization_method": "min-max"}
            if any(params[k] != v for k, v in required.items()):
                raise ValueError("Not current default: " + key)
            if config["smoke"] or config["seed"] != seed_start + repeat - 1 or config["seed"] != run["seed"]:
                raise ValueError("Smoke/seed mismatch: " + key)
            all_configs[key] = config
            log = (folder / "logging.txt").read_text(encoding="utf-8")
            epochs = [int(e) for e in re.findall(r"\[Train\] epoch:(\d+)", log)]
            if epochs != list(range(1, run["epochs_completed"] + 1)) or epochs[-1] > 200:
                raise ValueError("Epoch/log mismatch: " + key)
            y = np.load(folder / "true_label.npy", allow_pickle=False).ravel()
            p = np.load(folder / "pred_label.npy", allow_pickle=False).ravel()
            pooled = metrics(y, p)
            split = next(s for s in read(folder / "splits.json") if s["role"] == "test")
            if split["paths"] != [r["path"] for r in split["batteries"]]:
                raise ValueError("Split path/order mismatch")
            if len(y) != run["sample_counts"]["test"] or sum(b["n_pairs"] for b in split["batteries"]) != len(y):
                raise ValueError("Split/array boundary mismatch")
            start, local_cells, rebuilt_labels = 0, [], []
            for battery in split["batteries"]:
                end = start + battery["n_pairs"]
                rebuilt = cleaned_labels(args.repo / battery["path"], split["nominal_capacity"])
                if len(rebuilt) != battery["n_pairs"] or not np.array_equal(y[start:end], rebuilt):
                    raise ValueError("Independent CSV labels do not match: " + battery["path"])
                rebuilt_labels.append(rebuilt)
                cell = {"dataset": dataset, "repeat": repeat, "seed": config["seed"],
                        "CSV": battery["path"], "n_test": end - start,
                        **metrics(y[start:end], p[start:end])}
                local_cells.append(cell)
                cells.append(cell)
                start = end
            macro = {m: float(np.mean([c[m] for c in local_cells])) for m in pooled}
            for convention, values in [("pooled", pooled), ("CSV_macro", macro)]:
                rows.append({"dataset": dataset, "repeat": repeat, "seed": config["seed"],
                             "aggregation": convention, "n_test": len(y), "n_test_CSV": len(local_cells),
                             "epochs_completed": epochs[-1], **values})
            saved = read(folder / "metrics.json")
            for m in pooled:
                for label, original in [("manifest", run["metrics"][m]), ("metrics.json", saved[m])]:
                    difference = pooled[m] - original
                    if not np.isclose(pooled[m], original, rtol=3e-6, atol=1e-12):
                        raise ValueError("Saved float32 metrics mismatch: " + key + "/" + m)
                    saved_differences.append({"run": key, "metric": m, "original": label,
                                              "float64_minus_saved_float32": difference})
            per_run.append({"run": key, "config_hash": c_hash, "seed": config["seed"],
                            "status": "complete_arrays_and_provenance_verified",
                            "n_test": len(y), "n_test_CSV": len(local_cells),
                            "epochs_completed": epochs[-1], "independent_labels_bitwise_equal": True,
                            "original_checkpoint_and_arrays_sha_verified": True})
    summary = []
    for dataset in ["MIT", "HUST"]:
        for convention in ["pooled", "CSV_macro"]:
            group = [r for r in rows if r["dataset"] == dataset and r["aggregation"] == convention]
            result = {"dataset": dataset, "aggregation": convention, "n_runs": len(group)}
            for metric in ["MAE", "MAPE", "MSE", "RMSE"]:
                values = np.array([r[metric] for r in group])
                result[metric + "_mean"] = float(values.mean())
                result[metric + "_std_ddof0"] = float(values.std(ddof=0))
                result[metric + "_std_ddof1"] = float(values.std(ddof=1))
            summary.append(result)
    report = {"format": "pinn4soh-current-defaults-local-independent-audit-v1",
              "audited_at_utc": datetime.now(timezone.utc).isoformat(), "status": "passed_20",
              "results": str(args.results.resolve()), "reference_commit": inventory["reference_upstream_commit"],
              "source_fingerprint": fingerprint, "source_files_verified": len(source_checks),
              "original_evidence_checks": len(original_checks), "recovery_artifact_checks": len(recovery_checks),
              "current_loss": "relu((u2-u1)*(y1-y2)).sum()", "runs": per_run,
              "environment": read(args.results / "environment.json"), "summaries": summary,
              "arithmetic": "independent float64 metrics from unchanged original float32 arrays",
              "saved_metric_max_absolute_difference": max(abs(r["float64_minus_saved_float32"]) for r in saved_differences),
              "limits": ["No retraining and no checkpoint pickle loading or new forward verification",
                         "Original GPU forward verification covers representative repeat1 only",
                         "CSV macro is not physical-battery-UID macro; MIT May5/May6 duplicate identity remains",
                         "Full-trajectory scaling and capacity-dependent 3-sigma cleaning retained",
                         "Original random pair validation endpoint overlap retained",
                         "Current defaults differ from Note3 publication weights and both main2120 PINN arms",
                         "Twenty current-default runs do not establish all repository versions were tested",
                         "No external WeChat result can be adjudicated without its actual configs and arrays"]}
    args.output.mkdir(parents=True)
    write_csv(args.output / "metrics_per_run_20.csv", rows)
    write_csv(args.output / "metrics_per_CSV.csv", cells)
    write_csv(args.output / "metrics_summary_20.csv", summary)
    write_csv(args.output / "saved_metric_precision_differences.csv", saved_differences)
    for name, value in [("LOCAL_CURRENT_DEFAULT_AUDIT.json", report),
                        ("INPUT_SHA256_CHECKS.json", {"source": source_checks,
                         "original_evidence": original_checks, "recovery": recovery_checks}),
                        ("CURRENT_DEFAULT_CONFIGS_20.json", all_configs)]:
        (args.output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "summaries": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
