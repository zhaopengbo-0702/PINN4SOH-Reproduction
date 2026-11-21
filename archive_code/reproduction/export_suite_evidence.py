"""Export a compact, hash-audited, browser-downloadable full-suite evidence snapshot.

No model weights or full prediction arrays leave the cloud. Complete numerical
tables and loss histories are retained. Display curves use prespecified repeat1
and uniformly sampled retained-point indices, never test-error selection.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil

import build_suite_report as report


FORMAT = "pinn4soh-suite-evidence-export-v1"
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
LOG_VALUES = re.compile(r"(lr|data loss|PDE loss|physics loss|total loss|MSE|MAE|MAPE|RMSE):\s*(" + NUMBER + ")")
TABLES = ["task_audit", "per_repeat", "per_battery", "repeat_summaries", "per_battery_summaries",
          "published_comparisons", "comparison_effects", "figure_table_coverage", "claim_assessments"]


def encoded(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def stable_file(path, expected=None):
    path = Path(path)
    before = path.stat()
    sha = report.digest(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Artifact changed while exporting: " + str(path))
    if expected is not None and expected != sha:
        raise ValueError("Completed artifact hash mismatch: " + str(path))
    return {"path": str(path), "sha256": sha, "bytes": after.st_size}


class ObjectPool:
    """Content-addressed JSON structures; the resolver restores exact JSON values."""

    def __init__(self):
        self.nodes = {}

    def intern(self, value):
        if isinstance(value, dict):
            node = {key: self.intern(item) for key, item in value.items()}
        elif isinstance(value, list):
            node = [self.intern(item) for item in value]
        else:
            return value
        key = report.stable_hash(node)
        self.nodes.setdefault(key, node)
        return {"$ref": key}


def resolve(value, nodes):
    if isinstance(value, dict):
        if set(value) == {"$ref"}:
            return resolve(nodes[value["$ref"]], nodes)
        return {key: resolve(item, nodes) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve(item, nodes) for item in value]
    return value


class PartWriter:
    def __init__(self, output, max_bytes):
        self.output, self.max_bytes, self.files = Path(output), max_bytes, []

    def record(self, path, kind, rows=None):
        artifact = stable_file(path)
        artifact["path"] = path.name
        artifact["kind"] = kind
        if rows is not None:
            artifact["rows"] = rows
        self.files.append(artifact)
        return artifact

    def parts(self, name, values):
        records, current_bytes, number = [], 2, 0

        def flush():
            nonlocal records, current_bytes, number
            if not records:
                return
            number += 1
            path = self.output / (name + ".part" + str(number).zfill(3) + ".json")
            path.write_bytes(b"[" + b",".join(records) + b"]")
            self.record(path, name, len(records))
            records, current_bytes = [], 2

        for value in values:
            data = encoded(value)
            if len(data) + 2 > self.max_bytes:
                raise ValueError("One evidence record exceeds the JSON size cap: " + name)
            if records and current_bytes + len(data) + 1 > self.max_bytes:
                flush()
            records.append(data)
            current_bytes += len(data) + 1
        flush()


def log_evidence(folder, manifest, verification):
    path = folder / "logging.txt"
    artifact = stable_file(path, manifest.get("artifact_sha256", {}).get("logging.txt"))
    text = path.read_text(encoding="utf-8-sig")
    train, valid, tests = [], [], []
    train_columns = ["epoch", "lr", "data loss", "PDE loss", "physics loss", "total loss"]
    current_epoch = None
    unknown_relevant_lines = []
    for line in text.splitlines():
        tag = re.search(r"\[(Train|Valid|Test)\]", line)
        if tag is None:
            continue
        if re.search(r"(?:loss|MSE|lr):\s*[+-]?(?:nan|inf)\b", line, re.IGNORECASE):
            raise ValueError("Nonfinite training/validation log value: " + str(path))
        epoch = re.search(r"epoch:\s*(\d+)", line)
        values = {name: float(value) for name, value in LOG_VALUES.findall(line)}
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError("Nonfinite training log value: " + str(path))
        if tag[1] == "Train" and epoch:
            current_epoch = int(epoch[1])
            train.append([current_epoch] + [values.get(name) for name in train_columns[1:]])
        elif tag[1] == "Valid" and epoch and ("MSE" in values or "data loss" in values):
            label = "MSE" if "MSE" in values else "data loss"
            valid.append([int(epoch[1]), values[label], label])
        elif tag[1] == "Test" and values:
            tests.append([current_epoch] + [values.get(name) for name in ["MSE", "MAE", "MAPE", "RMSE"]])
        else:
            unknown_relevant_lines.append(line)
    if not train or not valid:
        raise ValueError("Complete task lacks parseable training/validation history: " + str(path))
    if len({row[0] for row in valid}) != len(valid) or len({row[0] for row in train}) != len(train):
        raise ValueError("Repeated epochs in immutable task log: " + str(path))
    minimum = min(valid, key=lambda row: row[1])
    recorded_epoch = manifest.get("best_epoch", verification.get("best_epoch"))
    if recorded_epoch is None:
        if any(row[2] == "data loss" for row in valid):
            raise ValueError("Rounded baseline log cannot uniquely establish missing best epoch")
        best_epoch, status = minimum[0], "derived_from_full_precision_PINN_validation_history"
    else:
        recorded_epoch = int(recorded_epoch)
        chosen = [row for row in valid if row[0] == recorded_epoch]
        if not chosen:
            raise ValueError("Recorded best epoch absent from validation log")
        tolerance = 1e-6 if chosen[0][2] == "data loss" else 1e-12
        if chosen[0][1] > minimum[1] + tolerance:
            raise ValueError("Best epoch disagrees with validation selection")
        best_epoch, status = recorded_epoch, "recorded_best_epoch_consistent_with_logged_precision"
    return {"file": artifact, "train_columns": train_columns, "train": train,
            "valid_columns": ["epoch", "value", "logged_metric_label"], "valid": valid,
            "author_test_columns": ["preceding_train_epoch", "MSE", "MAE", "MAPE", "RMSE"],
            "author_test": tests, "author_test_warning": "Historical author test MAPE can use reversed arguments; final independent metrics supersede it",
            "unknown_relevant_lines": unknown_relevant_lines, "last_train_epoch": train[-1][0],
            "best_epoch": best_epoch, "best_epoch_evidence": status,
            "minimum_logged_validation": minimum, "validation_points": len(valid)}


def task_snapshot(folder, pool):
    folder = Path(folder)
    manifest = report.read_json(folder / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("Included task is no longer complete: " + str(folder))
    artifacts = {name: stable_file(folder / name, sha)
                 for name, sha in manifest.get("artifact_sha256", {}).items()}
    for name in ["model.pth", "true_label.npy", "pred_label.npy", "config.json", "splits.json"]:
        if name not in artifacts:
            raise ValueError("Required complete artifact hash absent: " + name)
    documents = {}
    values = {}
    for name in ["manifest.json", "config.json", "splits.json", "metrics.json", "parameters.json",
                 "verification_checkpoint.json", "physical_loss_diagnostic.json", "source_inventory.json", "data_identity.json"]:
        path = folder / name
        if path.is_file():
            value = report.read_json(path)
            values[name] = value
            documents[name] = {"file": artifacts.get(name) or stable_file(path), "value": pool.intern(value),
                               "canonical_value_sha256": report.stable_hash(value)}
    verification = manifest.get("checkpoint_verification") or values.get("verification_checkpoint.json", {})
    if verification.get("passed") is not True:
        raise ValueError("Checkpoint reload verification did not pass")
    history = log_evidence(folder, manifest, verification)
    artifacts["logging.txt"] = history["file"]
    task = manifest.get("effective_task") or values["config.json"].get("task") or manifest["task"]
    snapshot = {"task_id": task["task_id"], "folder": str(folder), "status": "complete",
                "task": pool.intern(task), "documents": documents, "artifacts": artifacts,
                "best_epoch": history["best_epoch"], "best_epoch_evidence": history["best_epoch_evidence"],
                "epochs_completed": manifest.get("epochs_completed"),
                "checkpoint_verification": pool.intern(verification),
                "models_and_full_arrays_retained_on_cloud": True}
    return snapshot, {"task_id": task["task_id"], **history}, task, values["splits.json"]


def selected_curves(folder, task, splits, max_main=256, max_other=128):
    import numpy as np

    if int(task["repeat"]) != 1:
        return [], []
    family = "regular" if task["family"] == "full" else task["family"]
    if family not in {"regular", "small_sample", "transfer"}:
        return [], []
    if task.get("model", "PINN") == "PINN" and task.get("loss_variant") not in {
            "published_label_monotonicity", "paper_prediction_monotonicity"}:
        return [], []
    test = next(item for item in splits if item["role"] == "test")
    predictions, truths = [], []
    for evaluation, prefix in [("final", "")] + ([("source_only", "source_only_")] if family == "transfer" else []):
        truth = np.load(folder / (prefix + "true_label.npy"), allow_pickle=False).astype(np.float64).reshape(-1)
        prediction = np.load(folder / (prefix + "pred_label.npy"), allow_pickle=False).astype(np.float64).reshape(-1)
        if truth.shape != prediction.shape or sum(item["n_pairs"] for item in test["batteries"]) != len(truth):
            raise ValueError("Curve arrays disagree with exact battery boundaries")
        truth_sha = report.digest(folder / (prefix + "true_label.npy"))
        prediction_sha = report.digest(folder / (prefix + "pred_label.npy"))
        start = 0
        for battery in test["batteries"]:
            count = battery["n_pairs"]
            budget = max_main if family == "regular" else max_other
            indices = np.linspace(0, count - 1, min(budget, count), dtype=np.int64)
            value = {"battery": battery["path"], "n_full_retained_points": count,
                     "relative_retained_point_indices": indices.tolist(), "measured_SOH": truth[start + indices].tolist(),
                     "scope": "Uniform display sample of adjacent retained test-point sequence; original cycle IDs are not asserted"}
            truth_id = report.stable_hash(value)
            truths.append({"truth_id": truth_id, **value})
            predictions.append({"task_id": task["task_id"], "family": family,
                                "evaluation": evaluation, "truth_id": truth_id,
                                "predicted_SOH": prediction[start + indices].tolist(),
                                "selection": "prespecified repeat1; no test-error selection",
                                "full_truth_sha256": truth_sha,
                                "full_prediction_sha256": prediction_sha})
            start += count
    return predictions, truths


def scalar(value):
    if value == "":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    try:
        result = float(value)
        if math.isfinite(result):
            return result
    except ValueError:
        pass
    return value


def export(options):
    source, output = options.report.resolve(), options.output.resolve()
    if source == output or source in output.parents:
        raise ValueError("Export output must be separate from the source report directory")
    output.mkdir(parents=True, exist_ok=True)
    audit = report.read_json(source / "suite_report.json")
    if audit.get("format") != report.FORMAT or audit.get("metadata_only"):
        raise ValueError("Export requires an independently recomputed suite report")
    writer = PartWriter(output, int(options.max_json_mb * 1024 * 1024))
    pool = ObjectPool()
    snapshots, histories, curves, truths = [], [], [], {}
    for status in audit["tasks"]:
        if not status.get("included"):
            continue
        folder = Path(status["folder"])
        for name, key in [("config.json", "config_sha256"), ("splits.json", "splits_file_sha256"),
                          ("manifest.json", "manifest_sha256"), ("metrics.json", "metrics_sha256")]:
            stable_file(folder / name, status[key])
        reaudit, _, _, _ = report.inspect_task(folder)
        if not reaudit["included"] or reaudit["comparison_group"] != status["comparison_group"]:
            raise ValueError("Task no longer matches the independently recomputed report")
        snapshot, history, task, splits = task_snapshot(folder, pool)
        if snapshot["task_id"] != status["task_id"]:
            raise ValueError("Task identity differs from the independent report")
        snapshots.append(snapshot)
        histories.append(history)
        current, reference = selected_curves(folder, task, splits, options.main_curve_points, options.other_curve_points)
        curves.extend(current)
        truths.update({item["truth_id"]: item for item in reference})
    writer.parts("tasks", snapshots)
    writer.parts("training_histories", histories)
    writer.parts("curve_truth", truths.values())
    writer.parts("curve_predictions", curves)
    schemas = {}
    keep_columns = {
        "per_repeat": ["task_id", "repeat", "seed", "comparison_group", "evaluation", "aggregation", "convention",
                       "n_test", "n_batteries", *report.METRICS, *report.DIAGNOSTICS, "n_adjacent_retained_points"],
        "per_battery": ["task_id", "repeat", "seed", "comparison_group", "evaluation", "convention", "battery",
                        "start", "stop_exclusive", "n_test", *report.METRICS, *report.DIAGNOSTICS,
                        "n_adjacent_retained_points", "diagnostic_scope"]}
    for name in TABLES:
        path = source / (name + ".csv")
        before = stable_file(path)
        copied = output / path.name
        shutil.copy2(path, copied)
        copied_record = writer.record(copied, "csv_" + name)
        stable_file(path, before["sha256"])
        if copied_record["sha256"] != before["sha256"]:
            raise ValueError("Source CSV changed while copying")
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = [key for key in keep_columns.get(name, reader.fieldnames or []) if key in (reader.fieldnames or [])]
            values = [[scalar(row.get(key, "")) for key in columns] for row in reader]
        schemas[name] = columns
        writer.parts("table_" + name, values)
    writer.parts("json_nodes", ([key, value] for key, value in sorted(pool.nodes.items())))
    summary = {"format": FORMAT, "generated_at": datetime.now(timezone.utc).isoformat(),
               "source_report": stable_file(source / "suite_report.json"),
               "source_report_inputs": audit["inputs"], "policy": audit["policy"],
               "table_schemas": schemas,
               "counts": {"tasks_audited_complete": len(snapshots), "task_states": audit["task_status_counts"],
                          "curve_prediction_battery_series": len(curves), "unique_curve_truth_series": len(truths)},
               "prespecified_curve_policy": {"repeat": 1, "selection_uses_test_error": False,
                   "regular_groups_required": report.REGULAR_GROUPS,
                   "regular_methods_required": ["published_label_monotonicity PINN", "paper_prediction_monotonicity PINN", "MLP", "CNN"],
                   "regular_complete_task_count_when_available": 44,
                   "main_max_display_points_per_battery": options.main_curve_points,
                   "small_transfer_max_display_points_per_battery": options.other_curve_points,
                   "indices": "uniform retained-point indices including both endpoints; curves may be display-sampled",
                   "metric_arrays": "All metrics and monotonicity diagnostics use full saved arrays, not the displayed sample"},
               "additional_diagnostic_policy": "Prediction/truth rise and opposite-increment signs concern adjacent retained points only; extra diagnostics do not replace paper MAE/MAPE/RMSE",
               "content_addressed_documents": {"reference_form": {"$ref": "node ID"},
                   "node_files": "json_nodes.part*.json contain [node ID, JSON structure] entries",
                   "resolution": "Recursively replace each sole-key $ref object using the node dictionary; exact parsed JSON values are restored",
                   "provenance": "Each document retains original file SHA256/bytes/path and canonical parsed-value hash"},
               "files": writer.files, "max_json_file_bytes": writer.max_bytes,
               "cloud_retained_artifacts": "Complete original arrays, full text logs and model checkpoints remain in task folders with hashes in task records"}
    data = encoded(summary)
    if len(data) > writer.max_bytes:
        raise ValueError("Export index exceeds configured JSON cap")
    (output / "suite_evidence_index.json").write_bytes(data)
    readme = ["# PINN4SOH suite evidence export", "", "Start with suite_evidence_index.json.",
              "Every listed part is ordinary UTF-8 JSON or CSV with its SHA256 and byte count in the index.",
              "The task parts retain original manifest/config/splits/metrics and checkpoint/array/log hashes.",
              "JSON nodes deduplicate shared objects; export_suite_evidence.resolve restores original parsed JSON values.",
              "Training histories retain every logged train/validation/test point and the audited validation-selected best epoch.",
              "Baseline validation is printed to six decimals, so ties are interpreted at printed precision.",
              "Curve tasks are selected by repeat1 before observing test error; every available group and method is included.",
              "Curves may be uniformly sampled for display; exact raw arrays and model checkpoints remain on the cloud.",
              "All numerical metrics and additional adjacent-retained-point diagnostics use full arrays.",
              "A missing task or curve is pending, never silently replaced with a better-performing repeat.", ""]
    (output / "README.md").write_text("\n".join(readme), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-json-mb", type=float, default=18)
    parser.add_argument("--main-curve-points", type=int, default=256)
    parser.add_argument("--other-curve-points", type=int, default=128)
    options = parser.parse_args()
    if options.max_json_mb <= 0 or min(options.main_curve_points, options.other_curve_points) < 2:
        parser.error("JSON size cap must be positive and curve point budgets must be >=2")
    result = export(options)
    print(json.dumps({"output": str(options.output.resolve()), "counts": result["counts"],
                      "export_files": len(result["files"]), "maximum_part_bytes": max(
                          (item["bytes"] for item in result["files"] if item["path"].endswith(".json")), default=0)}))


if __name__ == "__main__":
    main()
