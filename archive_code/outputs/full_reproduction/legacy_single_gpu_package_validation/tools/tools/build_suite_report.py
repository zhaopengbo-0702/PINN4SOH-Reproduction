"""Independently audit and summarize full-suite predictions without training.

The primary conclusion convention is exact-battery macro, measured-denominator
MAPE. Pooled metrics and predicted-denominator MAPE are always separate outputs.
Published Source Data remains a reference; it is never a reproduced experiment.
"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


FORMAT = "pinn4soh-independent-suite-report-v1"
XJTU = ["2C", "3C", "R2.5", "R3", "RW", "satellite"]
METRICS = ["MAE", "MAPE", "MSE", "RMSE", "MAE_pp", "MAPE_pct", "RMSE_pp"]
DIAGNOSTICS = ["prediction_rise_fraction", "prediction_total_positive_increment",
               "truth_rise_fraction", "truth_total_positive_increment", "increment_opposite_sign_fraction"]
REGULAR_GROUPS = ["XJTU/" + str(i) for i in range(1, 7)] + [
    "TJU/" + str(i) for i in range(1, 4)] + ["MIT", "HUST"]
SMALL_GROUPS = ["XJTU/" + str(i) for i in range(1, 7)] + ["HUST"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list))
                         else value for key, value in row.items()} for row in rows)


def paper_group(dataset, batch):
    if dataset == "XJTU":
        if batch in XJTU:
            return "XJTU/" + str(XJTU.index(batch) + 1)
        return "XJTU/" + str(int(batch) + 1)
    if dataset == "TJU":
        return "TJU/" + str(int(batch) + 1)
    return dataset


def clean_parameters(parameters):
    return {key: value for key, value in parameters.items() if key not in {
        "save_folder", "log_dir", "pretrain_model"}}


def canonical_splits(splits):
    return [{"role": item["role"], "paths": item["paths"],
             "nominal_capacity": item.get("nominal_capacity")}
            for item in splits]


def recompute(truth_path, prediction_path, splits):
    import numpy as np

    truth = np.load(truth_path, allow_pickle=False).astype(np.float64).reshape(-1)
    prediction = np.load(prediction_path, allow_pickle=False).astype(np.float64).reshape(-1)
    if not len(truth) or truth.shape != prediction.shape:
        raise ValueError("Empty or inconsistent prediction arrays")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Nonfinite prediction arrays")
    test = [item for item in splits if item["role"] == "test"]
    if len(test) != 1 or not test[0].get("batteries"):
        raise ValueError("One test split with exact battery boundaries is required")
    batteries = test[0]["batteries"]
    counts = [item["n_pairs"] for item in batteries]
    if any(not isinstance(value, int) or value <= 0 for value in counts) or sum(counts) != len(truth):
        raise ValueError("Battery boundary counts do not match saved predictions")
    if len({item["path"] for item in batteries}) != len(batteries):
        raise ValueError("Duplicate test battery paths")
    if test[0].get("paths") != [item["path"] for item in batteries]:
        raise ValueError("Test path order does not match boundary metadata")

    def metric(y, p, reversed_mape=False):
        error = y - p
        mae, mse = float(np.abs(error).mean()), float(np.square(error).mean())
        denominator = p if reversed_mape else y
        mape = float((np.abs(error) / np.maximum(np.abs(denominator), np.finfo(np.float64).eps)).mean())
        return {"MAE": mae, "MAPE": mape, "MSE": mse, "RMSE": math.sqrt(mse),
                "MAE_pp": mae * 100, "MAPE_pct": mape * 100, "RMSE_pp": math.sqrt(mse) * 100}

    summaries, rows = [], []
    for convention in ["standard", "reversed"]:
        reverse = convention == "reversed"
        start, current = 0, []
        for battery in batteries:
            end = start + battery["n_pairs"]
            measured_increment, predicted_increment = np.diff(truth[start:end]), np.diff(prediction[start:end])
            n_adjacent = len(measured_increment)
            diagnostic = {"n_adjacent_retained_points": n_adjacent,
                "prediction_rise_fraction": float((predicted_increment > 0).mean()) if n_adjacent else None,
                "prediction_total_positive_increment": float(np.maximum(predicted_increment, 0).sum()),
                "truth_rise_fraction": float((measured_increment > 0).mean()) if n_adjacent else None,
                "truth_total_positive_increment": float(np.maximum(measured_increment, 0).sum()),
                "increment_opposite_sign_fraction": float((measured_increment * predicted_increment < 0).mean()) if n_adjacent else None,
                "diagnostic_scope": "additional diagnostic; adjacent retained test points, not necessarily consecutive original cycles"}
            row = {"battery": battery["path"], "start": start, "stop_exclusive": end,
                   "n_test": end - start, "convention": convention,
                   **metric(truth[start:end], prediction[start:end], reverse), **diagnostic}
            current.append(row)
            rows.append(row)
            start = end
        adjacent = sum(row["n_adjacent_retained_points"] for row in current)
        pooled_diagnostic = {"n_adjacent_retained_points": adjacent}
        macro_diagnostic = {"n_adjacent_retained_points": adjacent}
        for name in DIAGNOSTICS:
            valid_rows = [row for row in current if row[name] is not None]
            macro_diagnostic[name] = statistics.mean(row[name] for row in valid_rows) if valid_rows else None
            if name.endswith("total_positive_increment"):
                pooled_diagnostic[name] = sum(row[name] for row in valid_rows)
            else:
                pooled_diagnostic[name] = sum(row[name] * row["n_adjacent_retained_points"] for row in valid_rows) / adjacent if adjacent else None
        summaries.append({"aggregation": "pooled", "convention": convention,
                          "n_test": len(truth), "n_batteries": len(batteries),
                          **metric(truth, prediction, reverse), **pooled_diagnostic})
        summaries.append({"aggregation": "battery_macro", "convention": convention,
                          "n_test": len(truth), "n_batteries": len(batteries),
                          **{name: statistics.mean(row[name] for row in current) for name in METRICS}, **macro_diagnostic})
    return summaries, rows


def read_metadata_metrics(folder, metrics, splits, prefix=""):
    """Support both runner schemas; unavailable pooled reversed MAPE stays null."""
    transfer = "metrics_battery_macro" in metrics
    pooled = metrics.get(prefix + "metrics") if transfer else metrics.get("pooled_standard")
    macro = metrics.get(prefix + "metrics_battery_macro") if transfer else metrics.get("battery_macro_standard")
    if prefix and transfer:
        macro = metrics.get("source_only_battery_macro")
    if pooled is None or macro is None:
        raise ValueError("Unrecognized or incomplete metrics schema")
    path = folder / (prefix + "metrics_per_battery.csv")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        stored_rows = list(csv.DictReader(stream))
    batteries = next(item["batteries"] for item in splits if item["role"] == "test")
    if len(stored_rows) != len(batteries):
        raise ValueError("Stored battery metric row count differs from split")
    rows, start = [], 0
    for stored, battery in zip(stored_rows, batteries):
        if stored["battery"] != battery["path"]:
            raise ValueError("Stored battery order differs from split")
        end = start + battery["n_pairs"]
        for convention in ["standard", "reversed"]:
            values = {}
            for name in METRICS:
                key = name if convention == "standard" else name + "_original"
                if convention == "reversed" and name == "MAPE" and "MAPE_author_reversed" in stored:
                    key = "MAPE_author_reversed"
                raw = stored.get(key)
                values[name] = float(raw) if raw not in {None, ""} else None
            if convention == "reversed":
                for name in ["MAE", "MSE", "RMSE", "MAE_pp", "RMSE_pp"]:
                    values[name] = float(stored[name])
                if values["MAPE"] is not None:
                    values["MAPE_pct"] = values["MAPE"] * 100
            rows.append({"battery": battery["path"], "start": start, "stop_exclusive": end,
                         "n_test": end - start, "convention": convention, **values})
        start = end
    if pooled.get("n_test", start) != start:
        raise ValueError("Stored pooled sample count differs from split")
    summaries = []
    for convention in ["standard", "reversed"]:
        for aggregation, stored in [("pooled", pooled), ("battery_macro", macro)]:
            values = {name: stored.get(name) for name in METRICS}
            if convention == "reversed":
                current = [row for row in rows if row["convention"] == "reversed"]
                values["MAPE"] = (statistics.mean(row["MAPE"] for row in current)
                                  if aggregation == "battery_macro" and all(row["MAPE"] is not None for row in current)
                                  else None)
                values["MAPE_pct"] = values["MAPE"] * 100 if values["MAPE"] is not None else None
            summaries.append({"aggregation": aggregation, "convention": convention,
                              "n_test": start, "n_batteries": len(batteries), **values})
    return summaries, rows


def discover(suites):
    found = []
    for suite in suites:
        suite = Path(suite).resolve()
        if not suite.is_dir():
            raise ValueError("Suite directory is missing: " + str(suite))
        root = suite / "tasks" if (suite / "tasks").is_dir() else suite
        for path in sorted(root.rglob("config.json")):
            relative = path.relative_to(root)
            if any(".previous." in part or part.startswith(".") for part in relative.parts):
                continue
            if (path.parent / "manifest.json").is_file():
                found.append(path.parent)
    return sorted(set(found))


def inspect_task(folder, expected_repeats=10, metadata_only=False):
    manifest = read_json(folder / "manifest.json")
    config = read_json(folder / "config.json")
    task = manifest.get("effective_task") or config.get("task") or manifest.get("task")
    if not isinstance(task, dict):
        raise ValueError("Explicit task metadata is required; legacy runs need an audited migration")
    task_id = task.get("task_id", manifest.get("task_id"))
    if not task_id:
        raise ValueError("Missing task identity")
    status = {"task_id": task_id, "folder": str(folder), "runner_status": manifest.get("status"),
              "included": False, "reason": None}
    if manifest.get("status") != "complete":
        status["reason"] = "Task is not complete"
        return status, [], [], None
    if config.get("smoke") or task.get("smoke") or task.get("epochs", 200) < 200 or task.get("adaptation_epochs", 200) < 200:
        status["reason"] = "Smoke/short run excluded from numerical conclusions"
        return status, [], [], None
    verification = manifest.get("checkpoint_verification")
    if verification is None and (folder / "verification_checkpoint.json").is_file():
        verification = read_json(folder / "verification_checkpoint.json")
    if not verification or verification.get("passed") is not True:
        raise ValueError("Complete task lacks passed checkpoint reload verification")
    hashes = manifest.get("artifact_sha256", {})
    for name in ["config.json", "splits.json", "true_label.npy", "pred_label.npy"]:
        if name not in hashes or digest(folder / name) != hashes[name]:
            raise ValueError("Missing or inconsistent completed artifact hash: " + name)
    splits = read_json(folder / "splits.json")
    test = next(item for item in splits if item["role"] == "test")
    train = next(item for item in splits if item["role"] in {"train", "train_validation"})
    if not train["paths"] or not test["paths"] or set(train["paths"]) & set(test["paths"]):
        raise ValueError("Training/test battery identities overlap or are empty")
    valid = next((item for item in splits if item["role"] == "valid"), None)
    if valid and set(valid["paths"]) & (set(train["paths"]) | set(test["paths"])):
        raise ValueError("Explicit validation batteries overlap training/test")
    parameters = config.get("parameters")
    if parameters is None:
        parameters = read_json(folder / "parameters.json")
    if parameters.get("epochs", task.get("epochs", 200)) < 200:
        status["reason"] = "Effective epochs identify a short verification run"
        return status, [], [], None
    family = task.get("family", manifest.get("family"))
    family = "regular" if family == "full" else family
    dataset = task.get("target", task.get("dataset", manifest.get("dataset")))
    batch = task.get("batch", manifest.get("batch"))
    model = task.get("model", manifest.get("model", "PINN"))
    group = paper_group(dataset, batch)
    environment = config.get("environment", {})
    environment = {key: value for key, value in environment.items() if key not in {
        "CUDA_VISIBLE_DEVICES", "recorded_at"}}
    base = {"family": family, "dataset": dataset, "batch": batch, "paper_group": group,
            "model": model, "sample_size": task.get("sample_size"),
            "protocol_id": task.get("protocol_id", "unknown"),
            "loss_variant": config.get("loss_variant", task.get("loss_variant", "author_current")),
            "validation_protocol": task.get("validation_protocol", "author_sample_split"),
            "source": task.get("source") if family == "transfer" else None,
            "source_batch": task.get("source_batch") if family == "transfer" else None,
            "source_mode": task.get("source_mode"), "transfer_design": task.get("transfer_design"),
            "source_fingerprint": config.get("source_fingerprint"),
            "environment_sha256": stable_hash(environment),
            "parameters_sha256": stable_hash(clean_parameters(parameters)),
            "splits_sha256": stable_hash(canonical_splits(splits)),
            "test_split_sha256": stable_hash(test["paths"]),
            "implementation_sha256": stable_hash(manifest.get("implementation_sha256", {
                "runner": config.get("runner_sha256"), "helper": config.get("helper_sha256")}))}
    if not base["source_fingerprint"] or base["protocol_id"] == "unknown":
        raise ValueError("Missing source fingerprint or protocol identity")
    comparison_id = stable_hash(base)
    common = {"task_id": task_id, "repeat": int(task["repeat"]), "seed": int(task["seed"]),
              "comparison_group": comparison_id, **base, "folder": str(folder),
              "metric_evidence": "stored_metadata" if metadata_only else "independent_saved_array_recomputation"}
    metrics = read_json(folder / "metrics.json")
    repeat_rows, battery_rows = [], []
    evaluations = [("final", "")]
    if family == "transfer":
        evaluations.append(("source_only", "source_only_"))
    for evaluation, prefix in evaluations:
        if prefix:
            for name in [prefix + "true_label.npy", prefix + "pred_label.npy"]:
                if name not in hashes or digest(folder / name) != hashes[name]:
                    raise ValueError("Source-only artifact hash missing/changed: " + name)
        if metadata_only:
            summaries, rows = read_metadata_metrics(folder, metrics, splits, prefix)
        else:
            summaries, rows = recompute(folder / (prefix + "true_label.npy"), folder / (prefix + "pred_label.npy"), splits)
            stored, _ = read_metadata_metrics(folder, metrics, splits, prefix)
            for new, old in zip(summaries, stored):
                for name in METRICS:
                    if old[name] is not None and not math.isclose(new[name], old[name], rel_tol=1e-5, abs_tol=1e-6):
                        raise ValueError("Saved metric disagrees with array recomputation: " + evaluation + "/" + name)
        repeat_rows.extend({**common, "evaluation": evaluation, **row} for row in summaries)
        battery_rows.extend({**common, "evaluation": evaluation, **row} for row in rows)
    status.update(included=True, reason="Audited", comparison_group=comparison_id,
                  config_sha256=digest(folder / "config.json"), splits_file_sha256=digest(folder / "splits.json"),
                  manifest_sha256=digest(folder / "manifest.json"), metrics_sha256=digest(folder / "metrics.json"))
    proof = {"task_id": task_id, "expected_repeats": expected_repeats,
             "parameters": clean_parameters(parameters), "config": config,
             "splits": splits, "frozen_dynamical_F": manifest.get("frozen_dynamical_F"),
             "solution_u_updated": manifest.get("solution_u_updated"),
             "source_checkpoint_lineage": manifest.get("source_checkpoint_lineage"),
             "artifact_sha256": hashes}
    return status, repeat_rows, battery_rows, proof


def summarize_rows(rows, expected_repeats):
    groups = defaultdict(list)
    for row in rows:
        key = (row["comparison_group"], row["evaluation"], row["aggregation"], row["convention"])
        groups[key].append(row)
    summaries = []
    for key, current in sorted(groups.items()):
        first = current[0]
        identity = {name: first[name] for name in ["comparison_group", "family", "dataset", "batch",
            "paper_group", "model", "sample_size", "protocol_id", "loss_variant", "validation_protocol",
            "source", "source_batch", "source_mode", "transfer_design", "source_fingerprint",
            "environment_sha256", "parameters_sha256", "splits_sha256", "test_split_sha256",
            "implementation_sha256", "evaluation", "aggregation", "convention", "metric_evidence"]}
        repeat_ids = [row["repeat"] for row in current]
        seeds = [row["seed"] for row in current]
        valid = len(set(repeat_ids)) == len(current) and len(set(seeds)) == len(current)
        summary = {**identity, "n_repeats": len(current), "expected_repeats": expected_repeats,
                   "repeat_ids": sorted(repeat_ids), "seeds": sorted(seeds), "unique_repeats_and_seeds": valid,
                   "complete_repeat_set": valid and sorted(repeat_ids) == list(range(1, expected_repeats + 1))}
        for name in METRICS + DIAGNOSTICS:
            values = [row[name] for row in current if row.get(name) is not None]
            if any(not math.isfinite(value) for value in values):
                raise ValueError("Nonfinite summary metric")
            summary[name + "_n"] = len(values)
            summary[name + "_mean"] = statistics.mean(values) if values else None
            summary[name + "_sd_ddof0"] = statistics.pstdev(values) if values else None
            summary[name + "_sd_ddof1"] = statistics.stdev(values) if len(values) > 1 else None
        summaries.append(summary)
    return summaries


def summarize_batteries(rows, expected_repeats):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["comparison_group"], row["evaluation"], row["convention"], row["battery"])].append(row)
    summaries = []
    for key, current in sorted(grouped.items()):
        first = current[0]
        result = {name: first[name] for name in ["comparison_group", "family", "dataset", "paper_group",
            "model", "sample_size", "protocol_id", "loss_variant", "evaluation", "convention", "battery"]}
        ids, seeds = [row["repeat"] for row in current], [row["seed"] for row in current]
        result.update(n_repeats=len(current), expected_repeats=expected_repeats,
                      complete_repeat_set=sorted(ids) == list(range(1, expected_repeats + 1)) and len(set(seeds)) == len(current))
        for name in METRICS + DIAGNOSTICS:
            values = [row[name] for row in current if row.get(name) is not None]
            result[name + "_mean"] = statistics.mean(values) if values else None
            result[name + "_sd_ddof0"] = statistics.pstdev(values) if values else None
            result[name + "_sd_ddof1"] = statistics.stdev(values) if len(values) > 1 else None
        summaries.append(result)
    return summaries


def published_comparisons(summaries, protocol, table_s7):
    rows = []
    tables = [("Table 2", protocol.get("published_table_2", {}), "regular"),
              ("Table 3", protocol.get("published_table_3", {}), "small_sample"),
              ("Table S7", table_s7, "small_sample")]
    for summary in summaries:
        for label, table, family in tables:
            if summary["family"] != family or summary["evaluation"] != "final":
                continue
            group = summary["paper_group"]
            if family == "small_sample":
                group += "/n" + str(summary["sample_size"])
            values = table.get("rows", {}).get(group)
            if values is None:
                continue
            columns = table.get("column_order", table.get("columns", []))
            for metric in ["MAPE", "RMSE"]:
                column = summary["model"] + "_" + metric
                if column not in columns:
                    continue
                published, observed = values[columns.index(column)], summary[metric + "_mean"]
                rows.append({"table": label, "group": group, "model": summary["model"],
                             "metric": metric, "comparison_group": summary["comparison_group"],
                             "protocol_id": summary["protocol_id"], "loss_variant": summary["loss_variant"],
                             "aggregation": summary["aggregation"], "convention": summary["convention"],
                             "n_repeats": summary["n_repeats"], "complete_repeat_set": summary["complete_repeat_set"],
                             "published": published, "observed": observed,
                             "difference": observed - published if observed is not None else None,
                             "within_printed_4dp_half_unit": abs(observed - published) <= 0.00005 if observed is not None else None,
                             "numeric_match_is_not_protocol_equivalence": True})
        if summary["family"] == "small_sample" and summary["model"] == "PINN" and summary["sample_size"] in [1, 2]:
            table = protocol.get("published_table_4", {})
            domains = table.get("domain_order", [])
            if summary["paper_group"] in domains and summary["evaluation"] == "final":
                index = domains.index(summary["paper_group"])
                published = table["target_only_n" + str(summary["sample_size"])][index]
                observed = summary["RMSE_mean"]
                rows.append({"table": "Table 4", "group": summary["paper_group"], "model": "PINN",
                             "metric": "RMSE", "evaluation": "target_only_n" + str(summary["sample_size"]),
                             "comparison_group": summary["comparison_group"], "protocol_id": summary["protocol_id"],
                             "loss_variant": summary["loss_variant"], "aggregation": summary["aggregation"],
                             "convention": summary["convention"], "n_repeats": summary["n_repeats"],
                             "complete_repeat_set": summary["complete_repeat_set"], "published": published,
                             "observed": observed, "difference": observed - published if observed is not None else None,
                             "within_printed_4dp_half_unit": abs(observed - published) <= 0.00005 if observed is not None else None,
                             "numeric_match_is_not_protocol_equivalence": True})
        if summary["family"] != "transfer":
            continue
        table = protocol.get("published_table_4", {})
        domains = table.get("domain_order", [])
        source_group = paper_group(summary["source"], summary["source_batch"])
        if summary["paper_group"] not in domains or source_group not in domains:
            continue
        target_index, source_index = domains.index(summary["paper_group"]), domains.index(source_group)
        label = "fine_tuning" if summary["evaluation"] == "final" else "source_only"
        published = table[label][target_index][source_index]
        observed = summary["RMSE_mean"]
        rows.append({"table": "Table 4", "group": source_group + "->" + summary["paper_group"],
                     "model": "PINN", "metric": "RMSE", "evaluation": summary["evaluation"],
                     "comparison_group": summary["comparison_group"], "protocol_id": summary["protocol_id"],
                     "loss_variant": summary["loss_variant"], "aggregation": summary["aggregation"],
                     "convention": summary["convention"], "n_repeats": summary["n_repeats"],
                     "complete_repeat_set": summary["complete_repeat_set"], "published": published,
                     "observed": observed, "difference": observed - published if observed is not None else None,
                     "within_printed_4dp_half_unit": abs(observed - published) <= 0.00005 if observed is not None else None,
                     "numeric_match_is_not_protocol_equivalence": True})
    return rows


def describe_coverage(summaries, inventory):
    primary = [row for row in summaries if row["aggregation"] == "battery_macro" and
               row["convention"] == "standard" and row["complete_repeat_set"]]
    rows = []
    for item in inventory:
        relevant = [row for row in primary if row["family"] in item.get("families", [])]
        if item.get("sample_groups"):
            relevant = [row for row in relevant if row["paper_group"] in item["sample_groups"]]
        if item.get("evaluation"):
            relevant = [row for row in relevant if row["evaluation"] == item["evaluation"]]
        rows.append({**item, "complete_numeric_groups_available": len(relevant),
                     "status": "numeric_artifacts_available_review_required" if relevant else "pending_or_external_evidence_required",
                     "status_does_not_assert_figure_rendered_or_claim_supported": True})
    return rows


def comparison_effects(summaries, repeat_rows, proofs):
    """Describe prespecified comparisons; never select the best metric convention."""
    primary = [row for row in summaries if row["aggregation"] == "battery_macro" and
               row["convention"] == "standard" and row["complete_repeat_set"]]
    results = []
    rows_by_group = defaultdict(list)
    for row in repeat_rows:
        rows_by_group[row["comparison_group"]].append(row)
    proofs_by_task = {proof["task_id"]: proof for proof in proofs}
    proofs_by_group = {key: [proofs_by_task[task_id] for task_id in sorted({row["task_id"] for row in rows})
                            if task_id in proofs_by_task] for key, rows in rows_by_group.items()}
    cohort_fields = ["dataset", "paper_group", "protocol_id", "validation_protocol",
                     "source_fingerprint", "environment_sha256", "test_split_sha256"]

    def matching(left, right, fields):
        return all(left[name] == right[name] for name in fields)

    def compare(left, candidates, kind, metrics, extra=None):
        base = {"kind": kind, "protocol_id": left["protocol_id"], "loss_variant": left["loss_variant"],
                "paper_group": left["paper_group"], "sample_size": left["sample_size"],
                "source": left["source"], "source_mode": left["source_mode"],
                "left_comparison_group": left["comparison_group"],
                "primary_convention": "standard/battery_macro", **(extra or {})}
        if len(candidates) != 1:
            results.append({**base, "status": "missing_or_ambiguous_comparator", "comparator_count": len(candidates)})
            return
        right = candidates[0]
        if left["seeds"] != right["seeds"] or left["repeat_ids"] != right["repeat_ids"]:
            results.append({**base, "status": "unmatched_repeat_seed_sets"})
            return
        for metric in metrics:
            difference = left[metric + "_mean"] - right[metric + "_mean"]
            results.append({**base, "status": "descriptive_comparison_available",
                            "right_comparison_group": right["comparison_group"], "metric": metric,
                            "left_mean": left[metric + "_mean"], "right_mean": right[metric + "_mean"],
                            "left_minus_right": difference,
                            "direction": "lower" if difference < -1e-12 else "higher" if difference > 1e-12 else "tie",
                            "left_sd_ddof0": left[metric + "_sd_ddof0"], "right_sd_ddof0": right[metric + "_sd_ddof0"],
                            "left_sd_ddof1": left[metric + "_sd_ddof1"], "right_sd_ddof1": right[metric + "_sd_ddof1"],
                            "no_significance_or_equivalence_claim": True})

    for left in primary:
        if left["model"] != "PINN" or left["evaluation"] != "final":
            continue
        if left["family"] in {"regular", "small_sample"}:
            for model in ["MLP", "CNN"]:
                candidates = [right for right in primary if right["model"] == model and
                              right["family"] == left["family"] and right["sample_size"] == left["sample_size"] and
                              right["evaluation"] == "final" and matching(left, right, cohort_fields)]
                compare(left, candidates, "PINN_vs_" + model, ["MAE", "MAPE", "RMSE"],
                        {"shared_baseline": True, "baseline_has_no_PINN_physical_loss": True})
        if left["family"] != "transfer":
            continue
        candidates = [right for right in primary if right["comparison_group"] == left["comparison_group"]
                      and right["evaluation"] == "source_only"]
        current_proofs = proofs_by_group.get(left["comparison_group"], [])
        frozen_ok = bool(current_proofs) and all(proof.get("frozen_dynamical_F", {}).get("unchanged") is True
                        and proof["frozen_dynamical_F"].get("before_sha256") == proof["frozen_dynamical_F"].get("after_sha256")
                        and proof.get("solution_u_updated") is True for proof in current_proofs)
        paired = defaultdict(dict)
        for row in rows_by_group[left["comparison_group"]]:
            if row["aggregation"] == "battery_macro" and row["convention"] == "standard":
                paired[row["repeat"]][row["evaluation"]] = row["RMSE"]
        deltas = [values["final"] - values["source_only"] for values in paired.values()
                  if "final" in values and "source_only" in values]
        compare(left, candidates, "fine_tune_vs_source_only", ["RMSE"],
                {"frozen_G_and_updated_F_verified": frozen_ok, "paired_repeats": len(deltas),
                 "paired_repeats_improved": sum(value < 0 for value in deltas),
                 "paired_RMSE_delta_mean": statistics.mean(deltas) if deltas else None,
                 "paired_RMSE_delta_sd_ddof1": statistics.stdev(deltas) if len(deltas) > 1 else None})
        for size in [1, 2]:
            candidates = [right for right in primary if right["family"] == "small_sample" and
                          right["model"] == "PINN" and right["sample_size"] == size and right["evaluation"] == "final"
                          and right["loss_variant"] == left["loss_variant"] and matching(left, right, cohort_fields)]
            if len(candidates) == 1:
                right = candidates[0]
                right_proofs = {int(proof["config"]["task"]["repeat"]): proof
                                for proof in proofs_by_group.get(right["comparison_group"], [])}
                matched = True
                for proof in current_proofs:
                    repeat = int(proof["config"]["task"]["repeat"])
                    target = right_proofs.get(repeat)
                    if target is None:
                        matched = False
                        break
                    role = lambda entries: next(item["paths"] for item in entries if item["role"] in {"train", "train_validation"})
                    adaptation_paths, baseline_paths = role(proof["splits"]), role(target["splits"])
                    if adaptation_paths != baseline_paths[:1] or len(baseline_paths) != size:
                        matched = False
                        break
                if not matched:
                    results.append({"kind": "fine_tune_vs_target_only_n" + str(size),
                                    "left_comparison_group": left["comparison_group"],
                                    "status": "unmatched_target_training_battery_paths"})
                    continue
            compare(left, candidates, "fine_tune_vs_target_only_n" + str(size), ["RMSE"],
                    {"different_method_hyperparameters_remain_recorded": True})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="append", type=Path, required=True)
    parser.add_argument("--plan", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/full_reproduction/paper_protocol.json")
    parser.add_argument("--rules", type=Path, default=Path(__file__).with_name("suite_evidence_rules.json"))
    parser.add_argument("--expected-repeats", type=int, default=10)
    parser.add_argument("--metadata-only", action="store_true")
    options = parser.parse_args()
    if options.expected_repeats < 1:
        parser.error("expected-repeats must be positive")
    output = options.output.resolve()
    suites = [path.resolve() for path in options.suite]
    if any(output == suite or suite in output.parents and "tasks" in output.relative_to(suite).parts for suite in suites):
        parser.error("Report output must not replace the suite or live task directory")
    output.mkdir(parents=True, exist_ok=True)
    protocol, rules = read_json(options.protocol), read_json(options.rules)
    tasks, repeats, batteries, proofs = [], [], [], []
    for folder in discover(suites):
        try:
            status, current, battery, proof = inspect_task(folder, options.expected_repeats, options.metadata_only)
            tasks.append(status)
            repeats.extend(current)
            batteries.extend(battery)
            if proof:
                proofs.append(proof)
        except (ValueError, KeyError, OSError, StopIteration, TypeError) as error:
            tasks.append({"folder": str(folder), "included": False, "reason": str(error), "report_audit_error": True})
    seen = {item.get("task_id") for item in tasks}
    planned_ids = set()
    for path in options.plan:
        for task in read_json(path)["tasks"]:
            task_id = task["task_id"]
            if task_id in planned_ids:
                raise ValueError("Task occurs in more than one supplied plan: " + task_id)
            planned_ids.add(task_id)
            if task_id not in seen:
                tasks.append({"task_id": task_id, "included": False, "runner_status": "not_materialized",
                              "reason": "Planned task has no readable task config and manifest in supplied suite roots"})
    summaries = summarize_rows(repeats, options.expected_repeats)
    battery_summaries = summarize_batteries(batteries, options.expected_repeats)
    s7_path = options.protocol.parent / "source_data/paper_table_s7.json"
    s7 = read_json(s7_path) if s7_path.is_file() else {}
    comparisons = published_comparisons(summaries, protocol, s7)
    effects = comparison_effects(summaries, repeats, proofs)
    coverage = describe_coverage(summaries, rules["figure_table_inventory"])
    claims = [{**rule, "assessment": "pending_rule_evaluation",
               "published_statement": next((claim["statement"] for claim in protocol.get("claims", [])
                                            if claim["id"] == rule["id"]), None),
               "warning": "Training completion or a rounded numeric match alone does not establish this claim"}
              for rule in rules["claim_rules"]]
    report = {"format": FORMAT, "generated_at": datetime.now(timezone.utc).isoformat(),
              "inputs": {"suites": [str(path) for path in suites],
                         "plans": [{"path": str(path), "sha256": digest(path)} for path in options.plan],
                         "protocol": str(options.protocol),
                         "protocol_sha256": digest(options.protocol), "rules_sha256": digest(options.rules)},
              "policy": rules["policy"], "metadata_only": options.metadata_only,
              "task_count": len(tasks), "task_status_counts": dict(Counter("included" if item["included"] else
                  "audit_error" if item.get("report_audit_error") else "excluded" for item in tasks)),
              "repeat_summary_count": len(summaries), "tasks": tasks, "summaries": summaries,
              "published_comparisons": comparisons, "comparison_effects": effects,
              "coverage": coverage, "claim_assessments": claims}
    write_json(output / "suite_report.json", report)
    write_json(output / "task_evidence.json", proofs)
    for name, rows in [("task_audit", tasks), ("per_repeat", repeats), ("per_battery", batteries),
                       ("per_battery_summaries", battery_summaries),
                       ("repeat_summaries", summaries), ("published_comparisons", comparisons),
                       ("comparison_effects", effects),
                       ("figure_table_coverage", coverage), ("claim_assessments", claims)]:
        write_csv(output / (name + ".csv"), rows)
    counts = report["task_status_counts"]
    text = ["# PINN4SOH independent suite report", "", "Generated: " + report["generated_at"], "",
            "Included full tasks: " + str(counts.get("included", 0)),
            "Excluded incomplete/smoke tasks: " + str(counts.get("excluded", 0)),
            "Audit errors: " + str(counts.get("audit_error", 0)), "",
            "Primary convention: equal-weight exact-battery macro metrics; MAPE divides by measured SOH.",
            "Every report also exports pooled metrics and reversed MAPE as distinct sensitivity conventions.",
            "MAPE and RMSE are fractions; MAPE_pct is percent, MAE_pp/RMSE_pp are SOH percentage points.",
            "Repeat summaries identify population SD (ddof=0) and sample SD (ddof=1).", "",
            "Additional monotonicity diagnostics use within-battery adjacent retained points, never cross-battery boundaries.",
            "Rise fractions count strictly positive increments; opposite signs require measured_increment*predicted_increment < 0.",
            "Positive increment totals have SOH-fraction units; zeros are not counted as opposite signs.",
            "These diagnostics do not assert that retained neighbors were consecutive original cycles and do not replace paper metrics.", "",
            "A ten-repeat summary requires repeat IDs 1..10 and ten unique seeds in the same scientific configuration.",
            "Settings, loss, validation, source inventory, implementation and environments are separated in summaries.",
            "Rounded agreement is evaluated against 0.00005, the half-unit for four printed decimal places.",
            "This numerical tolerance is not a claim of statistical equivalence or successful protocol reproduction.", "",
            "Figure/table coverage records available numeric artifacts, not finished figures or supported conclusions.",
            "Explicit decisions and external evidence requirements are in claim_assessments.csv and suite_evidence_rules.json.",
            "Published Source Data is reference evidence only and is never counted as an independent training run.", ""]
    (output / "README.md").write_text("\n".join(text), encoding="utf-8")
    print(json.dumps({"output": str(output), "tasks": counts, "summary_groups": len(summaries)}))


if __name__ == "__main__":
    main()
