"""Array audit for selected extensions, composing immutable repeat1 + new2..10.

No 40-task restriction and no file copying or task-identity rewriting. Uses the
frozen array auditor and existing pair-matching predicates as read-only helpers.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

import numpy as np

import build_suite_report as audit
from compare_legacy_control import load_task, match_reasons, write_csv, direction, METRICS
from prepare_legacy_extension import (read, digest, require, verify_reused,
    verify_completed, same_legacy_cohort, verify_modern_frozen, verify_composite)


def validate_lineage(item, document, source_item, source_document, record, modern=False):
    """Prove the checkpoint came from the exact selected same-repeat source."""
    task = item["modern_task" if modern else "legacy_task"]
    source_task = source_item["modern_task" if modern else "legacy_task"]
    lineage = document["proof"].get("source_checkpoint_lineage") or {}
    require(lineage == document["proof"]["config"].get("source_checkpoint_lineage"), "Config/manifest lineage differs")
    require(source_task["repeat"] == task["repeat"] and source_task["loss_variant"] == task["loss_variant"] and
            source_task["dataset"] == task["source"], "Transfer source repeat/loss/dataset differs")
    require(task["source_run"].split("/tasks/", 1)[1] == source_task["task_id"], "Alternate source task selected")
    require(lineage.get("source_run") == task["source_run"] and lineage.get("path") == task["source_checkpoint"], "Source run/checkpoint path changed")
    require(lineage.get("training_seed") == source_task["seed"], "Source training seed differs")
    require(lineage.get("source_dataset") == task["source"] and lineage.get("source_batch") == task["source_batch"], "Source dataset/batch lineage differs")
    for field, name in [("sha256", "model.pth"), ("source_config_sha256", "config.json"),
                        ("source_splits_sha256", "splits.json"), ("source_manifest_sha256", "manifest.json")]:
        checksum = digest(source_document["folder"] / name)
        require(lineage.get(field) == checksum and record.get("input_sha256", {}).get(task["source_run"] + "/" + name) == checksum,
                "Source lineage/completion input changed: " + name)
    require(lineage.get("source_paths") == source_document["proof"]["splits"], "Source training split identity differs")
    require(source_document["proof"]["config"].get("checkpoint_snapshot_deepcopy") is True, "Source best-validation snapshot unverified")
    require(lineage.get("source_best_validation_mse") == source_document["optimization"]["best_validation_MSE"], "Source validation selection proof changed")
    return lineage


def distribution(values):
    return {"n": len(values), "mean": statistics.mean(values),
            "sd_ddof0": statistics.pstdev(values), "sd_ddof1": statistics.stdev(values) if len(values) > 1 else None}


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["logical_group"], row["evaluation"], row["metric"])].append(row)
    results = []
    for key, values in sorted(groups.items()):
        repeats = [row["repeat"] for row in values]
        complete = sorted(repeats) == list(range(1, 11)) and len(set(row["seed"] for row in values)) == 10 and all(row["comparable"] for row in values)
        result = {"logical_group": key[0], "evaluation": key[1], "metric": key[2],
                  "repeat_ids": sorted(repeats), "complete_ten_repeat_set": complete,
                  "comparable_repeats": sum(row["comparable"] for row in values)}
        if complete:
            result.update(legacy=distribution([row["legacy"] for row in values]),
                          modern=distribution([row["modern"] for row in values]),
                          paired_legacy_minus_modern=distribution([row["legacy"] - row["modern"] for row in values]))
        results.append(result)
    return results


def compare(extension, legacy_control_suite, legacy_extension_suite, current_suites, output):
    extension, old, new, output = [Path(path).resolve() for path in
        [extension, legacy_control_suite, legacy_extension_suite, output]]
    current_suites = [Path(path).resolve() for path in current_suites]
    require(not output.exists() and output.parent.is_dir(), "Comparison output must be fresh")
    design, manifest = read(extension / "extension_design.json"), read(extension / "extension_analysis_manifest.json")
    require(digest(extension / "extension_plan.json") == design["plan_sha256"] and
            digest(extension / "extension_analysis_manifest.json") == design["analysis_manifest_sha256"], "Extension support hashes differ")
    require(verify_reused(manifest["tasks"], old) == manifest["reuse_evidence"], "Reused repeat1 changed")
    state = read(new / "manifest.json")
    new_plan = read(extension / "extension_plan.json")
    verify_composite(new_plan, manifest, design)
    require(state["frozen"]["plan_sha256"] == audit.stable_hash(new_plan), "New suite plan mismatch")
    same_legacy_cohort(manifest["reuse_evidence"]["control_frozen"], state["frozen"])
    require(set(state["tasks"]) == {task["task_id"] for task in new_plan["tasks"]}, "New suite task set differs from extension plan")
    for source in [old, new, *current_suites]:
        require(output != source and source not in output.parents, "Comparison must be outside suites")
    modern_lookup, modern_states = {}, {}
    needed = {item["modern_task_id"] for item in manifest["tasks"]}
    for suite in current_suites:
        modern_states[suite] = read(suite / "manifest.json")
        verify_modern_frozen(modern_states[suite]["frozen"])
        for folder in audit.discover([suite]):
            task_id = read(folder / "config.json").get("task", {}).get("task_id")
            if task_id in needed:
                require(task_id not in modern_lookup, "Duplicate modern task")
                modern_lookup[task_id] = (suite, folder)
    statuses, deltas, batteries, proof_rows, effects = [], [], [], [], []
    method_values = {}
    for item in manifest["tasks"]:
        identity = {key: item[key] for key in ["logical_group", "repeat", "seed", "storage", "selection_role", "legacy_task_id", "modern_task_id"]}
        legacy_folder = (old if item["repeat"] == 1 else new) / "tasks" / item["legacy_task_id"]
        modern_entry = modern_lookup.get(item["modern_task_id"])
        if modern_entry is None or not (legacy_folder / "manifest.json").is_file():
            statuses.append(dict(identity, status="pending"))
            continue
        modern_suite, modern_folder = modern_entry
        if read((old if item["repeat"] == 1 else new) / "manifest.json")["tasks"][item["legacy_task_id"]].get("status") != "complete" or modern_states[modern_suite]["tasks"][item["modern_task_id"]].get("status") != "complete":
            statuses.append(dict(identity, status="pending"))
            continue
        try:
            modern_suite, modern_folder = modern_entry
            legacy_suite = old if item["repeat"] == 1 else new
            _, record, _, _ = verify_completed(legacy_suite, item["legacy_task"])
            _, modern_record, _, _ = verify_completed(modern_suite, item["modern_task"], modern_states[modern_suite])
            legacy, modern = load_task(legacy_folder), load_task(modern_folder)
            left, right = legacy["repeats"][0], modern["repeats"][0]
            reasons = match_reasons(left, right, legacy["proof"], modern["proof"])
            if left.get("environment_sha256") == right.get("environment_sha256"):
                reasons.append("software_environment_not_distinct")
            require(legacy["proof"]["config"]["task"].get("original_task_id") == item["modern_task_id"], "Original modern identity mismatch")
            if item["legacy_task"]["family"] == "transfer":
                # Target and source seed offsets differ by dataset; use exact source task,
                # never require source training seed to equal the target seed.
                source_id = item["legacy_task"]["source_run"].split("/tasks/", 1)[1]
                source_item = next(row for row in manifest["tasks"] if row["legacy_task_id"] == source_id)
                source_folder, _, _, _ = verify_completed(legacy_suite, source_item["legacy_task"])
                modern_source_suite, modern_source_folder = modern_lookup[source_item["modern_task_id"]]
                verify_completed(modern_source_suite, source_item["modern_task"], modern_states[modern_source_suite])
                legacy_source, modern_source = load_task(source_folder), load_task(modern_source_folder)
                lineage = validate_lineage(item, legacy, source_item, legacy_source, record)
                modern_lineage = validate_lineage(item, modern, source_item, modern_source, modern_record, modern=True)
                require(lineage["training_seed"] == modern_lineage["training_seed"], "Modern/legacy source seeds differ")
                reasons.extend("source_" + reason for reason in match_reasons(legacy_source["repeats"][0], modern_source["repeats"][0],
                    legacy_source["proof"], modern_source["proof"]))
                if not np.array_equal(np.load(source_folder / "true_label.npy", allow_pickle=False),
                                      np.load(modern_source_folder / "true_label.npy", allow_pickle=False)):
                    reasons.append("source_different_truth")
            staged_deltas, staged_batteries, staged_metrics = [], [], {}
            for lrow in legacy["repeats"]:
                rrow = next(row for row in modern["repeats"] if row["evaluation"] == lrow["evaluation"])
                evaluation = lrow["evaluation"]
                prefix = "source_only_" if evaluation == "source_only" else ""
                truth_l = np.load(legacy_folder / (prefix + "true_label.npy"), allow_pickle=False).reshape(-1)
                truth_r = np.load(modern_folder / (prefix + "true_label.npy"), allow_pickle=False).reshape(-1)
                equal = np.array_equal(truth_l, truth_r)
                comparable = not reasons and equal
                pred_l = np.load(legacy_folder / (prefix + "pred_label.npy"), allow_pickle=False).reshape(-1)
                pred_r = np.load(modern_folder / (prefix + "pred_label.npy"), allow_pickle=False).reshape(-1)
                drift = pred_l.astype(float) - pred_r.astype(float) if comparable else None
                for metric in METRICS:
                    staged_deltas.append(dict(identity, evaluation=evaluation, metric=metric, comparable=comparable,
                        legacy=lrow[metric], modern=rrow[metric], paired_difference=lrow[metric] - rrow[metric] if comparable else None,
                        unmatched_reasons=reasons + ([] if equal else ["different_truth"]),
                        prediction_mean_abs_drift=float(np.mean(np.abs(drift))) if comparable else None,
                        prediction_RMS_drift=float(np.sqrt(np.mean(drift ** 2))) if comparable else None))
                for battery in legacy["batteries"]:
                    if battery["evaluation"] != evaluation:
                        continue
                    other = [row for row in modern["batteries"] if row["evaluation"] == evaluation and row["battery"] == battery["battery"]]
                    for metric in METRICS:
                        staged_batteries.append(dict(identity, evaluation=evaluation, battery=battery["battery"], metric=metric,
                            comparable=comparable and len(other) == 1, legacy=battery[metric],
                            modern=other[0][metric] if len(other) == 1 else None,
                            paired_difference=battery[metric] - other[0][metric] if comparable and len(other) == 1 else None))
                staged_metrics[evaluation] = (lrow, rrow, comparable,
                    digest(legacy_folder / (prefix + "true_label.npy")), digest(modern_folder / (prefix + "true_label.npy")))
            deltas.extend(staged_deltas)
            batteries.extend(staged_batteries)
            method_values[(item["logical_group"], item["repeat"])] = staged_metrics
            statuses.append(dict(identity, status="matched" if all(row["comparable"] for row in staged_deltas) else "unmatched",
                                 reasons=list(dict.fromkeys(reasons + [reason for row in staged_deltas for reason in row["unmatched_reasons"]])),
                                 legacy_optimization=legacy["optimization"], modern_optimization=modern["optimization"]))
            proof_rows.append(dict(identity, legacy_artifacts=legacy["proof"]["artifact_sha256"], modern_artifacts=modern["proof"]["artifact_sha256"],
                                   source_lineage=legacy["proof"].get("source_checkpoint_lineage"),
                                   modern_source_lineage=modern["proof"].get("source_checkpoint_lineage")))
        except (ValueError, KeyError, OSError, StopIteration, TypeError) as error:
            statuses.append(dict(identity, status="audit_failed", error=repr(error)))
    # Per-repeat method effects are matched by group/batch/repeat, never first baseline.
    group_tasks = {row["logical_group"]: row["legacy_task"] for row in manifest["tasks"]}
    for item in manifest["tasks"]:
        task, key, repeat = item["legacy_task"], item["logical_group"], item["repeat"]
        values = method_values.get((key, repeat))
        if not values:
            continue
        comparators = []
        if task["family"] == "transfer":
            if "source_only" in values:
                comparators.append(("fine_tune-minus-source_only", values["source_only"]))
        elif task["model"] == "PINN":
            for model in ["MLP", "CNN"]:
                baseline_keys = [k for k, candidate in group_tasks.items() if candidate["family"] == "full" and candidate["model"] == model and
                                 candidate["dataset"] == task["dataset"] and candidate["batch"] == task["batch"]]
                if len(baseline_keys) == 1 and (baseline_keys[0], repeat) in method_values:
                    comparators.append(("PINN-minus-" + model, method_values[baseline_keys[0], repeat]["final"]))
        for label, comparator in comparators:
            final = values["final"]
            comparable = final[2] and comparator[2] and final[0]["test_split_sha256"] == comparator[0]["test_split_sha256"] and final[3:] == comparator[3:]
            for metric in METRICS:
                ldelta, rdelta = final[0][metric] - comparator[0][metric], final[1][metric] - comparator[1][metric]
                effects.append(dict(logical_group=key, evaluation=label, metric=metric, repeat=repeat, seed=item["seed"],
                    comparable=comparable, legacy=ldelta, modern=rdelta,
                    empirical_direction_changed=direction(ldelta) != direction(rdelta) if comparable else None))
    summaries, effect_summaries = aggregate(deltas), aggregate(effects)
    output.mkdir()
    for name, rows in [("task_matching", statuses), ("per_repeat_metric_differences", deltas), ("per_battery_differences", batteries),
                       ("per_repeat_method_effects", effects), ("ten_repeat_summaries", summaries), ("ten_repeat_effect_summaries", effect_summaries)]:
        write_csv(output / (name + ".csv"), rows)
    audit.write_json(output / "comparison_evidence.json", proof_rows)
    counts = dict(Counter(row["status"] for row in statuses))
    summary = {"format": "pinn4soh-selected-legacy-extension-comparison-v1", "task_counts": counts,
        "expected_tasks": len(manifest["tasks"]), "expected_repeats": 10,
        "reused_repeat1": design["reused_repeat1_count"], "new_training_repeats": list(range(2, 11)),
        "assessment": "complete_selected_ten_repeat_sensitivity" if counts == {"matched": len(manifest["tasks"])} else "pending_or_unmatched",
        "selection": manifest["selection"], "no_significance_or_equivalence_claim": True,
        "summaries": summaries, "method_effect_summaries": effect_summaries,
        "analysis_manifest_sha256": digest(extension / "extension_analysis_manifest.json"),
        "input_suite_manifest_sha256": {str(path): digest(path / "manifest.json") for path in [old, new, *current_suites]},
        "analysis_scripts_sha256": {path.name: digest(path) for path in [Path(__file__), Path(audit.__file__),
            Path(__file__).with_name("prepare_legacy_extension.py"), Path(__file__).with_name("compare_legacy_control.py")]},
        "training_performed": False, "limitation": "Post-screen selected groups; cannot establish all2120 environment-independence or reconstruct historical backend defaults"}
    audit.write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["extension", "legacy-control-suite", "legacy-extension-suite", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--current-suites", type=Path, nargs="+", required=True)
    print(json.dumps(compare(**vars(parser.parse_args())), indent=2))
