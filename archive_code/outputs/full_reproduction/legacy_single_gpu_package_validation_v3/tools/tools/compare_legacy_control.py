"""Independently recompute matched current/legacy control differences without pooling stacks."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

import build_suite_report as audit


METRICS = ["MAE", "MAPE", "MSE", "RMSE"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                          for key, value in row.items()} for row in rows)


def match_reasons(legacy, current, proof_legacy, proof_current):
    fields = ["family", "dataset", "batch", "model", "sample_size", "protocol_id", "loss_variant",
        "validation_protocol", "source", "source_batch", "source_mode", "transfer_design",
        "source_fingerprint", "parameters_sha256", "splits_sha256", "test_split_sha256", "repeat", "seed"]
    reasons = ["different_" + key for key in fields if legacy.get(key) != current.get(key)]
    for label, proof in [("legacy", proof_legacy), ("current", proof_current)]:
        if legacy["family"] == "transfer":
            frozen = proof.get("frozen_dynamical_F") or {}
            if (frozen.get("unchanged") is not True or not frozen.get("before_sha256")
                    or frozen.get("before_sha256") != frozen.get("after_sha256")):
                reasons.append(label + "_frozen_G_unverified")
            if proof.get("solution_u_updated") is not True:
                reasons.append(label + "_F_update_unverified")
            lineage = proof.get("source_checkpoint_lineage") or {}
            if not lineage.get("training_history_verified") or lineage.get("source_selection") != "Prespecified same-index independent source repeat; no target-test selection":
                reasons.append(label + "_prespecified_reproduced_source_unverified")
    return reasons


def direction(value):
    return "lower" if value < 0 else "higher" if value > 0 else "equal"


def load_task(folder):
    status, repeats, batteries, proof = audit.inspect_task(folder, expected_repeats=1, metadata_only=False)
    if not status["included"]:
        raise ValueError(status["reason"])
    manifest = read_json(folder / "manifest.json")
    if "model.pth" not in manifest["artifact_sha256"] or audit.digest(folder / "model.pth") != manifest["artifact_sha256"]["model.pth"]:
        raise ValueError("Completed checkpoint hash missing or changed")
    repeats = [row for row in repeats if row["aggregation"] == "battery_macro" and row["convention"] == "standard"]
    batteries = [row for row in batteries if row["convention"] == "standard"]
    log = (folder / "logging.txt").read_text(encoding="utf-8")
    validations = [(int(epoch), float(value)) for epoch, value in re.findall(r"\[Valid\] epoch:(\d+), MSE: ([^\s]+)", log)]
    train_epochs = [int(value) for value in re.findall(r"\[Train\] epoch:(\d+)", log)]
    optimization = dict(best_validation_epoch=min(validations, key=lambda row: row[1])[0] if validations else None,
                        best_validation_MSE=min(value for _, value in validations) if validations else None,
                        completed_epochs=max(train_epochs) if train_epochs else None,
                        declared_best_epoch=manifest.get("best_epoch"))
    return dict(folder=folder, repeats=repeats, batteries=batteries, proof=proof, status=status,
                manifest=manifest, optimization=optimization)


def compare(plan_path, legacy_suite, current_suites, output):
    plan_path, legacy_suite, output = [Path(path).resolve() for path in [plan_path, legacy_suite, output]]
    current_suites = [Path(path).resolve() for path in current_suites]
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new or empty comparison output directory")
    for source in [legacy_suite, *current_suites]:
        if output == source or source in output.parents:
            raise ValueError("Comparison output must be outside suites")
    plan = read_json(plan_path)
    if len(plan["tasks"]) != 40 or plan.get("stage") != "legacy_matched_formal_control":
        raise ValueError("Expected the frozen 40-task formal control plan")
    lookup = {}
    needed = {task["original_task_id"] for task in plan["tasks"]}
    for suite in current_suites:
        for folder in audit.discover([suite]):
            config = read_json(folder / "config.json")
            task_id = config.get("task", {}).get("task_id")
            if task_id not in needed:
                continue
            if task_id in lookup:
                raise ValueError("Duplicate current task identity across suites")
            lookup[task_id] = folder
    tasks, deltas, per_battery, evidence, errors = [], [], [], [], []
    raw_metrics = {}
    for task in plan["tasks"]:
        identity = {key: task.get(key) for key in ["task_id", "original_task_id", "family", "dataset", "batch", "model",
            "loss_variant", "source", "target", "source_mode", "repeat", "seed"]}
        legacy_folder = legacy_suite / "tasks" / task["task_id"]
        current_folder = lookup.get(task["original_task_id"])
        if not (legacy_folder / "manifest.json").is_file() or current_folder is None:
            tasks.append({**identity, "status": "pending", "legacy_folder": str(legacy_folder),
                          "current_folder": str(current_folder) if current_folder else None})
            continue
        try:
            legacy, current = load_task(legacy_folder), load_task(current_folder)
            staged_deltas, staged_batteries, staged_metrics = [], [], {}
            for label, document in [("legacy", legacy), ("current", current)]:
                evidence.append({"stack": label, "task_id": task["task_id"],
                    "config": document["proof"]["config"], "artifact_sha256": document["proof"]["artifact_sha256"],
                    "optimization": document["optimization"], "frozen_dynamical_F": document["proof"].get("frozen_dynamical_F"),
                    "solution_u_updated": document["proof"].get("solution_u_updated"),
                    "source_checkpoint_lineage": document["proof"].get("source_checkpoint_lineage")})
            primary_legacy, primary_current = legacy["repeats"][0], current["repeats"][0]
            reasons = match_reasons(primary_legacy, primary_current, legacy["proof"], current["proof"])
            if legacy["proof"]["config"]["task"].get("original_task_id") != task["original_task_id"]:
                reasons.append("legacy_original_task_identity_differs")
            if primary_legacy.get("environment_sha256") == primary_current.get("environment_sha256"):
                reasons.append("software_environment_not_distinct")
            evaluation_details = []
            for left in legacy["repeats"]:
                right = next((row for row in current["repeats"] if row["evaluation"] == left["evaluation"]), None)
                if right is None:
                    raise ValueError("Current stack lacks matching evaluation")
                evaluation = left["evaluation"]
                prefix = "source_only_" if evaluation == "source_only" else ""
                truths = [np.load(document["folder"] / (prefix + "true_label.npy"), allow_pickle=False).reshape(-1)
                          for document in [legacy, current]]
                predictions = [np.load(document["folder"] / (prefix + "pred_label.npy"), allow_pickle=False).reshape(-1)
                               for document in [legacy, current]]
                equal_truth = np.array_equal(*truths)
                comparable = not reasons and equal_truth
                prediction_difference = predictions[0].astype(float) - predictions[1].astype(float) if comparable else None
                detail = dict(evaluation=evaluation, comparable=comparable, equal_truth=equal_truth,
                    prediction_max_abs_drift=float(np.max(np.abs(prediction_difference))) if comparable else None,
                    prediction_mean_abs_drift=float(np.mean(np.abs(prediction_difference))) if comparable else None,
                    prediction_RMS_drift=float(np.sqrt(np.mean(prediction_difference ** 2))) if comparable else None,
                    legacy_n_test=len(truths[0]), current_n_test=len(truths[1]))
                evaluation_details.append(detail)
                for metric in METRICS:
                    staged_deltas.append({**identity, **detail, "metric": metric, "legacy": left[metric], "current": right[metric],
                        "legacy_minus_current": left[metric] - right[metric] if comparable else None,
                        "descriptive_unmatched_difference": left[metric] - right[metric] if not comparable else None,
                        "unmatched_reasons": reasons + ([] if equal_truth else ["different_retained_measured_SOH"]),
                        "legacy_best_validation_epoch": legacy["optimization"]["best_validation_epoch"],
                        "current_best_validation_epoch": current["optimization"]["best_validation_epoch"]})
                for row in legacy["batteries"]:
                    if row["evaluation"] != evaluation:
                        continue
                    matches = [other for other in current["batteries"] if other["evaluation"] == evaluation and other["battery"] == row["battery"]]
                    if len(matches) != 1:
                        if comparable:
                            raise ValueError("Per-battery identity matching failed despite matching split")
                        for metric in METRICS:
                            staged_batteries.append({**identity, "evaluation": evaluation, "battery": row["battery"], "metric": metric,
                                "legacy": row[metric], "current": None, "comparable": False, "legacy_minus_current": None,
                                "status": "unmatched_battery_identity"})
                        continue
                    for metric in METRICS:
                        staged_batteries.append({**identity, "evaluation": evaluation, "battery": row["battery"], "metric": metric,
                            "legacy": row[metric], "current": matches[0][metric], "comparable": comparable,
                            "legacy_minus_current": row[metric] - matches[0][metric] if comparable else None})
                staged_metrics[task["task_id"], evaluation] = (left, right, comparable)
            deltas.extend(staged_deltas)
            per_battery.extend(staged_batteries)
            raw_metrics.update(staged_metrics)
            tasks.append({**identity, "status": "matched" if not reasons and all(row["equal_truth"] for row in evaluation_details) else "unmatched",
                "matching_issues": reasons, "evaluations": evaluation_details,
                "legacy_optimization": legacy["optimization"], "current_optimization": current["optimization"]})
        except (ValueError, KeyError, OSError, StopIteration, TypeError) as error:
            errors.append({**identity, "error": repr(error)})
            tasks.append({**identity, "status": "audit_failed", "error": repr(error)})
    method_effects, transfer_effects = [], []
    regular = [task for task in plan["tasks"] if task["family"] == "full"]
    for task in regular:
        if task["model"] != "PINN" or (task["task_id"], "final") not in raw_metrics:
            continue
        left, right, comparable = raw_metrics[task["task_id"], "final"]
        for model in ["MLP", "CNN"]:
            baseline = next(row for row in regular if row["dataset"] == task["dataset"] and row["model"] == model)
            item = raw_metrics.get((baseline["task_id"], "final"))
            if item is None:
                continue
            for metric in METRICS:
                shared = comparable and item[2] and left["test_split_sha256"] == item[0]["test_split_sha256"]
                legacy_effect, current_effect = left[metric] - item[0][metric], right[metric] - item[1][metric]
                method_effects.append(dict(dataset=task["dataset"], batch=task["batch"], loss_variant=task["loss_variant"],
                    comparison="PINN-minus-" + model, metric=metric, comparable=shared,
                    legacy_difference=legacy_effect, current_difference=current_effect,
                    empirical_direction_changed=direction(legacy_effect) != direction(current_effect) if shared else None,
                    legacy_direction=direction(legacy_effect), current_direction=direction(current_effect)))
    for task in [row for row in plan["tasks"] if row["family"] == "transfer"]:
        final, source = raw_metrics.get((task["task_id"], "final")), raw_metrics.get((task["task_id"], "source_only"))
        if final is None or source is None:
            continue
        for metric in METRICS:
            legacy_effect, current_effect = final[0][metric] - source[0][metric], final[1][metric] - source[1][metric]
            comparable = final[2] and source[2]
            transfer_effects.append(dict(source=task["source"], target=task["target"], loss_variant=task["loss_variant"],
                metric=metric, comparable=comparable, legacy_finetune_minus_source=legacy_effect,
                current_finetune_minus_source=current_effect, legacy_direction=direction(legacy_effect),
                current_direction=direction(current_effect),
                empirical_direction_changed=direction(legacy_effect) != direction(current_effect) if comparable else None))
    output.mkdir(parents=True, exist_ok=True)
    for name, values in [("task_matching", tasks), ("all_metric_differences", deltas), ("per_battery_differences", per_battery),
                         ("method_rank_changes", method_effects), ("transfer_effect_changes", transfer_effects)]:
        write_csv(output / (name + ".csv"), values)
    audit.write_json(output / "comparison_evidence.json", evidence)
    counts = dict(Counter(row["status"] for row in tasks))
    summary = dict(format="pinn4soh-legacy-control-comparison-v1", task_counts=counts, expected_tasks=40,
        independent_array_recomputation=True, aggregation="battery_macro", convention="standard measured-SOH MAPE",
        difference="legacy-minus-current; raw SOH fractions; every matched task/evaluation retained",
        method_direction_changes=sum(row["empirical_direction_changed"] is True for row in method_effects),
        transfer_direction_changes=sum(row["empirical_direction_changed"] is True for row in transfer_effects),
        no_statistical_significance_or_equivalence_assumed=True, repeat_count=1, errors=errors,
        assessment="complete_single_seed_version_screen" if counts == {"matched": 40} else "pending_or_unmatched",
        limitations=["One prespecified seed cannot estimate training stability or statistical significance",
            "Different software versions need not produce identical GPU arithmetic or RNG streams",
            "Matched source training is separately repeated in each stack; source drift is part of full transfer pipeline sensitivity",
            "No result from this subset establishes environment independence of the entire paper or all2120 tasks"],
        input_plan_sha256=audit.digest(plan_path), script_sha256=audit.digest(__file__),
        analyzer_sha256=audit.digest(Path(audit.__file__)), training_performed=False,
        artifacts={str(path.relative_to(output)):audit.digest(path) for path in output.rglob("*") if path.is_file()})
    audit.write_json(output / "summary.json", summary)
    brief = ["# 原版软件栈配对对照", "", "按事先固定 repeat1 配对，独立从预测数组重算指标，未合并两套环境。",
        "", "任务状态：" + json.dumps(counts, ensure_ascii=False),
        "- 所有差值均为旧栈减现代栈；主口径为逐电池等权宏平均、实测 SOH 作 MAPE 分母。",
        "- 参数、划分、电池顺序、源数据、损失及种子不匹配时保留原值并标为 unmatched，不作可比差值。",
        "- 方法排序变化数：" + str(summary["method_direction_changes"]) + "；迁移效果方向变化数：" + str(summary["transfer_direction_changes"]) + "。",
        "- 这些变化按四种指标逐格列出，不能把指标格数量当作独立重复次数。",
        "- 单种子仅用于版本敏感性筛查；结论反转或显著优化路径变化应触发受影响配置的十次配对扩展。",
        "- 未设论文未报告的等效误差阈值，不能据此宣称两环境等价或全部复现结论与环境无关。", ""]
    (output / "旧版软件栈对照说明.md").write_text("\n".join(brief), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--legacy-suite", type=Path, required=True)
    parser.add_argument("--current-suite", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    result = compare(options.plan, options.legacy_suite, options.current_suite, options.output)
    print(json.dumps({key:result[key] for key in ["task_counts", "assessment", "method_direction_changes", "transfer_direction_changes"]}))
