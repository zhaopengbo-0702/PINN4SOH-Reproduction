"""Read-only supplement for transfer/target-only schema mismatches.

Never changes the frozen report, training outputs, or training code. Aliases
alone are insufficient: every normalized environment must be corroborated by
the completed coordinator's recorded probe and task/thread/hash evidence.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import sys


FORMAT = "pinn4soh-transfer-environment-schema-supplement-v1"
REQUIRED_ENV = {"python", "torch", "numpy", "pandas", "scipy", "sklearn",
                "cuda", "cuda_available", "gpu"}
ALIASES = {"scikit_learn": "sklearn", "cuda_runtime": "cuda"}
COHORT = ["dataset", "paper_group", "protocol_id", "validation_protocol",
          "source_fingerprint", "test_split_sha256"]
EXPECTED_REPORT_SHA = "ebdc9f8a7ddeef608ee93a4b4aa6215b8ad06ac4b26e75d744233419f76d9738"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def require(condition, explanation):
    if not condition:
        raise ValueError(explanation)


def canonical_environment(raw):
    """Exact aliases only. Conflicting aliases and unexpected keys fail closed."""
    result = {}
    for key, value in raw.items():
        if key in {"CUDA_VISIBLE_DEVICES", "recorded_at", "threads"}:
            continue
        key = ALIASES.get(key, key)
        require(key in REQUIRED_ENV, "Unreviewed environment key: " + key)
        require(key not in result or result[key] == value,
                "Conflicting environment aliases: " + key)
        result[key] = value
    return result


def corroborate_environment(raw, task, frozen, coordinator, manifest):
    actual = canonical_environment(raw)
    require(REQUIRED_ENV - {"scipy"} <= set(actual), "Missing essential runtime evidence")
    signatures = frozen.get("environment_signatures", {})
    for key, value in signatures.items():
        require(stable(value) == key, "Corrupt coordinator environment signature")
    candidates = [(key, value) for key, value in signatures.items()
                  if set(value) == REQUIRED_ENV and
                  all(value.get(key) == val for key, val in actual.items())]
    require(len(candidates) == 1, "Actual runtime lacks unique matching coordinator probe")
    signature_sha, signature = candidates[0]
    require(task.get("threads") == frozen.get("threads"), "Task/coordinator thread mismatch")
    require(raw.get("threads", task["threads"]) == task["threads"], "Raw thread mismatch")
    visibility = raw.get("CUDA_VISIBLE_DEVICES", manifest.get("assigned_cuda_visible_devices"))
    require(str(visibility) == str(coordinator.get("assigned_gpu")), "GPU allocation mismatch")
    normalized = dict(signature, threads=task["threads"])
    return {"raw": raw, "raw_report_environment_sha256": stable({k: v for k, v in raw.items()
            if k not in {"CUDA_VISIBLE_DEVICES", "recorded_at"}}),
            "normalized": normalized, "normalized_sha256": stable(normalized),
            "coordinator_probe_sha256": signature_sha,
            "filled_from_completed_coordinator_probe": sorted(REQUIRED_ENV - set(actual)),
            "threads_provenance": "config.task.threads + worker task + frozen.threads; runner set_num_threads source checked",
            "assigned_gpu": coordinator.get("assigned_gpu"), "visibility": visibility,
            "physical_gpu_uuid_or_driver_not_recorded": True}


def relocated(workspace, cloud_path):
    parts = PurePosixPath(cloud_path).parts
    require(parts[:2] == ("/", "workspace"), "Outside recovered workspace: " + cloud_path)
    require(".." not in parts, "Parent path is forbidden")
    return workspace.joinpath(*parts[2:])


def role(splits, selected):
    values = [item for item in splits if item["role"] in selected]
    require(len(values) == 1, "Expected one split role: " + str(selected))
    return values[0]


def compare_split_evidence(left, right, size):
    lt, rt = role(left["splits"], {"test"}), role(right["splits"], {"test"})
    la = role(left["splits"], {"train", "train_validation"})
    ra = role(right["splits"], {"train", "train_validation"})
    checks = {"ordered_test_split_and_boundaries_equal": lt == rt,
              "adaptation_paths_equal_target_prefix": la["paths"] == ra["paths"][:1],
              "target_baseline_battery_count_correct": len(ra["paths"]) == size,
              "adaptation_nominal_capacity_equal": la["nominal_capacity"] == ra["nominal_capacity"],
              "adaptation_battery_boundary_equal": la.get("batteries") == ra.get("batteries", [])[:1]}
    if size == 1:
        checks["entire_n1_training_split_equal"] = la == ra
        checks["n1_train_valid_sample_counts_equal"] = left["sample_counts"] == right["sample_counts"]
    return checks, la["paths"], ra["paths"]


def validate_task_identity(config_task, raw_task, effective_task, planned, coordinator_sha, worker_sha):
    """Respect the runner's separately recorded raw and effective task layers."""
    effective_planned = dict(planned)
    if planned["runner"] == "extended":
        effective_planned.setdefault("smoke", False)
    require(raw_task == planned, "Raw task differs from frozen plan")
    require(config_task == effective_planned, "Effective task differs from frozen plan/defaults")
    require(effective_task == config_task, "Effective task mismatch")
    require(worker_sha == coordinator_sha == stable(planned), "Raw task hash mismatch")


def descriptive_stats(values):
    return {"n": len(values), "mean": statistics.mean(values),
            "sd_ddof0": statistics.pstdev(values),
            "sd_ddof1": statistics.stdev(values) if len(values) > 1 else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    workspace, output = options.workspace.resolve(), options.output.resolve()
    require(not output.exists(), "Refusing to overwrite supplemental output")
    original_sha = sha(options.report)
    require(original_sha == EXPECTED_REPORT_SHA, "Unexpected frozen report")
    report = read(options.report)
    require(report["task_count"] == 2120 and report["metadata_only"] is False,
            "Expected final independently recomputed 2120-task report")
    reporter_path = workspace / "full_report_tools/build_suite_report.py"
    spec = importlib.util.spec_from_file_location("frozen_suite_report", reporter_path)
    reporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reporter)
    # Source code is imported only for JSON/NumPy auditing; no PyTorch model is loaded.
    suites, evidence_files, verified_files, file_hash_cache = {}, {}, {}, {}

    def actual_sha(path):
        key = str(path)
        if key not in file_hash_cache:
            file_hash_cache[key] = sha(path)
        return file_hash_cache[key]

    for suite_path in report["inputs"]["suites"]:
        folder = relocated(workspace, suite_path)
        manifest, plan, inventory = (read(folder / name) for name in
                                     ["manifest.json", "plan.json", "source_inventory.json"])
        require(manifest["status"] == "complete", "Suite is incomplete")
        require(stable(plan) == manifest["frozen"]["plan_sha256"], "Frozen plan mismatch")
        require(stable(inventory["sha256"]) == inventory["fingerprint"] ==
                manifest["frozen"]["source_fingerprint"], "Source inventory mismatch")
        require(stable(inventory["data_directory_order"]) ==
                manifest["frozen"]["data_directory_order_sha256"], "Enumeration mismatch")
        plans = {task["task_id"]: task for task in plan["tasks"]}
        suites[folder.name] = {"manifest": manifest, "plan": plans, "inventory": inventory,
                               "folder": folder}
        for name in ["manifest.json", "plan.json", "source_inventory.json"]:
            evidence_files[str((folder / name).relative_to(workspace))] = actual_sha(folder / name)
        for cloud_file, checksum in manifest["frozen"]["implementation_sha256"].items():
            local = relocated(workspace, cloud_file)
            require(actual_sha(local) == checksum, "Frozen implementation changed: " + cloud_file)
            evidence_files[str(local.relative_to(workspace))] = checksum
    inventories = [value["inventory"] for value in suites.values()]
    require(inventories[0] == inventories[1], "Different stage source inventories")
    inventory = inventories[0]
    loader_path = workspace / "PINN4SOH-main/dataloader/dataloader.py"
    require(actual_sha(loader_path) == inventory["sha256"]["dataloader/dataloader.py"], "Loader changed")
    evidence_files[str(loader_path.relative_to(workspace))] = actual_sha(loader_path)
    primary = [row for row in report["summaries"] if row["aggregation"] == "battery_macro" and
               row["convention"] == "standard" and row["complete_repeat_set"] and row["evaluation"] == "final"]
    transfers = [row for row in primary if row["family"] == "transfer"]
    require(len(transfers) == 48, "Expected 48 separate transfer groups")
    candidates, group_tasks, original_status = [], defaultdict(list), {}
    for item in report["tasks"]:
        if item["included"]:
            group_tasks[item["comparison_group"]].append(item)
            original_status[item["task_id"]] = item
    for left in transfers:
        for size in [1, 2]:
            right = [row for row in primary if row["family"] == "small_sample" and
                     row["model"] == "PINN" and row["sample_size"] == size and
                     row["loss_variant"] == left["loss_variant"] and
                     all(row[key] == left[key] for key in COHORT)]
            require(len(right) == 1, "Comparator missing or ambiguous after environment exclusion")
            candidates.append((left, right[0], size))
    selected_groups = {row["comparison_group"] for left, right, _ in candidates for row in [left, right]}
    audited, environments, failures = {}, [], []
    for group in sorted(selected_groups):
        for task_status in group_tasks[group]:
            task_id = task_status["task_id"]
            try:
                folder = relocated(workspace, task_status["folder"])
                suite = suites[folder.relative_to(workspace).parts[0]]
                state, frozen = suite["manifest"]["tasks"][task_id], suite["manifest"]["frozen"]
                require(state["status"] == "complete" and state.get("returncode") == 0,
                        "Coordinator task not complete")
                # Every recovered task file is checked against the coordinator's completion-time hashes.
                verified = state["verified_file_sha256"]
                require(set(["config.json", "splits.json", "model.pth", "metrics.json"]) <= set(verified),
                        "Missing coordinator file evidence")
                for name, expected in verified.items():
                    p = folder / name
                    require(actual_sha(p) == expected, "Changed task file: " + name)
                require(actual_sha(folder / "manifest.json") == state["worker_manifest_sha256"],
                        "Worker manifest changed")
                status, repeat_rows, battery_rows, proof = reporter.inspect_task(folder)
                require(status["included"] and status["comparison_group"] == group, "Original group changed")
                for key in ["config_sha256", "splits_file_sha256", "manifest_sha256", "metrics_sha256"]:
                    require(status[key] == task_status[key], "Frozen report/task hash mismatch: " + key)
                config, splits = proof["config"], proof["splits"]
                manifest = read(folder / "manifest.json")
                task = config["task"]
                planned = dict(suite["plan"][task_id])
                planned.setdefault("threads", frozen["threads"])
                validate_task_identity(task, manifest["task"], manifest.get("effective_task", manifest["task"]),
                                       planned, state["task_sha256"], manifest["task_sha256"])
                require(config["source_fingerprint"] == inventory["fingerprint"], "Source fingerprint mismatch")
                env = corroborate_environment(config["environment"], task, frozen, state, manifest)
                require(env["raw_report_environment_sha256"] == next(row["environment_sha256"]
                        for row in repeat_rows), "Raw report environment identity differs")
                if (folder / "environment.json").exists():
                    require(read(folder / "environment.json") == config["environment"], "Standalone environment differs")
                parameters = proof["parameters"]
                require(parameters["normalization_method"] == "min-max", "Unexpected normalization")
                require(parameters["batch_size"] == 128, "Unexpected target batch size")
                reference_cloud = task.get("split_json") or task["reference_run"] + "/splits.json"
                reference_file = relocated(workspace, reference_cloud)
                require(actual_sha(reference_file) == state["input_sha256"][reference_cloud], "Reference split changed")
                reference = read(reference_file)
                train, test = role(splits, {"train", "train_validation"}), role(splits, {"test"})
                reference_train, reference_test = role(reference, {"train", "train_validation"}), role(reference, {"test"})
                size = 1 if task["family"] == "transfer" else task["sample_size"]
                require(train["paths"] == reference_train["paths"][:size], "Selected training paths differ")
                require(test == reference_test, "Test split differs from reference")
                for split in splits:
                    for csv in split["paths"]:
                        expected = inventory["sha256"][csv]
                        require(actual_sha(workspace / "PINN4SOH-main" / csv) == expected, "CSV changed: " + csv)
                        require(state["input_sha256"]["/workspace/PINN4SOH-main/" + csv] == expected,
                                "Coordinator input CSV differs")
                        verified_files[csv] = expected
                lineage = proof.get("source_checkpoint_lineage")
                if lineage:
                    require(actual_sha(relocated(workspace, lineage["path"])) == lineage["sha256"],
                            "Source checkpoint changed")
                    require(state["input_sha256"][lineage["path"]] == lineage["sha256"], "Source input hash mismatch")
                if task["runner"] == "extended":
                    data_identity = read(folder / "data_identity.json")
                    require(data_identity["sample_counts"] == manifest["sample_counts"], "Sample count identity differs")
                    for csv, expected in data_identity["input_file_sha256"].items():
                        require(expected == verified_files[csv], "Input identity differs")
                env.update(task_id=task_id, suite=folder.relative_to(workspace).parts[0],
                           config_sha256=actual_sha(folder / "config.json"))
                environments.append(env)
                final = next(row for row in repeat_rows if row["evaluation"] == "final" and
                             row["aggregation"] == "battery_macro" and row["convention"] == "standard")
                source_only = next((row for row in repeat_rows if row["evaluation"] == "source_only" and
                                    row["aggregation"] == "battery_macro" and row["convention"] == "standard"), None)
                audited[task_id] = {"task": task, "folder": folder, "splits": splits,
                    "environment": env, "parameters": parameters, "final": final, "source_only": source_only,
                    "sample_counts": manifest["sample_counts"], "epochs_completed": manifest.get("epochs_completed"),
                    "artifact_sha256": dict(verified), "frozen_G": proof.get("frozen_dynamical_F"),
                    "solution_F_updated": proof.get("solution_u_updated"), "lineage": lineage,
                    "reference_split_sha256": actual_sha(reference_file)}
            except Exception as error:
                failures.append({"task_id": task_id, "error": str(error)})
    comparisons, repeat_checks = [], []
    for left, right, size in candidates:
        left_tasks = {int(item["task_id"].split("Experiment")[-1]): item["task_id"]
                      for item in group_tasks[left["comparison_group"]]}
        right_tasks = {int(item["task_id"].split("Experiment")[-1]): item["task_id"]
                       for item in group_tasks[right["comparison_group"]]}
        identity = {"kind": "fine_tune_vs_target_only_n" + str(size),
            "loss_variant": left["loss_variant"], "source_mode": left["source_mode"],
            "source": left["source"], "source_batch": left["source_batch"],
            "target": left["paper_group"], "target_only_batteries": size,
            "claim_scope": "C13" if size == 1 else ("C14" if left["paper_group"] in {"XJTU/1", "TJU/3"}
                                                     else "additional_n2_outside_C14"),
            "left_comparison_group": left["comparison_group"], "right_comparison_group": right["comparison_group"],
            "raw_environment_hashes_differ": left["environment_sha256"] != right["environment_sha256"],
            "metric": "RMSE", "primary_convention": "standard/battery_macro",
            "original_strict_report_unchanged": True, "no_significance_or_equivalence_claim": True}
        matched = set(left_tasks) == set(right_tasks) == set(range(1, 11))
        left_values, right_values, source_values, per_repeat_deltas, hyperparameters = [], [], [], [], {}
        for repeat in sorted(left_tasks):
            l, r = audited.get(left_tasks[repeat]), audited.get(right_tasks.get(repeat))
            record = dict(identity, repeat=repeat, left_task=left_tasks[repeat], right_task=right_tasks.get(repeat))
            if l is None or r is None:
                record.update(passed=False, unresolved="One or both tasks failed independent audit")
                matched = False
                repeat_checks.append(record)
                continue
            checks, adaptation_paths, baseline_paths = compare_split_evidence(l, r, size)
            checks.update(seed_equal=l["task"]["seed"] == r["task"]["seed"],
                          normalized_environment_equal=l["environment"]["normalized"] == r["environment"]["normalized"],
                          true_label_file_sha256_equal=l["artifact_sha256"]["true_label.npy"] == r["artifact_sha256"]["true_label.npy"],
                          source_fingerprint_equal=l["final"]["source_fingerprint"] == r["final"]["source_fingerprint"],
                          reference_split_sha256_equal=l["reference_split_sha256"] == r["reference_split_sha256"],
                          normalization_equal=l["parameters"]["normalization_method"] == r["parameters"]["normalization_method"],
                          final_and_source_only_truth_equal=l["artifact_sha256"]["true_label.npy"] == l["artifact_sha256"]["source_only_true_label.npy"],
                          frozen_G_unchanged=l["frozen_G"]["unchanged"] is True and
                              l["frozen_G"]["before_sha256"] == l["frozen_G"]["after_sha256"],
                          solution_F_updated=l["solution_F_updated"] is True)
            differences = {key: {"transfer": l["parameters"].get(key), "target_only": r["parameters"].get(key)}
                           for key in sorted(set(l["parameters"]) | set(r["parameters"]))
                           if l["parameters"].get(key) != r["parameters"].get(key)}
            hyperparameters[stable(differences)] = differences
            delta = l["final"]["RMSE"] - r["final"]["RMSE"]
            record.update(passed=all(checks.values()), checks=checks, seed=l["task"]["seed"],
                          normalized_environment_sha256=l["environment"]["normalized_sha256"],
                          adaptation_paths=adaptation_paths, baseline_paths=baseline_paths,
                          transfer_RMSE=l["final"]["RMSE"], target_only_RMSE=r["final"]["RMSE"],
                          transfer_minus_target_RMSE=delta, source_only_RMSE=l["source_only"]["RMSE"],
                          transfer_epochs=l["epochs_completed"], target_only_epochs=r["epochs_completed"],
                          transfer_sample_counts=l["sample_counts"], target_only_sample_counts=r["sample_counts"],
                          hyperparameter_differences_sha256=stable(differences))
            matched = matched and record["passed"]
            repeat_checks.append(record)
            left_values.append(l["final"]["RMSE"])
            right_values.append(r["final"]["RMSE"])
            source_values.append(l["source_only"]["RMSE"])
            per_repeat_deltas.append(delta)
        result = dict(identity, status="descriptive_comparison_available" if matched else "unresolved_evidence_mismatch",
                      paired_repeats=len(left_values), hyperparameter_differences=list(hyperparameters.values()))
        if matched:
            # Cross-check the independent recomputation against the unchanged report.
            for values, original in [(left_values, left), (right_values, right)]:
                require(math.isclose(statistics.mean(values), original["RMSE_mean"], rel_tol=1e-12, abs_tol=1e-12),
                        "Independent RMSE mean differs from frozen report")
            difference = statistics.mean(left_values) - statistics.mean(right_values)
            result.update(transfer=descriptive_stats(left_values), target_only=descriptive_stats(right_values),
                          source_only=descriptive_stats(source_values),
                          left_minus_right=difference,
                          direction="lower" if difference < -1e-12 else "higher" if difference > 1e-12 else "tie",
                          paired_delta=descriptive_stats(per_repeat_deltas),
                          paired_repeats_transfer_lower=sum(value < 0 for value in per_repeat_deltas))
        comparisons.append(result)
    summary = {"candidate_comparisons": len(comparisons), "independently_audited_tasks": len(audited),
               "task_audit_failures": len(failures), "repeat_pair_checks": len(repeat_checks),
               "repeat_pair_checks_passed": sum(row["passed"] for row in repeat_checks),
               "normalized_environment_hashes": sorted({row["normalized_sha256"] for row in environments}),
               "raw_environment_hashes": dict(Counter(row["raw_report_environment_sha256"] for row in environments)),
               "status_counts": dict(Counter(row["status"] for row in comparisons)),
               "scope_direction_counts": {scope: dict(Counter(row.get("direction", "unresolved") for row in comparisons
                    if row["claim_scope"] == scope)) for scope in ["C13", "C14", "additional_n2_outside_C14"]}}
    lanes = defaultdict(list)
    for row in comparisons:
        lanes[(row["claim_scope"], row["loss_variant"], row["source_mode"])].append(row)
    lane_summary = [{"claim_scope": key[0], "loss_variant": key[1], "source_mode": key[2],
                     "comparisons": len(rows), **dict(Counter(row.get("direction", "unresolved") for row in rows))}
                    for key, rows in sorted(lanes.items())]
    risks = [
        "The frozen strict report is preserved. This supplement changes only the environment matching interpretation after independent corroboration.",
        "SciPy is not recorded per extended worker; 1.10.1 is corroborated from its completed suite coordinator probe, not inferred from the transfer worker.",
        "Thread evidence is task/frozen configuration plus the verified runner set_num_threads call; no per-worker get_num_threads measurement was saved.",
        "GPU model/CUDA runtime match and assigned visibility is checked. Physical GPU UUID, CUDA driver, cuDNN, deterministic-algorithm flags, and BLAS build were not recorded, so bitwise hardware equivalence is not asserted.",
        "Both methods reuse per-CSV 3-sigma filtering and min-max feature scaling, followed by author 20% cycle-sample validation with random_state=420. Test features are scaled using their entire CSV as in the author loader; this remains an author-protocol limitation.",
        "For n1 both methods use the same ordered battery/sample split. For n2 the target baseline trains on two batteries while transfer adapts on the first, so sample partitions are intentionally different and per-repeat common seeds do not mean identical training randomness.",
        "Transfer and target-only loss weights, patience, learning-rate schedules and other recorded hyperparameters differ. Effects describe those methods; they do not isolate pretraining as the only cause.",
        "Supplied-author checkpoints lack original training seed/split/history provenance; reproduced-source checkpoints have separate lineage and are never merged with supplied weights.",
        "No retraining, model unpickling, inference replay, significance test, equivalence margin, or physical-parameter identifiability assertion is performed here."]
    payload = {"format": FORMAT, "generated_at": datetime.now(timezone.utc).isoformat(),
        "frozen_report": {"path": str(options.report.resolve()), "sha256_before": original_sha,
                          "sha256_after": sha(options.report), "unchanged": sha(options.report) == original_sha},
        "source": {"workspace": str(workspace), "source_fingerprint": inventory["fingerprint"],
                   "frozen_reporter_sha256": sha(reporter_path), "diagnostic_script_sha256": sha(__file__),
                   "files": evidence_files, "selected_csv_sha256": verified_files},
        "matching_policy": {"cohort_fields_unchanged": COHORT, "environment_aliases": ALIASES,
            "environment_required_fields": sorted(REQUIRED_ENV | {"threads"}),
            "required_repeat_ids": list(range(1, 11)), "seed_matching": "per-repeat exact",
            "target_training_paths": "transfer first battery equals baseline ordered prefix; baseline count exactly n",
            "array_audit": "all task files match completion-time coordinator hashes; independent NumPy saved-array metric recomputation; truth and ordered battery boundaries exactly match"},
        "summary": summary, "lane_summary": lane_summary, "risks": risks, "task_failures": failures,
        "task_evidence": [{"task_id": task_id, "folder": str(item["folder"].relative_to(workspace)),
             "repeat": item["task"]["repeat"], "seed": item["task"]["seed"],
             "sample_counts": item["sample_counts"], "epochs_completed": item["epochs_completed"],
             "parameters": item["parameters"], "source_checkpoint_lineage": item["lineage"],
             "reference_split_sha256": item["reference_split_sha256"],
             "completion_time_artifact_hashes_verified_locally": item["artifact_sha256"]}
             for task_id, item in sorted(audited.items())],
        "environment_evidence": environments, "comparisons": comparisons,
        "repeat_pair_checks": repeat_checks}
    require(payload["frozen_report"]["unchanged"], "Frozen report changed during diagnostic")
    output.mkdir(parents=True)
    reporter.write_json(output / "supplement.json", payload)
    reporter.write_csv(output / "comparisons.csv", comparisons)
    reporter.write_csv(output / "repeat_pair_checks.csv", repeat_checks)
    reporter.write_csv(output / "environment_evidence.csv", environments)
    reporter.write_csv(output / "lane_summary.csv", lane_summary)
    lines = ["# C13/C14 环境记录格式补充诊断", "", "本补充只读核对已恢复的实验产物；冻结的 2120 项报告保持不变。", "",
             f"- 核对 {len(audited)} 个任务，{len(comparisons)} 个候选比较，{len(repeat_checks)} 个逐重复配对。",
             f"- 任务审计失败：{len(failures)}；逐重复配对通过：{summary['repeat_pair_checks_passed']}。",
             "- 原始环境 SHA 不同源于 sklearn/CUDA 键名及 SciPy/threads 记录范围差异；补充仅在原始值与各自已完成 coordinator 探测一致时规范化。",
             "- 实测版本：Python 3.10.13、PyTorch 1.13.1+cu116、NumPy 1.24.4、pandas 1.5.3、SciPy 1.10.1、scikit-learn 1.2.2、CUDA 11.6、RTX 3090；线程配置 2。", "",
             "方向均指 fine-tune 的 RMSE 相对目标独训；按电池等权宏平均后跨 10 次重复汇总。", "",
             "| 范围 | fine-tune 较低 | 较高 | 持平 |", "| --- | ---: | ---: | ---: |"]
    for scope, counts in summary["scope_direction_counts"].items():
        lines.append(f"| {scope} | {counts.get('lower', 0)} | {counts.get('higher', 0)} | {counts.get('tie', 0)} |")
    lines.extend(["", "| 范围 | 损失 | 来源 | 较低 | 较高 |", "| --- | --- | --- | ---: | ---: |"])
    for row in lane_summary:
        lines.append(f"| {row['claim_scope']} | {row['loss_variant']} | {row['source_mode']} | {row.get('lower', 0)} | {row.get('higher', 0)} |")
    lines.extend(["", "C13 包含 12 方向 × 2 损失 × 2 来源，共 48 比较；C14 仅 XJTU/1 和 TJU/3 两个目标，共 24 比较。另 24 个 n2 比较单独标为额外范围。来源和损失均分开保留，不能合并为独立重复。", "",
                  "风险与证据边界：", ""] + ["- " + value for value in risks] +
                 ["", "详见 supplement.json、comparisons.csv、repeat_pair_checks.csv 和 environment_evidence.csv。"])
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    reporter.write_json(output / "OUTPUT_SHA256.json", {path.name: sha(path) for path in sorted(output.iterdir()) if path.is_file()})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if failures or any(not row["passed"] for row in repeat_checks) else 0


if __name__ == "__main__":
    sys.exit(main())
