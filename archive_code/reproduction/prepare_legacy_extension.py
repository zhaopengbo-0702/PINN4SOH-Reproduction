"""Prepare only explicitly selected ten-repeat runtime extensions; never train.

Reuse repeat1 in its immutable original control directory. New suite contains
only repeats2..10. The separate analysis manifest composes the two locations.
"""

import argparse
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
import re
from pathlib import Path, PurePosixPath


FORMAT = "pinn4soh-legacy-selected-extension-v1"
SELECTION_FORMAT = "pinn4soh-legacy-extension-selection-v1"
CONTROL_SHA = "f415319506907e48683038f00be801d5bca6f008890663391082bb9893717338"
STAGE1_SHA = "e6f443c022c2ed9177e385f8079e357df12d6f37494edd3ce09e98d813c9e18d"
STAGE2_SHA = "cdd144587df0a01677b88a52ccfc1e23825bf96cb7a35f8baa0d4cad54aed4e3"
SOURCE_POLICY = "Prespecified same-index independent source repeat; no target-test selection"
MODERN_CODE_SHA = {
    "run_reproduction.py": "29322fff682bf66dfe08bc5715e9ddfad9188b1c85cfe7b09941d87bade5bad5",
    "run_extended.py": "24c3e482fb3a9c751c478072e04b77a238b37571d9b970ae1f69bea600fdba17",
    "run_transfer.py": "4f2f19642f74f0904511b83dbafecd211ee6e254458fb052d1bca0d4b3a8945f",
    "run_full_suite.py": "c9730b49492f835538080fdd1c12d13b21393eadb37947f12365a535d17a3402"}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def effective_task(task):
    value = deepcopy(task)
    if value["runner"] == "extended":
        value.setdefault("smoke", False)
        value.setdefault("overrides", {})
    return value


def runtime_signature(frozen):
    signatures = frozen.get("environment_signatures", {})
    require(len(signatures) == 1, "Exactly one frozen legacy runtime signature required")
    signature = next(iter(signatures.values()))
    require(set(signature) >= {"python", "torch", "numpy", "pandas", "sklearn", "cuda", "cuda_available", "gpu"},
            "Incomplete frozen runtime signature")
    require(signature["cuda_available"] is True and frozen.get("threads") == 2 and
            frozen.get("smoke") is False and frozen.get("allow_cpu") is False, "Unexpected frozen GPU execution policy")
    return signature


def implementation(task, frozen):
    codes = {}
    for path, checksum in frozen["implementation_sha256"].items():
        name = PurePosixPath(path).name
        require(name not in codes or codes[name] == checksum, "Conflicting frozen implementation filenames")
        codes[name] = checksum
    names = ["run_extended.py", "run_reproduction.py"] if task["runner"] == "extended" else ["run_transfer.py", "run_reproduction.py"]
    if task["runner"] == "extended" and task.get("model") == "PINN" and task.get("loss_variant", "author_current") != "author_current":
        names.append("run_transfer.py")
    if task["runner"] == "transfer" and task.get("validation_protocol") == "explicit_battery_split":
        names.append("run_extended.py")
    return {("helpers" if task["runner"] == "extended" and name == "run_reproduction.py" else name): codes[name] for name in names}


def same_legacy_cohort(old, new):
    require(runtime_signature(old) == runtime_signature(new), "Legacy runtime cohort changed across reused/new repeats")
    for key in ["source_fingerprint", "data_directory_order_sha256", "threads", "smoke", "allow_cpu"]:
        require(old[key] == new[key], "Legacy frozen source/execution cohort changed: " + key)
    old_codes = {PurePosixPath(path).name: checksum for path, checksum in old["implementation_sha256"].items()}
    for path, checksum in new["implementation_sha256"].items():
        require(old_codes.get(PurePosixPath(path).name) == checksum, "Legacy implementation changed: " + path)


def verify_modern_frozen(frozen):
    runtime_signature(frozen)
    for path, checksum in frozen["implementation_sha256"].items():
        require(MODERN_CODE_SHA.get(PurePosixPath(path).name) == checksum, "Frozen modern runner is outside approved implementation")


def verify_composite(plan, manifest, design):
    tasks = manifest["tasks"]
    require(0 < len(plan["tasks"]) == design["new_task_count"] <= 360 and
            0 < len(tasks) == design["analysis_task_count"] <= 400, "Prepared task counts differ")
    require(stable(manifest["selection"]) == plan["selection_sha256"], "Prepared explicit selection differs")
    require({row["legacy_task_id"]: row["legacy_task"] for row in tasks if row["repeat"] > 1} ==
            {row["task_id"]: row for row in plan["tasks"]}, "New plan and analysis task identities differ")
    require(len({row["legacy_task_id"] for row in tasks}) == len(tasks) == len({row["modern_task_id"] for row in tasks}), "Duplicate analysis identity")
    groups = {}
    for row in tasks:
        groups.setdefault(row["logical_group"], []).append(row)
        require(row["legacy_task_id"] == row["legacy_task"]["task_id"] and row["modern_task_id"] == row["modern_task"]["task_id"] and
                row["repeat"] == row["legacy_task"]["repeat"] == row["modern_task"]["repeat"] and
                row["seed"] == row["legacy_task"]["seed"] == row["modern_task"]["seed"], "Analysis task/seed identity differs")
        require(row["legacy_task"].get("original_task_id") == row["modern_task_id"], "Legacy original task identity differs")
        expected_storage = "immutable_original_repeat1" if row["repeat"] == 1 else "new_repeat2_to10"
        require(row["storage"] == expected_storage, "Analysis storage identity differs")
    for rows in groups.values():
        require(sorted(row["repeat"] for row in rows) == list(range(1, 11)) and len({row["seed"] for row in rows}) == 10, "Composite group lacks exact ten unique repeats")
    require(sum(row["repeat"] == 1 for row in tasks) == design["reused_repeat1_count"], "Repeat1 reuse count differs")


def verify_completed(suite, task, state=None):
    """Validate exact planned raw/effective identity and all completion-time bytes."""
    suite = Path(suite).resolve()
    state = read(suite / "manifest.json") if state is None else state
    task_id = task["task_id"]
    folder = (suite / "tasks" / task_id).resolve()
    require(suite / "tasks" in folder.parents, "Task path escapes suite")
    record = state["tasks"][task_id]
    require(record.get("status") == "complete" and record.get("task_sha256") == stable(task), "Completed task specification changed: " + task_id)
    files = record.get("verified_file_sha256", {})
    required = {"config.json", "splits.json", "true_label.npy", "pred_label.npy", "model.pth", "manifest.json", "logging.txt", "metrics.json", "metrics_per_battery.csv"}
    if task["family"] == "transfer":
        required.update(["source_only_true_label.npy", "source_only_pred_label.npy", "source_only_metrics_per_battery.csv", "parameters.json"])
    require(required <= set(files), "Missing completion-time hashes: " + task_id)
    for name, checksum in files.items():
        path = (folder / name).resolve()
        require(folder in path.parents and path.is_file() and digest(path) == checksum, "Completion-time artifact changed: " + task_id + "/" + name)
    manifest, config = read(folder / "manifest.json"), read(folder / "config.json")
    require(manifest.get("task") == task and manifest.get("task_sha256") == stable(task), "Worker raw task changed: " + task_id)
    require(config.get("task") == effective_task(task) and manifest.get("effective_task", config.get("task")) == effective_task(task), "Worker effective task changed: " + task_id)
    require(digest(folder / "manifest.json") == record.get("worker_manifest_sha256"), "Worker manifest changed: " + task_id)
    frozen = state["frozen"]
    require(manifest.get("source_fingerprint") == config.get("source_fingerprint") == frozen["source_fingerprint"], "Worker source fingerprint changed")
    require(manifest.get("implementation_sha256") == implementation(task, frozen), "Worker implementation changed")
    environment = config.get("environment", {})
    aliases = {"scikit_learn": "sklearn", "cuda_runtime": "cuda"}
    normalized = {aliases.get(key, key): value for key, value in environment.items() if aliases.get(key, key) in runtime_signature(frozen)}
    require(set(normalized) >= {"python", "torch", "numpy", "pandas", "sklearn", "cuda", "cuda_available", "gpu"} and
            all(runtime_signature(frozen)[key] == value for key, value in normalized.items()), "Worker runtime differs from frozen cohort")
    order = config.get("data_directory_order_sha256") if task["runner"] == "extended" else stable(config.get("source_directory_order"))
    require(order == frozen["data_directory_order_sha256"], "Worker data directory order changed")
    if task["family"] == "transfer":
        parameters = read(folder / "parameters.json")
        logged = dict(re.findall(r" - CRITICAL - \t([^:\r\n]+):([^\r\n]*)", (folder / "logging.txt").read_text(encoding="utf-8")))
        require(all(logged.get(key) == str(value) for key, value in parameters.items()), "Transfer parameters differ from logged training arguments")
        require(all(parameters.get(key) == value for key, value in task.get("overrides", {}).items()), "Transfer parameters differ from frozen overrides")
        for key in ["adaptation_lr", "adaptation_epochs", "source_data", "source_batch", "target_data", "target_batch"]:
            task_key = {"source_data": "source", "target_data": "target"}.get(key, key)
            require(parameters.get(key) == task.get(task_key), "Transfer parameter differs from frozen task: " + key)
        require(parameters.get("pretrain_model") == task.get("source_checkpoint"), "Transfer parameter source checkpoint differs")
    return folder, record, manifest, config


def stem(task_id):
    parent, repeat = task_id.rsplit("/Experiment", 1)
    require(repeat.isdigit() and 1 <= int(repeat) <= 10, "Noncanonical experiment identity")
    return parent


def absolute_cloud(value):
    path = PurePosixPath(value)
    require(path.is_absolute() and str(path) == value and ".." not in path.parts and "\\" not in value,
            "Expected canonical absolute Linux path")
    return path


def build(selection, control, stage1, stage2, cloud_suite):
    """Pure, testable exact-catalog selection and same-repeat dependency closure."""
    cloud = absolute_cloud(cloud_suite)
    old_cloud = absolute_cloud(control["expected_suite_output"])
    require(cloud != old_cloud and old_cloud not in cloud.parents and cloud not in old_cloud.parents,
            "Extension suite must be separate from frozen control")
    require(selection.get("format") == SELECTION_FORMAT, "Unsupported explicit selection manifest")
    require(set(selection) == {"format", "reason", "regular_groups", "transfer_repeat1_task_ids", "evidence_refs"},
            "Unexpected selection fields")
    require(isinstance(selection["reason"], str) and selection["reason"].strip(), "Selection requires a review reason")
    require(isinstance(selection["evidence_refs"], list) and selection["evidence_refs"] and
            all(isinstance(item, str) and item.strip() for item in selection["evidence_refs"]), "Explicit evidence references required")
    require(isinstance(selection["regular_groups"], list) and isinstance(selection["transfer_repeat1_task_ids"], list), "Selections must be lists")
    require(control.get("stage") == "legacy_matched_formal_control" and len(control["tasks"]) == 40, "Expected frozen 40-task catalog")
    controls = {task["task_id"]: task for task in control["tasks"]}
    require(len(controls) == 40 and all(task["repeat"] == 1 for task in controls.values()), "Control catalog identity invalid")
    catalog, modern = {}, {}
    for task in stage1["tasks"] + stage2["tasks"]:
        require(task["task_id"] not in modern, "Duplicate modern task identity")
        modern[task["task_id"]] = task
    for task in controls.values():
        key = stem(task["original_task_id"])
        require(key not in catalog, "Duplicate control method/direction")
        repeats = [modern.get(key + "/Experiment%02d" % repeat) for repeat in range(1, 11)]
        require(all(repeats) and [row["repeat"] for row in repeats] == list(range(1, 11)), "Missing exact modern repeat catalog")
        require(len({row["seed"] for row in repeats}) == 10, "Repeated modern seeds")
        catalog[key] = {"control": task, "modern": repeats}
    chosen, reasons = set(), {}
    valid_groups = {(task["dataset"], str(task["batch"])) for task in controls.values() if task["family"] == "full"}
    groups = []
    for item in selection["regular_groups"]:
        require(isinstance(item, dict) and set(item) == {"dataset", "batch"}, "Regular group must name dataset/batch exactly")
        group = (item["dataset"], str(item["batch"]))
        require(group in valid_groups and group not in groups, "Unknown/duplicate selected regular group")
        groups.append(group)
        for key, value in catalog.items():
            task = value["control"]
            if task["family"] == "full" and (task["dataset"], str(task["batch"])) == group:
                chosen.add(key)
                reasons[key] = "explicit_regular_group"
    selected_transfer_ids = selection["transfer_repeat1_task_ids"]
    require(len(selected_transfer_ids) == len(set(selected_transfer_ids)), "Duplicate selected transfer direction")
    for task_id in selected_transfer_ids:
        task = controls.get(task_id)
        require(task and task["family"] == "transfer" and task["source_mode"] == "reproduced_checkpoint",
                "Unknown/off-catalog/non-reproduced selected transfer task")
        key = stem(task["original_task_id"])
        chosen.add(key)
        reasons[key] = "explicit_transfer_direction_and_loss"
    require(chosen, "Empty selection does not authorize an extension")
    # Add only necessary same-loss source PINNs, not arbitrary extra groups.
    for key in list(chosen):
        for original in catalog[key]["modern"]:
            if original["family"] != "transfer":
                continue
            source_task_id = original["source_run"].split("/tasks/", 1)[1]
            source = modern.get(source_task_id)
            require(source and source["family"] == "full" and source["model"] == "PINN", "Missing source PINN")
            require(source["repeat"] == original["repeat"] and source["loss_variant"] == original["loss_variant"],
                    "Source repeat/loss does not match transfer")
            require(original.get("source_selection") == SOURCE_POLICY, "Source selection policy changed")
            require(source["dataset"] == original["source"], "Wrong source dataset")
            expected_batch = ["2C", "3C", "R2.5", "R3", "RW", "satellite"][original["source_batch"]] if original["source"] == "XJTU" else str(original["source_batch"]) if original["source"] == "TJU" else "all"
            require(str(source["batch"]) == expected_batch, "Wrong source batch")
            source_key = stem(source_task_id)
            require(source_key in catalog, "Source dependency outside bounded 40-task catalog")
            chosen.add(source_key)
            reasons.setdefault(source_key, "same_repeat_transfer_source_dependency")
    new_tasks, analysis = [], []
    for key in sorted(chosen):
        entry = catalog[key]
        for repeat, original in enumerate(entry["modern"], 1):
            expected = deepcopy(original)
            if repeat == 1:
                legacy = deepcopy(entry["control"])
                folder = old_cloud / "tasks" / legacy["task_id"]
                storage = "immutable_original_repeat1"
            else:
                legacy = deepcopy(original)
                legacy.update(task_id="legacy_extension/" + original["task_id"], original_task_id=original["task_id"],
                              runtime_control="readme_versions_legacy_vs_current", paper_coverage=[])
                legacy.setdefault("protocol_evidence", []).append("Selected ten-repeat runtime extension; exact same-index modern seed/split; repeat1 reused in original frozen directory")
                if legacy["family"] == "transfer":
                    source_original_id = original["source_run"].split("/tasks/", 1)[1]
                    source_id = "legacy_extension/" + source_original_id
                    source_run = str(cloud / "tasks" / source_id)
                    legacy.update(original_source_run=original["source_run"], source_run=source_run,
                                  source_checkpoint=source_run + "/model.pth", source_manifest=source_run + "/manifest.json",
                                  depends_on=[source_id])
                new_tasks.append(legacy)
                folder = cloud / "tasks" / legacy["task_id"]
                storage = "new_repeat2_to10"
            require(legacy["repeat"] == original["repeat"] and legacy["seed"] == original["seed"], "Seed/repeat changed")
            require(legacy.get("epochs") == 200 and not legacy.get("smoke") and legacy.get("threads") == 2, "Unexpected training budget")
            analysis.append({"logical_group": key, "selection_role": reasons[key], "storage": storage,
                "repeat": repeat, "seed": original["seed"], "modern_task_id": original["task_id"],
                "legacy_task_id": legacy["task_id"], "legacy_folder": str(folder), "legacy_task": legacy,
                "modern_task": expected})
    require(0 < len(new_tasks) <= 360 and len(analysis) <= 400, "Extension exceeds bounded 40x10 catalog")
    lookup = {task["task_id"]: task for task in new_tasks}
    for task in new_tasks:
        for dependency in task.get("depends_on", []):
            require(dependency in lookup and lookup[dependency]["repeat"] == task["repeat"], "Broken same-repeat source DAG")
    new_tasks.sort(key=lambda task: (task["runner"] == "transfer", task["repeat"], task["task_id"]))
    plan = {"schema": stage1["schema"], "stage": FORMAT, "diagnostic_only": True, "training_launched": False,
            "expected_suite_output": str(cloud), "selection_sha256": stable(selection), "tasks": new_tasks}
    return plan, analysis


def verify_reused(analysis, suite, require_complete=True):
    """Hash original repeat1 evidence without rewriting any metadata or lineage."""
    suite = Path(suite).resolve()
    state = read(suite / "manifest.json")
    if require_complete:
        require(state.get("status") == "complete" and len(state["tasks"]) == 40 and
                all(row.get("status") == "complete" for row in state["tasks"].values()), "40-task screening suite not complete")
    result = []
    for item in analysis:
        if item["repeat"] != 1:
            continue
        task_id = item["legacy_task_id"]
        folder, record, manifest, config = verify_completed(suite, item["legacy_task"], state)
        # Keep original logical cloud identity so byte-exact local recovery audits
        # do not fail merely because the recovery directory has another name.
        result.append({"legacy_task_id": task_id, "original_folder": item["legacy_folder"], "files": record["verified_file_sha256"],
                       "worker_manifest_sha256": record["worker_manifest_sha256"]})
    return {"control_suite_manifest_sha256": digest(suite / "manifest.json"),
            "control_frozen": state["frozen"], "reused_tasks": result}


def verify_screen_binding(screen, report_path, suite, control):
    state = read(Path(suite) / "manifest.json")
    require(state["frozen"]["plan_sha256"] == stable(control) and
            set(state["tasks"]) == {task["task_id"] for task in control["tasks"]}, "Screened suite does not match frozen40 plan")
    # The immutable original40 report does not hash its suite manifest. Bind its
    # own artifact inventory and per-task observed config/artifact hashes to this
    # exact suite instead of modifying that completed report schema.
    report_path = Path(report_path)
    for name, checksum in screen.get("artifacts", {}).items():
        path = (report_path.parent / name).resolve()
        require(report_path.parent.resolve() in path.parents and digest(path) == checksum, "Screen report artifact changed")
    evidence = read(report_path.parent / "comparison_evidence.json")
    legacy = {row["task_id"]: row for row in evidence if row.get("stack") == "legacy"}
    require(len(legacy) == 40 and set(legacy) == set(state["tasks"]), "Screen report lacks exact40 legacy evidence")
    for task in control["tasks"]:
        _, _, manifest, config = verify_completed(suite, task, state)
        require(legacy[task["task_id"]]["config"] == config and legacy[task["task_id"]]["artifact_sha256"] == manifest["artifact_sha256"],
                "Screen report evidence differs from completed original task")


def prepare(selection_path, control_plan, stage1_plan, stage2_plan, screening_report, legacy_control_suite, output, cloud_suite):
    output = Path(output).resolve()
    require(not output.exists() and output.parent.is_dir(), "Prepared support path must be fresh with existing parent")
    paths = {"selection": Path(selection_path), "control_plan": Path(control_plan), "stage1_plan": Path(stage1_plan),
             "stage2_plan": Path(stage2_plan), "screening_report": Path(screening_report)}
    require(digest(paths["control_plan"]) == CONTROL_SHA and digest(paths["stage1_plan"]) == STAGE1_SHA and
            digest(paths["stage2_plan"]) == STAGE2_SHA, "Frozen screening/modern plan identity differs")
    screen = read(paths["screening_report"])
    require(screen.get("task_counts") == {"matched": 40} and not screen.get("errors") and
            screen.get("input_plan_sha256") == CONTROL_SHA, "Need complete matched 40-task screen before selecting extension")
    selection = read(paths["selection"])
    control = read(paths["control_plan"])
    verify_screen_binding(screen, paths["screening_report"], legacy_control_suite, control)
    plan, analysis = build(selection, control, read(paths["stage1_plan"]), read(paths["stage2_plan"]), cloud_suite)
    reuse = verify_reused(analysis, legacy_control_suite)
    # Bind all available 40-screen result tables so row references remain reviewable.
    for path in paths["screening_report"].parent.glob("*.csv"):
        paths["screening_table:" + path.name] = path
    output.mkdir()
    write(output / "extension_plan.json", plan)
    write(output / "extension_analysis_manifest.json", {"format": FORMAT, "selection": selection,
        "new_suite": cloud_suite, "repeat1_suite": str(PurePosixPath(read(control_plan)["expected_suite_output"])),
        "tasks": analysis, "reuse_evidence": reuse})
    write(output / "extension_design.json", {"format": FORMAT, "training_launched": False,
        "new_task_count": len(plan["tasks"]), "analysis_task_count": len(analysis),
        "reused_repeat1_count": len(reuse["reused_tasks"]), "group_count": len(analysis) // 10,
        "new_repeats": list(range(2, 11)), "analysis_repeats": list(range(1, 11)),
        "new_family_counts": dict(Counter(task["family"] for task in plan["tasks"])),
        "selection": selection, "deployment": {"gpus": ["0"], "workers_per_gpu": 1, "threads": 2},
        "plan_sha256": digest(output / "extension_plan.json"),
        "analysis_manifest_sha256": digest(output / "extension_analysis_manifest.json"),
        "inputs": {key: {"path": str(path.resolve()), "sha256": digest(path)} for key, path in paths.items()},
        "script_sha256": digest(__file__),
        "repeat1_policy": "Read immutable original paths and completion-time hashes; do not copy, rebase metadata, or retrain",
        "conclusion_limit": "Post-screen selected sensitivity analysis; no full2120 environment-independence or unplanned equivalence/significance assertion"})
    return read(output / "extension_design.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["selection-path", "control-plan", "stage1-plan", "stage2-plan", "screening-report", "legacy-control-suite", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cloud-suite", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(**vars(args)), indent=2))
