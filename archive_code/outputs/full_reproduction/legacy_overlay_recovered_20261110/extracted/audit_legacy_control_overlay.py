"""Additional read-only identity/lineage gates for the frozen 40-task comparator.

This stdlib-only overlay never changes frozen comparisons, models, or training.
Missing evidence stays pending, genuine mismatches fail, and no failure retrains.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def canonical_input(value):
    value = value.replace("\\", "/")
    if value.startswith("data/"):
        return value
    if "/data/" in value:
        return "data/" + value.split("/data/", 1)[1]
    raise ValueError("Unrecognized input path: " + value)


def locate(suites, task_id):
    values = [root / "tasks" / task_id for root in suites
              if (root / "tasks" / task_id / "manifest.json").is_file()]
    if len(values) > 1:
        raise ValueError("Ambiguous task identity: " + task_id)
    return values[0] if values else None


def environment_checks(config, stack):
    environment = config["environment"]
    expected = {"legacy": {"python":"3.7.10", "torch":"1.7.1+cu110", "numpy":"1.20.3",
        "pandas":"1.3.5", "sklearn":"0.24.2", "cuda":"11.0"},
        "current":{"python":"3.10.13", "torch":"1.13.1+cu116", "numpy":"1.24.4",
        "pandas":"1.5.3", "sklearn":"1.2.2", "cuda":"11.6"}}[stack]
    normalized = dict(environment, sklearn=environment.get("sklearn", environment.get("scikit_learn")),
        cuda=environment.get("cuda", environment.get("cuda_runtime")))
    issues = ["different_"+stack+"_"+key for key,value in expected.items()
              if (not str(normalized.get(key, "")).startswith(value+" ") and normalized.get(key) != value)]
    if environment.get("cuda_available") is not True or "3090" not in environment.get("gpu", ""):
        issues.append("unexpected_recorded_gpu")
    if config["task"].get("threads") != 2 or environment.get("threads", 2) != 2:
        issues.append("unexpected_recorded_thread_count")
    return issues


def inspect(folder, plan_task, stack, repo, compatibility, cache):
    config, manifest = read(folder / "config.json"), read(folder / "manifest.json")
    problems, pending, facts = [], [], {}
    task = config["task"]
    expected_id = plan_task["task_id"] if stack == "legacy" else plan_task["original_task_id"]
    if manifest.get("status") != "complete":
        return dict(status="pending", problems=[], pending=["task_not_complete"], facts={})
    keys = ["family","dataset","batch","model","repeat","seed","epochs","threads","protocol_id",
            "loss_variant","validation_protocol","source","source_batch","target","target_batch",
            "adaptation_lr","adaptation_epochs","sample_size","source_mode","transfer_design","overrides"]
    for key in keys:
        default = {} if key == "overrides" else None
        if task.get(key, default) != plan_task.get(key, default):
            problems.append("config_vs_plan_"+key)
    if task.get("task_id") != expected_id or manifest.get("task_id") != expected_id:
        problems.append("task_identity_conflict")
    effective = manifest.get("effective_task")
    if effective is not None and effective != task:
        problems.append("effective_task_vs_config_conflict")
    raw = manifest.get("task", {})
    for key in keys+["task_id"]:
        default = {} if key == "overrides" else None
        if raw.get(key, default) != task.get(key, default):
            problems.append("manifest_task_vs_config_"+key)
    if stack == "legacy" and task.get("original_task_id") != plan_task["original_task_id"]:
        problems.append("legacy_original_identity_conflict")
    for name, checksum in manifest.get("artifact_sha256", {}).items():
        path = folder / name
        if not path.is_file() or digest(path) != checksum:
            problems.append("artifact_hash_"+name)
    for name in ["config.json","splits.json","model.pth","true_label.npy","pred_label.npy"]:
        if name not in manifest.get("artifact_sha256", {}):
            problems.append("missing_artifact_hash_"+name)
    expected_config = dict(config)
    if task["family"] == "transfer":
        expected_config = json.loads(json.dumps(config))
        expected_config["environment"].pop("CUDA_VISIBLE_DEVICES", None)
    if manifest.get("config_hash") != stable(expected_config):
        problems.append("config_identity_hash_conflict")
    problems.extend(environment_checks(config, stack))
    actual_implementation = manifest.get("implementation_sha256", {})
    for name,checksum in actual_implementation.items():
        source_name = "run_reproduction.py" if name == "helpers" else name
        approved = compatibility["runner_files"].get(source_name)
        if not approved or checksum != approved["compat_sha256" if stack == "legacy" else "original_sha256"]:
            problems.append("unapproved_implementation_"+name)
    inventory = read(folder / "source_inventory.json")
    if stable(inventory.get("sha256", {})) != config.get("source_fingerprint"):
        problems.append("inventory_fingerprint_conflict")
    for name,checksum in inventory.get("sha256",{}).items():
        if not (name.endswith(".py") or name in {"README.md","fix.md"}):
            continue
        path=(repo / name).resolve()
        if repo not in path.parents:
            problems.append("source_escaped_repository")
            continue
        if str(path) not in cache:
            cache[str(path)]=digest(path)
        if cache[str(path)]!=checksum:
            problems.append("author_source_vs_inventory_"+name)
    facts["inventory_sha256"] = digest(folder / "source_inventory.json")
    facts["source_fingerprint"] = config.get("source_fingerprint")
    splits = read(folder / "splits.json")
    input_hashes = {}
    for split in splits:
        for name in split["paths"]:
            key = canonical_input(name)
            path = (repo / key).resolve()
            if repo not in path.parents:
                problems.append("input_escaped_repository")
                continue
            if str(path) not in cache:
                cache[str(path)] = digest(path)
            input_hashes[key] = cache[str(path)]
            if inventory.get("sha256", {}).get(key) != input_hashes[key]:
                problems.append("input_vs_inventory_"+key)
    facts["input_file_sha256"] = input_hashes
    identity_path = folder / "data_identity.json"
    if identity_path.is_file():
        identity = read(identity_path)
        if manifest.get("artifact_sha256", {}).get("data_identity.json") != digest(identity_path):
            problems.append("data_identity_artifact_hash_missing_or_changed")
        recorded = {canonical_input(name):checksum for name,checksum in identity["input_file_sha256"].items()}
        if recorded != input_hashes:
            problems.append("training_input_file_hashes_conflict")
        facts.update(tensor_sha256=identity["tensor_sha256"], sample_counts=identity["sample_counts"],
                     training_time_tensor_identity=True)
    else:
        facts["training_time_tensor_identity"] = False
        pending.append("read_only_tensor_reconstruction_required")
    if task.get("model") == "PINN" and task["family"] == "full":
        expected_loss = {"published_label_monotonicity":"relu(y2-y1).sum()",
                         "paper_prediction_monotonicity":"relu(u2-u1).sum()"}[task["loss_variant"]]
        if config.get("physical_loss", {}).get("expression") != expected_loss:
            problems.append("physical_loss_expression_conflict")
    if task["family"] == "transfer":
        expected_gradient="identically_zero" if task["loss_variant"]=="published_label_monotonicity" else "generally_nonzero"
        if config.get("physical_loss_parameter_gradient")!=expected_gradient:
            problems.append("transfer_physical_loss_gradient_record_conflict")
    log = (folder / "logging.txt").read_text(encoding="utf-8")
    if task["family"]=="transfer":
        parameters=read(folder / "parameters.json")
        logged=dict(re.findall(r" - CRITICAL - \t([^:\r\n]+):([^\r\n]*)",log))
        for key,value in parameters.items():
            if logged.get(key)!=str(value):
                problems.append("transfer_parameter_vs_logged_argument_"+key)
        for key,value in task.get("overrides",{}).items():
            if parameters.get(key)!=value:
                problems.append("transfer_effective_parameter_vs_override_"+key)
        for key in ["adaptation_lr","adaptation_epochs","source_data","source_batch","target_data","target_batch"]:
            task_key={"source_data":"source","target_data":"target"}.get(key,key)
            if parameters.get(key)!=task.get(task_key):
                problems.append("transfer_effective_parameter_vs_task_"+key)
        if parameters.get("pretrain_model")!=task.get("source_checkpoint"):
            problems.append("transfer_parameter_source_checkpoint_conflict")
        facts.update(parameters_sha256=digest(folder / "parameters.json"),
                     logging_sha256=digest(folder / "logging.txt"),
                     note_parameter_provenance="parameters.json is not artifact-hashed by original transfer runner; all fields independently cross-checked against logged Args and frozen task")
    validations = [(int(e),float(v)) for e,v in re.findall(r"\[Valid\] epoch:(\d+), MSE: ([^\s]+)", log)]
    facts["printed_validation_min_epoch"] = min(validations,key=lambda row:row[1])[0] if validations else None
    facts["declared_best_epoch"] = manifest.get("best_epoch")
    # Printed MSE is rounded. Preserve disagreement descriptively; it is not a checkpoint-selection proof.
    facts["printed_vs_declared_best_epoch_agree"] = (facts["printed_validation_min_epoch"] == manifest.get("best_epoch"))
    facts["config_sha256"] = digest(folder / "config.json")
    facts["splits_sha256"] = digest(folder / "splits.json")
    return dict(status="failed" if problems else "pending" if pending else "passed", problems=problems,
                pending=pending, facts=facts)


def lineage_check(folder, plan_task, stack, suites, plan_lookup):
    config, manifest = read(folder / "config.json"), read(folder / "manifest.json")
    lineage = manifest.get("source_checkpoint_lineage") or {}
    issues, facts = [], {}
    if lineage != config.get("source_checkpoint_lineage"):
        issues.append("manifest_vs_config_lineage_conflict")
    if "/tasks/" not in plan_task["source_run"]:
        return ["unrecognized_planned_source_path"], facts
    legacy_source_id = plan_task["source_run"].split("/tasks/",1)[1]
    source_plan = plan_lookup.get(legacy_source_id)
    if not source_plan:
        return ["source_not_in_frozen_control_closure"], facts
    source_id = legacy_source_id if stack == "legacy" else source_plan["original_task_id"]
    source_folder = locate(suites, source_id)
    if source_folder is None or read(source_folder / "manifest.json").get("status") != "complete":
        return ["source_missing_or_incomplete"], facts
    source_config, source_manifest = read(source_folder / "config.json"), read(source_folder / "manifest.json")
    source_task = source_config["task"]
    for name,lineage_key in [("model.pth","sha256"),("config.json","source_config_sha256"),
            ("splits.json","source_splits_sha256"),("manifest.json","source_manifest_sha256")]:
        if digest(source_folder / name) != lineage.get(lineage_key):
            issues.append("source_lineage_rehash_"+name)
    source_path = lineage.get("source_run", "")
    if "/tasks/" not in source_path or source_path.split("/tasks/",1)[1] != source_id:
        issues.append("source_stack_or_task_identity_conflict")
    if lineage.get("path") != source_path + "/model.pth" or task_path(config["task"].get("source_checkpoint")) != task_path(lineage.get("path")):
        issues.append("source_checkpoint_path_conflict")
    if source_task.get("repeat") != plan_task["repeat"] or source_task.get("seed") != source_plan["seed"]:
        issues.append("source_repeat_or_prespecified_seed_conflict")
    if lineage.get("training_seed") != source_task.get("seed"):
        issues.append("source_recorded_training_seed_conflict")
    if source_task.get("dataset") != plan_task["source"] or source_task.get("loss_variant") != plan_task["loss_variant"]:
        issues.append("source_dataset_or_loss_conflict")
    expected_batch=(["2C","3C","R2.5","R3","RW","satellite"][plan_task["source_batch"]]
        if plan_task["source"]=="XJTU" else str(plan_task["source_batch"]) if plan_task["source"]=="TJU" else "all")
    if source_task.get("batch")!=expected_batch or source_task.get("model")!="PINN":
        issues.append("source_batch_or_model_conflict")
    expected_run=plan_task["source_run"] if stack=="legacy" else plan_task["original_source_run"]
    if config["task"].get("source_run")!=expected_run or source_path!=expected_run:
        issues.append("source_logical_suite_path_conflict")
    if source_config.get("environment") != config.get("environment"):
        # Source extended and target transfer runners use distinct key schemas and CUDA visibility.
        issues.extend("source_"+issue for issue in environment_checks(source_config,stack))
    if lineage.get("source_paths") != read(source_folder / "splits.json"):
        issues.append("source_retained_split_record_conflict")
    if (lineage.get("training_history_verified") is not True or lineage.get("source_selection") !=
            "Prespecified same-index independent source repeat; no target-test selection"):
        issues.append("source_selection_or_history_unverified")
    verification = source_manifest.get("checkpoint_verification")
    if verification is None and (source_folder / "verification_checkpoint.json").is_file():
        verification=read(source_folder / "verification_checkpoint.json")
    if not verification or verification.get("passed") is not True:
        issues.append("source_checkpoint_reload_unverified")
    facts.update(source_id=source_id, source_repeat=source_task.get("repeat"), source_seed=source_task.get("seed"),
        source_folder=str(source_folder), checkpoint_sha256=digest(source_folder / "model.pth"),
        note="Source dataset seed offsets are preserved; source seed is compared to its own frozen source task")
    return issues, facts


def task_path(value):
    return str(value or "").replace("\\", "/").rstrip("/")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--design",type=Path,required=True)
    parser.add_argument("--compatibility",type=Path,required=True)
    parser.add_argument("--legacy-suite",type=Path,required=True)
    parser.add_argument("--current-suite",type=Path,action="append",required=True)
    parser.add_argument("--repo",type=Path,required=True)
    parser.add_argument("--comparison",type=Path,help="Frozen comparator output directory, when available")
    parser.add_argument("--legacy-reconstruction",type=Path)
    parser.add_argument("--current-reconstruction",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    plan,design,compatibility=read(args.plan),read(args.design),read(args.compatibility)
    if digest(args.plan)!=design["plan_sha256"] or len(plan["tasks"])!=40:
        raise ValueError("Plan differs from frozen design")
    if args.output.exists():
        raise ValueError("Use a fresh overlay evidence file")
    suites={"legacy":[args.legacy_suite.resolve()],"current":[p.resolve() for p in args.current_suite]}
    if any(root in args.output.resolve().parents for root in [args.repo.resolve()]+suites["legacy"]+suites["current"]):
        raise ValueError("Overlay output must be outside source/suites")
    reconstruction={}
    for stack,path in [("legacy",args.legacy_reconstruction),("current",args.current_reconstruction)]:
        if path:
            value=read(path)
            if (value.get("plan_sha256")!=digest(args.plan) or value.get("stack")!=stack
                    or value.get("training_performed") is not False or value.get("checkpoint_loaded") is not False
                    or value.get("script_sha256")!=digest(Path(__file__).with_name("reconstruct_control_data_identity.py"))
                    or value.get("loader_sha256")!=digest(args.repo / "dataloader/dataloader.py")):
                raise ValueError("Reconstruction provenance mismatch")
            reconstruction_environment=value.get("environment",{})
            reconstruction_config={"environment":dict(reconstruction_environment,
                cuda="11.0" if stack=="legacy" else "11.6",cuda_available=True,gpu="NVIDIA GeForce RTX 3090",
                threads=2),"task":{"threads":2}}
            if environment_checks(reconstruction_config,stack):
                raise ValueError("Reconstruction packages differ from the requested software stack")
            reconstruction[stack]={r["task_id"]:r for r in value["tasks"]}
    rows,cache=[],{}
    lookup={t["task_id"]:t for t in plan["tasks"]}
    comparator={}
    if args.comparison:
        comparison_summary=read(args.comparison / "summary.json")
        if (comparison_summary.get("script_sha256")!=digest(Path(__file__).with_name("compare_legacy_control.py"))
                or comparison_summary.get("analyzer_sha256")!=digest(Path(__file__).with_name("build_suite_report.py"))
                or comparison_summary.get("independent_array_recomputation") is not True
                or comparison_summary.get("expected_tasks")!=40):
            raise ValueError("Comparator/analyzer provenance mismatch")
        for name,checksum in comparison_summary.get("artifacts",{}).items():
            if digest(args.comparison / name)!=checksum:
                raise ValueError("Comparator artifact changed: "+name)
        with (args.comparison / "task_matching.csv").open(encoding="utf-8-sig",newline="") as stream:
            comparator={r["task_id"]:r for r in csv.DictReader(stream)}
        if comparison_summary["input_plan_sha256"] != digest(args.plan):
            raise ValueError("Comparator used a different plan")
    for task in plan["tasks"]:
        row=dict(task_id=task["task_id"],original_task_id=task["original_task_id"],family=task["family"],stacks={})
        for stack in ["legacy","current"]:
            identity=task["task_id"] if stack=="legacy" else task["original_task_id"]
            try:
                folder=locate(suites[stack],identity)
                if folder is None:
                    result=dict(status="pending",problems=[],pending=["task_missing"],facts={})
                else:
                    result=inspect(folder,task,stack,args.repo.resolve(),compatibility,cache)
                    if result["status"]!="pending" or read(folder / "manifest.json").get("status")=="complete":
                        if task["family"]=="transfer":
                            issues,facts=lineage_check(folder,task,stack,suites[stack],lookup)
                            result["problems"].extend(issues)
                            result["facts"]["source_lineage"]=facts
                        record=reconstruction.get(stack,{}).get(identity)
                        if record:
                            if record.get("status")!="verified" or any(record.get(key)!=result["facts"].get(key)
                                    for key in ["config_sha256","splits_sha256"]):
                                result["problems"].append("reconstruction_failed_or_wrong_artifacts")
                            elif record.get("input_file_sha256")!=result["facts"]["input_file_sha256"]:
                                result["problems"].append("reconstructed_input_hashes_conflict")
                            elif result["facts"].get("tensor_sha256") and record["tensor_sha256"]!=result["facts"]["tensor_sha256"]:
                                result["problems"].append("reconstructed_vs_training_tensor_conflict")
                            else:
                                result["facts"].update(tensor_sha256=record["tensor_sha256"],sample_counts=record["sample_counts"],
                                    reconstructed_tensor_identity=True)
                                result["pending"]=[p for p in result["pending"] if p!="read_only_tensor_reconstruction_required"]
                        result["status"]="failed" if result["problems"] else "pending" if result["pending"] else "passed"
            except Exception as error:
                result=dict(status="failed",problems=[repr(error)],pending=[],facts={})
            row["stacks"][stack]=result
        pair_issues=[]
        left,right=[row["stacks"][s]["facts"] for s in ["legacy","current"]]
        for key in ["input_file_sha256","tensor_sha256","sample_counts","source_fingerprint"]:
            if key in left and key in right and left[key]!=right[key]:
                pair_issues.append("different_"+key)
        row["pair_issues"]=pair_issues
        row["comparator_status"]=comparator.get(task["task_id"],{}).get("status")
        statuses=[r["status"] for r in row["stacks"].values()]
        row["status"]="failed" if pair_issues or "failed" in statuses else "pending" if "pending" in statuses else "passed"
        if args.comparison and row["comparator_status"]!="matched" and row["status"]=="passed":
            row["status"]="pending"
            row["pair_issues"].append("frozen_comparator_not_matched")
        rows.append(row)
    document=dict(format="pinn4soh-legacy-control-overlay-v1",created_at=datetime.now(timezone.utc).isoformat(),
        script_sha256=digest(__file__),plan_sha256=digest(args.plan),design_sha256=digest(args.design),
        compatibility_sha256=digest(args.compatibility),task_counts=dict(Counter(r["status"] for r in rows)),
        assessment=("passed_all40" if args.comparison else "identity_evidence_only_no_comparator")
            if all(r["status"]=="passed" for r in rows) else "pending_or_failed",tasks=rows,
        training_performed=False,checkpoint_loaded=False,source_modified=False,
        interpretation="Additional gate only. The frozen comparator and all outputs remain unchanged. Any failure requires audit, never automatic retraining.",
        limits=["Reconstruction of transfer tensors is retrospective; exact training-time tensor hashes were not recorded by its runner",
            "Recorded worker threads and exact package versions are checked; actual-process backend flags require the separate runtime observation sidecars",
            "Printed validation MSE is rounded and cannot independently identify an unrounded minimum",
            "No statistical significance, equivalence margin or environment independence is inferred"])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(document,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({k:document[k] for k in ["assessment","task_counts"]}))


if __name__=="__main__":
    main()
