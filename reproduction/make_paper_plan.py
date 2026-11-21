"""Freeze the full main/supplement training matrix without importing torch."""

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

from run_transfer import transfer_configurations, source_checkpoint_name


BATCHES = {"XJTU": ["2C", "3C", "R2.5", "R3", "RW", "satellite"],
           "MIT": ["all"], "HUST": ["all"], "TJU": ["0", "1", "2"]}
LANES = {"published_code": "published_label_monotonicity",
         "paper_equation": "paper_prediction_monotonicity"}
WEIGHTS = {"XJTU": (.7, 20), "MIT": (1, 50), "HUST": (.5, 80), "TJU": (1, 50)}
HISTORY = "Publication-era commit d26bc1f40dc0cdf34405cd5e599a6f6af09742e9"


def seed_for(dataset, batch, repeat):
    return 20261109 + list(BATCHES).index(dataset) * 10000 + BATCHES[dataset].index(batch) * 100 + repeat - 1


def reference_run(dataset, batch, repeat, reference):
    return f"{reference}/{dataset}/{batch}/Experiment{repeat:02d}"


def extended_task(dataset, batch, model, lane, repeat, family, reference, size=None):
    identifier = f"{family}/{lane}/{dataset}/{batch}/{model}"
    if size is not None:
        identifier += f"/n{size}"
    identifier += f"/Experiment{repeat:02d}"
    task = dict(task_id=identifier, runner="extended", family=family,
                dataset=dataset, batch=batch, model=model, repeat=repeat,
                seed=seed_for(dataset, batch, repeat), epochs=200, threads=2,
                protocol_id="author_current", validation_protocol="author_sample_split",
                reference_run=reference_run(dataset, batch, repeat, reference),
                loss_variant=LANES[lane] if model == "PINN" else "author_current",
                protocol_evidence=[HISTORY, "Supplementary Note 3; peer review pp29-30 sample validation clarification"],
                paper_coverage=["Table 2", "Fig. 4", "Tables S3-S6"] if family == "full" else ["Table 3", "Fig. 5", "Table S7", "Fig. S5"])
    if size is not None:
        task["sample_size"] = size
    if model == "PINN":
        alpha, beta = WEIGHTS[dataset]
        if size is not None and dataset == "XJTU":
            alpha, beta = .5, 10
        task["overrides"] = dict(alpha=alpha, beta=beta,
                                 batch_size=128 if size is not None else 256 if dataset == "XJTU" else 512,
                                 early_stop=10 if dataset == "MIT" else 20)
        task["protocol_evidence"].append("Original main_*.py defaults; small_sample() batch128 and XJTU alpha0.5/beta10")
    else:
        task["protocol_evidence"].append("Original main_comparision.py defaults batch512; hardcoded patience10; corresponding dataset loader adaptation")
    return task


def make_stage_one(reference):
    tasks = []
    for dataset, batches in BATCHES.items():
        for batch in batches:
            for model, lane in [("PINN", "published_code"), ("PINN", "paper_equation"),
                                ("MLP", "shared_baseline"), ("CNN", "shared_baseline")]:
                for repeat in range(1, 11):
                    tasks.append(extended_task(dataset, batch, model, lane, repeat, "full", reference))
    for dataset, batches in {"XJTU": BATCHES["XJTU"], "HUST": ["all"]}.items():
        for batch in batches:
            for size in range(1, 5):
                for model, lane in [("PINN", "published_code"), ("PINN", "paper_equation"),
                                    ("MLP", "shared_baseline"), ("CNN", "shared_baseline")]:
                    for repeat in range(1, 11):
                        tasks.append(extended_task(dataset, batch, model, lane, repeat,
                                                   "small_sample", reference, size))
    for dataset, batch in [("MIT", "all"), ("TJU", "2")]:
        for size in [1, 2]:
            for lane in LANES:
                for repeat in range(1, 11):
                    task = extended_task(dataset, batch, "PINN", lane, repeat,
                                         "small_sample", reference, size)
                    task["paper_coverage"] = ["Table 4 target-only"]
                    tasks.append(task)
    tasks.extend(make_transfer_tasks(reference, "supplied_author_checkpoint"))
    assert len(tasks) == 1880
    return tasks


def make_transfer_tasks(reference, source_mode, source_suite=None):
    tasks = []
    for lane, loss in LANES.items():
        for mapping in transfer_configurations("paper_fixed12"):
            source, target = mapping["source"], mapping["target"]
            sb, tb = mapping["source_batch"], mapping["target_batch"]
            source_batch = BATCHES[source][sb] if source in {"XJTU", "TJU"} else "all"
            target_batch = BATCHES[target][tb] if target in {"XJTU", "TJU"} else "all"
            for repeat in range(1, 11):
                task = copy.deepcopy(mapping)
                task.update(task_id=f"transfer/{lane}/{source_mode}/{source}_{sb}-to-{target}_{tb}/Experiment{repeat:02d}",
                            runner="transfer", family="transfer", dataset=target, batch=target_batch,
                            model="PINN", repeat=repeat, seed=seed_for(target, target_batch, repeat),
                            epochs=200, adaptation_epochs=200, threads=2, sample_size=1,
                            protocol_id="author_current", loss_variant=loss,
                            validation_protocol="author_sample_split", source_mode=source_mode,
                            split_json=reference_run(target, target_batch, repeat, reference) + "/splits.json",
                            overrides=dict(alpha=.7, beta=50, batch_size=128, early_stop=10),
                            protocol_evidence=[HISTORY, "Original adaptation defaults alpha0.7/beta50/batch128/patience10",
                                               "Table 4 fixed XJTU index0 and TJU index2; original target-specific adaptation LR"],
                            paper_coverage=["Table 4", "Fig. 6b"])
                if source_mode == "supplied_author_checkpoint":
                    task["source_checkpoint"] = source_checkpoint_name(source, sb)
                else:
                    run_id = f"full/{lane}/{source}/{source_batch}/PINN/Experiment{repeat:02d}"
                    folder = f"{source_suite}/tasks/{run_id}"
                    task.update(source_run=folder, source_checkpoint=folder + "/model.pth",
                                source_manifest=folder + "/manifest.json",
                                source_selection="Prespecified same-index independent source repeat; no target-test selection")
                tasks.append(task)
    assert len(tasks) == 240
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", default="/workspace/pinn4soh_main_20261109")
    parser.add_argument("--source-suite", default="/workspace/pinn4soh_full_paper_stage1")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plans = {
        "stage1_plan.json": dict(schema="pinn4soh-full-paper-plan-v1", stage=1,
                                 tasks=make_stage_one(args.reference)),
        "stage2_reproduced_transfer_plan.json": dict(schema="pinn4soh-full-paper-plan-v1", stage=2,
                  prerequisite="All referenced source tasks in stage1 complete and hash-verified",
                  tasks=make_transfer_tasks(args.reference, "reproduced_checkpoint", args.source_suite))}
    smoke_tasks = [copy.deepcopy(task) for task in plans["stage1_plan.json"]["tasks"]
                   if task["repeat"] == 1 and
                   ((task["family"] == "full" and
                     (task["model"] == "PINN" and task["batch"] == BATCHES[task["dataset"]][0]
                      or task["dataset"] == "XJTU" and task["batch"] == "2C"))
                    or task["family"] == "transfer" and task["source"] == "TJU" and task["target"] == "XJTU")]
    assert len(smoke_tasks) == 12
    plans["smoke_plan.json"] = dict(schema="pinn4soh-full-paper-plan-v1", stage="smoke", tasks=smoke_tasks)
    summary = {"total_new_training_tasks": 2120, "existing_current_PINN_main_tasks": 110,
               "seed_policy": "10 prespecified paired seeds per dataset/batch; not recovered historical seeds",
               "lanes": LANES, "shared_baselines": True,
               "small_sample_baseline_gap": "No published dedicated baseline settings; original comparison batch512 retained",
               "target_only_reuse": "XJTU1/HUST n1/n2 use same small-sample tasks; MIT/TJU3 add40 per two PINN lanes",
               "numeric_exactness": "Unknown author seeds/checkpoint provenance and documented code/paper differences remain explicit",
               "stages": {}}
    for name, plan in plans.items():
        ids = [task["task_id"] for task in plan["tasks"]]
        assert len(ids) == len(set(ids))
        payload = json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
        (args.output / name).write_text(payload, encoding="utf-8")
        summary["stages"][name] = dict(tasks=len(ids), sha256=hashlib.sha256(payload.encode()).hexdigest(),
                by_family=dict(Counter(t["family"] for t in plan["tasks"])),
                by_model=dict(Counter(t["model"] for t in plan["tasks"])))
    (args.output / "plan_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
