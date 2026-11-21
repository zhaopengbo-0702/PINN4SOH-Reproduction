"""Freeze a separate, matched 40-task legacy-runtime control; never launch it."""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath


LOSSES = {"published_label_monotonicity", "paper_prediction_monotonicity"}
GROUPS = {("XJTU", "2C"), ("MIT", "all"), ("HUST", "all"), ("TJU", "2")}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(stage1_path, stage2_path, output, cloud_suite):
    stage1_path, stage2_path, output = [Path(path).resolve() for path in [stage1_path, stage2_path, output]]
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new or empty control-plan directory")
    cloud = PurePosixPath(cloud_suite)
    if not cloud.is_absolute() or ".." in cloud.parts:
        raise ValueError("cloud-suite must be an explicit absolute Linux path")
    stages = [json.loads(path.read_text(encoding="utf-8-sig")) for path in [stage1_path, stage2_path]]
    regular = [deepcopy(row) for row in stages[0]["tasks"] if row["repeat"] == 1 and row["family"] == "full"
               and (row["dataset"], row["batch"]) in GROUPS and row["model"] in {"PINN", "MLP", "CNN"}]
    transfer = [deepcopy(row) for row in stages[1]["tasks"] if row["repeat"] == 1 and row["family"] == "transfer"
                and row["source_mode"] == "reproduced_checkpoint" and row["loss_variant"] in LOSSES]
    if len(regular) != 16 or len(transfer) != 24:
        raise ValueError("Expected four groups x four methods plus twelve directions x two losses")
    lookup = {row["task_id"]: "legacy_control/" + row["task_id"] for row in regular}
    for row in regular + transfer:
        row["original_task_id"] = row["task_id"]
        row["task_id"] = "legacy_control/" + row["task_id"]
        row["runtime_control"] = "readme_versions_legacy_vs_current"
        row["paper_coverage"] = []
        row["protocol_evidence"].append("Separate original-runtime control; same seed/splits/hyperparameters; no test-based selection")
        if row.get("smoke") or row["epochs"] != 200:
            raise ValueError("Matched control requires formal epoch budget")
        if row["family"] == "transfer":
            marker = "/tasks/"
            if marker not in row["source_run"]:
                raise ValueError("Reproduced source run lacks canonical tasks path")
            old_source = row["source_run"].split(marker, 1)[1]
            if old_source not in lookup:
                raise ValueError("Matched source repeat is absent from regular legacy control")
            new_source_id = lookup[old_source]
            row["depends_on"] = [new_source_id]
            row["original_source_run"] = row["source_run"]
            row["source_run"] = str(cloud / "tasks" / new_source_id)
            row["source_checkpoint"] = str(cloud / "tasks" / new_source_id / "model.pth")
            row["source_manifest"] = str(cloud / "tasks" / new_source_id / "manifest.json")
    plan = dict(schema=stages[0]["schema"], stage="legacy_matched_formal_control",
        diagnostic_only=True, launch_prerequisite="12/12 original-runtime GPU smoke complete",
        training_launched=False, expected_suite_output=str(cloud), tasks=regular + transfer)
    output.mkdir(parents=True)
    plan_path = output / "legacy_control_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    design = dict(format="pinn4soh-legacy-formal-control-design-v1", plan_sha256=digest(plan_path),
        stage1_sha256=digest(stage1_path), stage2_sha256=digest(stage2_path), task_count=40,
        counts_by_family=dict(Counter(row["family"] for row in plan["tasks"])),
        regular_groups=sorted([list(group) for group in GROUPS]), repeat=1,
        scope="16 regular models (8 PINN,4 MLP,4 CNN) +24 fine-tunes with paired source-only target evaluation",
        source_policy="Prespecified repeat1 independently retrained in legacy stack, same-index target/source seeds; no best-run selection",
        matching=["source/data hashes", "exact ordered battery paths", "normalization", "seeds", "effective hyperparameters",
                  "checkpoint selection", "loss expression", "thread count", "GPU model and assigned visibility recorded"],
        outcomes_to_report=["Code/checkpoint verification failures", "All signed metric differences by group/method/direction",
            "Best validation epoch and completed epochs", "Exact split membership and final prediction drift",
            "PINN-v-MLP/CNN ranking reversals", "Fine-tune-v-source-only sign changes", "Frozen-G and F-update evidence"],
        decision_policy=["No unreported numerical equivalence margin or arbitrary tolerance is invented",
            "A single seed is a version-sensitivity screen, not a stability estimate or significance test",
            "Observed claim reversal or material prediction/optimization changes justify matched ten-repeat expansion in affected groups",
            "If discrepancies are widespread, freeze a broader independently reviewed plan before any old-stack full rerun",
            "Absence of reversal in this subset does not prove the full 2120-task results environment-independent"],
        exclusions=["Does not replace or merge into current-stack formal suite", "No small-sample accuracy conclusion",
                    "No supplied-author-source formal matrix; smoke only checks that load path", "No exact publication-machine claim"],
        training_launched=False, script_sha256=digest(__file__))
    (output / "legacy_control_design.json").write_text(json.dumps(design, indent=2), encoding="utf-8")
    return design


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cloud-suite", default="/workspace/pinn4soh_legacy_formal_control_20261110")
    args = parser.parse_args()
    print(json.dumps(prepare(args.stage1, args.stage2, args.output, args.cloud_suite), indent=2))
