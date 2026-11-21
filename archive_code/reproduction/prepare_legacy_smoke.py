"""Copy frozen runners with two isolated API adaptations and select 12 smoke tasks."""

import argparse
from copy import deepcopy
import difflib
import hashlib
import json
from pathlib import Path


NAMES = ["run_reproduction.py", "run_extended.py", "run_transfer.py", "run_full_suite.py"]
LOSSES = ["published_label_monotonicity", "paper_prediction_monotonicity"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compatibility_text(name, text):
    if name == "run_transfer.py":
        if text.count(", weights_only=True") != 2:
            raise ValueError("Expected exactly two PyTorch newer-API call sites")
        return text.replace(", weights_only=True", "")
    if name == "run_full_suite.py":
        before = "        temporary.unlink(missing_ok=True)"
        after = "        try:\n            temporary.unlink()\n        except FileNotFoundError:\n            pass"
        if text.count(before) != 1:
            raise ValueError("Expected exactly one Python newer-API call site")
        return text.replace(before, after)
    return text


def select_tasks(plan):
    candidates, selected = plan["tasks"], []

    def choose(predicate):
        current = [row for row in candidates if row["repeat"] == 1 and predicate(row)]
        if len(current) != 1:
            raise ValueError("Smoke selection must identify exactly one existing task")
        task = deepcopy(current[0])
        task["original_task_id"] = task["task_id"]
        task["task_id"] = "legacy_smoke/" + task["task_id"]
        task["smoke"], task["epochs"] = True, 2
        if task["family"] == "transfer":
            task["adaptation_epochs"] = 2
        task["legacy_runtime_diagnostic"] = True
        selected.append(task)

    for dataset, batch in [("XJTU", "2C"), ("MIT", "all"), ("HUST", "all"), ("TJU", "2")]:
        for loss in LOSSES:
            choose(lambda row: row["family"] == "full" and row["dataset"] == dataset
                   and row["batch"] == batch and row["model"] == "PINN" and row["loss_variant"] == loss)
    for model in ["MLP", "CNN"]:
        choose(lambda row: row["family"] == "full" and row["dataset"] == "XJTU"
               and row["batch"] == "2C" and row["model"] == model)
    for loss in LOSSES:
        choose(lambda row: row["family"] == "transfer" and row["source"] == "XJTU" and row["target"] == "HUST"
               and row["source_batch"] == 0 and row["source_mode"] == "supplied_author_checkpoint"
               and row["loss_variant"] == loss)
    if len(selected) != 12:
        raise AssertionError("Expected 8 PINN + 2 architectures + 2 transfer smoke tasks")
    return selected


def prepare(source, plan_path, output):
    source, plan_path, output = Path(source).resolve(), Path(plan_path).resolve(), Path(output).resolve()
    if output == source or source in output.parents:
        raise ValueError("Compatibility output must be outside frozen runner directory")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new or empty compatibility output directory")
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    output.mkdir(parents=True, exist_ok=True)
    compat = output / "legacy_compat"
    compat.mkdir()
    records, patches = {}, []
    for name in NAMES:
        path = source / name
        raw = path.read_bytes()
        text = raw.decode("utf-8").replace("\r\n", "\n")
        changed = compatibility_text(name, text)
        destination = compat / name
        if changed == text:
            destination.write_bytes(raw)
        else:
            destination.write_text(changed, encoding="utf-8")
        patches.extend(difflib.unified_diff(text.splitlines(True), changed.splitlines(True),
            fromfile="frozen/" + name, tofile="legacy_compat/" + name))
        records[name] = {"original_sha256": digest(path), "compat_sha256": digest(destination), "modified": changed != text}
    smoke = {"schema": plan["schema"], "stage": "legacy_runtime_smoke", "scientific_accuracy_experiment": False,
             "source_plan_sha256": digest(plan_path), "tasks": select_tasks(plan)}
    (output / "legacy_smoke_plan.json").write_text(json.dumps(smoke, indent=2), encoding="utf-8")
    (output / "compatibility.patch").write_text("".join(patches), encoding="utf-8")
    manifest = dict(format="pinn4soh-legacy-compatibility-v1", source_plan=str(plan_path),
        source_plan_sha256=digest(plan_path), source_runner_directory=str(source), runner_files=records,
        plan_sha256=digest(output / "legacy_smoke_plan.json"), task_count=12,
        changes=["torch.load: remove weights_only=True only in legacy transfer copy; trusted checkpoint hashes retained",
                 "Path.unlink: replace missing_ok=True with FileNotFoundError handling only in legacy coordinator copy"],
        science_unchanged=["networks", "data", "splits", "seeds", "loss expressions", "hyperparameters except explicit two-epoch smoke"],
        training_started=False, script_sha256=digest(__file__))
    (output / "legacy_compat_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    print(json.dumps(prepare(options.source, options.plan, options.output), indent=2))
