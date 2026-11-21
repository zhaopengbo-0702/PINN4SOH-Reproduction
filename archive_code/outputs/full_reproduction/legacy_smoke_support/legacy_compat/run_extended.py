"""One independent full, small-sample, or explicitly specified PINN ablation task."""

import argparse
import contextlib
import copy
import hashlib
import importlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import time

FORMAT = "pinn4soh-extended-task-v1"
BATCHES = {"XJTU": ["2C", "3C", "R2.5", "R3", "RW", "satellite"],
           "MIT": ["all"], "HUST": ["all"], "TJU": ["0", "1", "2"]}
LOADERS = {"XJTU": "load_data", "MIT": "load_MIT_data", "HUST": "load_HUST_data", "TJU": "load_TJU_data"}
LEGACY_GAPS = {"XJTU": 0.05, "MIT": 0.05, "HUST": 0.1, "TJU": 0.07}
METRICS = ["MAE", "MAPE", "MSE", "RMSE", "MAE_pp", "MAPE_pct", "RMSE_pp"]
LOSS_VARIANTS = {
    "author_current": {
        "expression": "relu((u2-u1)*(y1-y2)).sum()",
        "reference_commit": "cc3cc5053caf38f04e0665f7f88cb109144d035e",
        "reference_model_blob_sha1": "e838b590e580003e2a194de29146682523ed220f",
        "reference_model_sha256": "b146b179a7267a50bed231a99ad5b161b14f8c527b62c376bf286aa7995e0fca",
        "physical_gradient": "Generally differentiable with respect to solution_u; it can be zero for an inactive ReLU batch",
    },
    "paper_prediction_monotonicity": {
        "expression": "relu(u2-u1).sum()",
        "reference_commit": "1e94ec8ed9af35134c8afbd6c576b80f3228b7a0",
        "reference_model_blob_sha1": "e23ebf93155989c5e016425b72a2daf92ec7a2d3",
        "reference_model_sha256": "ab5fb875873c5b803f69315a71ee0751d1d5ede9e1ac521493561f61359295a9",
        "physical_gradient": "Generally differentiable with respect to solution_u; it can be zero for an inactive ReLU batch",
    },
    "published_label_monotonicity": {
        "expression": "relu(y2-y1).sum()",
        "reference_commit": "d26bc1f40dc0cdf34405cd5e599a6f6af09742e9",
        "reference_model_blob_sha1": "151b3f0df6c3cd367772292040209929a7d8faad",
        "reference_model_sha256": "cfdb1636a22bdc9be9c4848082239667ba787db7fe9942be0602e02125f20ef7",
        "physical_gradient": "Identically zero with respect to every model parameter because the penalty depends only on labels",
    },
}


def load_helpers(path):
    spec = importlib.util.spec_from_file_location("extended_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def normalize_task(value, smoke=False):
    task = copy.deepcopy(value)
    for key in ["task_id", "family", "dataset", "model", "repeat", "seed", "protocol_id"]:
        if key not in task:
            raise ValueError("Missing task field: " + key)
    if task["family"] not in {"full", "small_sample", "ablation"} or task["model"] not in {"PINN", "CNN", "MLP"}:
        raise ValueError("Unsupported task family/model")
    dataset = task["dataset"]
    if dataset not in BATCHES:
        raise ValueError("Unsupported dataset")
    task["batch"] = str(task.get("batch", "all"))
    if task["batch"] not in BATCHES[dataset]:
        raise ValueError("Invalid batch for " + dataset)
    for key, default in [("epochs", 200), ("threads", 2)]:
        task.setdefault(key, default)
    if (not isinstance(task["repeat"], int) or not 1 <= task["repeat"] <= 10
            or not isinstance(task["seed"], int) or task["seed"] < 0
            or task["threads"] not in {1, 2, 3, 4} or not isinstance(task["epochs"], int)
            or not 1 <= task["epochs"] <= 200):
        raise ValueError("Invalid repetition/seed/thread/epoch setting")
    task["smoke"] = bool(task.get("smoke", False) or smoke)
    if task["smoke"]:
        task["epochs"] = 2
    task.setdefault("protocol_evidence", [])
    task.setdefault("loss_variant", "author_current")
    if task["loss_variant"] not in LOSS_VARIANTS:
        raise ValueError("Unknown loss_variant")
    if task["model"] != "PINN" and task["loss_variant"] != "author_current":
        raise ValueError("Physical loss variants only apply to PINN")
    task.setdefault("validation_protocol", "author_sample_split")
    if task["validation_protocol"] not in {"author_sample_split", "explicit_battery_split"}:
        raise ValueError("Unknown validation_protocol")
    if task["protocol_id"] not in {"author_current", "paper_note3"}:
        raise ValueError("protocol_id must be author_current or paper_note3")
    if task["validation_protocol"] == "explicit_battery_split" and not task.get("split_json"):
        raise ValueError("Battery-level validation requires an explicit split_json; validation IDs are not guessed")
    if not isinstance(task["protocol_evidence"], list):
        raise ValueError("protocol_evidence must be a list")
    overrides = task.setdefault("overrides", {})
    if not isinstance(overrides, dict) or set(overrides) - {"batch_size", "alpha", "beta", "early_stop"}:
        raise ValueError("Only explicit batch_size/alpha/beta/early_stop overrides are supported")
    if overrides and not task["protocol_evidence"]:
        raise ValueError("Overrides require explicit protocol_evidence")
    for name, value in overrides.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid override: " + name)
        if name == "batch_size" and (not isinstance(value, int) or value < 1):
            raise ValueError("batch_size must be a positive integer")
        if name == "early_stop" and (not isinstance(value, int) or value < 1):
            raise ValueError("early_stop must be a positive integer")
        if name in {"alpha", "beta"} and task["model"] != "PINN":
            raise ValueError("Loss weights only apply to PINN")
    if task["family"] == "small_sample":
        if not isinstance(task.get("sample_size"), int) or not 1 <= task["sample_size"] <= 4:
            raise ValueError("small_sample requires sample_size in 1..4")
    elif task.get("sample_size") is not None:
        raise ValueError("sample_size is only defined for small_sample tasks")
    if task["family"] == "ablation":
        if task["model"] != "PINN" or not ({"alpha", "beta"} & set(overrides)) or not task["protocol_evidence"]:
            raise ValueError("Ablations require PINN, explicit loss weights, and a cited protocol")
    return task


def evaluate_predictions(truth, prediction, splits, dataset, helpers, output=None):
    import numpy as np
    truth, prediction = np.asarray(truth).reshape(-1), np.asarray(prediction).reshape(-1)
    pooled = helpers.calculate_metrics(truth, prediction)
    test = next(item for item in splits if item["role"] == "test")
    if sum(item["n_pairs"] for item in test["batteries"]) != len(truth):
        raise ValueError("Exact battery boundaries disagree with predictions")
    battery_rows, standard, original, start = [], [], [], 0
    for battery in test["batteries"]:
        end = start + battery["n_pairs"]
        direct = helpers.calculate_metrics(truth[start:end], prediction[start:end])
        reverse = helpers.calculate_metrics(prediction[start:end], truth[start:end])
        row = {"battery": battery["path"], "start": start, "stop_exclusive": end,
               "n_test": end - start, **direct,
               **{name + "_original": reverse[name] for name in METRICS}}
        battery_rows.append(row)
        standard.append(direct)
        original.append(reverse)
        start = end

    def macro(rows):
        return {name: float(np.mean([row[name] for row in rows])) for name in METRICS}

    endpoints = list(np.where(np.diff(truth) > LEGACY_GAPS[dataset])[0]) + [len(truth)]
    legacy_rows, empty, dropped, start = [], [], [], 0
    for index, end in enumerate(endpoints):
        end = int(end)
        if end <= start:
            empty.append(index)
        else:
            legacy_rows.append(helpers.calculate_metrics(prediction[start:end], truth[start:end]))
        if end < len(truth):
            dropped.append(end)
        start = end + 1
    result = {"pooled_standard": pooled,
              "battery_macro_standard": macro(standard), "battery_macro_original": macro(original),
              "n_batteries": len(battery_rows),
              "legacy_author_analysis": {"gap": LEGACY_GAPS[dataset], "n_segments": len(endpoints),
                  "matches_actual_battery_count": len(endpoints) == len(battery_rows),
                  "dropped_sample_indices": dropped, "empty_segments": empty,
                  "valid": not empty, "metrics": macro(legacy_rows) if legacy_rows and not empty else None},
              "conventions": {"standard": "MAPE denominator is measured SOH",
                  "original": "eval_metrix(predicted, measured): MAPE denominator is predicted SOH",
                  "battery_macro": "Equal-weight mean of exact battery metrics per run; then compare independent runs",
                  "legacy": "Author gap detection and end-exclusive slicing reproduced for diagnosis; not true boundaries",
                  "std": "Across-run sample standard deviation ddof=1; paper only says mean and std"}}
    if output is not None:
        helpers.write_json(output / "metrics.json", result)
        columns = ["battery", "start", "stop_exclusive", "n_test", *METRICS,
                   *[name + "_original" for name in METRICS]]
        helpers.write_csv(output / "metrics_per_battery.csv", battery_rows, columns)
    return result


def get_author_args(module, model, output):
    previous = sys.argv
    with tempfile.TemporaryDirectory(prefix="pinn4soh_args_") as temporary:
        try:
            sys.argv = [module.__name__ + ".py"]
            if model != "PINN":
                sys.argv += ["--save_folder", temporary]
            args = module.get_args()
        finally:
            sys.argv = previous
    args.save_folder, args.log_dir = str(output), "logging.txt"
    return args


def load_data(task, args, module, comparison, helpers, repo, reference):
    dataset, size = task["dataset"], task.get("sample_size")
    if reference is not None:
        from dataloader.dataloader import DF
        source = json.loads(reference.read_text(encoding="utf-8"))
        if task["validation_protocol"] == "explicit_battery_split":
            return load_explicit_battery_split(args, source, size)
        training = next(item for item in source if item["role"] == "train_validation")
        test = next(item for item in source if item["role"] == "test")
        train_paths = training["paths"][:size] if size else training["paths"]
        if size and len(train_paths) != size:
            raise ValueError("Requested more training batteries than the reference split contains")
        data = DF(args)
        train_loaders = data.load_all_battery(train_paths, training["nominal_capacity"])
        test_loaders = data.load_all_battery(test["paths"], test["nominal_capacity"])
        return {"train": train_loaders["train_2"], "valid": train_loaders["valid_2"],
                "test": test_loaders["test_3"]}
    if task["model"] != "PINN" and dataset == "XJTU":
        return comparison.load_XJTU_data(args, small_sample=size)
    return getattr(module, LOADERS[dataset])(args, small_sample=size)


def load_explicit_battery_split(args, split_records, sample_size=None):
    """Read explicit train/valid/test batteries; capture callers should assign these three roles."""
    from dataloader.dataloader import DF
    from torch.utils.data import DataLoader
    by_role = {item["role"]: item for item in split_records}
    if set(by_role) != {"train", "valid", "test"} or len(split_records) != 3:
        raise ValueError("Battery split must contain exactly train, valid, test records")
    paths = {role: list(record["paths"]) for role, record in by_role.items()}
    if sample_size is not None:
        paths["train"] = paths["train"][:sample_size]
        if len(paths["train"]) != sample_size:
            raise ValueError("Insufficient explicitly designated training batteries")
    sets = {role: {str(Path(path).resolve()) for path in items} for role, items in paths.items()}
    if any(not items for items in sets.values()) or any(
            sets[a] & sets[b] for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]):
        raise ValueError("Train, valid, and test battery IDs must be nonempty and disjoint")
    data, loaders = DF(args), {}
    for role in ["train", "valid", "test"]:
        loaded = data.load_all_battery(paths[role], by_role[role]["nominal_capacity"])["test_3"]
        loaders[role] = DataLoader(loaded.dataset, batch_size=args.batch_size, shuffle=role == "train")
    return loaders


def train_task(task, args, loaders, module, comparison, output):
    import torch
    if task["model"] == "PINN":
        class SnapshotPINN(module.PINN):
            def __setattr__(self, name, value):
                if name == "best_model" and isinstance(value, dict):
                    value = copy.deepcopy(value)
                super().__setattr__(name, value)

            def train_one_epoch(self, epoch, dataloader):
                self.current_epoch = epoch
                if task.get("loss_variant", "author_current") != "author_current":
                    from run_transfer import train_with_loss_variant
                    return train_with_loss_variant(self, epoch, dataloader, task["loss_variant"], False)
                return super().train_one_epoch(epoch, dataloader)

            def Test(self, testloader):
                self.best_epoch = self.current_epoch
                return super().Test(testloader)

        model = SnapshotPINN(args)
        diagnostic = {"loss_variant": task.get("loss_variant", "author_current"),
                      "parameter_gradient_expected_identically_zero": task.get("loss_variant") == "published_label_monotonicity"}

        def inspect_physical_penalty(relu, inputs, value):
            if "first_batch_output_requires_grad" in diagnostic:
                return
            diagnostic.update(first_batch_output_requires_grad=bool(value.requires_grad),
                              first_batch_physical_penalty_sum=float(value.detach().sum().cpu()),
                              first_batch_element_count=value.numel())
            if diagnostic["parameter_gradient_expected_identically_zero"] and value.requires_grad:
                raise ValueError("The publication-era label-only penalty unexpectedly depends on model parameters")

        hook = model.relu.register_forward_hook(inspect_physical_penalty)
        try:
            model.Train(trainloader=loaders["train"], validloader=loaders["valid"], testloader=loaders["test"])
        finally:
            hook.remove()
        (output / "physical_loss_diagnostic.json").write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
        checkpoint = torch.load(output / "model.pth", map_location=next(model.parameters()).device)
        model.solution_u.load_state_dict(checkpoint["solution_u"], strict=True)
        model.dynamical_F.load_state_dict(checkpoint["dynamical_F"], strict=True)
        truth, prediction = module.PINN.Test(model, loaders["test"])
        return truth, prediction, model.best_epoch

    class SnapshotTrainer(comparison.Trainer):
        def train_one_epoch(self, epoch):
            self.current_epoch = epoch
            return super().train_one_epoch(epoch)

        def test(self):
            result = super().test()
            self.best_epoch = self.current_epoch
            torch.save({"state_dict": copy.deepcopy(self.model.state_dict()),
                        "model": task["model"], "best_epoch": self.best_epoch}, output / "model.pth")
            return result

    args.model = task["model"]
    trainer = SnapshotTrainer(comparison.load_model(args), loaders["train"], loaders["valid"], loaders["test"], args)
    trainer.train()
    checkpoint = torch.load(output / "model.pth", map_location=trainer.device)
    trainer.model.load_state_dict(checkpoint["state_dict"], strict=True)
    trainer.model.eval()
    truth, prediction = [], []
    import numpy as np
    with torch.no_grad():
        for x, _, y, _ in loaders["test"]:
            truth.append(y.numpy())
            prediction.append(trainer.model(x.to(trainer.device)).cpu().numpy())
    return np.concatenate(truth), np.concatenate(prediction), trainer.best_epoch


@contextlib.contextmanager
def task_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+")
    try:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-json", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--helpers", type=Path, default=Path(__file__).with_name("run_reproduction.py"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    options = parser.parse_args()
    helper_path = options.helpers.resolve()
    helpers = load_helpers(helper_path)
    raw_task = json.loads(options.task_json.read_text(encoding="utf-8"))
    task = normalize_task(raw_task, options.smoke)
    repo, output = options.repo.resolve(), options.output.resolve()
    if output == repo or repo in output.parents:
        raise ValueError("Extended results must be outside the author repository")
    reference_run = Path(task["reference_run"]).resolve() if task.get("reference_run") else None
    if reference_run is not None and (output == reference_run or output in reference_run.parents or reference_run in output.parents):
        raise ValueError("Extended output must not contain or modify the reference run")
    reference = Path(task["split_json"]).resolve() if task.get("split_json") else (
        reference_run / "splits.json" if reference_run else None)
    assigned_visibility = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    os.environ["CUDA_VISIBLE_DEVICES"] = assigned_visibility
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(task["threads"])
    import numpy as np
    import torch
    import pandas as pd
    import sklearn
    torch.set_num_threads(task["threads"])
    if not torch.cuda.is_available() and not options.allow_cpu:
        raise RuntimeError("CUDA unavailable; use --allow-cpu only for an explicit CPU check")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    environment = {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                   "pandas": pd.__version__, "sklearn": sklearn.__version__, "threads": task["threads"],
                   "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    inventory = helpers.source_inventory(repo)
    if reference_run:
        reference_config = json.loads((reference_run / "config.json").read_text(encoding="utf-8"))
        if reference_config["source_fingerprint"] != inventory["fingerprint"]:
            raise ValueError("Reference source/data fingerprint differs from the current repository")
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    module = importlib.import_module("main_" + task["dataset"])
    comparison = importlib.import_module("main_comparision")
    os.environ["CUDA_VISIBLE_DEVICES"] = assigned_visibility
    author = module if task["model"] == "PINN" else comparison
    args = get_author_args(author, task["model"], output)
    args.epochs = task["epochs"]
    args.warmup_epochs = min(args.warmup_epochs, max(0, args.epochs - 1))
    args.data, args.dataset = task["dataset"], task["dataset"]
    args.batch = int(task["batch"]) if task["dataset"] == "TJU" else task["batch"]
    args.xjtu_batch, args.tju_batch, args.in_same_batch = task["batch"], args.batch, True
    gaps = []
    if task["model"] != "PINN" and task["dataset"] != "XJTU":
        gaps.append("Baseline architecture/Trainer from comparison entry; dataset loader adapted from the corresponding PINN entry")
    if task["family"] == "small_sample":
        if task["model"] == "PINN":
            args.batch_size = 128
            if task["dataset"] == "XJTU":
                args.alpha, args.beta = 0.5, 10
        else:
            gaps.append("Small-sample baseline settings reuse comparison defaults unless explicitly overridden; no native small-sample baseline loop")
    if task["protocol_id"] == "paper_note3":
        args.batch_size = 256 if task["dataset"] == "XJTU" else 512
        if task["model"] == "PINN":
            args.alpha, args.beta = {"XJTU": (0.7, 20), "MIT": (1, 50),
                                     "TJU": (1, 50), "HUST": (0.5, 80)}[task["dataset"]]
        gaps.append("Supplementary Note 3 general hyperparameters are applied with the explicitly selected loss_variant; parameter matching alone does not establish paper numerical equivalence")
        if task["family"] == "small_sample":
            gaps.append("Note 3 gives general batch/loss settings but no separate small-sample hyperparameters; these are an explicit general-setting interpretation")
    if task["validation_protocol"] == "author_sample_split":
        validation_evidence = "Author loader uses 20% cycle-sample validation from non-test batteries, clarified in the public peer-review response pp29-30"
    else:
        validation_evidence = "Explicit disjoint validation batteries; an extra sensitivity protocol rather than the author-reviewed original split"
        gaps.append(validation_evidence)
    if task["family"] == "ablation":
        gaps.append("Explicitly configured ablation; publication equivalence depends on the cited task protocol")
    for name, value in task["overrides"].items():
        setattr(args, name, value)
    if task["model"] != "PINN":
        args.model = task["model"]
    else:
        args.model = "PINN"
    reference_data = {}
    if reference is not None:
        for record in json.loads(reference.read_text(encoding="utf-8")):
            for path in record["paths"]:
                resolved = Path(path).resolve()
                try:
                    relative = resolved.relative_to(repo).as_posix()
                except ValueError:
                    relative = None
                reference_data[str(resolved)] = inventory["sha256"].get(relative) or helpers.digest(resolved)
    physical_loss = copy.deepcopy(LOSS_VARIANTS[task["loss_variant"]]) if task["model"] == "PINN" else None
    if physical_loss is not None:
        physical_loss["variant"] = task["loss_variant"]
        physical_loss["history_source_url"] = "https://github.com/wang-fujin/PINN4SOH/blob/" + physical_loss["reference_commit"] + "/Model/Model.py"
        physical_loss["actual_alpha"] = args.alpha
        physical_loss["actual_beta"] = args.beta
        physical_loss["supplementary_note3_alpha_beta"] = {"XJTU": [0.7, 20], "MIT": [1, 50], "TJU": [1, 50], "HUST": [0.5, 80]}[task["dataset"]]
        physical_loss["matches_note3_general_weights"] = [args.alpha, args.beta] == physical_loss["supplementary_note3_alpha_beta"]
        physical_loss["publication_era_main_defaults"] = {
            "alpha": physical_loss["supplementary_note3_alpha_beta"][0],
            "beta": physical_loss["supplementary_note3_alpha_beta"][1],
            "early_stop": 10 if task["dataset"] == "MIT" else 20,
            "batch_size": 256 if task["dataset"] == "XJTU" else 512,
        }
        physical_loss["actual_early_stop"] = args.early_stop
        physical_loss["matches_publication_era_main_patience"] = args.early_stop == physical_loss["publication_era_main_defaults"]["early_stop"]
        physical_loss["publication_era_small_sample_defaults"] = {
            "batch_size": 128,
            "alpha": 0.5 if task["dataset"] == "XJTU" else physical_loss["supplementary_note3_alpha_beta"][0],
            "beta": 10 if task["dataset"] == "XJTU" else physical_loss["supplementary_note3_alpha_beta"][1],
        }
    loss_helper_path = Path(__file__).with_name("run_transfer.py").resolve() if task["model"] == "PINN" and task["loss_variant"] != "author_current" else None
    config = {"task": task, "parameters": vars(args), "environment": environment,
              "source_fingerprint": inventory["fingerprint"], "runner_sha256": helpers.digest(Path(__file__).resolve()),
              "helper_sha256": helpers.digest(helper_path), "reference_split_sha256": helpers.digest(reference) if reference else None,
              "loss_variant": task["loss_variant"], "physical_loss": physical_loss,
              "loss_helper_sha256": helpers.digest(loss_helper_path) if loss_helper_path else None,
              "reference_data_sha256": reference_data,
              "data_directory_order_sha256": config_hash(inventory["data_directory_order"]),
              "checkpoint_snapshot_deepcopy": True, "protocol_gaps": gaps,
              "validation_protocol_evidence": validation_evidence,
              "validation_selection": "Author PINN sample-weighted MSE" if task["model"] == "PINN" else "Author baseline equal-batch mean MSE",
              "early_stop": args.early_stop if task["model"] == "PINN" else 10,
              "pair_id": f"{task['family']}/{task['dataset']}/{task['batch']}/n{task.get('sample_size', 'all')}/repeat{task['repeat']:02d}/seed{task['seed']}"}
    identity = config_hash(config)
    with task_lock(output.with_name("." + output.name + ".lock")):
        previous = None
        if (output / "manifest.json").exists():
            previous = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            if previous.get("format") != FORMAT:
                raise ValueError("Refusing to modify a directory owned by another runner")
            if previous.get("task_sha256") != config_hash(raw_task):
                raise ValueError("Raw task specification changed; use a new output directory")
            if previous.get("config_hash") != identity:
                raise ValueError("Task/config/source/environment changed; use a new output directory")
            if previous["status"] == "complete" and options.resume:
                if all((output / name).is_file() and helpers.digest(output / name) == checksum
                       for name, checksum in previous["artifact_sha256"].items()):
                    print("SKIP " + task["task_id"], flush=True)
                    return
                raise ValueError("Completed artifacts changed or are missing")
        if output.exists() and any(output.iterdir()):
            if not options.resume or previous is None:
                raise ValueError("Output is not empty; use --resume for the same incomplete task")
            output.rename(output.with_name(output.name + ".previous." + str(time.time_ns())))
        output.mkdir(parents=True, exist_ok=True)
        helpers.write_json(output / "config.json", config)
        helpers.write_json(output / "source_inventory.json", inventory)
        manifest = {"format": FORMAT, "status": "running", "task": raw_task, "effective_task": task,
                    "task_id": task["task_id"], "task_sha256": config_hash(raw_task),
                    "seed": task["seed"], "dataset": task["dataset"], "batch": task["batch"],
                    "model": task["model"], "family": task["family"], "source_fingerprint": inventory["fingerprint"],
                    "implementation_sha256": {"run_extended.py": config["runner_sha256"], "helpers": config["helper_sha256"]},
                    "task_hash": config_hash(raw_task), "config_hash": identity,
                    "started_at": helpers.utc_now(), "protocol_gaps": gaps,
                    "assigned_cuda_visible_devices": assigned_visibility}
        if loss_helper_path is not None:
            manifest["implementation_sha256"]["run_transfer.py"] = config["loss_helper_sha256"]
        helpers.write_json(output / "manifest.json", manifest)
        try:
            random.seed(task["seed"])
            np.random.seed(task["seed"])
            torch.manual_seed(task["seed"])
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(task["seed"])
            with (output / "console.txt").open("w", encoding="utf-8", buffering=1) as console:
                with contextlib.redirect_stdout(console), contextlib.redirect_stderr(console):
                    with helpers.capture_data_paths() as splits:
                        loaders = load_data(task, args, module, comparison, helpers, repo, reference)
                    if task["validation_protocol"] == "explicit_battery_split":
                        for item, role in zip(splits, ["train", "valid", "test"]):
                            item["role"] = role
                    helpers.write_json(output / "splits.json", splits)
                    sample_counts, tensor_hashes = {}, {}
                    for name, loader in loaders.items():
                        if not len(loader.dataset) or not all(torch.isfinite(tensor).all() for tensor in loader.dataset.tensors):
                            raise ValueError("Empty/non-finite " + name + " data")
                        sample_counts[name] = len(loader.dataset)
                        tensor_hashes[name] = [hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
                                               for tensor in loader.dataset.tensors]
                    helpers.write_json(output / "data_identity.json", {"sample_counts": sample_counts,
                        "tensor_sha256": tensor_hashes, "input_file_sha256": {str(path): helpers.digest(path)
                        for call in splits for path in call["paths"]}})
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    started = time.perf_counter()
                    reloaded_truth, reloaded_prediction, best_epoch = train_task(task, args, loaders, module, comparison, output)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    seconds = time.perf_counter() - started
            truth = np.load(output / "true_label.npy", allow_pickle=False)
            prediction = np.load(output / "pred_label.npy", allow_pickle=False)
            if not np.array_equal(truth, reloaded_truth) or not np.allclose(prediction, reloaded_prediction, atol=1e-6, rtol=1e-5):
                raise ValueError("Reloaded best checkpoint does not reproduce saved predictions")
            metrics = evaluate_predictions(truth, prediction, splits, task["dataset"], helpers, output)
            difference = np.abs(prediction.astype(np.float64) - reloaded_prediction.astype(np.float64))
            verification = {"passed": True, "run": str(output), "best_epoch": best_epoch,
                "labels_bitwise_equal": True, "predictions_bitwise_equal": bool(np.array_equal(prediction, reloaded_prediction)),
                "predictions_allclose": True, "max_absolute_prediction_difference": float(difference.max()),
                "atol": 1e-6, "rtol": 1e-5, "checkpoint_sha256": helpers.digest(output / "model.pth"),
                "saved_prediction_sha256": helpers.digest(output / "pred_label.npy")}
            helpers.write_json(output / "verification_checkpoint.json", verification)
            epochs = re.findall(r"\[Train\] epoch:(\d+)", (output / "logging.txt").read_text(encoding="utf-8"))
            artifacts = ["config.json", "splits.json", "data_identity.json", "true_label.npy", "pred_label.npy", "model.pth",
                         "logging.txt", "console.txt", "metrics.json", "metrics_per_battery.csv", "verification_checkpoint.json"]
            if task["model"] == "PINN":
                artifacts.append("physical_loss_diagnostic.json")
            manifest.update(status="complete", finished_at=helpers.utc_now(), training_and_reload_seconds=seconds,
                            best_epoch=best_epoch, epochs_completed=int(epochs[-1]), sample_counts=sample_counts,
                            metrics=metrics, artifact_sha256={name: helpers.digest(output / name) for name in artifacts})
            print("DONE " + task["task_id"] + " " + json.dumps(metrics["battery_macro_standard"]), flush=True)
        except BaseException as error:
            import traceback
            manifest.update(status="failed", finished_at=helpers.utc_now(), error=repr(error))
            (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise
        finally:
            logger = logging.getLogger("mylogger")
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)
            helpers.write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
