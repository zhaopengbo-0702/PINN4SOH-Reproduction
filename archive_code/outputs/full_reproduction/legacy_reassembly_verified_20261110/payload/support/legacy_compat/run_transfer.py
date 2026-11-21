"""Audited single-task execution of the author's transfer and source training."""

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
import shutil
import sys
import time
import traceback

import run_reproduction as common


DATASETS = ["XJTU", "TJU", "HUST", "MIT"]
XJTU_BATCHES = ["2C", "3C", "R2.5", "R3", "RW", "satellite"]
SOURCE_MODES = ["supplied_author_checkpoint", "reproduced_checkpoint"]
LOSS_VARIANTS = ["author_current", "paper_prediction_monotonicity", "published_label_monotonicity"]


def transfer_configurations(protocol="script_native19"):
    if protocol not in {"script_native19", "paper_fixed12"}:
        raise ValueError("Unknown transfer protocol: " + protocol)
    configurations = []
    for target_batch, (source_batch, lr) in enumerate(zip(
            [2, 2, 2, 1, 0, 1], [.0004, .01, .0005, .002, .003, .0006])):
        configurations.append(dict(source="TJU", source_batch=source_batch,
                                   target="XJTU", target_batch=target_batch,
                                   adaptation_lr=lr, author_function="FineTune_TJU2XJTU"))
    for target_batch, (source_batch, lr) in enumerate(zip([3, 3, 2], [.003, .002, .002])):
        configurations.append(dict(source="XJTU", source_batch=source_batch,
                                   target="TJU", target_batch=target_batch,
                                   adaptation_lr=lr, author_function="FineTune_XJTU2TJU"))
    for source, target, lr in [("HUST", "MIT", .005), ("MIT", "HUST", .0002)]:
        configurations.append(dict(source=source, source_batch=-1, target=target,
                                   target_batch=-1, adaptation_lr=lr,
                                   author_function=f"FineTune_{source}2{target}"))
    batches = dict(XJTU=0, TJU=2, HUST=-1, MIT=-1)
    for source in DATASETS:
        for target in DATASETS:
            if (source in {"XJTU", "TJU"}) == (target in {"XJTU", "TJU"}):
                continue
            configurations.append(dict(source=source, source_batch=batches[source],
                                       target=target, target_batch=batches[target],
                                       adaptation_lr=.001, author_function="FineTune"))
    if protocol == "paper_fixed12":
        configurations = [item for item in configurations
                          if item["target_batch"] == batches[item["target"]]]
        for item in configurations:
            original = item["source_batch"]
            item["source_batch"] = batches[item["source"]]
            item["paper_source_batch_override"] = original != item["source_batch"]
            item["source_batch_in_author_script"] = original
    return [dict(item, transfer_design=protocol) for item in configurations]


def source_checkpoint_name(dataset, batch):
    suffix = f"_{batch}" if dataset in {"XJTU", "TJU"} else ""
    return f"pretrained model/model_{dataset}{suffix}.pth"


def build_transfer_tasks(protocol="script_native19", source_mode="supplied_author_checkpoint",
                         repeats=10, seed=20261109, reproduced_sources=None):
    if source_mode not in SOURCE_MODES:
        raise ValueError("Unknown source mode")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    tasks = []
    for configuration in transfer_configurations(protocol):
        for repeat in range(1, repeats + 1):
            item = dict(configuration)
            source, target = item["source"], item["target"]
            sb, tb = item["source_batch"], item["target_batch"]
            batch = XJTU_BATCHES[tb] if target == "XJTU" else str(tb) if target == "TJU" else "all"
            key = f"{source}_{sb}-to-{target}_{tb}"
            item.update(task_id=f"transfer/{protocol}/{source_mode}/{key}/Experiment{repeat:02d}",
                        family="transfer", dataset=target, batch=batch, model="PINN",
                        repeat=repeat, seed=seed + DATASETS.index(target) * 10000 + max(tb, 0) * 100 + repeat - 1,
                        epochs=200, adaptation_epochs=200, threads=2, sample_size=1,
                        source_mode=source_mode, protocol_id="author_current", loss_variant="author_current",
                        validation_protocol="author_sample_split",
                        protocol_evidence=["main_adaptation - fine-tuning.py:" + item["author_function"]])
            if protocol == "paper_fixed12":
                item["protocol_evidence"].append("Paper Table 4 footnote: XJTU batch 1 and TJU batch 3")
                item["learning_rate_evidence"] = "Author script's value for this target batch; supplement confirmation required"
            if source_mode == "supplied_author_checkpoint":
                item["source_checkpoint"] = source_checkpoint_name(source, sb)
            else:
                source_run = (reproduced_sources or {}).get(f"{source}/{sb}")
                if source_run is None:
                    raise ValueError(f"Explicit reproduced source run is required: {source}/{sb}")
                item["source_run"] = str(source_run)
                item["source_checkpoint"] = str(Path(source_run) / "model.pth")
            tasks.append(item)
    return tasks


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def resolve_path(value, repo):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def checkpoint_lineage(task, repo, inventory):
    checkpoint = resolve_path(task["source_checkpoint"], repo)
    if not checkpoint.is_file():
        raise ValueError("Source checkpoint is missing: " + str(checkpoint))
    lineage = {"source_mode": task["source_mode"], "path": str(checkpoint),
               "sha256": common.digest(checkpoint), "source_dataset": task["source"],
               "source_batch": task["source_batch"]}
    if task["source_mode"] == "supplied_author_checkpoint":
        expected = (repo / source_checkpoint_name(task["source"], task["source_batch"])).resolve()
        if checkpoint != expected:
            raise ValueError("Author source mode requires the matching supplied dataset/batch checkpoint")
        lineage.update(training_seed=None, training_split=None, training_history_verified=False,
                       note="Supplied author weights; training seed, selection criterion and split provenance are not supplied")
    elif task["source_mode"] == "reproduced_checkpoint":
        source_run = resolve_path(task["source_run"], repo)
        if checkpoint != source_run / "model.pth":
            raise ValueError("Reproduced checkpoint must be the specified source run's model.pth")
        config = read_json(source_run / "config.json")
        splits = read_json(source_run / "splits.json")
        parameters = config.get("parameters") or read_json(source_run / "parameters.json")
        source = task["source"]
        if parameters.get("data") != source:
            raise ValueError("Source run dataset does not match transfer source")
        expected_batch = XJTU_BATCHES[task["source_batch"]] if source == "XJTU" else task["source_batch"]
        if source in {"XJTU", "TJU"} and parameters.get("batch") != expected_batch:
            raise ValueError("Source run batch does not match transfer source")
        if config.get("source_fingerprint") != inventory["fingerprint"]:
            raise ValueError("Source checkpoint was trained from a different source/data inventory")
        if config.get("smoke") or parameters.get("epochs", 0) < 200:
            raise ValueError("A smoke/short source run is not a full reproduced source checkpoint")
        if not config.get("checkpoint_snapshot_deepcopy"):
            raise ValueError("Source checkpoint must have a verified best-validation snapshot")
        log = (source_run / "logging.txt").read_text(encoding="utf-8")
        validations = [float(value) for value in re.findall(r"\[Valid\] epoch:\d+, MSE: ([^\s]+)", log)]
        if not validations or not all(math.isfinite(value) and value >= 0 for value in validations):
            raise ValueError("Source validation history is missing or invalid")
        source_manifest_path = task.get("source_manifest")
        if source_manifest_path:
            source_manifest_path = resolve_path(source_manifest_path, repo)
        else:
            source_manifest_path = source_run / "manifest.json"
            if not source_manifest_path.exists():
                source_manifest_path = source_run.parents[2] / "manifest.json"
        source_manifest = read_json(source_manifest_path)
        if "runs" in source_manifest:
            source_key = source_run.relative_to(source_manifest_path.parent).as_posix()
            source_state = source_manifest["runs"].get(source_key, {})
        else:
            source_state = source_manifest
        if source_state.get("status") != "complete":
            raise ValueError("Reproduced source manifest does not confirm a complete run")
        source_seed = config.get("seed", config.get("task", {}).get("seed"))
        lineage.update(training_seed=source_seed, training_history_verified=True,
                       source_config_sha256=common.digest(source_run / "config.json"),
                       source_splits_sha256=common.digest(source_run / "splits.json"),
                       source_manifest_sha256=common.digest(source_manifest_path),
                       source_run=str(source_run), source_best_validation_mse=min(validations),
                       source_paths=splits, source_selection=task.get("source_selection", "explicit source run; no target test selection"))
    else:
        raise ValueError("Unknown source mode")
    return lineage


def select_reproduced_source(output, dataset, source_batch):
    """Select a completed source run by source validation MSE, never target tests."""
    output = Path(output).resolve()
    manifest = read_json(output / "manifest.json")
    batch = XJTU_BATCHES[source_batch] if dataset == "XJTU" else str(source_batch) if dataset == "TJU" else "all"
    candidates = []
    for key, record in manifest["runs"].items():
        if record.get("status") != "complete" or record.get("dataset") != dataset or record.get("batch") != batch:
            continue
        folder = output / record["relative_folder"]
        log = (folder / "logging.txt").read_text(encoding="utf-8")
        values = [float(value) for value in re.findall(r"\[Valid\] epoch:\d+, MSE: ([^\s]+)", log)]
        if values and all(math.isfinite(value) and value >= 0 for value in values):
            candidates.append((min(values), key, folder))
    if not candidates:
        raise ValueError(f"No complete source-validation history for {dataset}/{batch}")
    best = min(candidates)
    return {"source_run": str(best[2]), "source_checkpoint": str(best[2] / "model.pth"),
            "source_selection": "Minimum source validation MSE across completed source repeats; ties by run key",
            "source_selection_value": best[0], "source_selection_candidates": len(candidates)}


def validate_task(task):
    for name in ["task_id", "family", "seed", "repeat", "protocol_id"]:
        if name not in task:
            raise ValueError("Missing task field: " + name)
    if task["family"] not in {"transfer", "transfer_pretrain"}:
        raise ValueError("run_transfer accepts transfer or transfer_pretrain tasks")
    if task.get("loss_variant", "author_current") not in LOSS_VARIANTS:
        raise ValueError("Unknown physical-loss variant")
    if task.get("validation_protocol", "author_sample_split") not in {"author_sample_split", "explicit_battery_split"}:
        raise ValueError("Unknown validation protocol")
    if task.get("validation_protocol") == "explicit_battery_split" and not task.get("split_json"):
        raise ValueError("Explicit train/valid/test battery records are required")
    overrides = task.get("overrides", {})
    if not isinstance(overrides, dict) or set(overrides) - {"alpha", "beta", "batch_size", "early_stop", "adaptation_lr"}:
        raise ValueError("Unsupported transfer hyperparameter override")
    if overrides and not task.get("protocol_evidence"):
        raise ValueError("Transfer overrides require an explicit protocol evidence record")
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in overrides.values()):
        raise ValueError("Invalid transfer hyperparameter override")
    source = task.get("source", task.get("dataset"))
    if source not in DATASETS or int(task["repeat"]) < 1:
        raise ValueError("Invalid source dataset or repetition")
    if task["family"] == "transfer":
        target = task.get("target", task.get("dataset"))
        if target not in DATASETS or source == target:
            raise ValueError("Transfer requires distinct source and target datasets")
        if task.get("source_mode") not in SOURCE_MODES:
            raise ValueError("An explicit source_mode is required")
        for dataset, batch in [(source, task.get("source_batch", -1)), (target, task.get("target_batch", -1))]:
            valid = range(6) if dataset == "XJTU" else range(3) if dataset == "TJU" else [-1]
            if batch not in valid:
                raise ValueError("Invalid source/target batch for " + dataset)
        if int(task.get("sample_size", 1)) != 1:
            raise ValueError("The audited transfer protocol adapts on exactly one target battery")
        if float(task.get("adaptation_lr", 0)) <= 0:
            raise ValueError("A positive adaptation_lr is required")
    else:
        batch = task.get("source_batch", -1)
        valid = range(6) if source == "XJTU" else range(3) if source == "TJU" else [-1]
        if batch not in valid:
            raise ValueError("Invalid pretraining source batch")


def load_adaptation_module(repo):
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    spec = importlib.util.spec_from_file_location("pinn4soh_author_adaptation", repo / "main_adaptation - fine-tuning.py")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        if visible_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    return module


def default_arguments(module):
    previous = sys.argv
    try:
        sys.argv = [str(module.__file__)]
        return module.get_args()
    finally:
        sys.argv = previous


def validate_splits(splits, task, repo):
    train = next(item for item in splits if item["role"] in {"train_validation", "train"})
    test = next(item for item in splits if item["role"] == "test")
    train_paths = [str(resolve_path(path, repo)) for path in train["paths"]]
    test_paths = [str(resolve_path(path, repo)) for path in test["paths"]]
    if not train_paths or not test_paths or set(train_paths) & set(test_paths):
        raise ValueError("Empty target split or target adaptation/test leakage")
    valid = next((item for item in splits if item["role"] == "valid"), None)
    if valid is not None:
        valid_paths = {str(resolve_path(path, repo)) for path in valid["paths"]}
        if not valid_paths or valid_paths & (set(train_paths) | set(test_paths)):
            raise ValueError("Battery-level validation overlaps target training/test batteries")
    if task["family"] == "transfer" and len(train_paths) != 1:
        raise ValueError("Expected exactly one target training/validation battery")
    reference = task.get("split_json")
    if reference:
        expected = read_json(resolve_path(reference, repo))
        source_train = next(item for item in expected if item["role"] in {"train_validation", "train"})
        source_test = next(item for item in expected if item["role"] == "test")
        expected_train = source_train["paths"][:1] if task["family"] == "transfer" else source_train["paths"]
        if train["paths"] != expected_train or test["paths"] != source_test["paths"]:
            raise ValueError("Actual target split differs from the reference regular split")
    return {"train_test_paths_disjoint": True, "target_training_batteries": len(train_paths),
            "reference_split_checked": bool(reference)}


def metrics_with_batteries(output, truth, prediction, splits, prefix=""):
    import numpy as np
    metrics = common.calculate_metrics(truth, prediction)
    test = next(item for item in splits if item["role"] == "test")
    if sum(item["n_pairs"] for item in test["batteries"]) != len(truth):
        raise ValueError("Saved test predictions do not match actual battery boundaries")
    rows, start = [], 0
    for battery in test["batteries"]:
        end = start + battery["n_pairs"]
        row = {"battery": battery["path"], **common.calculate_metrics(truth[start:end], prediction[start:end])}
        row["MAPE_author_reversed"] = common.calculate_metrics(prediction[start:end], truth[start:end])["MAPE"]
        rows.append(row)
        start = end
    common.write_csv(output / (prefix + "metrics_per_battery.csv"), rows,
                     ["battery", "n_test", *common.METRICS, "MAPE_author_reversed"])
    macro = {key: float(np.mean([row[key] for row in rows])) for key in common.METRICS}
    macro["MAPE_author_reversed"] = float(np.mean([row["MAPE_author_reversed"] for row in rows]))
    return metrics, macro


def train_with_loss_variant(model, epoch, dataloader, variant, adaptation):
    import torch
    from utils.util import AverageMeter
    model.solution_u.train() if adaptation else model.train()
    meters = [AverageMeter(), AverageMeter(), AverageMeter()]
    device = next(model.parameters()).device
    for x1, x2, y1, y2 in dataloader:
        x1, x2, y1, y2 = [value.to(device) for value in (x1, x2, y1, y2)]
        u1, f1 = model.forward(x1)
        u2, f2 = model.forward(x2)
        data_loss = .5 * model.loss_func(u1, y1) + .5 * model.loss_func(u2, y2)
        residual_loss = .5 * model.loss_func(f1, torch.zeros_like(f1)) + .5 * model.loss_func(f2, torch.zeros_like(f2))
        if variant == "published_label_monotonicity":
            physical_loss = model.relu(y2 - y1).sum()
        elif variant == "paper_prediction_monotonicity":
            physical_loss = model.relu(u2 - u1).sum()
        else:
            raise ValueError("Unexpected custom physical-loss variant")
        loss = data_loss + model.alpha * residual_loss + model.beta * physical_loss
        optimizers = [model.ada_optimizer] if adaptation else [model.optimizer1, model.optimizer2]
        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        for meter, value in zip(meters, (data_loss, residual_loss, physical_loss)):
            meter.update(value.item())
    return tuple(meter.avg for meter in meters)


def build_snapshot_class(base, device, loss_variant="author_current"):
    import torch

    class SnapshotModel(base):
        def __setattr__(self, name, value):
            if name == "best_model" and isinstance(value, dict):
                value = copy.deepcopy(value)
            super().__setattr__(name, value)

        def load_model(self, model_path):
            checkpoint = torch.load(model_path, map_location=device)
            self.solution_u.load_state_dict(checkpoint["solution_u"])
            self.dynamical_F.load_state_dict(checkpoint["dynamical_F"])
            for parameter in self.solution_u.parameters():
                parameter.requires_grad = True

        def train_one_epoch(self, epoch, dataloader):
            if loss_variant == "author_current":
                return super().train_one_epoch(epoch, dataloader)
            return train_with_loss_variant(self, epoch, dataloader, loss_variant, False)

        def adaptation_one_epoch(self, epoch, dataloader):
            if loss_variant == "author_current":
                return super().adaptation_one_epoch(epoch, dataloader)
            return train_with_loss_variant(self, epoch, dataloader, loss_variant, True)

    return SnapshotModel


def run_task(task, repo, output, resume=False, allow_cpu=False, smoke=False):
    validate_task(task)
    repo, output = Path(repo).resolve(), Path(output).resolve()
    if not (repo / "Model" / "Model.py").is_file():
        raise ValueError("Invalid author repository")
    allocated_devices = os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    import numpy as np
    import torch
    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError("CUDA is unavailable; --allow-cpu is for explicit local verification")
    torch.set_num_threads(int(task.get("threads", 2)))
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    inventory = common.source_inventory(repo)
    lineage = checkpoint_lineage(task, repo, inventory) if task["family"] == "transfer" else None
    import pandas
    import scipy
    import sklearn
    environment = {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                   "pandas": pandas.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
                   "CUDA_VISIBLE_DEVICES": allocated_devices,
                   "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    implementation = {"run_transfer.py": common.digest(Path(__file__)),
                      "run_reproduction.py": common.digest(Path(common.__file__))}
    if task.get("validation_protocol") == "explicit_battery_split":
        implementation["run_extended.py"] = common.digest(Path(__file__).with_name("run_extended.py"))
    config = {"task": task, "smoke": smoke, "environment": environment,
              "implementation_sha256": implementation,
              "source_fingerprint": inventory["fingerprint"],
              "source_directory_order": inventory["data_directory_order"],
              "source_checkpoint_lineage": lineage, "checkpoint_snapshot_deepcopy": True,
              "loss_variant": task.get("loss_variant", "author_current"),
              "physical_loss_parameter_gradient": "identically_zero" if task.get("loss_variant") == "published_label_monotonicity" else "generally_nonzero"}
    comparable_config = copy.deepcopy(config)
    comparable_config["environment"].pop("CUDA_VISIBLE_DEVICES", None)
    config_hash, task_hash = stable_hash(comparable_config), stable_hash(task)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        old = read_json(manifest_path)
        if old.get("status") == "complete":
            if not resume or old.get("config_hash") != config_hash:
                raise RuntimeError("Completed task exists or configuration changed; use --resume with identical inputs")
            for name, expected in old.get("artifact_sha256", {}).items():
                if not (output / name).is_file() or common.digest(output / name) != expected:
                    raise RuntimeError("Completed artifact is missing or changed: " + name)
            common.calculate_metrics(np.load(output / "true_label.npy"), np.load(output / "pred_label.npy"))
            return old
    if output.exists() and any(output.iterdir()):
        output.rename(output.with_name(output.name + ".previous." + str(time.time_ns())))
    output.mkdir(parents=True, exist_ok=True)
    common.write_json(output / "config.json", config)
    common.write_json(output / "environment.json", environment)
    common.write_json(output / "source_inventory.json", inventory)
    manifest = {"status": "running", "task": task, "task_id": task["task_id"],
                "task_sha256": task_hash, "config_hash": config_hash,
                "source_fingerprint": inventory["fingerprint"],
                "implementation_sha256": implementation,
                "seed": task["seed"], "dataset": task.get("target", task.get("dataset")),
                "batch": task.get("batch"), "model": task.get("model", "PINN"), "family": task["family"],
                "started_at": common.utc_now(), "source_checkpoint_lineage": lineage}
    common.write_json(manifest_path, manifest)
    previous_directory, previous_argv = Path.cwd(), sys.argv
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    model, handlers = None, []
    try:
        seed = int(task["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        with (output / "console.txt").open("w", encoding="utf-8", buffering=1) as stream:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                if task["family"] == "transfer":
                    module = load_adaptation_module(repo)
                    args = default_arguments(module)
                    args.source_data, args.source_batch = task["source"], task["source_batch"]
                    args.target_data, args.target_batch = task["target"], task["target_batch"]
                    args.pretrain_model = lineage["path"]
                    args.adaptation_lr = float(task["adaptation_lr"])
                    args.adaptation_epochs = 2 if smoke else int(task.get("adaptation_epochs", task.get("epochs", 200)))
                    base, loader_name = module.AdaPINN, "load_" + task["target"] + "_data"
                else:
                    dataset = task.get("source", task["dataset"])
                    try:
                        module = importlib.import_module("main_" + dataset)
                    finally:
                        os.environ["CUDA_VISIBLE_DEVICES"] = allocated_devices
                    args = default_arguments(module)
                    args.epochs = 2 if smoke else int(task.get("epochs", 200))
                    if dataset == "XJTU":
                        args.batch = XJTU_BATCHES[task["source_batch"]]
                    elif dataset == "TJU":
                        args.batch, args.in_same_batch = task["source_batch"], True
                    base, loader_name = module.PINN, common.LOADERS[dataset]
                args.save_folder, args.log_dir = str(output), "logging.txt"
                for name, value in task.get("overrides", {}).items():
                    setattr(args, name, value)
                args.warmup_epochs = min(args.warmup_epochs, max(0, args.epochs - 1))
                common.write_json(output / "parameters.json", vars(args))
                with common.capture_data_paths() as splits:
                    if task.get("validation_protocol") == "explicit_battery_split":
                        from run_extended import load_explicit_battery_split
                        reference = read_json(resolve_path(task["split_json"], repo))
                        loaders = load_explicit_battery_split(args, reference, 1 if task["family"] == "transfer" else None)
                    elif task.get("split_json"):
                        from dataloader.dataloader import DF
                        reference = read_json(resolve_path(task["split_json"], repo))
                        train = next(item for item in reference if item["role"] == "train_validation")
                        test = next(item for item in reference if item["role"] == "test")
                        data = DF(args)
                        train_paths = train["paths"][:1] if task["family"] == "transfer" else train["paths"]
                        train_loaders = data.load_all_battery(train_paths, train["nominal_capacity"])
                        test_loaders = data.load_all_battery(test["paths"], test["nominal_capacity"])
                        loaders = {"train": train_loaders["train_2"], "valid": train_loaders["valid_2"],
                                   "test": test_loaders["test_3"]}
                    else:
                        loader = getattr(module, loader_name)
                        loaders = loader(args, small_sample=1) if task["family"] == "transfer" else loader(args)
                if task.get("validation_protocol") == "explicit_battery_split":
                    for record, role in zip(splits, ["train", "valid", "test"]):
                        record["role"] = role
                common.write_json(output / "splits.json", splits)
                manifest["split_audit"] = validate_splits(splits, task, repo)
                manifest["sample_counts"] = {key: len(value.dataset) for key, value in loaders.items()}
                for role, loader in loaders.items():
                    if not len(loader.dataset) or not all(torch.isfinite(tensor).all() for tensor in loader.dataset.tensors):
                        raise ValueError("Empty/nonfinite " + role + " samples")
                model = build_snapshot_class(base, "cuda" if torch.cuda.is_available() else "cpu",
                                             task.get("loss_variant", "author_current"))(args)
                handlers = list(model.logger.handlers)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                started = time.perf_counter()
                if task["family"] == "transfer":
                    source_truth, source_prediction = model.Test(loaders["test"])
                    np.save(output / "source_only_true_label.npy", source_truth)
                    np.save(output / "source_only_pred_label.npy", source_prediction)
                    before_f = state_hash(model.dynamical_F.state_dict())
                    before_u = state_hash(model.solution_u.state_dict())
                    manifest["source_only_metrics"], manifest["source_only_battery_macro"] = metrics_with_batteries(
                        output, source_truth, source_prediction, splits, "source_only_")
                    model.Adaptation(trainloader=loaders["train"], validloader=loaders["valid"], testloader=loaders["test"])
                    after_f = state_hash(model.dynamical_F.state_dict())
                    if before_f != after_f or any(parameter.requires_grad for parameter in model.dynamical_F.parameters()):
                        raise RuntimeError("The dynamical_F source parameters changed during adaptation")
                    manifest["frozen_dynamical_F"] = {"before_sha256": before_f, "after_sha256": after_f,
                                                       "unchanged": True, "eval_mode": not model.dynamical_F.training}
                    manifest["solution_u_updated"] = before_u != state_hash(model.solution_u.state_dict())
                    shutil.copy2(output / "finetune model.pth", output / "model.pth")
                else:
                    model.Train(trainloader=loaders["train"], validloader=loaders["valid"], testloader=loaders["test"])
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                manifest["training_seconds"] = time.perf_counter() - started
                truth = np.load(output / "true_label.npy", allow_pickle=False)
                prediction = np.load(output / "pred_label.npy", allow_pickle=False)
                manifest["metrics"], manifest["metrics_battery_macro"] = metrics_with_batteries(output, truth, prediction, splits)
                checkpoint = torch.load(output / "model.pth", map_location="cuda" if torch.cuda.is_available() else "cpu")
                model.solution_u.load_state_dict(checkpoint["solution_u"])
                model.dynamical_F.load_state_dict(checkpoint["dynamical_F"])
                replay_truth, replay_prediction = model.Test(loaders["test"])
                difference = float(np.max(np.abs(replay_prediction - prediction)))
                if not np.array_equal(replay_truth, truth) or difference > 1e-6:
                    raise RuntimeError("Best checkpoint does not reproduce validation-selected predictions")
                manifest["checkpoint_verification"] = {"passed": True, "maximum_absolute_difference": difference}
                text = (output / "logging.txt").read_text(encoding="utf-8")
                epochs = re.findall(r"\[Train\] epoch:(\d+)", text)
                manifest["epochs_completed"] = int(epochs[-1]) if epochs else None
                manifest["artifact_sha256"] = {name: common.digest(output / name) for name in
                                               ["config.json", "model.pth", "true_label.npy", "pred_label.npy", "splits.json"]}
                if task["family"] == "transfer":
                    manifest["artifact_sha256"].update({name: common.digest(output / name) for name in
                                                       ["source_only_true_label.npy", "source_only_pred_label.npy"]})
                manifest["status"] = "complete"
                common.write_json(output / "metrics.json", {key: value for key, value in manifest.items()
                                                            if "metric" in key or "macro" in key})
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        manifest["finished_at"] = common.utc_now()
        common.write_json(manifest_path, manifest)
        for handler in handlers:
            handler.close()
        logging.getLogger("mylogger").handlers.clear()
        sys.argv = previous_argv
        os.chdir(previous_directory)
        if sys.path and sys.path[0] == str(repo):
            sys.path.pop(0)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-json", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    options = parser.parse_args()
    result = run_task(read_json(options.task_json), options.repo, options.output,
                      options.resume, options.allow_cpu, options.smoke)
    print(json.dumps({"task_id": result["task"]["task_id"], "status": result["status"],
                      "metrics": result.get("metrics")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
