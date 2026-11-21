"""Stage A improvement experiment: battery-macro selection and mean/eval monotonicity.

This is a new protocol, never a replacement for frozen paper-reproduction results.
One explicit dataset/model group per specification; dry-run is the default.
"""
import argparse
from contextlib import contextmanager
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

from search_replacement_grid import read, write, sha, identity
from run_extended import task_lock

FORMAT = "pinn4soh-stage-a-improvement-v1"
FIXED_KEYS = {"epochs", "min_epochs", "patience", "warmup_epochs", "batch_size", "threads", "normalization_method",
              "diagnostic_batch_size", "grad_clip_norm", "warmup_ratio", "final_lr_ratio", "deterministic", "allow_tf32"}
STRUCTURE_KEYS = {"u_layers", "u_hidden", "F_layers_num", "F_hidden_dim"}
CANDIDATE_KEYS = {"lr_u", "lr_F", "alpha", "beta_mean", "weight_decay"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name, minimum=0, integer=False):
    require(not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value >= minimum
            and (not integer or isinstance(value, int)), "Invalid numeric field: " + name)


def build_plan(spec, repo):
    require(set(spec) == {"label", "dataset", "model", "seeds", "splits", "fixed", "structure", "candidates", "protocol_evidence", "smoke"}, "Stage A spec fields differ")
    require(spec["model"] in {"PINN", "MLP", "CNN"} and spec["dataset"] in {"MIT", "HUST", "XJTU", "TJU"}, "Unknown model/dataset")
    require(isinstance(spec["label"], str) and spec["label"] and isinstance(spec["smoke"], bool), "Label and explicit smoke bool required")
    require(isinstance(spec["protocol_evidence"], list) and spec["protocol_evidence"], "Explicit protocol evidence required")
    seeds = spec["seeds"]
    require(isinstance(seeds, list) and seeds and len(seeds) == len(set(seeds)), "Explicit distinct seeds required")
    for seed in seeds:
        number(seed, "seed", integer=True)
        require(seed < 2 ** 32, "Seed must fit NumPy uint32")
    fixed, structure = spec["fixed"], spec["structure"]
    require(set(fixed) == FIXED_KEYS and set(structure) == STRUCTURE_KEYS, "Explicit fixed settings and structure required")
    for name in ("epochs", "min_epochs", "patience", "batch_size", "threads", "diagnostic_batch_size"):
        number(fixed[name], name, 1, True)
    number(fixed["warmup_epochs"], "warmup_epochs", 0, True)
    number(fixed["grad_clip_norm"], "grad_clip_norm")
    number(fixed["warmup_ratio"], "warmup_ratio", 1e-15)
    number(fixed["final_lr_ratio"], "final_lr_ratio", 1e-15)
    require(fixed["min_epochs"] <= fixed["epochs"] <= 200 and fixed["warmup_epochs"] < fixed["epochs"] and fixed["threads"] <= 4
            and fixed["normalization_method"] in {"min-max", "z-score"} and isinstance(fixed["deterministic"], bool)
            and isinstance(fixed["allow_tf32"], bool) and fixed["warmup_ratio"] <= 1 and fixed["final_lr_ratio"] <= 1, "Invalid fixed schedule")
    if not spec["smoke"]:
        require(fixed["min_epochs"] >= 30 and fixed["patience"] >= 30 and fixed["warmup_epochs"] == 5
                and fixed["normalization_method"] == "min-max", "Formal Stage A requires min30/patience30/warmup5/minmax")
    for name, value in structure.items():
        number(value, name, 2 if "layers" in name else 1, True)
    require(structure == {"u_layers": 3, "u_hidden": 60, "F_layers_num": 3, "F_hidden_dim": 60}, "Stage A fixes the author structure 3/60")
    splits, checksums, uid_roles, path_roles = {}, {}, {}, {}
    require(set(spec["splits"]) == {"train", "valid", "test"}, "Explicit train/valid/test required")
    for role in ("train", "valid", "test"):
        record = spec["splits"][role]
        require(set(record) == {"paths", "physical_ids", "nominal_capacity"} and isinstance(record["paths"], list) and record["paths"]
                and len(record["paths"]) == len(record["physical_ids"]), "Split must pair every CSV with a physical ID")
        number(record["nominal_capacity"], "nominal_capacity", 1e-15)
        paths = []
        for name, uid in zip(record["paths"], record["physical_ids"]):
            path = (repo / name).resolve()
            require(path.is_file() and path.suffix.lower() == ".csv" and isinstance(uid, str) and uid, "Missing CSV or physical identity")
            normalized = os.path.normcase(str(path))
            require(normalized not in path_roles, "Repeated CSV path")
            require(uid not in uid_roles or uid_roles[uid] == role, "Physical battery crosses split boundary")
            path_roles[normalized], uid_roles[uid] = role, role
            paths.append(str(path))
            checksums[str(path)] = sha(path)
        splits[role] = dict(paths=paths, physical_ids=record["physical_ids"], nominal_capacity=record["nominal_capacity"])
    require(isinstance(spec["candidates"], list) and 1 <= len(spec["candidates"]) <= 32, "Explicit small candidate list required")
    candidates, seen = [], set()
    for index, candidate in enumerate(spec["candidates"]):
        require(set(candidate) == CANDIDATE_KEYS, "Every candidate must expose all five parameters")
        for name, value in candidate.items():
            number(value, name, 1e-15 if name == "lr_u" else 0)
        require(candidate["lr_F"] > 0 if candidate["alpha"] > 0 else True, "Positive G rate required when PDE is active")
        if spec["model"] != "PINN":
            require(candidate["alpha"] == candidate["beta_mean"] == candidate["lr_F"] == 0, "Baseline cannot tune PINN-only parameters")
        if candidate["alpha"] == 0:
            require(candidate["lr_F"] == 0, "Disable G optimizer when alpha=0")
        key = identity(candidate)
        require(key not in seen, "Duplicate effective candidate")
        seen.add(key)
        candidates.append(dict(candidate_id="candidate_%02d" % (index + 1), parameters=candidate))
    if not spec["smoke"]:
        require(len(candidates) == 9 and len(seeds) == 3, "Formal Stage A is nine candidates by three seeds")
    source = sorted(repo.rglob("*.py")) + [Path(__file__).resolve(), Path(__file__).with_name("search_replacement_grid.py"), Path(__file__).with_name("run_extended.py")]
    return dict(format=FORMAT, specification=spec, repo=str(repo), source_sha256={str(p.resolve()): sha(p) for p in source},
                input_sha256=checksums, resolved_splits=splits, candidates=candidates, candidate_runs=len(candidates) * len(seeds),
                objective="mean_physical_battery_RMSE_fraction; then mean across selection seeds", workers=1,
                loss_protocol=dict(data="0.5*MSE(u1,y1)+0.5*MSE(u2,y2)", pde="0.5*MSE(f1,0)+0.5*MSE(f2,0); skipped if alpha=0",
                                   monotonicity="mean(relu(u2_eval-u1_eval)); solution_u.eval with autograd enabled",
                                   reduction="mean", mono_dropout="disabled only on mono branch; data/PDE branches remain train mode"),
                preprocessing="Author full-battery 3sigma over all columns including capacity, then full-battery minmax and cycle index; offline/transductive",
                model_protocol="MLP is exact author Solution_u including initialization; CNN is author Compare_Models.CNN; paired endpoint supervision for all")


def battery_metrics(truth, prediction, batteries):
    import numpy as np
    truth, prediction = np.asarray(truth).reshape(-1).astype(np.float64), np.asarray(prediction).reshape(-1).astype(np.float64)
    require(truth.shape == prediction.shape and truth.size and np.isfinite(truth).all() and np.isfinite(prediction).all(), "Invalid metric arrays")
    require(sum(row["n_pairs"] for row in batteries) == len(truth), "Battery boundaries disagree with arrays")
    by_uid, start = {}, 0
    for row in batteries:
        end = start + row["n_pairs"]
        by_uid.setdefault(row["physical_id"], []).append((truth[start:end], prediction[start:end]))
        start = end
    rows = []
    for uid, pieces in by_uid.items():
        y, pred = np.concatenate([p[0] for p in pieces]), np.concatenate([p[1] for p in pieces])
        diff = pred - y
        require(np.all(y > 0), "Positive measured SOH required for MAPE")
        rows.append(dict(physical_id=uid, n_pairs=len(y), MSE=float(np.mean(diff ** 2)), RMSE=float(np.sqrt(np.mean(diff ** 2))),
                         MAE=float(np.mean(np.abs(diff))), MAPE_pct=float(np.mean(np.abs(diff) / y) * 100)))
    return dict(battery_macro_RMSE=sum(row["RMSE"] for row in rows) / len(rows),
                battery_macro_MAE=sum(row["MAE"] for row in rows) / len(rows),
                battery_macro_MAPE_pct=sum(row["MAPE_pct"] for row in rows) / len(rows),
                pooled_MSE=float(np.mean((prediction - truth) ** 2)), per_physical_battery=rows)


@contextmanager
def mono_eval_with_grad(solution):
    previous = solution.training
    solution.eval()
    try:
        yield
    finally:
        solution.train(previous)


def losses(model, x1, x2, y1, y2, parameters, pinn):
    import torch
    if pinn and parameters["alpha"] > 0:
        u1, f1 = model.forward(x1)
        u2, f2 = model.forward(x2)
        pde = .5 * torch.nn.functional.mse_loss(f1, torch.zeros_like(f1)) + .5 * torch.nn.functional.mse_loss(f2, torch.zeros_like(f2))
    else:
        solution = model.solution_u if pinn else model
        u1, u2 = solution(x1), solution(x2)
        pde = torch.zeros((), device=u1.device)
    data = .5 * torch.nn.functional.mse_loss(u1, y1) + .5 * torch.nn.functional.mse_loss(u2, y2)
    solution = model.solution_u if pinn else model
    with mono_eval_with_grad(solution):
        v1, v2 = solution(x1), solution(x2)
        mono = torch.relu(v2 - v1).mean()
    total = data + parameters["alpha"] * pde + parameters["beta_mean"] * mono
    return dict(data=data, pde=pde, mono=mono, total=total, train_difference=u2 - u1, eval_difference=v2 - v1)


def grad_l2(loss, parameters, weight=1):
    import torch
    if not loss.requires_grad or weight == 0:
        return 0.0
    grads = torch.autograd.grad(loss * weight, parameters, retain_graph=True, allow_unused=True)
    return float(torch.sqrt(sum((g.detach().double() ** 2).sum() for g in grads if g is not None)).cpu())


def lr_ratio(epoch, fixed):
    warmup = fixed["warmup_epochs"]
    if warmup and epoch <= warmup:
        return fixed["warmup_ratio"] + (1 - fixed["warmup_ratio"]) * ((epoch - 1) / max(1, warmup - 1))
    progress = (epoch - warmup - 1) / max(1, fixed["epochs"] - warmup - 1)
    return fixed["final_lr_ratio"] + .5 * (1 - fixed["final_lr_ratio"]) * (1 + math.cos(math.pi * progress))


def should_stop(epoch, stale, fixed):
    return epoch >= fixed["min_epochs"] and stale >= fixed["patience"]


def worker_main(worker_path):
    worker = read(worker_path)
    output, fixed, params = Path(worker["output"]), worker["fixed"], worker["parameters"]
    require(Path(worker_path).resolve().parent == output.resolve(), "Worker output mismatch")
    for name, checksum in {**worker["source_sha256"], **worker["input_sha256"]}.items():
        require(sha(name) == checksum, "Source/input changed before worker")
    import numpy as np
    import torch
    import pandas as pd
    from types import SimpleNamespace
    from torch.utils.data import TensorDataset, DataLoader
    torch.set_num_threads(fixed["threads"])
    require(worker["device"] == "cpu" or torch.cuda.is_available(), "Requested GPU unavailable")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = fixed["deterministic"]
    torch.backends.cuda.matmul.allow_tf32 = fixed["allow_tf32"]
    torch.backends.cudnn.allow_tf32 = fixed["allow_tf32"]
    torch.use_deterministic_algorithms(fixed["deterministic"])
    random.seed(worker["seed"])
    np.random.seed(worker["seed"])
    torch.manual_seed(worker["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(worker["seed"])
    sys.path.insert(0, worker["repo"])
    module = importlib.import_module("Model.Model")
    from Model.Compare_Models import CNN
    from dataloader.dataloader import DF
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pinn = worker["model"] == "PINN"
    if pinn:
        args = SimpleNamespace(epochs=fixed["epochs"], warmup_epochs=fixed["warmup_epochs"], warmup_lr=params["lr_u"] * fixed["warmup_ratio"],
            lr=params["lr_u"], final_lr=params["lr_u"] * fixed["final_lr_ratio"], lr_F=params["lr_F"], F_layers_num=3, F_hidden_dim=60,
            alpha=params["alpha"], beta=params["beta_mean"], save_folder=str(output), log_dir="author_args.log")
        model = module.PINN(args)
        if params["alpha"] == 0:
            for parameter in model.dynamical_F.parameters():
                parameter.requires_grad_(False)
        solution = model.solution_u
    else:
        model = (module.Solution_u() if worker["model"] == "MLP" else CNN()).to(device)
        solution = model
    solution_parameters = list(solution.parameters())
    initial_solution_sha = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in solution_parameters)).hexdigest()
    # Reset training RNG after model creation. PINN alpha=beta=0 and exact MLP
    # then see the same data order/dropout masks as a controlled equivalence check.
    training_seed = (worker["seed"] + 101) % (2 ** 32)
    torch.manual_seed(training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_seed)
    args = SimpleNamespace(normalization_method=fixed["normalization_method"])
    data, loaders, tensor_identity = DF(args), {}, {}
    for role, record in worker["splits"].items():
        pieces, batteries = [], []
        for name, uid in zip(record["paths"], record["physical_ids"]):
            (x1, y1), (x2, y2) = data.load_one_battery(name, record["nominal_capacity"])
            values = [torch.tensor(v, dtype=torch.float32) for v in (x1, x2, y1[:, None], y2[:, None])]
            require(values[0].ndim == 2 and values[0].shape[1] == 17 and len(values[0]) > 0 and all(torch.isfinite(v).all() for v in values), "Invalid battery tensors")
            pieces.append(values)
            batteries.append(dict(path=name, physical_id=uid, n_pairs=len(values[0])))
        tensors = [torch.cat([row[i] for row in pieces]) for i in range(4)]
        generator = torch.Generator().manual_seed(training_seed + 1)
        loaders[role] = DataLoader(TensorDataset(*tensors), batch_size=fixed["batch_size"], shuffle=role == "train", num_workers=0, generator=generator)
        tensor_identity[role] = dict(batteries=batteries, sha256=[hashlib.sha256(v.numpy().tobytes()).hexdigest() for v in tensors])
    write(output / "data_identity.json", tensor_identity)

    def predict(loader):
        solution.eval()
        truth, prediction = [], []
        with torch.no_grad():
            for x1, _, y1, _ in loader:
                truth.append(y1.numpy())
                prediction.append(solution(x1.to(device)).cpu().numpy())
        return np.concatenate(truth), np.concatenate(prediction)

    environment = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__, cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        device=str(device), threads=torch.get_num_threads(), workers=1, deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    started = time.perf_counter()
    if worker["mode"] == "train":
        require(set(loaders) == {"train", "valid"}, "Train worker must not load test")
        optimizer_u = torch.optim.Adam(solution.parameters(), lr=params["lr_u"], weight_decay=params["weight_decay"])
        optimizer_F = torch.optim.Adam(model.dynamical_F.parameters(), lr=params["lr_F"], weight_decay=0) if pinn and params["alpha"] > 0 else None
        diagnostic_tensors = tuple(t[:min(fixed["diagnostic_batch_size"], len(t))].clone().to(device) for t in loaders["train"].dataset.tensors)
        write(output / "diagnostic_batch.json", dict(indices=list(range(len(diagnostic_tensors[0]))),
            sha256=[hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for v in diagnostic_tensors],
            data_branch_dropout="train", mono_branch_dropout="eval_with_grad", pde_computed=params["alpha"] > 0))
        best, stale, history, diagnostics, best_epoch, best_prediction = float("inf"), 0, [], [], None, None
        for epoch in range(1, fixed["epochs"] + 1):
            model.train()
            ratio = lr_ratio(epoch, fixed)
            for group in optimizer_u.param_groups:
                group["lr"] = params["lr_u"] * ratio
            if optimizer_F:
                for group in optimizer_F.param_groups:
                    group["lr"] = params["lr_F"] * ratio
            # Isolate diagnostic RNG so logging cannot alter training masks.
            devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed((worker["seed"] + 313) % (2 ** 32))
                values = losses(model, *(v.clone() for v in diagnostic_tensors), params, pinn)
                diag = dict(epoch=epoch, n=len(diagnostic_tensors[0]), mono_requires_grad=values["mono"].requires_grad,
                    unweighted_loss={name: float(values[name].detach().cpu()) for name in ("data", "pde", "mono")},
                    grad_l2_solution={name: grad_l2(values[name], solution_parameters) for name in ("data", "pde", "mono")},
                    weighted_grad_l2_solution=dict(data=grad_l2(values["data"], solution_parameters),
                        pde=grad_l2(values["pde"], solution_parameters, params["alpha"]), mono=grad_l2(values["mono"], solution_parameters, params["beta_mean"])),
                    pde_computed=params["alpha"] > 0, data_dropout="train", mono_dropout="eval_with_grad")
                diag["weighted_loss"] = dict(data=diag["unweighted_loss"]["data"], pde=params["alpha"] * diag["unweighted_loss"]["pde"],
                                             mono=params["beta_mean"] * diag["unweighted_loss"]["mono"])
                require(all(math.isfinite(value) for group in (diag["unweighted_loss"], diag["weighted_loss"],
                    diag["grad_l2_solution"], diag["weighted_grad_l2_solution"]) for value in group.values()), "Nonfinite fixed-batch diagnostics")
                diagnostics.append(diag)
            meters = {name: 0.0 for name in ("data", "pde", "mono", "total")}
            n_seen, sizes, clip_hits, train_delta, eval_delta = 0, [], 0, [], []
            for x1, x2, y1, y2 in loaders["train"]:
                batch = [v.to(device) for v in (x1, x2, y1, y2)]
                values = losses(model, *batch, params, pinn)
                require(all(torch.isfinite(values[name]).all().item() for name in meters), "Nonfinite loss")
                optimizer_u.zero_grad()
                if optimizer_F:
                    optimizer_F.zero_grad()
                values["total"].backward()
                active = list(solution.parameters()) + (list(model.dynamical_F.parameters()) if optimizer_F else [])
                require(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in active), "Nonfinite gradients")
                if fixed["grad_clip_norm"] > 0:
                    norm = torch.nn.utils.clip_grad_norm_(active, fixed["grad_clip_norm"])
                    clip_hits += int(norm > fixed["grad_clip_norm"])
                optimizer_u.step()
                if optimizer_F:
                    optimizer_F.step()
                count = len(x1)
                n_seen += count
                sizes.append(count)
                for name in meters:
                    meters[name] += float(values[name].detach().cpu()) * count
                train_delta.append(values["train_difference"].detach().cpu().numpy().reshape(-1))
                eval_delta.append(values["eval_difference"].detach().cpu().numpy().reshape(-1))
            truth, prediction = predict(loaders["valid"])
            metrics = battery_metrics(truth, prediction, tensor_identity["valid"]["batteries"])
            score = metrics["battery_macro_RMSE"]
            stale += 1
            if score < best:
                best, best_epoch, stale = score, epoch, 0
                state_dict = dict(solution_u=copy.deepcopy(solution.state_dict()))
                if pinn:
                    state_dict["dynamical_F"] = copy.deepcopy(model.dynamical_F.state_dict())
                torch.save(state_dict, output / "best_model.pth")
                best_prediction = prediction.copy()
            def delta_summary(arrays):
                values = np.concatenate(arrays).astype(np.float64)
                return dict(positive_fraction=float(np.mean(values > 0)), positive_mean=float(np.mean(np.maximum(values, 0))),
                    quantiles=dict(zip(("min", "median", "p90", "p99", "max"), map(float, np.quantile(values, [0, .5, .9, .99, 1])))))
            history.append(dict(epoch=epoch, losses={name: value / n_seen for name, value in meters.items()},
                weighted_losses=dict(data=meters["data"] / n_seen, pde=params["alpha"] * meters["pde"] / n_seen,
                                     mono=params["beta_mean"] * meters["mono"] / n_seen),
                lr_u=params["lr_u"] * ratio, lr_F=params["lr_F"] * ratio if optimizer_F else None,
                validation=metrics, stale_epochs=stale, batch_count=len(sizes), batch_min=min(sizes), batch_max=max(sizes),
                clip_trigger_batches=clip_hits, gradients_finite=True, train_monotonicity=delta_summary(train_delta),
                eval_monotonicity=delta_summary(eval_delta), mono_branch_mode="eval_with_grad", data_branch_mode="train",
                pde_computed=params["alpha"] > 0))
            write(output / "history.json", history)
            write(output / "gradient_diagnostics.json", diagnostics)
            if should_stop(epoch, stale, fixed):
                break
        checkpoint = torch.load(output / "best_model.pth", map_location=device, weights_only=True)
        solution.load_state_dict(checkpoint["solution_u"], strict=True)
        if pinn:
            model.dynamical_F.load_state_dict(checkpoint["dynamical_F"], strict=True)
        truth, prediction = predict(loaders["valid"])
        require(np.array_equal(best_prediction, prediction), "Reloaded best checkpoint predictions differ")
        metrics = battery_metrics(truth, prediction, tensor_identity["valid"]["batteries"])
        require(metrics["battery_macro_RMSE"] == best, "Reloaded validation objective differs")
        np.save(output / "valid_truth.npy", truth)
        np.save(output / "valid_prediction.npy", prediction)
        write(output / "validation_metrics.json", metrics)
        result = dict(objective=best, validation_battery_macro_RMSE=best, best_epoch=best_epoch, epochs_completed=len(history),
            test_loaded=False, checkpoint_reload_bitwise_prediction_equal=True, physical_validation_batteries=len(metrics["per_physical_battery"]),
            G_optimizer_enabled=optimizer_F is not None)
        names = ["best_model.pth", "valid_truth.npy", "valid_prediction.npy", "validation_metrics.json", "data_identity.json", "history.json", "gradient_diagnostics.json", "diagnostic_batch.json"]
    else:
        require(set(loaders) == {"test"} and sha(worker["selection"]) == worker["selection_sha256"], "Test requires frozen selection")
        selected = read(worker["selection"])["selected"]
        require(selected["candidate_id"] == worker["candidate_id"] and selected["parameters"] == params, "Test checkpoint not selected")
        require(sha(worker["checkpoint"]) == worker["checkpoint_sha256"], "Selected checkpoint SHA differs")
        checkpoint = torch.load(worker["checkpoint"], map_location=device, weights_only=True)
        solution.load_state_dict(checkpoint["solution_u"], strict=True)
        if pinn:
            model.dynamical_F.load_state_dict(checkpoint["dynamical_F"], strict=True)
        truth, prediction = predict(loaders["test"])
        metrics = battery_metrics(truth, prediction, tensor_identity["test"]["batteries"])
        np.save(output / "test_truth.npy", truth)
        np.save(output / "test_prediction.npy", prediction)
        write(output / "test_metrics.json", metrics)
        result = dict(test_battery_macro_RMSE=metrics["battery_macro_RMSE"], test_used_for_selection=False,
                      selection_sha256=worker["selection_sha256"])
        names = ["test_truth.npy", "test_prediction.npy", "test_metrics.json", "data_identity.json"]
    result.update(format=FORMAT, status="complete", worker_identity=identity(worker), model=worker["model"], initial_solution_sha256=initial_solution_sha,
                  environment=environment, seconds=time.perf_counter() - started, artifact_sha256={name: sha(output / name) for name in names})
    write(output / "result.json", result)


def check_completed(folder, worker):
    import numpy as np
    result = read(folder / "result.json")
    require(result["status"] == "complete" and result["worker_identity"] == identity(worker), "Completed worker identity changed")
    for name, checksum in result["artifact_sha256"].items():
        require(sha(folder / name) == checksum, "Completed artifact SHA changed")
    prefix, role = ("valid", "valid") if worker["mode"] == "train" else ("test", "test")
    metrics = battery_metrics(np.load(folder / (prefix + "_truth.npy"), allow_pickle=False), np.load(folder / (prefix + "_prediction.npy"), allow_pickle=False),
        read(folder / "data_identity.json")[role]["batteries"])
    key = "objective" if role == "valid" else "test_battery_macro_RMSE"
    require(metrics["battery_macro_RMSE"] == result[key], "Saved arrays disagree with objective")
    return result


def launch_worker(folder, worker, options):
    if (folder / "result.json").exists():
        require(options.resume, "Existing result requires --resume")
        return check_completed(folder, worker)
    if folder.exists() and any(folder.iterdir()):
        require(options.resume, "Incomplete directory exists; inspect before --resume")
        folder.rename(folder.with_name(folder.name + ".incomplete." + str(time.time_ns())))
    folder.mkdir(parents=True, exist_ok=True)
    write(folder / "worker.json", worker)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="" if options.device == "cpu" else options.device,
        OMP_NUM_THREADS=str(worker["fixed"]["threads"]), MKL_NUM_THREADS=str(worker["fixed"]["threads"]), CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUNBUFFERED="1")
    with (folder / "console.log").open("w", encoding="utf-8") as stream:
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker-json", str(folder / "worker.json")],
                                 env=environment, stdout=stream, stderr=subprocess.STDOUT)
    require(process.returncode == 0, "Stage A worker failed; inspect " + str(folder / "console.log"))
    return check_completed(folder, worker)


def select(rows, candidates, seeds):
    require(len(rows) == len(candidates) * len(seeds) and {(r["candidate_id"], r["seed"]) for r in rows} ==
            {(c["candidate_id"], seed) for c in candidates for seed in seeds}, "Selection requires complete candidate/seed matrix")
    ranking = []
    for candidate in candidates:
        scores = [row["objective"] for row in rows if row["candidate_id"] == candidate["candidate_id"]]
        require(all(math.isfinite(v) for v in scores), "Nonfinite selection objective")
        ranking.append(dict(**candidate, mean_validation_battery_macro_RMSE=sum(scores) / len(scores), per_seed_objectives=scores))
    ranking.sort(key=lambda row: (row["mean_validation_battery_macro_RMSE"], row["candidate_id"]))
    return dict(format=FORMAT, selected=ranking[0], ranking=ranking, test_metrics_used=False)


def run(options):
    require(options.device == "cpu" or options.device.isdigit(), "Use one GPU index")
    require(not options.execute or options.device != "cpu" or options.allow_cpu, "CPU execution requires --allow-cpu")
    repo, output = options.repo.resolve(), options.output.resolve()
    require(repo != output and repo not in output.parents, "Output must be outside source repo")
    plan = build_plan(read(options.spec), repo)
    plan["device"] = options.device
    if output.exists():
        require((output / "plan.json").is_file() and read(output / "plan.json") == plan, "Existing output differs from plan/source/data")
    else:
        output.mkdir(parents=True)
        write(output / "plan.json", plan)
    if not options.execute:
        print(json.dumps(dict(status="dry_run", runs=plan["candidate_runs"], smoke=plan["specification"]["smoke"], plan_sha256=identity(plan))))
        return
    with task_lock(output / ".stage_a.lock"):
        state = dict(format=FORMAT, status="running", completed_runs=0, total_runs=plan["candidate_runs"], plan_sha256=identity(plan), workers=1,
                     smoke=plan["specification"]["smoke"])
        write(output / "status.json", state)
        def pause():
            if options.pause_file and options.pause_file.exists():
                state.update(status="paused_after_completed_worker", pause_file=str(options.pause_file.resolve()))
                write(output / "status.json", state)
                return True
            return False
        try:
            rows = []
            for candidate in plan["candidates"]:
                for seed in plan["specification"]["seeds"]:
                    if pause():
                        return
                    folder = output / candidate["candidate_id"] / ("seed_%d" % seed)
                    worker = dict(mode="train", model=plan["specification"]["model"], repo=str(repo), output=str(folder), fixed=plan["specification"]["fixed"],
                        parameters=candidate["parameters"], candidate_id=candidate["candidate_id"], seed=seed, device=options.device,
                        source_sha256=plan["source_sha256"], splits={role: plan["resolved_splits"][role] for role in ("train", "valid")},
                        input_sha256={name: plan["input_sha256"][name] for role in ("train", "valid") for name in plan["resolved_splits"][role]["paths"]})
                    result = launch_worker(folder, worker, options)
                    rows.append(dict(candidate_id=candidate["candidate_id"], seed=seed, objective=result["objective"], result_sha256=sha(folder / "result.json")))
                    state["completed_runs"] += 1
                    write(output / "status.json", state)
            selection = select(rows, plan["candidates"], plan["specification"]["seeds"])
            selection.update(plan_sha256=identity(plan), candidate_results=rows)
            if (output / "selection.json").exists():
                require(read(output / "selection.json") == selection, "Selection differs from frozen record")
            else:
                write(output / "selection.json", selection)
            if options.evaluate_test:
                for seed in plan["specification"]["seeds"]:
                    if pause():
                        return
                    candidate = selection["selected"]
                    checkpoint = output / candidate["candidate_id"] / ("seed_%d" % seed) / "best_model.pth"
                    folder = output / "selected_test" / ("seed_%d" % seed)
                    worker = dict(mode="test", model=plan["specification"]["model"], repo=str(repo), output=str(folder), fixed=plan["specification"]["fixed"],
                        parameters=candidate["parameters"], candidate_id=candidate["candidate_id"], seed=seed, device=options.device,
                        source_sha256=plan["source_sha256"], splits={"test": plan["resolved_splits"]["test"]},
                        input_sha256={name: plan["input_sha256"][name] for name in plan["resolved_splits"]["test"]["paths"]},
                        checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), selection=str(output / "selection.json"), selection_sha256=sha(output / "selection.json"))
                    launch_worker(folder, worker, options)
            state.update(status="complete", selected_candidate=selection["selected"]["candidate_id"], test_evaluated=options.evaluate_test)
        except BaseException as error:
            state.update(status="failed", error=repr(error))
            raise
        finally:
            write(output / "status.json", state)
    print(json.dumps(state))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("spec", "repo", "output"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--pause-file", type=Path)
    parser.add_argument("--worker-json", type=Path)
    options = parser.parse_args()
    if options.worker_json:
        worker_main(options.worker_json)
    else:
        require(all((options.spec, options.repo, options.output)), "--spec/--repo/--output required")
        require(not options.evaluate_test or options.execute, "Test evaluation requires --execute")
        run(options)
