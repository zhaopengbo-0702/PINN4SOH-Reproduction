"""Explicit per-method validation search for MIT/HUST robustness tuning.

New experimental protocol, separate from frozen reproduction and XJTU search.
Defaults to dry-run. Each spec is one dataset/model/loss group with an explicit
candidate list and disjoint battery splits supplied by the protocol designer.
"""
import argparse
import copy
import importlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from search_replacement_grid import read, write, sha, identity, worker_main as pinn_worker
from run_extended import task_lock

FORMAT = "pinn4soh-robustness-tuning-v1"
FIXED = {"epochs", "early_stop", "batch_size", "threads", "normalization_method", "warmup_epochs",
         "warmup_lr", "lr", "final_lr", "lr_F"}
STRUCTURE = {"u_layers", "u_hidden", "F_layers_num", "F_hidden_dim", "alpha", "beta"}
TUNABLE = {"lr", "lr_F", "warmup_lr", "final_lr", "batch_size", "normalization_method", "alpha", "beta"}


def require(value, text):
    if not value:
        raise ValueError(text)


def plan_search(spec, repo):
    require(set(spec) == {"label", "dataset", "model", "loss_variant", "seeds", "splits", "fixed", "structure", "candidates", "protocol_evidence"},
            "Explicit spec fields required")
    require(spec["dataset"] in {"MIT", "HUST", "XJTU", "TJU"} and spec["model"] in {"PINN", "MLP", "CNN"}, "Unsupported dataset/model")
    require(spec["loss_variant"] in {"paper_prediction_monotonicity", "published_label_monotonicity", "author_current", "supervised_pair_mse"}, "Invalid loss")
    require((spec["model"] == "PINN") == (spec["loss_variant"] != "supervised_pair_mse"), "Baseline/PINN loss must be explicit")
    require(isinstance(spec["protocol_evidence"], list) and spec["protocol_evidence"], "Cited split/grid protocol required")
    seeds = spec["seeds"]
    require(isinstance(seeds, list) and seeds and len(seeds) == len(set(seeds)) and
            all(isinstance(s, int) and not isinstance(s, bool) and s >= 0 for s in seeds), "Distinct integer seeds required")
    require(set(spec["fixed"]) == FIXED and set(spec["structure"]) == STRUCTURE, "All fixed/structure values required")
    require(spec["model"] == "PINN" or (spec["structure"]["u_layers"] == 3 and spec["structure"]["u_hidden"] == 60),
            "Baseline author architectures are fixed; structure must identify u_layers=3/u_hidden=60")
    require(set(spec["splits"]) == {"train", "valid", "test"}, "Explicit battery splits required")
    splits, input_sha, path_sets = {}, {}, []
    for role, record in spec["splits"].items():
        require(set(record) == {"paths", "nominal_capacity"} and record["paths"] and record["nominal_capacity"] > 0, "Invalid split")
        paths = [str((repo / name).resolve()) for name in record["paths"]]
        require(len(paths) == len(set(paths)) and all(Path(p).is_file() for p in paths), "Missing/duplicate split file")
        splits[role] = dict(paths=paths, nominal_capacity=record["nominal_capacity"])
        input_sha.update({name: sha(name) for name in paths})
        path_sets.append({os.path.normcase(name) for name in paths})
    require(not any(a & b for i, a in enumerate(path_sets) for b in path_sets[i + 1:]), "Battery path overlap")
    candidates, seen = [], set()
    require(isinstance(spec["candidates"], list) and 1 <= len(spec["candidates"]) <= 128, "Use 1..128 explicit candidates")
    for index, values in enumerate(spec["candidates"]):
        require(isinstance(values, dict) and values and not set(values) - TUNABLE, "Unsupported candidate setting")
        fixed, structure = dict(spec["fixed"]), dict(spec["structure"])
        for name, value in values.items():
            (structure if name in STRUCTURE else fixed)[name] = value
        require(fixed["normalization_method"] in {"min-max", "z-score"}, "Unknown normalization")
        for name, value in {**{k: v for k, v in fixed.items() if k != "normalization_method"}, **structure}.items():
            require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0, "Invalid numeric setting")
        require(all(isinstance(fixed[name], int) and fixed[name] >= 1 for name in ("epochs", "early_stop", "batch_size", "threads")) and
                fixed["epochs"] <= 200 and fixed["threads"] <= 4 and isinstance(fixed["warmup_epochs"], int) and
                0 <= fixed["warmup_epochs"] < fixed["epochs"], "Invalid epoch/thread/batch schedule")
        require(all(isinstance(structure[name], int) and structure[name] >= (2 if "layers" in name else 1)
                    for name in ("u_layers", "u_hidden", "F_layers_num", "F_hidden_dim")), "Invalid structure")
        require(all(fixed[name] > 0 for name in ("lr", "warmup_lr", "lr_F")), "Positive optimizer rates required")
        require(spec["model"] == "PINN" or not ({"alpha", "beta", "lr_F"} & set(values)), "Baseline cannot tune unused PINN settings")
        require(spec["loss_variant"] != "published_label_monotonicity" or "beta" not in values, "Label-only beta has zero parameter gradient; do not tune it")
        key = identity(dict(fixed=fixed, parameters=structure))
        require(key not in seen, "Duplicate effective candidate")
        seen.add(key)
        candidates.append(dict(candidate_id="candidate_%04d" % (index + 1), fixed=fixed, parameters=structure, explicit_overrides=values))
    source = list(sorted(repo.rglob("*.py"))) + [Path(__file__).resolve(), Path(__file__).with_name("search_replacement_grid.py"),
        Path(__file__).with_name("run_extended.py"), Path(__file__).with_name("run_transfer.py"), Path(__file__).with_name("run_reproduction.py")]
    return dict(format=FORMAT, specification=spec, candidates=candidates, resolved_splits=splits, input_sha256=input_sha,
                source_sha256={str(p.resolve()): sha(p) for p in source}, repo=str(repo),
                candidate_runs=len(candidates) * len(seeds), selection="mean_seed_minimum_validation_sample_weighted_MSE",
                baseline_supervision="0.5*MSE(x1,y1)+0.5*MSE(x2,y2); same paired endpoint exposure as PINN",
                limitations=["Additional tuning protocol, not historical paper reproduction", "Per-battery full-trajectory normalization and capacity-dependent 3sigma inherited from author",
                             "Physical cell disjointness requires protocol evidence beyond file-path disjointness", "Fixed split/three seeds cannot guarantee population robustness"])


def completed(directory, worker):
    result = read(directory / "result.json")
    require(result["status"] == "complete" and result["worker_identity"] == identity(worker), "Completed worker identity differs")
    for name, checksum in result["artifact_sha256"].items():
        require(sha(directory / name) == checksum, "Completed artifact SHA differs")
    import numpy as np
    prefix = "valid" if worker["mode"] == "train" else "test"
    truth = np.load(directory / (prefix + "_truth.npy"), allow_pickle=False).astype(np.float64)
    pred = np.load(directory / (prefix + "_prediction.npy"), allow_pickle=False).astype(np.float64)
    require(truth.shape == pred.shape and truth.size and np.isfinite(truth).all() and np.isfinite(pred).all(), "Invalid evaluation arrays")
    key = "validation_mse" if prefix == "valid" else "test_mse"
    require(float(np.mean((truth - pred) ** 2)) == result[key], "Stored objective differs from arrays")
    return result


def launch_worker(directory, worker, device, resume):
    if (directory / "result.json").exists():
        require(resume, "Existing results require --resume")
        return completed(directory, worker)
    if directory.exists() and any(directory.iterdir()):
        require(resume, "Incomplete worker exists; inspect before --resume")
        directory.rename(directory.with_name(directory.name + ".incomplete." + str(time.time_ns())))
    directory.mkdir(parents=True, exist_ok=True)
    write(directory / "worker.json", worker)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="" if device == "cpu" else device,
                       OMP_NUM_THREADS=str(worker["fixed"]["threads"]), MKL_NUM_THREADS=str(worker["fixed"]["threads"]))
    with (directory / "console.log").open("w", encoding="utf-8") as stream:
        p = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker-json", str(directory / "worker.json")],
                           env=environment, stdout=stream, stderr=subprocess.STDOUT)
    require(p.returncode == 0, "Worker failed; inspect " + str(directory / "console.log"))
    return completed(directory, worker)


def baseline_worker(worker):
    import numpy as np
    import torch
    from types import SimpleNamespace
    from torch.utils.data import DataLoader, TensorDataset
    for name, checksum in {**worker["source_sha256"], **worker["input_sha256"]}.items():
        require(sha(name) == checksum, "Source/input changed")
    torch.set_num_threads(worker["fixed"]["threads"])
    require(worker["device"] == "cpu" or torch.cuda.is_available(), "GPU unavailable; no silent CPU fallback")
    random.seed(worker["seed"])
    np.random.seed(worker["seed"])
    torch.manual_seed(worker["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(worker["seed"])
    sys.path.insert(0, worker["repo"])
    from Model.Compare_Models import MLP, CNN
    from Model.Model import LR_Scheduler
    from dataloader.dataloader import DF
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (MLP() if worker["model"] == "MLP" else CNN()).to(device)
    fixed, output = worker["fixed"], Path(worker["output"])
    args = SimpleNamespace(**fixed, save_folder=str(output), log_dir=None)
    loaders, data_hashes = {}, {}
    data = DF(args)
    for role, record in worker["splits"].items():
        pieces, batteries = [], []
        for name in record["paths"]:
            (x1, y1), (x2, y2) = data.load_one_battery(name, record["nominal_capacity"])
            tensors = [torch.tensor(v, dtype=torch.float32) for v in (x1, x2, y1.reshape(-1, 1), y2.reshape(-1, 1))]
            require(tensors[0].ndim == 2 and tensors[0].shape[1] == 17 and len(tensors[0]) > 0 and all(torch.isfinite(t).all() for t in tensors), "Invalid data")
            pieces.append(tensors)
            batteries.append(dict(path=name, n_pairs=len(tensors[0])))
        tensors = [torch.cat([row[i] for row in pieces]) for i in range(4)]
        data_hashes[role] = dict(sha256=[__import__("hashlib").sha256(t.numpy().tobytes()).hexdigest() for t in tensors], batteries=batteries)
        loaders[role] = DataLoader(TensorDataset(*tensors), batch_size=fixed["batch_size"], shuffle=role == "train", num_workers=0)
    write(output / "data_identity.json", data_hashes)

    def predict(loader):
        model.eval()
        truth, pred = [], []
        with torch.no_grad():
            for x1, _, y1, _ in loader:
                truth.append(y1.numpy())
                pred.append(model(x1.to(device)).cpu().numpy())
        return np.concatenate(truth), np.concatenate(pred)

    if worker["mode"] == "train":
        require(set(loaders) == {"train", "valid"}, "No test loaders allowed during training")
        optimizer = torch.optim.Adam(model.parameters(), lr=fixed["warmup_lr"])
        scheduler = LR_Scheduler(optimizer, fixed["warmup_epochs"], fixed["warmup_lr"], fixed["epochs"], fixed["lr"], fixed["final_lr"])
        best, stale, history, best_epoch = float("inf"), 0, [], None
        for epoch in range(1, fixed["epochs"] + 1):
            model.train()
            losses = []
            for x1, x2, y1, y2 in loaders["train"]:
                x1, x2, y1, y2 = [v.to(device) for v in (x1, x2, y1, y2)]
                loss = .5 * torch.nn.functional.mse_loss(model(x1), y1) + .5 * torch.nn.functional.mse_loss(model(x2), y2)
                require(torch.isfinite(loss).item(), "Nonfinite training loss")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            scheduler.step()
            truth, pred = predict(loaders["valid"])
            score = float(np.mean((truth.astype(np.float64) - pred.astype(np.float64)) ** 2))
            require(math.isfinite(score), "Nonfinite validation MSE")
            history.append(dict(epoch=epoch, train_pair_mse=sum(losses) / len(losses), validation_mse=score))
            stale += 1
            if score < best:
                best, best_epoch, stale = score, epoch, 0
                torch.save(copy.deepcopy(model.state_dict()), output / "best_model.pth")
            write(output / "history.json", history)
            if stale > fixed["early_stop"]:
                break
        model.load_state_dict(torch.load(output / "best_model.pth", map_location=device, weights_only=True))
        truth, pred = predict(loaders["valid"])
        require(float(np.mean((truth.astype(np.float64) - pred.astype(np.float64)) ** 2)) == best, "Checkpoint reload failed")
        np.save(output / "valid_truth.npy", truth)
        np.save(output / "valid_prediction.npy", pred)
        result = dict(validation_mse=best, best_epoch=best_epoch, epochs_completed=len(history), test_loaded=False)
        names = ["best_model.pth", "history.json", "valid_truth.npy", "valid_prediction.npy", "data_identity.json"]
    else:
        require(set(loaders) == {"test"} and sha(worker["selection"]) == worker["selection_sha256"], "Test selection not frozen")
        selected = read(worker["selection"])["selected"]
        require(selected["candidate_id"] == worker["candidate_id"] and selected["fixed"] == fixed and selected["parameters"] == worker["parameters"], "Unselected checkpoint")
        require(sha(worker["checkpoint"]) == worker["checkpoint_sha256"], "Checkpoint changed")
        model.load_state_dict(torch.load(worker["checkpoint"], map_location=device, weights_only=True))
        truth, pred = predict(loaders["test"])
        difference = pred.astype(np.float64) - truth.astype(np.float64)
        np.save(output / "test_truth.npy", truth)
        np.save(output / "test_prediction.npy", pred)
        result = dict(test_mse=float(np.mean(difference ** 2)), test_mae=float(np.mean(np.abs(difference))),
                      test_used_for_selection=False, selection_sha256=worker["selection_sha256"])
        names = ["test_truth.npy", "test_prediction.npy", "data_identity.json"]
    result.update(format=FORMAT, status="complete", worker_identity=identity(worker), artifact_sha256={name: sha(output / name) for name in names},
                  environment=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
                                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, device=str(device), threads=torch.get_num_threads(), workers=1))
    write(output / "result.json", result)


def run(options):
    require(options.device == "cpu" or options.device.isdigit(), "One GPU only")
    require(not options.execute or options.device != "cpu" or options.allow_cpu, "CPU requires --allow-cpu")
    repo, output = options.repo.resolve(), options.output.resolve()
    require(repo != output and repo not in output.parents, "Output outside author repo required")
    plan = plan_search(read(options.spec), repo)
    plan["device"] = options.device
    if output.exists():
        require((output / "plan.json").is_file() and read(output / "plan.json") == plan, "Output exists with different plan/source/data")
    else:
        output.mkdir(parents=True)
        write(output / "plan.json", plan)
    if not options.execute:
        print(json.dumps(dict(status="dry_run", candidate_runs=plan["candidate_runs"], plan_sha256=identity(plan))))
        return
    with task_lock(output / ".search.lock"):
        state = dict(status="running", completed_runs=0, total_runs=plan["candidate_runs"], plan_sha256=identity(plan), workers=1)
        write(output / "status.json", state)
        try:
            rows = []
            def pause_requested():
                if options.pause_file and options.pause_file.exists():
                    state.update(status="paused_after_completed_worker", pause_file=str(options.pause_file.resolve()))
                    write(output / "status.json", state)
                    print(json.dumps(state), flush=True)
                    return True
                return False
            for candidate in plan["candidates"]:
                for seed in plan["specification"]["seeds"]:
                    if pause_requested():
                        return
                    folder = output / candidate["candidate_id"] / ("seed_%d" % seed)
                    worker = dict(mode="train", model=plan["specification"]["model"], repo=str(repo), output=str(folder),
                                  fixed=candidate["fixed"], loss_variant=plan["specification"]["loss_variant"], seed=seed,
                                  parameters=candidate["parameters"], candidate_id=candidate["candidate_id"], device=options.device,
                                  source_sha256=plan["source_sha256"], splits={role: plan["resolved_splits"][role] for role in ("train", "valid")},
                                  input_sha256={name: plan["input_sha256"][name] for role in ("train", "valid") for name in plan["resolved_splits"][role]["paths"]})
                    result = launch_worker(folder, worker, options.device, options.resume)
                    rows.append(dict(candidate_id=candidate["candidate_id"], seed=seed, validation_mse=result["validation_mse"], result_sha256=sha(folder / "result.json")))
                    state["completed_runs"] += 1
                    write(output / "status.json", state)
            ranking = []
            for candidate in plan["candidates"]:
                scores = [row["validation_mse"] for row in rows if row["candidate_id"] == candidate["candidate_id"]]
                require(len(scores) == len(plan["specification"]["seeds"]), "Incomplete candidate")
                ranking.append(dict(**candidate, mean_validation_mse=sum(scores) / len(scores), per_seed_validation_mse=scores))
            ranking.sort(key=lambda row: (row["mean_validation_mse"], row["candidate_id"]))
            if pause_requested():
                return
            selection = dict(format=FORMAT, selected=ranking[0], ranking=ranking, candidate_results=rows, test_metrics_used=False, plan_sha256=identity(plan))
            if (output / "selection.json").exists():
                require(read(output / "selection.json") == selection, "Selection changed")
            else:
                write(output / "selection.json", selection)
            if options.evaluate_test:
                candidate = ranking[0]
                for seed in plan["specification"]["seeds"]:
                    if pause_requested():
                        return
                    folder = output / "selected_test" / ("seed_%d" % seed)
                    checkpoint = output / candidate["candidate_id"] / ("seed_%d" % seed) / "best_model.pth"
                    worker = dict(mode="test", model=plan["specification"]["model"], repo=str(repo), output=str(folder),
                                  fixed=candidate["fixed"], loss_variant=plan["specification"]["loss_variant"], seed=seed,
                                  parameters=candidate["parameters"], candidate_id=candidate["candidate_id"], device=options.device,
                                  source_sha256=plan["source_sha256"], splits={"test": plan["resolved_splits"]["test"]},
                                  input_sha256={name: plan["input_sha256"][name] for name in plan["resolved_splits"]["test"]["paths"]},
                                  checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), selection=str(output / "selection.json"), selection_sha256=sha(output / "selection.json"))
                    launch_worker(folder, worker, options.device, options.resume)
            state.update(status="complete", selected_candidate=ranking[0]["candidate_id"], test_evaluated=options.evaluate_test)
        except BaseException as error:
            state.update(status="failed", error=repr(error))
            raise
        finally:
            write(output / "status.json", state)
    print(json.dumps(state))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--pause-file", type=Path, help="If present, stop before next worker; current worker completes and saves")
    parser.add_argument("--worker-json", type=Path)
    args = parser.parse_args()
    if args.worker_json:
        worker = read(args.worker_json)
        if worker["model"] == "PINN":
            pinn_worker(args.worker_json)
        else:
            baseline_worker(worker)
    else:
        require(all((args.spec, args.repo, args.output)), "--spec/--repo/--output required")
        require(not args.evaluate_test or args.execute, "--evaluate-test requires --execute")
        run(args)
