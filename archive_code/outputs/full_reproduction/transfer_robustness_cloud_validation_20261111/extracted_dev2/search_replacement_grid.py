"""Transparent replacement grid search; dry-run by default, never historical recovery.

Candidate workers receive train/validation paths only. Optional test evaluation is
dispatched after the validation selection file has been committed to disk.
"""
import argparse
import copy
import hashlib
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

FORMAT = "pinn4soh-replacement-search-v1"
DIMENSIONS = ("u_layers", "u_hidden", "F_layers_num", "F_hidden_dim", "alpha", "beta")
VARIANTS = ("published_label_monotonicity", "paper_prediction_monotonicity", "author_current")
FIXED = {"epochs", "early_stop", "batch_size", "threads", "normalization_method",
         "warmup_epochs", "warmup_lr", "lr", "final_lr", "lr_F"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def positive(value, name, minimum=1, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Invalid numeric field: " + name)
    if value < minimum or (integer and not isinstance(value, int)):
        raise ValueError("Invalid numeric range/type: " + name)


def build_plan(spec, repo, max_candidates):
    if set(spec) != {"label", "dataset", "loss_variant", "seeds", "splits", "fixed", "grid"}:
        raise ValueError("Specification must contain exactly label/dataset/loss_variant/seeds/splits/fixed/grid")
    if not isinstance(spec["label"], str) or not spec["label"] or spec["dataset"] not in {"XJTU", "MIT", "TJU", "HUST"}:
        raise ValueError("Specify a label and supported dataset")
    if spec["loss_variant"] not in VARIANTS:
        raise ValueError("Explicit loss_variant required")
    seeds = spec["seeds"]
    if not isinstance(seeds, list) or not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Use explicit distinct nonempty seeds")
    for seed in seeds:
        positive(seed, "seed", minimum=0, integer=True)
    fixed = spec["fixed"]
    if set(fixed) != FIXED:
        raise ValueError("All fixed settings must be explicit: " + str(sorted(FIXED)))
    for name in FIXED - {"normalization_method"}:
        positive(fixed[name], name, minimum=0 if name in {"warmup_epochs", "warmup_lr", "lr", "final_lr", "lr_F"} else 1,
                 integer=name in {"epochs", "early_stop", "batch_size", "threads", "warmup_epochs"})
    if fixed["epochs"] > 200 or fixed["threads"] > 4 or fixed["warmup_epochs"] >= fixed["epochs"]:
        raise ValueError("Require epochs<=200, threads<=4 and 0<=warmup_epochs<epochs")
    if fixed["normalization_method"] not in {"min-max", "z-score"}:
        raise ValueError("Unsupported normalization")
    grid = spec["grid"]
    if set(grid) != set(DIMENSIONS):
        raise ValueError("Grid must explicitly specify " + str(DIMENSIONS))
    for name, values in grid.items():
        if not isinstance(values, list) or not values or len({identity(v) for v in values}) != len(values):
            raise ValueError("Grid values must be distinct nonempty lists: " + name)
        for value in values:
            positive(value, name, minimum=2 if name in {"u_layers", "F_layers_num"} else (0 if name in {"alpha", "beta"} else 1),
                     integer=name not in {"alpha", "beta"})
    count = math.prod(len(grid[name]) for name in DIMENSIONS)
    if count > max_candidates:
        raise ValueError("Grid exceeds --max-candidates; review the explicit plan before raising the limit")
    if set(spec["splits"]) != {"train", "valid", "test"}:
        raise ValueError("Explicit train/valid/test battery splits required")
    splits, hashes, path_sets = {}, {}, []
    for role in ("train", "valid", "test"):
        record = spec["splits"][role]
        if set(record) != {"paths", "nominal_capacity"} or not isinstance(record["paths"], list) or not record["paths"]:
            raise ValueError("Each split requires paths and nominal_capacity")
        positive(record["nominal_capacity"], "nominal_capacity", minimum=1e-15)
        paths = []
        for name in record["paths"]:
            path = (repo / name).resolve()
            if not path.is_file() or path.suffix.lower() != ".csv":
                raise ValueError("Missing CSV: " + str(path))
            paths.append(str(path))
            hashes[str(path)] = sha(path)
        if len(paths) != len(set(paths)):
            raise ValueError("Duplicate battery paths within " + role)
        path_sets.append({os.path.normcase(name) for name in paths})
        splits[role] = dict(paths=paths, nominal_capacity=record["nominal_capacity"])
    if any(path_sets[a] & path_sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Battery IDs must be disjoint across train/valid/test")
    candidates = [dict(candidate_id="candidate_%04d" % (i + 1), parameters=dict(zip(DIMENSIONS, values)))
                  for i, values in enumerate(itertools.product(*(grid[name] for name in DIMENSIONS)))]
    source_files = sorted(repo.rglob("*.py")) + [Path(__file__).resolve(), Path(__file__).with_name("run_transfer.py"),
                                               Path(__file__).with_name("run_extended.py")]
    source_sha = {str(path.resolve()): sha(path) for path in source_files}
    return dict(format=FORMAT, implementation="replacement_not_recovered_author_grid", specification=spec,
                resolved_splits=splits, input_sha256=hashes, source_sha256=source_sha, candidates=candidates,
                candidate_runs=count * len(seeds), selection="mean across explicit seeds of minimum validation sample-weighted MSE",
                tie_break="lexicographic candidate_id; earliest best epoch within a run", workers=1,
                architecture="u: configurable author encoder 17->32 plus fixed author Predictor(32); F: configurable author MLP(35->1)",
                validation_protocol="Explicit disjoint battery split, a replacement protocol rather than author random sample split",
                normalization="Author DF per-battery normalization uses the full battery trajectory, including validation/test trajectories; retained and disclosed, not deployment-causal normalization",
                provenance="Supplementary Note 3 p14 names layers, neurons, alpha and beta; neither grid ranges nor search seeds are public")


def artifacts(directory, names):
    return {name: sha(directory / name) for name in names}


def check_result(directory, expected):
    result = read(directory / "result.json")
    if result.get("status") != "complete" or result.get("worker_identity") != identity(expected):
        raise ValueError("Completed worker identity changed: " + str(directory))
    for name, checksum in result["artifact_sha256"].items():
        if sha(directory / name) != checksum:
            raise ValueError("Completed artifact changed: " + str(directory / name))
    import numpy as np
    prefix = "valid" if expected["mode"] == "train" else "test"
    truth = np.load(directory / (prefix + "_truth.npy"), allow_pickle=False).astype(np.float64)
    pred = np.load(directory / (prefix + "_prediction.npy"), allow_pickle=False).astype(np.float64)
    if truth.shape != pred.shape or not truth.size or not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise ValueError("Invalid completed evaluation arrays")
    mse = float(np.mean((truth - pred) ** 2))
    key = "validation_mse" if expected["mode"] == "train" else "test_mse"
    if result[key] != mse:
        raise ValueError("Completed objective disagrees with saved arrays")
    return result


def launch(spec, directory, options):
    if (directory / "result.json").exists():
        if not options.resume:
            raise ValueError("Existing worker output requires --resume")
        return check_result(directory, spec)
    if directory.exists() and any(directory.iterdir()):
        if not options.resume:
            raise ValueError("Existing incomplete output requires --resume")
        directory.rename(directory.with_name(directory.name + ".incomplete." + str(time.time_ns())))
    directory.mkdir(parents=True, exist_ok=True)
    write(directory / "worker.json", spec)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="" if options.device == "cpu" else options.device,
                       OMP_NUM_THREADS=str(spec["fixed"]["threads"]), MKL_NUM_THREADS=str(spec["fixed"]["threads"]))
    with (directory / "console.log").open("w", encoding="utf-8") as stream:
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-json", str(directory / "worker.json")]
        process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=environment)
    if process.returncode:
        raise RuntimeError("Worker failed; inspect " + str(directory / "console.log"))
    return check_result(directory, spec)


def select(rows, candidates, seeds):
    expected = {(candidate["candidate_id"], seed) for candidate in candidates for seed in seeds}
    if len(rows) != len(expected) or {(r["candidate_id"], r["seed"]) for r in rows} != expected:
        raise ValueError("Selection requires all candidate/seed pairs exactly once")
    ranking = []
    for candidate in candidates:
        values = [row["validation_mse"] for row in rows if row["candidate_id"] == candidate["candidate_id"]]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Nonfinite validation objective")
        ranking.append(dict(candidate_id=candidate["candidate_id"], mean_validation_mse=sum(values) / len(values),
                            per_seed_validation_mse=values, parameters=candidate["parameters"]))
    ranking.sort(key=lambda row: (row["mean_validation_mse"], row["candidate_id"]))
    return dict(format=FORMAT, selected=ranking[0], ranking=ranking, test_metrics_used=False)


def orchestrate(options):
    if options.device != "cpu" and (not options.device.isdigit()):
        raise ValueError("Choose one numeric GPU index or cpu; multi-GPU visibility is forbidden")
    if options.device == "cpu" and options.execute and not options.allow_cpu:
        raise ValueError("CPU execution requires --allow-cpu")
    repo, output = options.repo.resolve(), options.output.resolve()
    if output == repo or repo in output.parents:
        raise ValueError("Output must be outside author repository")
    plan = build_plan(read(options.spec), repo, options.max_candidates)
    plan["repo"] = str(repo)
    plan["device"] = options.device
    plan_hash = identity(plan)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "plan.json").exists():
        if read(output / "plan.json") != plan:
            raise ValueError("Plan/source/data/device changed; use a new output directory")
    elif any(output.iterdir()):
        raise ValueError("Output directory contains unrelated files")
    else:
        write(output / "plan.json", plan)
    if not options.execute:
        print(json.dumps(dict(status="dry_run", candidate_runs=plan["candidate_runs"], plan_sha256=plan_hash,
                              test_evaluation="not started", plan=str(output / "plan.json"))))
        return
    from run_extended import task_lock
    with task_lock(output / ".search.lock"):
        rows = []
        state = dict(status="running", plan_sha256=plan_hash, completed_runs=0, total_runs=plan["candidate_runs"], workers=1)
        write(output / "status.json", state)
        try:
            for candidate in plan["candidates"]:
                for seed in plan["specification"]["seeds"]:
                    worker = dict(mode="train", repo=str(repo), output=str(output / candidate["candidate_id"] / ("seed_%d" % seed)),
                                  fixed=plan["specification"]["fixed"], loss_variant=plan["specification"]["loss_variant"],
                                  seed=seed, parameters=candidate["parameters"], candidate_id=candidate["candidate_id"],
                                  source_sha256=plan["source_sha256"], splits={role: plan["resolved_splits"][role] for role in ("train", "valid")},
                                  device=options.device, input_sha256={path: plan["input_sha256"][path]
                                      for role in ("train", "valid") for path in plan["resolved_splits"][role]["paths"]})
                    result = launch(worker, Path(worker["output"]), options)
                    rows.append(dict(candidate_id=candidate["candidate_id"], seed=seed, validation_mse=result["validation_mse"],
                                     result_sha256=sha(Path(worker["output"]) / "result.json")))
                    state["completed_runs"] += 1
                    write(output / "status.json", state)
            selection = select(rows, plan["candidates"], plan["specification"]["seeds"])
            selection.update(plan_sha256=plan_hash, candidate_results=rows)
            if (output / "selection.json").exists():
                if read(output / "selection.json") != selection:
                    raise ValueError("Prior validation selection changed")
            else:
                write(output / "selection.json", selection)
            if options.evaluate_test:
                selected = selection["selected"]
                for seed in plan["specification"]["seeds"]:
                    trained = output / selected["candidate_id"] / ("seed_%d" % seed)
                    worker = dict(mode="test", repo=str(repo), output=str(output / "selected_test" / ("seed_%d" % seed)),
                                  fixed=plan["specification"]["fixed"], loss_variant=plan["specification"]["loss_variant"],
                                  seed=seed, parameters=selected["parameters"], candidate_id=selected["candidate_id"],
                                  source_sha256=plan["source_sha256"], splits={"test": plan["resolved_splits"]["test"]}, device=options.device,
                                  input_sha256={path: plan["input_sha256"][path] for path in plan["resolved_splits"]["test"]["paths"]},
                                  checkpoint=str(trained / "best_model.pth"), checkpoint_sha256=sha(trained / "best_model.pth"),
                                  selection=str(output / "selection.json"), selection_sha256=sha(output / "selection.json"))
                    launch(worker, Path(worker["output"]), options)
            state.update(status="complete", selected_candidate=selection["selected"]["candidate_id"], test_evaluated=options.evaluate_test)
        except BaseException as error:
            state.update(status="failed", error=repr(error))
            raise
        finally:
            write(output / "status.json", state)
    print(json.dumps(state))


def worker_main(path):
    spec = read(path)
    output = Path(spec["output"])
    if Path(path).resolve().parent != output.resolve():
        raise ValueError("Worker spec/output disagreement")
    for name, checksum in {**spec["source_sha256"], **spec["input_sha256"]}.items():
        if sha(name) != checksum:
            raise ValueError("Source/input changed before worker execution: " + name)
    import numpy as np
    import torch
    from types import SimpleNamespace
    from torch.utils.data import DataLoader, TensorDataset
    torch.set_num_threads(spec["fixed"]["threads"])
    if spec["device"] != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("Requested GPU is unavailable; no silent CPU fallback")
    random.seed(spec["seed"])
    np.random.seed(spec["seed"])
    torch.manual_seed(spec["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(spec["seed"])
    sys.path.insert(0, spec["repo"])
    module = importlib.import_module("Model.Model")
    from dataloader.dataloader import DF
    parameters = spec["parameters"]

    class ConfigurableSolution(module.Solution_u):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.encoder = module.MLP(input_dim=17, output_dim=32, layers_num=parameters["u_layers"],
                                      hidden_dim=parameters["u_hidden"], droupout=0.2)
            self.predictor = module.Predictor(input_dim=32)
            self._init_()

    module.Solution_u = ConfigurableSolution
    args = SimpleNamespace(**spec["fixed"], F_layers_num=parameters["F_layers_num"],
                           F_hidden_dim=parameters["F_hidden_dim"], alpha=parameters["alpha"], beta=parameters["beta"],
                           save_folder=str(output), log_dir="author_args.log")
    model = module.PINN(args)
    data, loaders, tensor_hashes = DF(args), {}, {}
    for role, record in spec["splits"].items():
        all_tensors, counts = [], []
        for name in record["paths"]:
            (x1, y1), (x2, y2) = data.load_one_battery(name, record["nominal_capacity"])
            tensors = [torch.tensor(v, dtype=torch.float32) for v in (x1, x2, y1.reshape(-1, 1), y2.reshape(-1, 1))]
            if tensors[0].ndim != 2 or tensors[0].shape[1] != 17 or len(tensors[0]) == 0:
                raise ValueError("Require nonempty 16-feature CSV plus cycle index: " + name)
            if not all(torch.isfinite(tensor).all() for tensor in tensors):
                raise ValueError("Nonfinite normalized tensors: " + name)
            all_tensors.append(tensors)
            counts.append(dict(path=name, n_pairs=len(tensors[0])))
        tensors = [torch.cat([values[i] for values in all_tensors]) for i in range(4)]
        tensor_hashes[role] = dict(sha256=[hashlib.sha256(t.numpy().tobytes()).hexdigest() for t in tensors], batteries=counts)
        loaders[role] = DataLoader(TensorDataset(*tensors), batch_size=args.batch_size, shuffle=role == "train", num_workers=0)
    write(output / "data_identity.json", tensor_hashes)

    def predictions(loader):
        model.eval()
        truth, prediction = [], []
        with torch.no_grad():
            for x1, _, y1, _ in loader:
                truth.append(y1.numpy())
                prediction.append(model.predict(x1.to(module.device)).cpu().numpy())
        return np.concatenate(truth), np.concatenate(prediction)

    environment = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
                       device=module.device, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                       threads=torch.get_num_threads(), workers=1)
    if spec["mode"] == "train":
        if set(loaders) != {"train", "valid"}:
            raise ValueError("Candidate training must receive train/valid only")
        best, best_epoch, stale = float("inf"), None, 0
        history = []
        for epoch in range(1, args.epochs + 1):
            if spec["loss_variant"] == "author_current":
                losses = model.train_one_epoch(epoch, loaders["train"])
            else:
                from run_transfer import train_with_loss_variant
                losses = train_with_loss_variant(model, epoch, loaders["train"], spec["loss_variant"], False)
            model.scheduler.step()
            truth, pred = predictions(loaders["valid"])
            score = float(np.mean((truth.astype(np.float64) - pred.astype(np.float64)) ** 2))
            if not math.isfinite(score) or not all(math.isfinite(float(value)) for value in losses):
                raise ValueError("Nonfinite training loss or validation score")
            stale += 1
            history.append(dict(epoch=epoch, train_losses=list(map(float, losses)), validation_mse=score))
            if score < best:
                best, best_epoch, stale = score, epoch, 0
                torch.save(copy.deepcopy(dict(solution_u=model.solution_u.state_dict(), dynamical_F=model.dynamical_F.state_dict())), output / "best_model.pth")
            write(output / "history.json", history)
            if stale > args.early_stop:
                break
        checkpoint = torch.load(output / "best_model.pth", map_location=module.device, weights_only=True)
        model.solution_u.load_state_dict(checkpoint["solution_u"])
        model.dynamical_F.load_state_dict(checkpoint["dynamical_F"])
        truth, pred = predictions(loaders["valid"])
        reloaded = float(np.mean((truth.astype(np.float64) - pred.astype(np.float64)) ** 2))
        if reloaded != best:
            raise ValueError("Best validation checkpoint reload did not reproduce selection objective")
        np.save(output / "valid_truth.npy", truth)
        np.save(output / "valid_prediction.npy", pred)
        result = dict(validation_mse=best, best_epoch=best_epoch, epochs_completed=len(history), test_loaded=False,
                      artifact_sha256=artifacts(output, ["best_model.pth", "history.json", "valid_truth.npy", "valid_prediction.npy", "data_identity.json"]))
    elif spec["mode"] == "test":
        if set(loaders) != {"test"} or sha(spec["selection"]) != spec["selection_sha256"]:
            raise ValueError("Test requires previously frozen selection")
        selected = read(spec["selection"])["selected"]
        if selected["candidate_id"] != spec["candidate_id"] or selected["parameters"] != parameters:
            raise ValueError("Checkpoint configuration is not validation-selected")
        if sha(spec["checkpoint"]) != spec["checkpoint_sha256"]:
            raise ValueError("Selected checkpoint changed")
        checkpoint = torch.load(spec["checkpoint"], map_location=module.device, weights_only=True)
        model.solution_u.load_state_dict(checkpoint["solution_u"])
        model.dynamical_F.load_state_dict(checkpoint["dynamical_F"])
        truth, pred = predictions(loaders["test"])
        difference = pred.astype(np.float64) - truth.astype(np.float64)
        np.save(output / "test_truth.npy", truth)
        np.save(output / "test_prediction.npy", pred)
        result = dict(test_mse=float(np.mean(difference ** 2)), test_mae=float(np.mean(np.abs(difference))),
                      test_used_for_selection=False, selection_sha256=spec["selection_sha256"],
                      artifact_sha256=artifacts(output, ["test_truth.npy", "test_prediction.npy", "data_identity.json"]))
    else:
        raise ValueError("Unknown worker mode")
    result.update(format=FORMAT, status="complete", worker_identity=identity(spec), environment=environment)
    write(output / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="0", help="One physical GPU index, or cpu")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=128)
    parser.add_argument("--worker-json", type=Path, help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.worker_json:
        worker_main(options.worker_json)
    else:
        if not all((options.spec, options.repo, options.output)):
            parser.error("--spec, --repo and --output are required")
        if options.evaluate_test and not options.execute:
            parser.error("--evaluate-test requires --execute")
        orchestrate(options)


if __name__ == "__main__":
    main()
