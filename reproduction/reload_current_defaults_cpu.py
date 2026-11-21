"""Bounded, read-only CPU reload of the twenty audited current-default runs.

Run with python -I -B. Inputs and historical verification reports are never edited.
Only SHA-bound state dictionaries are loaded, with weights_only=True; no training.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace


INDEX_SHA256 = "272052e12ccd365a0f3037b1bbe4a5ec2d816b41017a9db4af48d37a35de73aa"
SOURCE_FINGERPRINT = "16406722f3e691fa9229b233704ae12633e3e8d11980f02b7ec248e5b68198c0"
ATOL, RTOL = 1e-6, 1e-5
RUNS = [f"{dataset}/all/Experiment{repeat:02d}"
        for dataset in ("MIT", "HUST") for repeat in range(1, 11)]
SMOKE_RUNS = ["MIT/all/Experiment01", "HUST/all/Experiment01"]
LIMITS = [
    "CPU Torch 2.7 results do not establish twenty GPU reloads or GPU bitwise equality.",
    "Input tensor hashes are newly reconstructed; no training-time GPU X tensor reference exists for these twenty historical runs.",
    "Source CSV/source SHA binding and an independent reconstruction are checked; they cannot retroactively supply missing training-time X hashes.",
    "Author full-trajectory scaling, capacity-dependent 3sigma cleaning, split order and batch size are retained.",
    "CSV macro metrics are not physical-battery-UID macro; historical MIT identity duplication is retained.",
    "This verifies the twenty current-default checkpoints, not all repository versions or the publication's Note3 loss/parameter arms.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def contained(root, name):
    root = root.resolve()
    path = (root / name).resolve()
    require(root in path.parents, "Path escaped input root: " + str(name))
    return path


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def preflight(args, output):
    workspace, repo, audit = args.repo.resolve().parent, args.repo.resolve(), args.audit_dir.resolve()
    tracked = {}

    def observe(path, expected, size=None):
        path = path.resolve()
        require(path.is_file() and not path.is_symlink(), "Missing/linked input: " + str(path))
        actual = digest(path)
        require(actual == expected, "Input SHA mismatch: " + str(path))
        nbytes = path.stat().st_size
        require(size is None or nbytes == size, "Input byte count mismatch: " + str(path))
        item = dict(path=str(path), bytes=nbytes, sha256=actual)
        require(str(path) not in tracked or tracked[str(path)] == item, "Conflicting input reference")
        tracked[str(path)] = item

    index_path = audit / "LOCAL_CURRENT_DEFAULT_DELIVERY_INDEX.json"
    observe(index_path, INDEX_SHA256)
    index = read(index_path)
    require(index["status"] == "passed_20" and index["trained_runs"] == 20, "Audit index status")
    for item in index["files"] + [index["script"]]:
        observe(contained(workspace, item["path"]), item["sha256"], item["bytes"])
    source_audit = read(audit / "LOCAL_CURRENT_DEFAULT_AUDIT.json")
    require(source_audit["status"] == "passed_20", "Audited runs not passed")
    require([row["run"] for row in source_audit["runs"]] == RUNS, "Unexpected audit run identities")
    require(source_audit["source_fingerprint"] == SOURCE_FINGERPRINT, "Audit source fingerprint")
    results = Path(source_audit["results"]).resolve()
    for source_root in (workspace / "PINN4SOH", audit, results):
        require(output != source_root and source_root not in output.parents
                and output not in source_root.parents, "Output overlaps an input directory")
    checks = read(audit / "INPUT_SHA256_CHECKS.json")
    require(len(checks["source"]) == 417, "Expected 417 source/data files")
    for item in checks["source"]:
        observe(contained(repo, item["path"]), item["sha256"])
    original = {item["path"]: item for item in checks["original_evidence"]}
    recovery = {item["path"]: item for item in checks["recovery"]}
    require(len(original) == 89 and len(recovery) == 202, "Unexpected evidence inventory cardinality")
    for name, item in original.items():
        observe(contained(results, name), item["sha256"])
    for name, item in recovery.items():
        prefix = results.name + "/"
        require(name.startswith(prefix), "Recovery root mismatch")
        observe(contained(results, name[len(prefix):]), item["sha256"], item["bytes"])
    inventory = read(results / "source_inventory.json")
    source_map = {item["path"]: item["sha256"] for item in checks["source"]}
    require(inventory["sha256"] == source_map, "Author inventory differs from audited source map")
    require(canonical_hash(source_map) == SOURCE_FINGERPRINT == inventory["fingerprint"],
            "Source fingerprint mismatch")
    actual_set = {p.relative_to(repo).as_posix() for p in repo.rglob("*") if p.is_file()
                  and (p.suffix == ".py" or p.name in {"README.md", "fix.md"}
                       or (p.suffix == ".csv" and "data" in p.relative_to(repo).parts))}
    require(actual_set == set(source_map), "Current repository source/data file set changed")
    manifest, configs = read(results / "manifest.json"), read(audit / "CURRENT_DEFAULT_CONFIGS_20.json")
    require(set(configs) == set(RUNS), "Expected exactly twenty audited configs")
    audit_runs = {row["run"]: row for row in source_audit["runs"]}
    prepared = {}
    for key in RUNS:
        folder = contained(results, key)
        for name in ("config.json", "splits.json", "model.pth", "pred_label.npy", "true_label.npy"):
            ref = recovery[results.name + "/" + key + "/" + name]
            observe(folder / name, ref["sha256"], ref["bytes"])
            if key + "/" + name in original:
                observe(folder / name, original[key + "/" + name]["sha256"])
        config, splits, record = read(folder / "config.json"), read(folder / "splits.json"), manifest["runs"][key]
        require(record["status"] == "complete" and record["relative_folder"] == key,
                "Incomplete/misdirected run: " + key)
        require(config == configs[key] and canonical_hash(config) == record["config_hash"]
                == audit_runs[key]["config_hash"], "Config provenance mismatch: " + key)
        require(config["source_fingerprint"] == SOURCE_FINGERPRINT and not config["smoke"],
                "Source/smoke mismatch: " + key)
        require(config["seed"] == record["seed"] == audit_runs[key]["seed"], "Seed binding mismatch")
        require(config["parameters"]["batch_size"] == 512 and config["threads"] == 2
                and config["parameters"]["normalization_method"] == "min-max", "Historical inference settings")
        test = [item for item in splits if item["role"] == "test"]
        require(len(test) == 1, "Expected one test split")
        test = test[0]
        require(test["paths"] == [item["path"] for item in test["batteries"]], "Test path/battery order")
        require(sum(item["n_pairs"] for item in test["batteries"]) == record["sample_counts"]["test"]
                == audit_runs[key]["n_test"], "Test sample boundary mismatch")
        for name in test["paths"]:
            require(name in source_map, "Test CSV absent from source inventory")
            contained(repo, name)
        prepared[key] = dict(folder=folder, config=config, split=test)
    before = sorted(tracked.values(), key=lambda row: row["path"])
    return repo, results, prepared, before


def array_record(value, np):
    value = np.ascontiguousarray(value)
    return dict(dtype=str(value.dtype), shape=list(value.shape),
                sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest())


def bitwise(a, b, np):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes(order="C") == b.tobytes(order="C")


def independent_csv(path, nominal, pd, np):
    # Independent expression of the author's cleaning/normalization/pair semantics.
    frame = pd.read_csv(path)
    frame.insert(frame.shape[1] - 1, "cycle index", np.arange(len(frame)))
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    keep = np.ones(len(frame), dtype=bool)
    for name in frame.columns:
        column = frame[name]
        keep &= ~((column < column.mean() - 3 * column.std()) |
                  (column > column.mean() + 3 * column.std())).to_numpy()
    frame = frame.loc[keep].reset_index(drop=True)
    features = frame.iloc[:, :-1]
    features = 2 * (features - features.min()) / (features.max() - features.min()) - 1
    x = features.to_numpy(dtype=np.float32)
    y = (frame.iloc[:, -1].to_numpy() / nominal).astype(np.float32).reshape(-1, 1)
    return x[:-1], x[1:], y[:-1], y[1:]


def metrics(truth, prediction, np):
    truth, prediction = truth.astype(np.float64).ravel(), prediction.astype(np.float64).ravel()
    delta = prediction - truth
    mse = float(np.mean(delta ** 2))
    return dict(MAE=float(np.mean(np.abs(delta))), MAPE=float(np.mean(np.abs(delta) /
                np.maximum(np.abs(truth), np.finfo(np.float64).eps))), MSE=mse, RMSE=float(np.sqrt(mse)))


def run_one(key, task, repo, output, DF, Solution_u, MLP, torch, np, pd):
    folder, config, split = task["folder"], task["config"], task["split"]
    target = output / "runs" / key
    target.mkdir(parents=True)
    started = time.perf_counter()
    record = dict(run=key, device="cpu", training_performed=False,
                  checkpoint_loaded=False, atol=ATOL, rtol=RTOL,
                  input_tensor_reference="newly_reconstructed_no_training_time_tensor_reference")
    try:
        params = dict(config["parameters"])
        params.update(log_dir=None, save_folder=None)
        loader = DF(SimpleNamespace(**params)).load_all_battery(
            [str(contained(repo, name)) for name in split["paths"]], split["nominal_capacity"])["test_3"]
        tensors = loader.dataset.tensors
        expected = sum(item["n_pairs"] for item in split["batteries"])
        require([tuple(t.shape) for t in tensors] == [(expected, 17), (expected, 17), (expected, 1), (expected, 1)],
                "Unexpected test tensor shapes")
        values = [t.detach().cpu().numpy() for t in tensors]
        require(all(t.dtype == torch.float32 for t in tensors) and all(np.isfinite(v).all() for v in values),
                "Invalid test tensor dtype/values")
        start, boundaries = 0, []
        for battery in split["batteries"]:
            end = start + battery["n_pairs"]
            rebuilt = independent_csv(contained(repo, battery["path"]), split["nominal_capacity"], pd, np)
            require(all(bitwise(a[start:end], b, np) for a, b in zip(values, rebuilt)),
                    "Independent test tensor reconstruction mismatch: " + battery["path"])
            boundaries.append(dict(path=battery["path"], start=start, stop=end,
                                   csv_sha256=digest(contained(repo, battery["path"]))))
            start = end
        saved_truth = np.load(folder / "true_label.npy", allow_pickle=False)
        saved_pred = np.load(folder / "pred_label.npy", allow_pickle=False)
        require(saved_truth.shape == saved_pred.shape == (expected, 1)
                and saved_truth.dtype == saved_pred.dtype == np.float32
                and np.isfinite(saved_truth).all() and np.isfinite(saved_pred).all(), "Saved array schema")
        require(bitwise(values[2], saved_truth, np), "Test labels are not bitwise equal")
        write(target / "INPUT_TENSOR_IDENTITY.json", dict(
            reference=record["input_tensor_reference"], matched_training_time_tensor_record=False,
            tensor_order=["X1", "X2", "Y1", "Y2"], tensors=[array_record(v, np) for v in values],
            independent_csv_reconstruction_bitwise_equal=True, labels_bitwise_equal=True,
            batch_size=512, shuffle=False, boundaries=boundaries))
        # No PINN wrapper is created: this avoids log files, optimizers and schedulers.
        u = Solution_u().cpu()
        g = MLP(input_dim=35, output_dim=1, layers_num=params["F_layers_num"],
                hidden_dim=params["F_hidden_dim"], droupout=0.2).cpu()
        checkpoint = torch.load(folder / "model.pth", map_location="cpu", weights_only=True)
        record["checkpoint_loaded"] = True
        require(isinstance(checkpoint, dict) and set(checkpoint) == {"solution_u", "dynamical_F"},
                "Unexpected checkpoint top-level keys")
        for name, model in (("solution_u", u), ("dynamical_F", g)):
            state = checkpoint[name]
            require(isinstance(state, dict) and set(state) == set(model.state_dict()), "State key mismatch")
            for tensor_name, tensor in state.items():
                reference = model.state_dict()[tensor_name]
                require(isinstance(tensor, torch.Tensor) and tensor.dtype == reference.dtype
                        and tuple(tensor.shape) == tuple(reference.shape)
                        and bool(torch.isfinite(tensor).all()), "Invalid checkpoint tensor: " + tensor_name)
            model.load_state_dict(state, strict=True)
            model.eval()
            model.requires_grad_(False)
        predictions, truths = [], []
        with torch.inference_mode():
            for x1, _, y1, _ in loader:
                predictions.append(u(x1).cpu().numpy())
                truths.append(y1.numpy())
        prediction, truth = np.concatenate(predictions), np.concatenate(truths)
        require(prediction.shape == saved_pred.shape and prediction.dtype == saved_pred.dtype,
                "Reloaded prediction schema mismatch")
        finite = all(np.isfinite(a).all() for a in (prediction, truth, saved_pred, saved_truth))
        labels_equal = bitwise(truth, saved_truth, np)
        pred_equal = bitwise(prediction, saved_pred, np)
        # Saved GPU prediction is the reference in the fixed asymmetric allclose bound.
        close = bool(np.allclose(prediction, saved_pred, atol=ATOL, rtol=RTOL, equal_nan=False))
        errors = np.abs(prediction.astype(np.float64) - saved_pred.astype(np.float64))
        bound = ATOL + RTOL * np.abs(saved_pred.astype(np.float64))
        np.save(target / "cpu_pred_label.npy", prediction, allow_pickle=False)
        np.save(target / "cpu_true_label.npy", truth, allow_pickle=False)
        record.update(status="passed_cpu_tolerance" if finite and labels_equal and close else "failed_comparison",
            passed=bool(finite and labels_equal and close), n_test=expected,
            checkpoint_sha256=digest(folder / "model.pth"), strict_network_loads=["solution_u", "dynamical_F"],
            g_loaded_but_not_used_for_inference=True, all_arrays_finite=bool(finite),
            labels_bitwise_equal=labels_equal, predictions_bitwise_equal=pred_equal,
            predictions_allclose=close, max_absolute_prediction_difference=float(errors.max()),
            mean_absolute_prediction_difference=float(errors.mean()),
            elements_over_fixed_bound=int(np.sum(errors > bound)),
            maximum_fixed_bound_ratio=float(np.max(errors / bound)),
            original_metrics_float64=metrics(saved_truth, saved_pred, np),
            cpu_metrics_float64=metrics(truth, prediction, np),
            cpu_prediction_tensor=array_record(prediction, np))
    except Exception as error:
        record.update(status="failed_exception", passed=False, error=type(error).__name__ + ": " + str(error))
        (target / "FAILURE_TRACEBACK.txt").write_text(traceback.format_exc(), encoding="utf-8")
    record["elapsed_seconds"] = time.perf_counter() - started
    write(target / "CPU_RELOAD.json", record)
    print(json.dumps({name: record.get(name) for name in ("run", "status", "predictions_bitwise_equal",
         "predictions_allclose", "max_absolute_prediction_difference", "error")}), flush=True)
    return record


def finish(output, report):
    write(output / "CPU_RELOAD_SUMMARY.json", report)
    text = ("# 当前默认checkpoint CPU复载\n\n"
            f"阶段：{report['phase']}；状态：{report['status']}。\n\n"
            f"通过 {report.get('passed_runs', 0)}/{report.get('expected_runs', 0)}；"
            f"prediction逐位一致 {report.get('prediction_bitwise_runs', 0)}。"
            f"固定容差atol={ATOL}、rtol={RTOL}；每次均报告两种判据。\n\n"
            "仅CPU推理，未训练；原始日志、checkpoint、数组、历史审计均不写回。"
            "原输入逐SHA在前后复核。INPUT_TENSOR_IDENTITY记录原DF与独立CSV重建的四张量、dtype、shape和边界。\n\n"
            "CPU跨Torch版本复载不能代替20/20 GPU复载或GPU逐位一致。"
            "新X哈希没有训练时GPU张量参考，不能追溯补造缺失的原始张量记录。\n\n"
            "结果、原/复载float64指标、失败与数值漂移详见CPU_RELOAD_SUMMARY.json及runs目录。\n")
    (output / "README.md").write_text(text, encoding="utf-8")
    records = [dict(path=p.relative_to(output).as_posix(), bytes=p.stat().st_size, sha256=digest(p))
               for p in sorted(output.rglob("*")) if p.is_file() and p.name != "SHA256_INDEX.json"]
    write(output / "SHA256_INDEX.json", dict(format="pinn4soh-cpu-reload-delivery-sha-v1", files=records,
                                            script_sha256=digest(__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["smoke", "full"], required=True)
    parser.add_argument("--smoke-summary", type=Path)
    args = parser.parse_args()
    require(sys.flags.isolated and sys.flags.dont_write_bytecode, "Use python -I -B")
    output = args.output.resolve()
    require(not output.exists(), "Use a new output directory")
    repo, audit = args.repo.resolve(), args.audit_dir.resolve()
    for path in (repo, audit):
        require(output != path and path not in output.parents and output not in path.parents, "Output overlaps input")
    output.mkdir(parents=True)
    report = dict(format="pinn4soh-current-defaults-cpu-reload-v1", phase=args.phase,
        started_at_utc=datetime.now(timezone.utc).isoformat(), status="preflight",
        script_sha256=digest(__file__), expected_runs=2 if args.phase == "smoke" else 20,
        training_performed=False, original_inputs_modified=False, device="cpu", atol=ATOL, rtol=RTOL, limits=LIMITS)
    before = []
    try:
        repo, results, prepared, before = preflight(args, output)
        write(output / "INPUTS_BEFORE.json", dict(files=before))
        report.update(repo=str(repo), results=str(results), source_files_verified=417,
                      source_fingerprint=SOURCE_FINGERPRINT, input_snapshot_sha256=canonical_hash(before),
                      delivery_index_sha256=INDEX_SHA256, original_gpu_reload_scope=SMOKE_RUNS)
        if args.phase == "full":
            require(args.smoke_summary is not None, "Full phase requires the completed two-run smoke summary")
            smoke = read(args.smoke_summary.resolve())
            require(smoke["phase"] == "smoke" and smoke["status"] == "passed_cpu_tolerance"
                    and smoke["passed_runs"] == 2 and [r["run"] for r in smoke["runs"]] == SMOKE_RUNS
                    and smoke["script_sha256"] == report["script_sha256"]
                    and smoke["input_snapshot_sha256"] == report["input_snapshot_sha256"]
                    and smoke["inputs_unchanged_after"] is True, "Two-run smoke binding failed")
            report["smoke_summary"] = dict(path=str(args.smoke_summary.resolve()), sha256=digest(args.smoke_summary))
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[key] = "2"
        import numpy as np
        import pandas as pd
        import scipy
        import sklearn
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        require(not torch.cuda.is_available(), "CPU verification must not acquire CUDA")
        sys.path.insert(0, str(repo))
        data_module = importlib.import_module("dataloader.dataloader")
        model_module = importlib.import_module("Model.Model")
        require(Path(data_module.__file__).resolve() == repo / "dataloader/dataloader.py"
                and Path(model_module.__file__).resolve() == repo / "Model/Model.py", "Author import origin mismatch")
        report["environment"] = dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
            numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
            threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
            cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"], cuda_available=False,
            isolated=True, dont_write_bytecode=True,
            imports={m.__name__: str(Path(m.__file__).resolve()) for m in (torch, np, pd, scipy, sklearn)})
        selected = SMOKE_RUNS if args.phase == "smoke" else RUNS
        report["runs"] = [run_one(key, prepared[key], repo, output, data_module.DF,
            model_module.Solution_u, model_module.MLP, torch, np, pd) for key in selected]
        report.update(passed_runs=sum(r["passed"] for r in report["runs"]),
            prediction_bitwise_runs=sum(r.get("predictions_bitwise_equal", False) for r in report["runs"]),
            prediction_allclose_runs=sum(r.get("predictions_allclose", False) for r in report["runs"]))
        report["status"] = "passed_cpu_tolerance" if report["passed_runs"] == len(selected) else "failed_runs"
    except Exception as error:
        report.update(status="failed_gate", error=type(error).__name__ + ": " + str(error))
        (output / "FAILURE_TRACEBACK.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        if before:
            changed = [row["path"] for row in before if not Path(row["path"]).is_file()
                       or Path(row["path"]).stat().st_size != row["bytes"] or digest(row["path"]) != row["sha256"]]
            write(output / "INPUTS_AFTER.json", dict(checked_files=len(before), changed=changed,
                                                     unchanged=not changed))
            report["inputs_unchanged_after"] = not changed
            if changed:
                report["status"] = "failed_input_mutation"
                report["original_inputs_modified"] = True
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        finish(output, report)
    print(json.dumps({k: report.get(k) for k in ("phase", "status", "passed_runs", "expected_runs",
          "prediction_bitwise_runs", "inputs_unchanged_after", "error")}), flush=True)
    if report["status"] != "passed_cpu_tolerance":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
