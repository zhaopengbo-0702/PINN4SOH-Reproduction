"""Run the author's current PINN4SOH loaders and training without editing them."""

import argparse
import contextlib
import copy
import csv
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import random
import re
import sys
import time
import traceback
from datetime import datetime, timezone

UPSTREAM_COMMIT = "cc3cc5053caf38f04e0665f7f88cb109144d035e"
BATCHES = {"XJTU": ["2C", "3C", "R2.5", "R3", "RW", "satellite"],
           "MIT": ["all"], "HUST": ["all"], "TJU": ["0", "1", "2"]}
LOADERS = {"XJTU": "load_data", "MIT": "load_MIT_data",
           "HUST": "load_HUST_data", "TJU": "load_TJU_data"}
METRICS = ["MAE", "MAPE", "MSE", "RMSE", "MAE_pp", "MAPE_pct", "RMSE_pp"]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_inventory(repo):
    files = [p for p in repo.rglob("*") if p.is_file() and
             (p.suffix == ".py" or p.name in {"README.md", "fix.md"} or
              (p.suffix == ".csv" and "data" in p.relative_to(repo).parts))]
    hashes = {p.relative_to(repo).as_posix(): digest(p) for p in sorted(files)}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    directory_order = {}
    for root, dirs, names in os.walk(repo / "data"):
        directory_order[Path(root).relative_to(repo).as_posix()] = {
            "directories": dirs, "files": names}
    return {"reference_upstream_commit": UPSTREAM_COMMIT,
            "reference_note": "Reference commit, verified separately; hashes identify actual local files.",
            "sha256": hashes, "fingerprint": fingerprint,
            "data_directory_order": directory_order}


def calculate_metrics(truth, prediction):
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
    from sklearn.metrics import mean_squared_error
    truth = np.asarray(truth).reshape(-1)
    prediction = np.asarray(prediction).reshape(-1)
    if not len(truth) or truth.shape != prediction.shape:
        raise ValueError("Empty or mismatched prediction arrays")
    if not (np.isfinite(truth).all() and np.isfinite(prediction).all()):
        raise ValueError("Non-finite prediction arrays")
    mae = float(mean_absolute_error(truth, prediction))
    mape = float(mean_absolute_percentage_error(truth, prediction))
    mse = float(mean_squared_error(truth, prediction))
    rmse = float(np.sqrt(mse))
    return {"MAE": mae, "MAPE": mape, "MSE": mse, "RMSE": rmse,
            "MAE_pp": mae * 100, "MAPE_pct": mape * 100,
            "RMSE_pp": rmse * 100, "n_test": int(len(truth))}


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


@contextlib.contextmanager
def capture_data_paths():
    from dataloader.dataloader import DF
    original_all, original_one = DF.load_all_battery, DF.load_one_battery
    calls, active = [], []

    def capture_all(self, path_list, nominal_capacity):
        record = {"role": "train_validation" if not calls else "test",
                  "paths": list(path_list), "nominal_capacity": nominal_capacity,
                  "batteries": []}
        calls.append(record)
        active.append(record)
        try:
            return original_all(self, path_list, nominal_capacity)
        finally:
            active.pop()

    def capture_one(self, path, nominal_capacity=None):
        result = original_one(self, path, nominal_capacity)
        if active:
            active[-1]["batteries"].append({"path": str(path), "n_pairs": len(result[0][0])})
        return result

    DF.load_all_battery, DF.load_one_battery = capture_all, capture_one
    try:
        yield calls
    finally:
        DF.load_all_battery, DF.load_one_battery = original_all, original_one


def plot_predictions(folder, truth, prediction):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    truth, prediction = np.asarray(truth).reshape(-1), np.asarray(prediction).reshape(-1)
    fig, ax = plt.subplots(figsize=(5.4, 5.1))
    ax.scatter(truth, prediction, s=3, alpha=0.3, rasterized=True)
    lower = float(min(truth.min(), prediction.min()))
    upper = float(max(truth.max(), prediction.max()))
    ax.plot([lower, upper], [lower, upper], color="#bd3838", linewidth=1)
    ax.set(xlabel="Measured SOH (fraction)", ylabel="Predicted SOH (fraction)",
           title="Validation-selected predictions on held-out batteries")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(folder / "test_predictions.png", dpi=160)
    plt.close(fig)


def summarize(output, make_plots=False):
    import numpy as np
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for key, run in manifest["runs"].items():
        if run["status"] != "complete":
            continue
        folder = output / run["relative_folder"]
        truth = np.load(folder / "true_label.npy", allow_pickle=False)
        prediction = np.load(folder / "pred_label.npy", allow_pickle=False)
        row = {"run": key, "dataset": run["dataset"], "batch": run["batch"],
               "repeat": run["repeat"], "seed": run["seed"],
               "training_seconds": run["training_seconds"], **calculate_metrics(truth, prediction)}
        write_json(folder / "metrics.json", row)
        split_path = folder / "splits.json"
        if split_path.exists():
            splits = json.loads(split_path.read_text(encoding="utf-8"))
            test_split = next(item for item in splits if item["role"] == "test")
            if sum(item["n_pairs"] for item in test_split["batteries"]) != len(truth):
                raise ValueError("Test battery boundaries disagree with saved predictions")
            battery_rows, start = [], 0
            for battery in test_split["batteries"]:
                end = start + battery["n_pairs"]
                battery_rows.append({"battery": battery["path"],
                                     **calculate_metrics(truth[start:end], prediction[start:end])})
                start = end
            write_csv(folder / "metrics_per_battery.csv", battery_rows, ["battery", "n_test", *METRICS])
        rows.append(row)
        if make_plots:
            plot_predictions(folder, truth, prediction)
    columns = ["run", "dataset", "batch", "repeat", "seed", "n_test",
               "training_seconds", *METRICS]
    write_csv(output / "metrics_per_run.csv", rows, columns)
    grouped = []
    for dataset, batch in sorted({(r["dataset"], r["batch"]) for r in rows}):
        selected = [r for r in rows if (r["dataset"], r["batch"]) == (dataset, batch)]
        row = {"dataset": dataset, "batch": batch, "n_runs": len(selected)}
        for metric in [*METRICS, "training_seconds"]:
            values = [r[metric] for r in selected]
            row[metric + "_mean"] = float(np.mean(values))
            row[metric + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        grouped.append(row)
    group_columns = ["dataset", "batch", "n_runs"]
    for metric in [*METRICS, "training_seconds"]:
        group_columns += [metric + "_mean", metric + "_std"]
    write_csv(output / "metrics_summary.csv", grouped, group_columns)
    statuses = {status: sum(r["status"] == status for r in manifest["runs"].values())
                for status in ["complete", "running", "failed"]}
    lines = ["# PINN4SOH 云端复现结果", "",
             f"更新时间（UTC）：{utc_now()}",
             f"已完成 {statuses['complete']}；运行中 {statuses['running']}；失败 {statuses['failed']}。", "",
             "## 范围与口径", "",
             "这是作者 2026 年 9 月修复 bug 后代码的功能复现，使用仓库预处理特征数据。",
             "作者 fix.md 说明 alpha、beta 已与论文补充材料不同，结果不能直接当作 2024 论文数值的精确复现。",
             "保留作者网络、损失、数据加载器、每批训练参数、早停及验证集选择逻辑；新增固定随机种子、独立输出和可恢复运行记录。",
             "云端 Python/PyTorch/数值库版本可能与 README 原始环境不同，实际版本见 environment.json。",
             "--smoke 为两轮连通性验证，不是收敛实验；正式训练默认最多 200 轮，早停设置沿用作者。", "",
             "每次指标以该次验证集最优时保存的 true_label.npy 和 pred_label.npy 计算，合并该次全部测试样本（按样本加权）。",
             "该口径不同于论文分析脚本先按电池再求均值的口径，不应直接逐项比较表格。",
             "MAPE 使用真实 SOH 作分母，等同 sklearn 的 mean_absolute_percentage_error(y_true, y_pred)。",
             "作者训练日志将这两个参数反向传入，日志 MAPE 可能与本汇总不同。",
             "MAE、RMSE 的原始单位为 SOH 比例（例如 0.01 = 1 个百分点）；MAE_pp、RMSE_pp 乘 100；MAPE_pct 是相对误差百分数。MSE 单位为比例平方。",
             "各批次汇总是独立重复的等权均值，标准差使用样本标准差 ddof=1；仅一次结果的标准差留空。",
             "training_seconds 包含模型初始化及 Train，不含数据读取和汇总，CUDA 在计时边界同步。", "",
             "## 已知代码限制", "",
             "TJU 以 os.listdir 返回顺序选择批次和测试电池；本脚本沿用该行为，并在 source_inventory.json 和每次 logging.txt 记录顺序与实际路径。",
             "HUST 入口把 CUDA_VISIBLE_DEVICES 写为 1；独立运行器在导入后恢复单卡容器的 0，不编辑作者文件。",
             "作者 best_model 的 state_dict 未做深拷贝。独立运行器仅在赋值给 best_model 时深拷贝快照，使 model.pth 与最佳验证时的预测 npy 一致；此修正不改变训练算法、验证选择或超参数。",
             "每次 splits.json 记录实际调用数据加载器的训练/验证与测试路径和每块电池样本数，metrics_per_battery.csv 使用这些精确边界。",
             "预处理由作者按每块完整电池数据计算归一化；这里保持该设定，不能据此声称严格在线或因果评估。", "",
             "## 汇总", "", "| 数据集 | 批次 | 次数 | MAE (百分点) | MAPE (%) | RMSE (百分点) |",
             "|---|---|---:|---:|---:|---:|"]
    for row in grouped:
        lines.append(f"| {row['dataset']} | {row['batch']} | {row['n_runs']} | "
                     f"{row['MAE_pp_mean']:.4f} | {row['MAPE_pct_mean']:.4f} | {row['RMSE_pp_mean']:.4f} |")
    (output / "RESULTS_说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": statuses, "completed": len(rows), "output": str(output)}, ensure_ascii=False), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1] / "PINN4SOH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=[*BATCHES, "all"], default="all")
    parser.add_argument("--batches", nargs="+", help="Batch names or indices (one dataset only)")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--threads", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    options = parser.parse_args()
    options.output = options.output.resolve()
    if options.summarize:
        summarize(options.output, options.plots)
        return
    if options.repeats < 1 or options.epochs < 1:
        parser.error("repeats and epochs must be positive")
    if options.batches and options.dataset == "all":
        parser.error("--batches requires one --dataset")
    options.repo = options.repo.resolve()
    if not (options.repo / "Model" / "Model.py").is_file():
        parser.error("--repo must point to the PINN4SOH repository")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["OMP_NUM_THREADS"] = str(options.threads)
    os.environ["MKL_NUM_THREADS"] = str(options.threads)
    import numpy as np
    import torch
    import pandas as pd
    import sklearn
    torch.set_num_threads(options.threads)
    if not torch.cuda.is_available() and not options.allow_cpu:
        raise RuntimeError("CUDA unavailable; refusing accidental CPU training")
    options.output.mkdir(parents=True, exist_ok=True)
    environment = {"recorded_at": utc_now(), "python": sys.version, "torch": torch.__version__,
                   "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
                   "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   "threads": options.threads}
    environment_path = options.output / "environment.json"
    if environment_path.exists():
        previous_environment = json.loads(environment_path.read_text(encoding="utf-8"))
        stable_fields = [key for key in environment if key != "recorded_at"]
        if any(previous_environment.get(key) != environment[key] for key in stable_fields):
            raise RuntimeError("Runtime environment changed; choose a new --output directory")
    else:
        write_json(environment_path, environment)
    inventory = source_inventory(options.repo)
    inventory_path = options.output / "source_inventory.json"
    if inventory_path.exists():
        old_inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        if old_inventory["fingerprint"] != inventory["fingerprint"]:
            raise RuntimeError("Source/data changed; choose a new --output directory")
        if old_inventory["data_directory_order"] != inventory["data_directory_order"]:
            raise RuntimeError("Data directory order changed; choose a new --output directory")
    else:
        write_json(inventory_path, inventory)
    manifest_path = options.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
        "created_at": utc_now(), "runs": {}}
    sys.path.insert(0, str(options.repo))
    os.chdir(options.repo)
    datasets = list(BATCHES) if options.dataset == "all" else [options.dataset]
    failures = []
    for dataset in datasets:
        module = importlib.import_module("main_" + dataset)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        class SnapshotPINN(module.PINN):
            def __setattr__(self, name, value):
                if name == "best_model" and isinstance(value, dict):
                    value = copy.deepcopy(value)
                super().__setattr__(name, value)

        old_argv = sys.argv
        try:
            sys.argv = ["main_" + dataset + ".py"]
            default_args = module.get_args()
        finally:
            sys.argv = old_argv
        batches = options.batches or BATCHES[dataset]
        if any(batch not in BATCHES[dataset] for batch in batches):
            parser.error(f"Valid {dataset} batches: {BATCHES[dataset]}")
        if options.smoke:
            batches = batches[:1]
        for batch in batches:
            repeats = 1 if options.smoke else options.repeats
            for repeat in range(1, repeats + 1):
                run_key = f"{dataset}/{batch}/Experiment{repeat:02d}"
                folder = options.output / run_key
                seed_offset = list(BATCHES).index(dataset) * 10000 + BATCHES[dataset].index(batch) * 100
                seed = options.seed + seed_offset + repeat - 1
                args = argparse.Namespace(**vars(default_args))
                args.epochs = 2 if options.smoke else options.epochs
                args.warmup_epochs = min(args.warmup_epochs, max(0, args.epochs - 1))
                args.save_folder = str(folder)
                args.log_dir = "logging.txt"
                if dataset == "XJTU":
                    args.batch = batch
                elif dataset == "TJU":
                    args.batch, args.in_same_batch = int(batch), True
                config = {"parameters": vars(args), "seed": seed, "smoke": options.smoke,
                          "threads": options.threads, "source_fingerprint": inventory["fingerprint"],
                          "checkpoint_snapshot_deepcopy": True}
                config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
                previous = manifest["runs"].get(run_key)
                if previous and previous["status"] == "complete":
                    if not options.resume:
                        raise RuntimeError(f"{run_key} already complete; use --resume or new output")
                    if previous.get("config_hash") != config_hash:
                        raise RuntimeError(f"{run_key} configuration changed; use new output")
                    if all((folder / name).is_file() for name in ["true_label.npy", "pred_label.npy", "model.pth"]):
                        calculate_metrics(np.load(folder / "true_label.npy", allow_pickle=False),
                                          np.load(folder / "pred_label.npy", allow_pickle=False))
                        print("SKIP " + run_key, flush=True)
                        continue
                if folder.exists() and any(folder.iterdir()):
                    archive = folder.with_name(folder.name + ".previous." + str(time.time_ns()))
                    folder.rename(archive)
                folder.mkdir(parents=True, exist_ok=True)
                write_json(folder / "config.json", config)
                run = {"dataset": dataset, "batch": batch, "repeat": repeat, "seed": seed,
                       "relative_folder": run_key, "config_hash": config_hash,
                       "status": "running", "started_at": utc_now()}
                manifest["runs"][run_key] = run
                write_json(manifest_path, manifest)
                print("START " + run_key + " seed=" + str(seed), flush=True)
                model = None
                saved_handlers = []
                try:
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    with (folder / "console.txt").open("w", encoding="utf-8", buffering=1) as console:
                        with contextlib.redirect_stdout(console), contextlib.redirect_stderr(console):
                            with capture_data_paths() as split_paths:
                                loaders = getattr(module, LOADERS[dataset])(args)
                            write_json(folder / "splits.json", split_paths)
                            run["sample_counts"] = {k: len(v.dataset) for k, v in loaders.items()}
                            for split, loader in loaders.items():
                                if not len(loader.dataset) or not all(torch.isfinite(t).all() for t in loader.dataset.tensors):
                                    raise ValueError("Empty or non-finite " + split + " data")
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            started = time.perf_counter()
                            model = SnapshotPINN(args)
                            saved_handlers = list(model.logger.handlers)
                            model.Train(trainloader=loaders["train"], validloader=loaders["valid"],
                                        testloader=loaders["test"])
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            run["training_seconds"] = time.perf_counter() - started
                    truth = np.load(folder / "true_label.npy", allow_pickle=False)
                    prediction = np.load(folder / "pred_label.npy", allow_pickle=False)
                    run["metrics"] = calculate_metrics(truth, prediction)
                    train_log = (folder / "logging.txt").read_text(encoding="utf-8")
                    epochs = re.findall(r"\[Train\] epoch:(\d+)", train_log)
                    run["epochs_completed"] = int(epochs[-1]) if epochs else None
                    run["status"] = "complete"
                    run["finished_at"] = utc_now()
                    print("DONE " + run_key + " " + json.dumps(run["metrics"]), flush=True)
                except Exception as error:
                    run["status"], run["error"] = "failed", repr(error)
                    run["finished_at"] = utc_now()
                    (folder / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                    failures.append(run_key)
                    print("FAILED " + run_key + " " + repr(error), flush=True)
                finally:
                    for handler in saved_handlers:
                        handler.close()
                    logger = logging.getLogger("mylogger")
                    logger.handlers.clear()
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    write_json(manifest_path, manifest)
                summarize(options.output)
    summarize(options.output, options.plots)
    if failures:
        raise SystemExit("Failed runs: " + ", ".join(failures))


if __name__ == "__main__":
    main()
