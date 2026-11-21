"""Audit downloaded PINN4SOH metrics and create a bounded local report."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics


GROUPS = [("XJTU", batch) for batch in ["2C", "3C", "R2.5", "R3", "RW", "satellite"]]
GROUPS += [("MIT", "all"), ("HUST", "all"), ("TJU", "0"), ("TJU", "1"), ("TJU", "2")]
EXPECTED_REPEATS = 10
EXPECTED_RUNS = {f"{dataset}/{batch}/Experiment{repeat:02d}"
                 for dataset, batch in GROUPS for repeat in range(1, EXPECTED_REPEATS + 1)}
METRICS = ["MAE", "MAPE", "MSE", "RMSE", "MAE_pp", "MAPE_pct", "RMSE_pp", "training_seconds"]
DISPLAY_METRICS = ["MAE_pp", "MAPE_pct", "RMSE_pp", "training_seconds"]
COMMIT = "cc3cc5053caf38f04e0665f7f88cb109144d035e"


def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value, name, allow_empty=False):
    if value in (None, "") and allow_empty:
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"Invalid nonnegative finite value for {name}: {value!r}")
    return result


def same_number(actual, expected, name):
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError(f"Missing value mismatch for {name}")
    elif not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-10):
        raise ValueError(f"Inconsistent {name}: {actual} versus {expected}")


def parse_summary(path):
    parsed = {}
    for raw in load_csv(path):
        key = (raw["dataset"], raw["batch"])
        if key not in GROUPS or key in parsed:
            raise ValueError(f"Unexpected or duplicate summary group: {key}")
        n = int(raw["n_runs"])
        if not 1 <= n <= EXPECTED_REPEATS:
            raise ValueError(f"Invalid repetition count for {key}: {n}")
        row = {"dataset": key[0], "batch": key[1], "n_runs": n}
        for metric in METRICS:
            for statistic in ["mean", "std"]:
                field = metric + "_" + statistic
                row[field] = number(raw.get(field), field, allow_empty=statistic == "std" and n == 1)
                if statistic == "std" and n == 1 and row[field] is not None:
                    raise ValueError(f"A single run cannot supply a sample standard deviation: {key} {metric}")
        for ratio, percent in [("MAE", "MAE_pp"), ("MAPE", "MAPE_pct"), ("RMSE", "RMSE_pp")]:
            for statistic in ["mean", "std"]:
                source = row[ratio + "_" + statistic]
                same_number(row[percent + "_" + statistic], None if source is None else source * 100,
                            f"{key} {percent} {statistic} scaling")
        parsed[key] = row
    if not parsed:
        raise ValueError("The summary contains no completed runs")
    return parsed


def audit_per_run(path, summary):
    rows, by_key = [], {}
    for raw in load_csv(path):
        key = raw["run"]
        if key not in EXPECTED_RUNS or key in by_key:
            raise ValueError(f"Unexpected or duplicate run: {key}")
        dataset, batch, experiment = key.split("/")
        repeat = int(raw["repeat"])
        if (raw["dataset"], raw["batch"], f"Experiment{repeat:02d}") != (dataset, batch, experiment):
            raise ValueError(f"Run identity columns disagree: {key}")
        row = {"run": key, "dataset": dataset, "batch": batch, "repeat": repeat,
               "seed": int(raw["seed"]), "n_test": int(raw["n_test"])}
        if row["n_test"] <= 0:
            raise ValueError(f"Empty test set: {key}")
        for metric in METRICS:
            row[metric] = number(raw[metric], key + " " + metric)
        for ratio, percent in [("MAE", "MAE_pp"), ("MAPE", "MAPE_pct"), ("RMSE", "RMSE_pp")]:
            same_number(row[percent], row[ratio] * 100, key + " " + percent + " scaling")
        same_number(row["RMSE"] ** 2, row["MSE"], key + " RMSE squared")
        rows.append(row)
        by_key[key] = row
    counts = Counter((row["dataset"], row["batch"]) for row in rows)
    if dict(counts) != {key: row["n_runs"] for key, row in summary.items()}:
        raise ValueError("Per-run CSV and summary cover different runs; download one consistent snapshot")
    for key, group in summary.items():
        selected = [row for row in rows if (row["dataset"], row["batch"]) == key]
        if len({row["n_test"] for row in selected}) != 1:
            raise ValueError(f"Test sample counts change between repeats in {key}")
        if len({row["seed"] for row in selected}) != len(selected):
            raise ValueError(f"Repeated random seeds within a group are not independent repetitions: {key}")
        for metric in METRICS:
            values = [row[metric] for row in selected]
            same_number(group[metric + "_mean"], statistics.mean(values), str(key) + " " + metric + " mean")
            std = statistics.stdev(values) if len(values) > 1 else None
            same_number(group[metric + "_std"], std, str(key) + " " + metric + " sample std")
    return rows, by_key


def audit_manifest(path, summary, per_run):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    runs = manifest["runs"]
    unexpected = sorted(set(runs) - EXPECTED_RUNS)
    if unexpected:
        raise ValueError(f"Unexpected run identifiers in manifest: {unexpected}")
    complete = {key: run for key, run in runs.items() if run["status"] == "complete"}
    counts = Counter((run["dataset"], run["batch"]) for run in complete.values())
    if dict(counts) != {key: row["n_runs"] for key, row in summary.items()}:
        raise ValueError("Manifest and summary contain different completed counts; obtain a consistent snapshot")
    if per_run is not None and set(complete) != set(per_run):
        raise ValueError("Manifest and per-run CSV contain different completed run identities")
    for key, run in complete.items():
        dataset, batch, experiment = key.split("/")
        if (run["dataset"], str(run["batch"]), f"Experiment{int(run['repeat']):02d}") != (dataset, batch, experiment):
            raise ValueError("Manifest identity columns disagree: " + key)
        if per_run is not None:
            row = per_run[key]
            if int(run["seed"]) != row["seed"]:
                raise ValueError("Seed disagrees with per-run CSV: " + key)
            if int(run["metrics"]["n_test"]) != row["n_test"]:
                raise ValueError("Test sample count disagrees with per-run CSV: " + key)
            for metric in METRICS:
                value = run["training_seconds"] if metric == "training_seconds" else run["metrics"][metric]
                same_number(number(value, key + " " + metric), row[metric], key + " manifest " + metric)
    return manifest, dict(Counter(run["status"] for run in runs.values()))


def format_measure(row, metric):
    if row is None:
        return "暂无结果"
    mean, std = row[metric + "_mean"], row[metric + "_std"]
    digits = 1 if metric == "training_seconds" else 4
    return f"{mean:.{digits}f}" if std is None else f"{mean:.{digits}f} ± {std:.{digits}f}"


def render_figure(summary, output, verified, style_kernel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.text import Text

    if not style_kernel.is_file():
        raise FileNotFoundError("Supply --figure-style-kernel pointing to the figure-style skill's kernel.py")
    spec = importlib.util.spec_from_file_location("pinsoh_figure_style", style_kernel)
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    style.apply_figure_style(font="Arial", sizes=(10, 9, 8))
    fig = plt.figure(figsize=(7.2, 6.8))
    ax = fig.add_axes([0.31, 0.28, 0.66, 0.575])
    colors = {"XJTU": "#0072B2", "MIT": "#E69F00", "HUST": "#009E73", "TJU": "#CC79A7"}
    positions, labels, row_position = [], [], 0.0
    previous_dataset = None
    separators = []
    for dataset, batch in GROUPS:
        if previous_dataset is not None and dataset != previous_dataset:
            separators.append(row_position - 0.3)
            row_position += 0.55
        key, y = (dataset, batch), row_position
        positions.append(y)
        row = summary.get(key)
        n = row["n_runs"] if row else 0
        batch_name = "all batteries" if batch == "all" else batch
        labels.append(f"{dataset} / {batch_name}   (n = {n}/10)")
        if row:
            mean, std = row["MAE_pp_mean"], row["MAE_pp_std"]
            filled = n == EXPECTED_REPEATS
            ax.errorbar(mean, y, xerr=std if std is not None else None, fmt="o" if filled else "D",
                        markersize=5.5, capsize=3, elinewidth=1.15, color=colors[dataset],
                        markerfacecolor=colors[dataset] if filled else "white",
                        markeredgecolor=colors[dataset], markeredgewidth=1.15, zorder=3)
        else:
            ax.text(0.025, y, "No results in this snapshot", transform=ax.get_yaxis_transform(),
                    color="#777777", va="center", fontsize=9)
        row_position += 1
        previous_dataset = dataset
    for y in separators:
        ax.axhline(y, color="#dddddd", linewidth=0.55, zorder=0)
    upper = max(row["MAE_pp_mean"] + (row["MAE_pp_std"] or 0) for row in summary.values())
    target_step = max(0.1, upper * 1.15) / 6
    magnitude = 10 ** math.floor(math.log10(target_step))
    step = next(value * magnitude for value in [1, 2, 2.5, 5, 10]
                if value * magnitude >= target_step)
    steps = math.ceil(max(0.1, upper * 1.15) / step)
    ax.set_xlim(0, steps * step)
    ax.set_xticks([index * step for index in range(steps + 1)])
    ax.set_ylim(positions[-1] + 0.65, -0.65)
    ax.set_yticks(positions, labels)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_xlabel("Test MAE (SOH percentage points)", labelpad=9)
    ax.xaxis.grid(True, color="#e4e4e4", linewidth=0.6)
    ax.spines["left"].set_visible(False)
    total = sum(row["n_runs"] for row in summary.values())
    fig.text(0.035, 0.965, "PINN errors on held-out battery data", va="top", fontsize=10)
    status = "Verified completed run set" if verified else "Partial / unaudited snapshot"
    fig.text(0.035, 0.925, f"{status}: {total}/110 runs represented", va="top", fontsize=9, color="#555555")
    fig.text(0.97, 0.89, "Lower is better", ha="right", fontsize=9)
    handles = [Line2D([], [], color="#444444", marker="o", linestyle="none", markersize=5.5,
                      label="10/10 repeats"),
               Line2D([], [], color="#444444", marker="D", markerfacecolor="white", linestyle="none",
                      markersize=5.5, label="Fewer than 10 repeats")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.3, 0.147), ncol=2,
               columnspacing=1.5, handletextpad=0.5)
    fig.text(0.035, 0.102,
             "Points: mean over independent runs. Error bars: sample SD (ddof = 1).\n"
             "Author's September 2026 corrected code; max 200 epochs; validation early stopping (20).\n"
             "Each run pools its test samples. These are not battery-equal paper-table estimates.",
             va="top", fontsize=9, linespacing=1.45)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [(text, text.get_window_extent(renderer)) for text in fig.findobj(Text)
             if text.get_text().strip() and text.get_visible()]
    collisions = [(a.get_text(), b.get_text()) for index, (a, box_a) in enumerate(texts)
                  for b, box_b in texts[index + 1:] if box_a.overlaps(box_b)]
    outside = [text.get_text() for text, box in texts
               if box.x0 < fig.bbox.x0 - 1 or box.y0 < fig.bbox.y0 - 1
               or box.x1 > fig.bbox.x1 + 1 or box.y1 > fig.bbox.y1 + 1]
    qa = {"text_collisions": collisions, "texts_outside_figure": outside,
          "plotted_means": {f"{key[0]}/{key[1]}": row["MAE_pp_mean"] for key, row in summary.items()},
          "missing_groups_drawn_as_zero": False}
    (output / "figure_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    if collisions or outside:
        plt.close(fig)
        raise RuntimeError("Figure text layout failed; inspect figure_qa.json")
    for extension in ["png", "pdf"]:
        fig.savefig(output / ("mae_by_dataset_batch." + extension), dpi=300, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--per-run", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--source-inventory", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "final_report")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--figure-style-kernel", type=Path,
                        default=Path.home() / ".codex" / "skills" / "figure-style" / "kernel.py")
    options = parser.parse_args()
    summary = parse_summary(options.summary)
    per_run_rows, per_run = audit_per_run(options.per_run, summary) if options.per_run else (None, None)
    manifest, statuses = audit_manifest(options.manifest, summary, per_run) if options.manifest else (None, None)
    complete_coverage = all(summary.get(key, {}).get("n_runs") == EXPECTED_REPEATS for key in GROUPS)
    verified = bool(complete_coverage and per_run is not None and manifest is not None
                    and set(per_run) == EXPECTED_RUNS
                    and statuses.get("complete") == len(EXPECTED_RUNS)
                    and not any(value for key, value in statuses.items() if key != "complete"))
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    represented = sum(row["n_runs"] for row in summary.values())
    generated = datetime.now(timezone(timedelta(hours=8))).isoformat()
    inputs = {name: {"path": str(path.resolve()), "sha256": sha256(path)}
              for name, path in [("summary", options.summary), ("per_run", options.per_run),
                                 ("manifest", options.manifest), ("environment", options.environment),
                                 ("source_inventory", options.source_inventory)] if path}
    audit = {"generated_at": generated, "represented_runs": represented, "planned_runs": 110,
             "complete_summary_coverage": complete_coverage, "completion_verified": verified,
             "per_run_statistics_audited": per_run is not None,
             "manifest_audited": manifest is not None, "manifest_statuses": statuses,
             "inputs": inputs, "reference_upstream_commit": COMMIT,
             "note": "Reference commit is task context; actual source identity requires source_inventory.json."}
    (output / "report_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    title = "PINN4SOH 云端主实验结果" if verified else "PINN4SOH 部分结果快照（非最终完成报告）"
    lines = ["# " + title, "", f"生成时间：{generated}（Asia/Shanghai）。", ""]
    if verified:
        lines.append("11 个数据集/批次分组各 10 次，共 110 次；逐次 CSV、批次汇总和运行清单相互一致，清单均为 complete。")
    else:
        lines.append(f"当前汇总表覆盖 {represented}/110 次。尚未满足全部完成的审计条件，不能据此宣称 110 次主实验全部完成。")
        if not options.per_run or not options.manifest:
            lines.append("逐次结果 CSV 或 manifest 未提供，当前只能核对汇总表内部单位与重复次数，不能独立确认每次运行的完成状态。")
    if statuses is not None:
        lines.append("清单状态：" + "；".join(f"{key}={value}" for key, value in sorted(statuses.items())) + "。")
    lines += ["", "## 误差汇总", "",
              "数值为独立重复的均值 ± 样本标准差（ddof=1）；仅一次时不计算标准差。缺失分组表示当前快照没有已完成结果，可能仍在运行或尚未包含于下载文件，不代表误差为零。", "",
              "| 数据集 | 批次 | 结果次数/计划 | MAE（SOH百分点） | MAPE（%） | RMSE（SOH百分点） | 训练时间（秒） |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for dataset, batch in GROUPS:
        row = summary.get((dataset, batch))
        values = " | ".join(format_measure(row, metric) for metric in DISPLAY_METRICS)
        lines.append(f"| {dataset} | {batch} | {row['n_runs'] if row else 0}/10 | {values} |")
    xjtu = [summary.get(("XJTU", batch)) for batch in ["2C", "3C", "R2.5", "R3", "RW", "satellite"]]
    if all(row and row["n_runs"] == 10 for row in xjtu):
        mean = statistics.mean(row["MAE_pp_mean"] for row in xjtu)
        lines += ["", f"XJTU 六批次 MAE 均值的等权平均为 {mean:.4f} 个 SOH 百分点，仅作为批次摘要。它既不是全体测试样本合并误差，也不是各电池等权平均，不能直接替代论文表格口径。"]
    lines += ["", "## 口径与适用范围", "",
              "本任务复现作者 2026 年 9 月修正版，参考提交为 `" + COMMIT + "`。作者 fix.md 说明修复原始 bug 后 alpha/beta 与论文补充材料不同；现代云环境也与 README 原始版本不同，因此不能声称精确恢复 2024 论文数值。",
              "每次 MAE/MAPE/MSE/RMSE 来自验证集选择的最佳时刻预测，合并该次全部测试样本后计算。批次汇总再对独立重复等权求均值；图中的误差条是重复间样本标准差，不是置信区间，也不是电池间差异。",
              "SOH 原始标签为比例：0.01 相当于 1 个 SOH 百分点。MAE_pp 和 RMSE_pp 是比例误差乘 100，不是相对误差百分数；MAPE_pct 才是相对误差百分数。",
              "标准 MAPE 使用真实 SOH 作分母。作者日志 eval_metrix(pred_label, true_label) 把参数反向传入，可能得到不同 MAPE；日志 MAE/MSE/RMSE 不受该参数反向影响。",
              "训练时间含模型初始化和 Train，不含数据读取、进程启动与汇总，不等于任务总墙钟耗时。并行阶段单次训练时间受资源竞争影响，不能把串行与并行的单次秒数直接解释为速度提升；只有任务起止时间和调度记录支持总吞吐比较。",
              "作者源码保持原样；独立运行器在 best_model 赋值时 deepcopy 权重，修复快照引用问题，不改变网络、损失或超参数。重载一致性是否已实证通过，必须另看 verification_checkpoint.json，本报告不从误差 CSV 推断。",
              "TJU 沿用作者 os.listdir 顺序划分，实际路径和电池样本边界记录在各次 splits.json；本报告不凭批次编号猜测电池身份。",
              "本报告范围为四数据集 110 次 PINN 主实验，不包含 CNN/MLP、小样本和迁移实验。", "",
              "## 审计证据", "",
              f"汇总内部缩放核对：通过。逐次统计核对：{'通过' if per_run is not None else '未提供，未核对'}。运行清单核对：{'通过' if manifest is not None else '未提供，未核对'}。",
              "本报告从输入 CSV/JSON 汇总，不直接重新训练、重新计算预测数组或独立验证论文结论。输入文件路径和 SHA-256 记录在 report_audit.json。"]
    if options.environment:
        environment = json.loads(options.environment.read_text(encoding="utf-8"))
        lines += ["", "实际环境记录：`" + json.dumps(environment, ensure_ascii=False) + "`。"]
    if options.source_inventory:
        inventory = json.loads(options.source_inventory.read_text(encoding="utf-8"))
        lines += ["", "源码/数据指纹：`" + inventory["fingerprint"] + "`。参考提交：`"
                  + str(inventory.get("reference_upstream_commit", "未记录")) + "`。"]
    if not options.no_plots:
        render_figure(summary, output, verified, options.figure_style_kernel)
        lines += ["", "![各数据集批次误差汇总](mae_by_dataset_batch.png)"]
    report_name = "复现结果报告.md" if verified else "部分结果快照_非最终.md"
    (output / report_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output / report_name), **audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
