"""Read-only descriptive review of the frozen 1324-task report snapshot."""

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "outputs/full_reproduction/partial_results_1324/suite_report.json"
EXPECTED_SHA = "75806c381d68b752c557d6c0b00469a31541938efdc21a00e65f00dd87ca20e6"
OUT = REPORT.parent / "independent_review"
LANES = ["published_label_monotonicity", "paper_prediction_monotonicity"]
CONFIGS = [("PINN", LANES[0]), ("PINN", LANES[1]), ("MLP", "author_current"), ("CNN", "author_current")]
LABELS = {LANES[0]: "标签单调PINN", LANES[1]: "预测单调PINN"}
METRICS = ["MAE", "MAPE", "RMSE"]
BATCHES = [("XJTU", b) for b in ["2C", "3C", "R2.5", "R3", "RW", "satellite"]]
BATCHES += [("MIT", "all"), ("HUST", "all"), *[("TJU", str(n)) for n in range(3)]]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def main():
    actual_sha = hashlib.sha256(REPORT.read_bytes()).hexdigest()
    if actual_sha != EXPECTED_SHA:
        raise ValueError("Expected the immutable 1324 snapshot")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if report["metadata_only"] or any(row.get("report_audit_error") for row in report["tasks"]):
        raise ValueError("Expected full array audit and zero report audit errors")
    primary = [row for row in report["summaries"] if row["aggregation"] == "battery_macro"
        and row["convention"] == "standard" and row["evaluation"] == "final"]
    lookup = {}
    for row in primary:
        key = (row["family"], row["dataset"], row["batch"], row["sample_size"], row["model"], row["loss_variant"])
        if key in lookup:
            raise ValueError("Ambiguous scientific configuration")
        lookup[key] = row
    cohorts, completeness = {}, []
    for family in ["regular", "small_sample"]:
        expected = [(dataset, batch, None) for dataset, batch in BATCHES] if family == "regular" else [
            (dataset, batch, n) for dataset, batch in BATCHES if dataset in {"XJTU", "HUST"} for n in range(1, 5)]
        for dataset, batch, size in expected:
            records = [lookup.get((family, dataset, batch, size, model, lane)) for model, lane in CONFIGS]
            counts = [row["n_repeats"] if row else 0 for row in records]
            complete = all(row and row["complete_repeat_set"] for row in records)
            completeness.append(dict(family=family, dataset=dataset, batch=batch, sample_size=size,
                complete=complete, **{model + "_" + lane + "_repeats": count for (model, lane), count in zip(CONFIGS, counts)}))
            if complete:
                # Same cohort, paired seeds, same test split and numerical environment.
                for field in ["source_fingerprint", "validation_protocol", "environment_sha256", "test_split_sha256", "repeat_ids", "seeds"]:
                    if any(row[field] != records[0][field] for row in records):
                        raise ValueError("Unmatched cohort field: " + field)
                cohorts[(family, dataset, batch, size)] = records
    regular = {key: rows for key, rows in cohorts.items() if key[0] == "regular"}
    small = {key: rows for key, rows in cohorts.items() if key[0] == "small_sample"}
    if len(regular) != 11 or len(small) != 22:
        raise ValueError("Unexpected snapshot completeness")
    effects, counts, tables = [], [], {"regular": [], "small_sample": []}
    for key, rows in sorted(cohorts.items()):
        family, dataset, batch, size = key
        for row in rows:
            tables[family].append({"family": family, "dataset": dataset, "batch": batch,
                "sample_size": size, "model": row["model"], "loss_variant": row["loss_variant"],
                "n_repeats": row["n_repeats"], "comparison_group": row["comparison_group"],
                **{metric + suffix: row[metric + suffix] for metric in METRICS for suffix in ["_mean", "_sd_ddof0", "_sd_ddof1"]}})
        for left in rows[:2]:
            for right in rows[2:]:
                for metric in METRICS:
                    delta = left[metric + "_mean"] - right[metric + "_mean"]
                    effects.append(dict(family=family, dataset=dataset, batch=batch, sample_size=size,
                        loss_variant=left["loss_variant"], comparator=right["model"], metric=metric,
                        left_mean=left[metric + "_mean"], right_mean=right[metric + "_mean"],
                        left_minus_right=delta, direction="lower" if delta < -1e-12 else "higher" if delta > 1e-12 else "tie",
                        left_sd_ddof1=left[metric + "_sd_ddof1"], right_sd_ddof1=right[metric + "_sd_ddof1"]))
    # Cross-check every independent effect against the saved report's comparator selection.
    saved = {(row["left_comparison_group"], row.get("right_comparison_group"), row.get("metric")): row
             for row in report["comparison_effects"] if row["status"] == "descriptive_comparison_available"}
    for key, rows in cohorts.items():
        for left in rows[:2]:
            for right in rows[2:]:
                for metric in METRICS:
                    effect = saved[left["comparison_group"], right["comparison_group"], metric]
                    if abs(effect["left_minus_right"] - (left[metric + "_mean"] - right[metric + "_mean"])) > 1e-15:
                        raise ValueError("Independent effect disagrees with saved report")
    for family, collection in [("regular", regular), ("small_sample", small)]:
        for lane in LANES:
            for metric in METRICS:
                c = Counter()
                for rows in collection.values():
                    left = next(row for row in rows if row["loss_variant"] == lane)
                    base = rows[2:]
                    c["lower_both_mean"] += all(left[metric + "_mean"] < r[metric + "_mean"] for r in base)
                    c["lower_MLP_mean"] += left[metric + "_mean"] < base[0][metric + "_mean"]
                    c["lower_CNN_mean"] += left[metric + "_mean"] < base[1][metric + "_mean"]
                    c["lower_both_SD"] += all(left[metric + "_sd_ddof1"] < r[metric + "_sd_ddof1"] for r in base)
                counts.append(dict(family=family, loss_variant=lane, metric=metric, cohorts=len(collection), **c))
    trends = []
    for dataset, batch in BATCHES:
        if any(("small_sample", dataset, batch, size) not in cohorts for size in range(1, 5)):
            continue
        for model, lane in CONFIGS:
            rows = [lookup["small_sample", dataset, batch, size, model, lane] for size in range(1, 5)]
            for metric in METRICS:
                values = [row[metric + "_mean"] for row in rows]
                trends.append(dict(dataset=dataset, batch=batch, model=model, loss_variant=lane, metric=metric,
                    **{"n" + str(n): v for n, v in enumerate(values, 1)},
                    n4_lower_n1=values[3] < values[0], every_step_lower=all(b < a for a, b in zip(values[:-1], values[1:])),
                    increases=[f"n{n+1}->n{n+2}" for n, (a, b) in enumerate(zip(values[:-1], values[1:])) if b > a]))
    numeric = [row for row in report["published_comparisons"] if row["aggregation"] == "battery_macro"
        and row["convention"] == "standard" and row["complete_repeat_set"] and row["table"] in {"Table 2", "Table 3", "Table S7"}]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in [("regular_metrics.csv", tables["regular"]), ("complete_small_sample_metrics.csv", tables["small_sample"]),
        ("matched_effects.csv", effects), ("cohort_completeness.csv", completeness), ("direction_counts.csv", counts), ("sample_size_trends.csv", trends)]:
        write_csv(OUT / name, rows)
    review = dict(source_report=str(REPORT.relative_to(ROOT)), source_report_sha256=actual_sha,
        status_counts=report["task_status_counts"], audit_errors=0,
        regular_completed_tasks=440, complete_small_sample_cohorts=len(small), complete_small_sample_tasks=880,
        incomplete_small_sample="satellite n3: label-only PINN repeats1..4 only; all other lanes/n4 and HUST unavailable in this snapshot",
        direction_counts=counts, numeric_matches=[row for row in numeric if row["within_printed_4dp_half_unit"]],
        limitations=["Descriptive mean and sample-SD comparisons, without significance or equivalence claims",
            "Full main+supplement suite and transfer/sensitivity conclusions remain pending",
            "MIT battery_macro is a released-CSV file macro; mixed CSV5 is not one physical cell",
            "Ten seeds are prespecified reproduction seeds, not recovered author seeds",
            "The report audits saved arrays on cloud; local review verifies the immutable report and recomputes comparator differences from summaries"])
    (OUT / "independent_review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 1324 任务快照独立初审", "",
        f"审读对象为固定快照 `{REPORT.name}`，SHA-256 `{actual_sha}`。本初审没有改动数据或训练配置。", "",
        "快照包含 1324 个通过报告审计的任务，796 个未纳入，报告审计错误为 0。常规实验 440/440 已齐（11 组 × 两种 PINN 损失与 MLP/CNN × 10 次）。小样本仅 22 个完整对照组、880 个任务已齐：XJTU 的 2C/3C/R2.5/R3/RW 五批 n=1..4，以及 satellite 的 n=1/2。satellite n3 只有标签单调 PINN 的 4 次完成，不能形成完整对照；satellite n4、HUST 小样本、迁移及 40 次物理身份诊断均不能由本快照给出完整结论。", "",
        "以下固定使用 standard measured-SOH denominator 和 battery_macro，均值与样本标准差来自十个独立 seed。RMSE 乘 100 后单位为 SOH 百分点（pp）。未进行显著性或等效性判断，也未从多种指标约定中挑选最接近论文的一种。MIT 的 battery_macro 按发布 CSV 文件等权，其中混合 CSV5 含两个物理身份，应称发布文件宏平均，不能称真实物理电池等权平均。", "",
        "## 已齐的常规实验", "",
        "标签单调 PINN 在 MAE、MAPE、RMSE 上均为 9/11 组低于两种基线，支持‘大多数设置更小’的限定描述；MIT 和 HUST 均未优于任一基线。预测单调 PINN 在三个指标上均为 0/11 组同时低于两基线，11/11 均高于 MLP，仅 XJTU 3C/RW 低于 CNN。这两条实现通道的结论差异很大，不能合并成同一方法结果。", "",
        "RMSE 的十次样本 SD：两条 PINN 通道均在 11/11 组低于两基线。对 MAE/MAPE，标签单调通道为 10/11，预测单调通道为 11/11。预测单调通道的误差均值明显较高而 SD 很小，说明跨 seed 接近同一个较差解也会得到低 SD；低 SD 不能替代准确率。单靠本比较不能把优势因果归于物理知识，标签单调项本身对模型参数的梯度为零。", "",
        "| 组别 | 标签单调PINN RMSE±SD | 预测单调PINN RMSE±SD | MLP RMSE±SD | CNN RMSE±SD |", "|---|---:|---:|---:|---:|"]
    for dataset, batch in BATCHES:
        rows = regular["regular", dataset, batch, None]
        lines.append(f"| {dataset}/{batch} | " + " | ".join(f"{r['RMSE_mean']*100:.4f}±{r['RMSE_sd_ddof1']*100:.4f}" for r in rows) + " |")
    lines += ["", "## 已齐的小样本对照", "",
        "标签单调 PINN 在 MAE/MAPE/RMSE 三项上都是 21/22 组低于两基线，唯一例外为 RW n1：RMSE 为 20.1488 pp，高于 MLP 的 8.9649 pp，低于 CNN 的 25.4596 pp。预测单调 PINN 的对应胜出组数分别为 MAE 4/22、MAPE 5/22、RMSE 5/22；RMSE 胜出为 2C n1、3C n1/n2/n3、RW n3。小样本优势同样依赖实现通道和具体设置。", "",
        "在 n1..4 全齐的五个 XJTU 批次，标签单调 PINN 的 RMSE 从 n1 到 n4 均下降，但 3C 从 n3 的 1.2152 pp 回升到 n4 的 1.3191 pp；另外四批逐步下降。MLP/CNN 从 n1 到 n4 也都下降，MLP 只有 3C 每步下降，CNN 除 2C 的 n2→n3 回升外其余四批每步下降。预测单调通道仅 R2.5 每步下降，2C 的 n4 比 n1 略高（4.7684 vs 4.7592 pp）。因此可描述首末趋势，不能声称增加电池数必然改善每一步。", "",
        "对于‘1 电池 PINN 接近 3/4 电池基线’：在这五批中，标签单调 PINN n1 的 RMSE 分别在 7/10 个跨模型比较中低于 n3 基线、7/10 低于 n4 基线；未胜出包括 3C 对 MLP 与 RW 对两基线。‘接近’尚无预设等效界值，这些胜负计数不能证明等效。", "",
        "| 组别 | n | 标签单调PINN RMSE | 预测单调PINN RMSE | MLP RMSE | CNN RMSE |", "|---|---:|---:|---:|---:|---:|"]
    for key in [("small_sample", dataset, batch, size) for dataset, batch in BATCHES for size in range(1, 5) if ("small_sample", dataset, batch, size) in small]:
        _, dataset, batch, size = key
        lines.append(f"| {dataset}/{batch} | {size} | " + " | ".join(f"{r['RMSE_mean']*100:.4f}" for r in small[key]) + " |")
    lines += ["", "## 与论文数值的边界", "",
        "固定主约定下，Table2 共 88 个已齐数值对照只有 HUST/CNN 的 RMSE 落在论文四位小数半单位内（0.0087323 对 0.0087）；标签单调 PINN 22 个主表 MAPE/RMSE 对照均未达到该打印精度。已齐 Table3 的 32 个对照中，只有 XJTU/1 n4 标签单调 PINN RMSE 匹配（0.0076066 对 0.0076）。TableS7 还有 R2.5 n2 标签单调 PINN MAPE 匹配（0.0095823 对 0.0096），以及同一个 n4 RMSE 的重复列示。零星打印精度相符不等于协议完全相同；自设 seed、损失实现及已记录数据问题仍需保留。", "",
        "完整 claims 在原报告中仍标为 pending_rule_evaluation。本初审仅支持上述已齐实验范围内的描述性结论，不得写成全部论文结论已复现。独立数值表、完整性、所有均值差值和样本量趋势见本目录 CSV/JSON；所有主比较差值均与原报告保存的 matched comparison_effects 再次核对一致。", ""]
    (OUT / "1324快照_常规与已齐小样本_独立初审.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"source_report_sha256": actual_sha, "regular_cohorts": len(regular), "complete_small_cohorts": len(small),
        "effect_checks": len(effects), "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
