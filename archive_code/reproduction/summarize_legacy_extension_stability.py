"""Describe seed directions separately from ten-repeat means; never train.

Consumes a complete audited selected-extension comparison. Reconstructs effects
from the metric table, cross-checks the comparator's effect table and summaries,
and writes an independent report without changing those inputs.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import textwrap


METRICS = ("MAE", "MAPE", "MSE", "RMSE")
REPEATS = list(range(1, 11))
STACKS = ("legacy", "modern")
METHODS = ("PINN published-code", "PINN paper-formula", "MLP", "CNN")


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as reader:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else
            format(value, ".17g") if isinstance(value, float) else value for key, value in row.items()} for row in rows)


def number(value):
    result = float(value)
    require(math.isfinite(result), "Nonfinite statistic")
    return result


def close(left, right):
    return math.isclose(number(left), number(right), rel_tol=2e-12, abs_tol=1e-15)


def direction(value):
    return "lower" if value < 0 else "higher" if value > 0 else "exact_zero"


def stats(values):
    require(values and all(math.isfinite(x) for x in values), "Finite nonempty values required")
    return {"n": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
        "sd_ddof1": statistics.stdev(values) if len(values) > 1 else None,
        "minimum": min(values), "maximum": max(values)}


def direction_summary(values):
    counts = dict(Counter(direction(value) for value in values))
    counts = {key: counts.get(key, 0) for key in ("lower", "exact_zero", "higher")}
    mean_direction = direction(statistics.mean(values))
    majority = next((key for key, count in counts.items() if count > len(values) / 2), "no_strict_majority")
    return {**stats(values), "direction_counts": counts, "direction_of_mean": mean_direction,
        "direction_of_median": direction(statistics.median(values)), "strict_majority_direction": majority,
        "number_same_direction_as_mean": counts[mean_direction], "all_observed_seeds_same_direction": max(counts.values()) == len(values),
        "smallest_nonzero_absolute_effect": min((abs(x) for x in values if x != 0), default=None)}


def method_name(task):
    if task["model"] != "PINN":
        return task["model"]
    return "PINN published-code" if task["loss_variant"] == "published_label_monotonicity" else "PINN paper-formula"


def ranks(values):
    """Ascending exact-value midranks; ties are numeric equality, not equivalence."""
    order = sorted(values, key=values.get)
    return {name: 1 + sum(other < values[name] for other in values.values()) +
        (sum(other == values[name] for other in values.values()) - 1) / 2 for name in order}


def summarize_effects(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["logical_group"], row["evaluation"], row["metric"])].append(row)
    summaries, pairs = [], []
    for key, current in sorted(grouped.items()):
        current = sorted(current, key=lambda x: x["repeat"])
        require([x["repeat"] for x in current] == REPEATS and len({x["seed"] for x in current}) == 10, "Effect needs exact ten unique repeats/seeds")
        legacy, modern = [x["legacy"] for x in current], [x["modern"] for x in current]
        identity = {"logical_group": key[0], "evaluation": key[1], "metric": key[2]}
        joint = Counter(direction(left) + "__" + direction(right) for left, right in zip(legacy, modern))
        row = dict(identity, legacy=direction_summary(legacy), modern=direction_summary(modern),
            paired_legacy_minus_modern=stats([left - right for left, right in zip(legacy, modern)]),
            paired_joint_direction_counts=dict(sorted(joint.items())),
            paired_direction_changes=sum(direction(left) != direction(right) for left, right in zip(legacy, modern)),
            mean_directions_differ=direction(statistics.mean(legacy)) != direction(statistics.mean(modern)),
            selected_repeat1={"legacy": legacy[0], "modern": modern[0], "legacy_direction": direction(legacy[0]),
                "modern_direction": direction(modern[0]), "directions_differ": direction(legacy[0]) != direction(modern[0])},
            new_repeat2_to10={"legacy": direction_summary(legacy[1:]), "modern": direction_summary(modern[1:]),
                "paired_direction_changes": sum(direction(left) != direction(right) for left, right in zip(legacy[1:], modern[1:]))})
        summaries.append(row)
        for item in current:
            pairs.append(dict(item, legacy_direction=direction(item["legacy"]), modern_direction=direction(item["modern"]),
                paired_direction_changed=direction(item["legacy"]) != direction(item["modern"]),
                legacy_minus_modern_effect=item["legacy"] - item["modern"], screened_repeat1=item["repeat"] == 1))
    return summaries, pairs


def audit_inputs(comparison, extension):
    comparison, extension = Path(comparison).resolve(), Path(extension).resolve()
    names = ["summary.json", "task_matching.csv", "per_repeat_metric_differences.csv", "per_repeat_method_effects.csv"]
    paths = [comparison / name for name in names] + [extension / name for name in
        ["extension_design.json", "extension_analysis_manifest.json", "extension_plan.json"]]
    hashes = {str(path): digest(path) for path in paths}
    result, design = read(comparison / "summary.json"), read(extension / "extension_design.json")
    manifest = read(extension / "extension_analysis_manifest.json")
    require(result.get("format") == "pinn4soh-selected-legacy-extension-comparison-v1" and
        result.get("assessment") == "complete_selected_ten_repeat_sensitivity" and result.get("expected_repeats") == 10,
        "Requires a complete audited ten-repeat selected comparison")
    require(result["analysis_manifest_sha256"] == design["analysis_manifest_sha256"] == hashes[str(extension / "extension_analysis_manifest.json")] and
        design["plan_sha256"] == hashes[str(extension / "extension_plan.json")], "Prepared/comparison support SHA differs")
    tasks = manifest["tasks"]
    require(result["expected_tasks"] == len(tasks) == design["analysis_task_count"] and
        result["task_counts"] == {"matched": len(tasks)} and result["selection"] == manifest["selection"], "Incomplete or changed selected task accounting")
    lookup = {(row["logical_group"], row["repeat"]): row for row in tasks}
    require(len(lookup) == len(tasks), "Duplicate selected logical repeat")
    matches = read_csv(comparison / "task_matching.csv")
    require(len(matches) == len(tasks) and {(x["logical_group"], int(x["repeat"])) for x in matches} == set(lookup) and
        all(x["status"] == "matched" and int(x["seed"]) == lookup[x["logical_group"], int(x["repeat"])]["seed"] for x in matches), "Task matching table changed/incomplete")
    metrics = {}
    for raw in read_csv(comparison / "per_repeat_metric_differences.csv"):
        group, repeat = raw["logical_group"], int(raw["repeat"])
        item = lookup[group, repeat]
        require(raw["comparable"] == "True" and int(raw["seed"]) == item["seed"] and raw["metric"] in METRICS and
            raw["legacy_task_id"] == item["legacy_task_id"] and raw["modern_task_id"] == item["modern_task_id"], "Metric row is not exact comparable selected task")
        key = group, repeat, raw["evaluation"], raw["metric"]
        require(key not in metrics, "Duplicate per-repeat metric")
        row = {**raw, "repeat": repeat, "seed": item["seed"], "legacy": number(raw["legacy"]), "modern": number(raw["modern"])}
        require(close(row["legacy"] - row["modern"], raw["paired_difference"]), "Metric paired difference changed")
        metrics[key] = row
    expected = {(x["logical_group"], x["repeat"], evaluation, metric) for x in tasks for metric in METRICS
                for evaluation in (["final", "source_only"] if x["legacy_task"]["family"] == "transfer" else ["final"])}
    require(set(metrics) == expected, "Metric row set differs from selected tasks")
    # Independently reconstruct the declared same-repeat effects; never choose a
    # convenient baseline or a source checkpoint using target test performance.
    effects = []
    for item in tasks:
        group, repeat, task = item["logical_group"], item["repeat"], item["legacy_task"]
        comparators = []
        if task["family"] == "transfer":
            comparators = [("fine_tune-minus-source_only", group, "source_only")]
        elif task["model"] == "PINN":
            for model in ["MLP", "CNN"]:
                candidates = [x for x in tasks if x["repeat"] == repeat and x["legacy_task"]["family"] == "full" and
                    x["legacy_task"]["model"] == model and x["legacy_task"]["dataset"] == task["dataset"] and x["legacy_task"]["batch"] == task["batch"]]
                require(len(candidates) <= 1, "Ambiguous selected method comparator")
                if candidates:
                    require(candidates[0]["seed"] == item["seed"], "PINN/baseline same-repeat seeds differ")
                    comparators.append(("PINN-minus-" + model, candidates[0]["logical_group"], "final"))
        for label, other_group, other_evaluation in comparators:
            for metric in METRICS:
                final, other = metrics[group, repeat, "final", metric], metrics[other_group, repeat, other_evaluation, metric]
                effects.append({"logical_group": group, "evaluation": label, "metric": metric, "repeat": repeat, "seed": item["seed"],
                    "legacy": final["legacy"] - other["legacy"], "modern": final["modern"] - other["modern"]})
    declared_rows = read_csv(comparison / "per_repeat_method_effects.csv")
    declared = {(x["logical_group"], x["evaluation"], x["metric"], int(x["repeat"])): x for x in declared_rows}
    require(len(declared_rows) == len(declared) == len(effects), "Effect row count/duplicates differ")
    for row in effects:
        other = declared[row["logical_group"], row["evaluation"], row["metric"], row["repeat"]]
        require(other["comparable"] == "True" and int(other["seed"]) == row["seed"] and all(close(row[key], other[key]) for key in STACKS), "Declared method effect differs from independently reconstructed metric difference")
        require((other["empirical_direction_changed"] == "True") == (direction(row["legacy"]) != direction(row["modern"])), "Declared direction flag changed")
    summaries, pairs = summarize_effects(effects)
    supplied = {(x["logical_group"], x["evaluation"], x["metric"]): x for x in result["method_effect_summaries"]}
    require(set(supplied) == {(x["logical_group"], x["evaluation"], x["metric"]) for x in summaries}, "Effect summary group set changed")
    for row in summaries:
        other = supplied[row["logical_group"], row["evaluation"], row["metric"]]
        require(other["complete_ten_repeat_set"] is True and other["repeat_ids"] == REPEATS, "Upstream ten-repeat effect summary incomplete")
        for stack in STACKS:
            require(other[stack]["n"] == 10 and close(row[stack]["mean"], other[stack]["mean"]) and
                close(row[stack]["sd_ddof1"], other[stack]["sd_ddof1"]), "Ten-repeat effect summary disagrees with rows")
    return result, manifest, metrics, summaries, pairs, hashes


def method_rankings(manifest, metrics):
    selected = {(x["dataset"], str(x["batch"])) for x in manifest["selection"]["regular_groups"]}
    grouped = defaultdict(list)
    for item in manifest["tasks"]:
        task = item["legacy_task"]
        if task["family"] == "full" and (task["dataset"], str(task["batch"])) in selected:
            grouped[task["dataset"], str(task["batch"])].append(item)
    seed_rows, summary_rows = [], []
    for (dataset, batch), items in sorted(grouped.items()):
        for metric in METRICS:
            for stack in STACKS:
                observed = defaultdict(list)
                for repeat in REPEATS:
                    rows = [x for x in items if x["repeat"] == repeat]
                    values = {method_name(x["legacy_task"]): metrics[x["logical_group"], repeat, "final", metric][stack] for x in rows}
                    require(set(values) == set(METHODS) and len(rows) == 4, "Selected regular group lacks exact four methods")
                    ranks_seed = ranks(values)
                    for item in rows:
                        method = method_name(item["legacy_task"])
                        observed[method].append(values[method])
                        seed_rows.append({"dataset": dataset, "batch": batch, "metric": metric, "stack": stack,
                            "repeat": repeat, "seed": item["seed"], "method": method, "error": values[method], "exact_value_midrank": ranks_seed[method],
                            "screened_repeat1": repeat == 1})
                means = {method: statistics.mean(values) for method, values in observed.items()}
                mean_ranks = ranks(means)
                for method, values in observed.items():
                    current = [x["exact_value_midrank"] for x in seed_rows if x["dataset"] == dataset and x["batch"] == batch and
                        x["metric"] == metric and x["stack"] == stack and x["method"] == method]
                    summary_rows.append({"dataset": dataset, "batch": batch, "metric": metric, "stack": stack, "method": method,
                        "n_repeats": 10, "ten_repeat_error": stats(values), "rank_of_ten_repeat_mean_error": mean_ranks[method],
                        "mean_of_seedwise_rank": statistics.mean(current), "seedwise_rank_counts": dict(sorted(Counter(current).items()))})
    return seed_rows, summary_rows


def label(group, evaluation):
    parts = group.split("/")
    loss = "published-code" if "/published_code/" in group else "paper-formula"
    if parts[0] == "transfer":
        return parts[-1].replace("-to-", " → ") + " | " + loss + " | fine-tune − source-only"
    return parts[2] + "/" + parts[3] + " | " + loss + " | " + evaluation.replace("PINN-minus-", "PINN − ")


def render(pairs, summaries, output, synthetic):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    # Same typography, colors, DPI and vector font settings as main v4 figures.
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": .6, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False})
    figures = []
    configurations = sorted({(x["logical_group"], x["evaluation"]) for x in pairs})
    for index, (group, evaluation) in enumerate(configurations, 1):
        fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.6))
        fig.subplots_adjust(left=.11, right=.98, bottom=.19, top=.89, hspace=.43, wspace=.36)
        fig.suptitle(textwrap.fill(label(group, evaluation), width=88), fontsize=10, y=.985)
        for ax, metric in zip(axes.flat, METRICS):
            current = sorted([x for x in pairs if x["logical_group"] == group and x["evaluation"] == evaluation and x["metric"] == metric], key=lambda x: x["repeat"])
            summary = next(x for x in summaries if x["logical_group"] == group and x["evaluation"] == evaluation and x["metric"] == metric)
            scale = 1 if metric == "MSE" else 100
            ax.axhline(0, color="#252525", linewidth=.7, zorder=0)
            for row in current:
                y = [row["legacy"] * scale, row["modern"] * scale]
                # Fixed x-only spread reveals identical y values without
                # changing an observed effect or breaking repeat pairing.
                spread = (row["repeat"] - 5.5) * .014
                ax.plot([spread, 1 + spread], y, color="#888888", alpha=.6, linewidth=.65, zorder=1)
                for x, color in [(0, "#b96537"), (1, "#197c80")]:
                    ax.scatter(x + spread, y[x], color=color, marker="*" if row["repeat"] == 1 else "o",
                        s=70 if row["repeat"] == 1 else 18, edgecolors="white", linewidths=.35, zorder=2)
            for x, stack in enumerate(STACKS):
                mean = summary[stack]["mean"] * scale
                ax.scatter(x + .18, mean, color="#252525", marker="D", s=30, zorder=4)
            counts = [summary[stack]["direction_counts"]["lower"] for stack in STACKS]
            ax.set_title(metric + f" | <0 legacy {counts[0]}/10, modern {counts[1]}/10", loc="left")
            ax.set_xticks([0, 1], ["Legacy", "Modern"])
            ax.set_xlim(-.35, 1.45)
            ax.set_ylabel("Difference (SOH²)" if metric == "MSE" else "Difference (MAPE %)" if metric == "MAPE" else "Difference (SOH pp)")
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)
            ax.margins(y=.15)
        handles = [Line2D([], [], color="#888888", marker="o", markersize=4, linewidth=.65, label="Paired repeats 2–10"),
            Line2D([], [], color="#888888", marker="*", markersize=8, linewidth=0, label="Screened repeat 1"),
            Line2D([], [], color="#252525", marker="D", markersize=5, linewidth=0, label="Ten-repeat mean")]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .055), ncol=3)
        note = "SYNTHETIC FIXTURE — NOT EXPERIMENT RESULTS" if synthetic else "Post-screen selected | descriptive n=10 | <0 favors PINN or fine-tuning | no equivalence threshold"
        fig.text(.5, .022, note, fontsize=7, ha="center", va="center")
        stem = "effect_seed_directions_%02d" % index
        for suffix in ["png", "pdf"]:
            fig.savefig(output / (stem + "." + suffix), dpi=300, facecolor="white")
        plt.close(fig)
        figures.append({"filename_stem": stem, "logical_group": group, "evaluation": evaluation, "synthetic": synthetic})
    return figures


def report(comparison, extension, output, no_plots=False):
    comparison, extension, output = [Path(x).resolve() for x in [comparison, extension, output]]
    require(not output.exists() and output.parent.is_dir(), "Use a fresh independent report output")
    require(all(output != x and x not in output.parents and output not in x.parents for x in [comparison, extension]), "Report must be separate from audited inputs")
    source, manifest, metrics, summaries, pairs, hashes = audit_inputs(comparison, extension)
    seed_ranks, mean_ranks = method_rankings(manifest, metrics)
    synthetic = bool(source.get("synthetic_fixture"))
    output.mkdir()
    write_csv(output / "paired_seed_effect_directions.csv", pairs)
    write_csv(output / "effect_direction_stability.csv", summaries)
    write_csv(output / "seedwise_method_ranks.csv", seed_ranks)
    write_csv(output / "ten_repeat_method_ranks.csv", mean_ranks)
    figures = render(pairs, summaries, output, synthetic) if not no_plots else []
    limitations = ["Post-screen selected sensitivity analysis: repeat1 helped select these groups and is included in ten-repeat statistics; repeats2–10 are separately described, not a new prespecified confirmatory sample.",
        "Seed direction counts, direction of the mean, average rank and rank of average error answer different descriptive questions. A mean sign does not establish most seeds have that sign.",
        "Exact zero means numeric equality only. No tolerance, practical equivalence margin, p-value, confidence interval or significance claim was prespecified or introduced.",
        "Metrics and batteries are not independent training repeats. MSE and RMSE share squared-error information; battery-macro RMSE averages per-battery square roots, so its ordering need not duplicate battery-macro MSE.",
        "Method effects use PINN minus baseline; transfer effects use fine-tuned minus source-only error. Negative values favor PINN or fine-tuning. Pairing follows same-index repeats and their planned seeds; source and target seeds can differ by dataset.",
        "These selected fixed data/split/hyperparameter experiments cannot establish all2120 environment-independence, generalization to unseen datasets, expected population win probability or historical backend settings.",
        "This report consumes the independently array-audited comparison; it does not retrain, load checkpoints or repeat the raw-array audit."]
    lines = ["# SYNTHETIC FIXTURE — NOT EXPERIMENT RESULTS" if synthetic else "# 入选扩展：种子方向与十次均值", "",
        "以下按每个配置、损失和指标分别描述。负效应表示PINN优于基线，或微调优于source-only；计数是已观察的10次训练，不是概率估计。", "",
        "repeat1参与筛查后选组，也包含在十次统计中；新增2–10次另列。所有完整精度值与逐种子配对保存在CSV中。", "",
        "| 配置/比较 | 指标 | 旧栈<0/=0/>0 | 现代<0/=0/>0 | 十次均值(旧/现代) | 均值方向 | 配对方向变化 | 新2–10均值(旧/现代) |",
        "|---|---|---|---|---|---|---|---|"]
    for row in summaries:
        count = lambda stack: "/".join(str(row[stack]["direction_counts"][key]) for key in ["lower", "exact_zero", "higher"])
        new = row["new_repeat2_to10"]
        lines.append("| " + " | ".join([label(row["logical_group"], row["evaluation"]).replace("|", " / "), row["metric"], count("legacy"), count("modern"),
            f"{row['legacy']['mean']:.8g} / {row['modern']['mean']:.8g}", row["legacy"]["direction_of_mean"] + " / " + row["modern"]["direction_of_mean"],
            str(row["paired_direction_changes"]) + "/10", f"{new['legacy']['mean']:.8g} / {new['modern']['mean']:.8g}"]) + " |")
    lines += ["", "原始效应单位为SOH小数、MAPE比值或SOH²；图中MAE/RMSE转SOH百分点、MAPE转百分数，MSE保留SOH²。CSV区分rank_of_ten_repeat_mean_error与mean_of_seedwise_rank，均不合并损失通道。", ""]
    lines += ["- " + x for x in limitations]
    (output / "种子方向与十次均值说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    require(all(digest(path) == checksum for path, checksum in hashes.items()), "Input changed while reporting")
    result = {"format": "pinn4soh-selected-extension-direction-stability-v1", "synthetic_fixture": synthetic,
        "complete_ten_repeat_analysis": True, "effect_metric_groups": len(summaries), "effect_configurations": len({(x["logical_group"], x["evaluation"]) for x in summaries}),
        "seed_rows": len(pairs), "summaries": summaries, "method_mean_ranks": mean_ranks, "figures": figures,
        "input_sha256": hashes, "script_sha256": digest(__file__), "training_performed": False,
        "post_screen_selected": True, "no_p_values_or_equivalence_claims": True, "limitations": limitations,
        "artifacts": {path.name: digest(path) for path in output.iterdir() if path.is_file()}}
    write_json(output / "direction_stability_summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["comparison", "extension", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    result = report(**vars(parser.parse_args()))
    print(json.dumps({key: value for key, value in result.items() if key not in {"summaries", "method_mean_ranks", "artifacts", "input_sha256", "limitations"}}, indent=2))
