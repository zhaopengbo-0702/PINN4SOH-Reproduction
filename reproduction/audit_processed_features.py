"""Audit all released feature CSVs and redraw descriptive Figures 2/3 separately."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


FEATURES = ["voltage mean", "voltage std", "voltage kurtosis", "voltage skewness", "CC Q", "CC charge time",
            "voltage slope", "voltage entropy", "current mean", "current std", "current kurtosis",
            "current skewness", "CV Q", "CV charge time", "current slope", "current entropy"]
XJTU_BATCHES = ["2C", "3C", "R2.5", "R3", "RW", "satellite"]
NOMINAL = {"XJTU": 2.0, "HUST": 1.1, "MIT": 1.1, "TJU1": 3.5, "TJU2": 3.5, "TJU3": 2.5}
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_for(path, repo):
    relative = path.relative_to(repo / "data")
    dataset = relative.parts[0].split()[0]
    if dataset == "XJTU":
        batch = next(name for name in XJTU_BATCHES if name in path.name)
        return dataset, "XJTU" + str(XJTU_BATCHES.index(batch) + 1), NOMINAL[dataset]
    if dataset == "TJU":
        number = relative.parts[1].split("_")[1]
        return dataset, "TJU" + number, NOMINAL["TJU" + number]
    return dataset, dataset, NOMINAL[dataset]


def clean_author(frame):
    working = frame.copy()
    working.insert(len(FEATURES), "cycle index", np.arange(len(frame)))
    finite = working.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    remove = np.zeros(len(finite), dtype=bool)
    for column in finite.columns:
        mean, sd = finite[column].mean(), finite[column].std()
        remove |= ((finite[column] < mean - 3 * sd) | (finite[column] > mean + 3 * sd)).to_numpy()
    return finite.loc[~remove].reset_index(drop=True), len(finite), int(remove.sum())


def normalized(frame):
    features = frame[FEATURES].replace([np.inf, -np.inf], np.nan)
    return 2 * (features - features.min()) / (features.max() - features.min()) - 1


def save_figure(fig, output, name, qa):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [(item, item.get_window_extent(renderer)) for item in fig.findobj(matplotlib.text.Text)
             if item.get_text().strip() and item.get_visible()]
    overlaps = [(first.get_text(), second.get_text()) for index, (first, box) in enumerate(texts)
                for second, other in texts[index + 1:] if box.overlaps(other)]
    outside = [item.get_text() for item, box in texts if not fig.bbox.contains(box.x0, box.y0)
               or not fig.bbox.contains(box.x1, box.y1)]
    qa[name] = dict(text_overlap_count=len(overlaps), text_overlap_examples=overlaps[:12], outside_text=outside)
    for suffix in ["png", "pdf"]:
        fig.savefig(output / (name + "." + suffix), dpi=300)
    plt.close(fig)


def run(repo, output, style):
    output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("feature_figure_style", style)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.apply_figure_style(sizes=(9, 8, 7), font="Arial")
    sources = sorted((repo / "data").rglob("*.csv"))
    hashes = {path.relative_to(repo).as_posix(): sha(path) for path in sources}
    count_rows, correlation_rows, frames, clean_frames = [], [], {}, {}
    for path in sources:
        frame = pd.read_csv(path)
        if list(frame.columns) != FEATURES + ["capacity"]:
            raise ValueError("Unexpected released feature schema: " + path.name)
        dataset, group, nominal = group_for(path, repo)
        if not all(pd.api.types.is_numeric_dtype(frame[column]) for column in frame):
            raise ValueError("Non-numeric released feature column")
        cleaned, finite, sigma_removed = clean_author(frame)
        record = dict(dataset=dataset, paper_group=group, battery=path.relative_to(repo / "data").as_posix(),
                      raw_rows=len(frame), finite_rows=finite, author_sigma_removed=sigma_removed,
                      author_cleaned_rows=len(cleaned), author_adjacent_pairs=max(0, len(cleaned) - 1),
                      nominal_capacity_Ah=nominal, negative_voltage_kurtosis=int((frame["voltage kurtosis"] < 0).sum()),
                      negative_current_kurtosis=int((frame["current kurtosis"] < 0).sum()),
                      constant_input_columns="|".join(column for column in FEATURES if cleaned[column].max() == cleaned[column].min()),
                      capacity_Ah_min=float(frame["capacity"].min()), capacity_Ah_max=float(frame["capacity"].max()),
                      sha256=hashes[path.relative_to(repo).as_posix()])
        count_rows.append(record)
        for scope, source_frame in [("raw", frame), ("author_cleaned", cleaned)]:
            soh = source_frame["capacity"] / nominal
            raw_features = source_frame[FEATURES].copy()
            scaled = normalized(source_frame)
            for aggregation, features in [(scope + "_pooled", raw_features), (scope + "_per_battery_minmax_pooled", scaled)]:
                data = features.copy()
                data["SOH"] = soh.to_numpy()
                data["battery"] = path.stem
                frames.setdefault((aggregation, group), []).append(data)
            for number, feature in enumerate(FEATURES, start=1):
                valid = source_frame[[feature, "capacity"]].replace([np.inf, -np.inf], np.nan).dropna()
                correlation_rows.append(dict(scope=scope, group=group, battery=record["battery"], feature_number=number,
                                             feature=feature, n=len(valid), pearson_r=float(valid[feature].corr(valid["capacity"]))))
        if group == "XJTU1":
            clean_frames[path.stem] = cleaned
    counts = pd.DataFrame(count_rows)
    counts.to_csv(output / "battery_inventory.csv", index=False)
    groups = counts.groupby(["dataset", "paper_group"], sort=False).agg(
        batteries=("battery", "size"), raw_rows=("raw_rows", "sum"), finite_rows=("finite_rows", "sum"),
        author_cleaned_rows=("author_cleaned_rows", "sum"), author_adjacent_pairs=("author_adjacent_pairs", "sum"))
    groups.to_csv(output / "group_counts.csv")
    per_battery = pd.DataFrame(correlation_rows)
    per_battery.to_csv(output / "pearson_per_battery.csv", index=False)
    order = ["XJTU" + str(index) for index in range(1, 7)] + ["TJU1", "TJU2", "TJU3", "MIT", "HUST"]
    matrices = {}
    for aggregation in sorted(set(key[0] for key in frames)):
        matrix = pd.DataFrame(index=order, columns=FEATURES, dtype=float)
        for group in order:
            group_frame = pd.concat(frames[(aggregation, group)], ignore_index=True)
            for feature in FEATURES:
                finite_pair = group_frame[[feature, "SOH"]].replace([np.inf, -np.inf], np.nan).dropna()
                matrix.loc[group, feature] = finite_pair[feature].corr(finite_pair["SOH"])
        matrix.to_csv(output / ("pearson_" + aggregation + ".csv"))
        matrices[aggregation] = matrix
    battery_mean = per_battery[per_battery["scope"] == "raw"].groupby(["group", "feature"], sort=False)["pearson_r"].mean().unstack().reindex(index=order, columns=FEATURES)
    battery_mean.to_csv(output / "pearson_raw_battery_mean.csv")
    qa = {}
    fig, ax = plt.subplots(figsize=(7.4, 3.9), layout="constrained")
    for index, batch in enumerate(XJTU_BATCHES):
        for path in sources:
            if "XJTU data" in path.parts and batch in path.name:
                capacity = pd.read_csv(path)["capacity"].to_numpy()
                ax.plot(np.arange(len(capacity) - 1), capacity[1:], color=COLORS[index], linewidth=.7, alpha=.85)
    ax.set(xlabel="Cycle Row After First Row", ylabel="Capacity (Ah)", title="Released XJTU Capacity Trajectories")
    ax.margins(.03)
    low, high = ax.get_ylim()
    left, right = ax.get_xlim()
    ax.set_xticks([value for value in ax.get_xticks() if left <= value <= right])
    ax.set_yticks([value for value in ax.get_yticks() if low <= value <= high])
    ax.legend(handles=[Line2D([0], [0], color=color, label=f"Batch {index + 1}") for index, color in enumerate(COLORS)], ncol=3, loc="upper right")
    save_figure(fig, output, "figure2_released_capacity", qa)
    fig, axes = plt.subplots(4, 4, figsize=(9.6, 9.0), sharex=True, sharey=True, layout="constrained")
    battery_colors = plt.cm.coolwarm(np.linspace(.05, .95, len(clean_frames)))
    for battery_index, (battery, frame) in enumerate(sorted(clean_frames.items())):
        scaled = normalized(frame)
        for index, feature in enumerate(FEATURES):
            ax = axes.flat[index]
            ax.scatter(frame["capacity"] / 2, scaled[feature], s=1.3, alpha=.6,
                       color=battery_colors[battery_index], rasterized=True)
            ax.set_title(f"{index + 1}. {feature}")
    for ax in axes[-1]:
        ax.set_xlabel("SOH")
    for ax in axes[:, 0]:
        ax.set_ylabel("Feature, Battery Min-Max")
    for ax in axes.flat:
        ax.set_xticks([.8, .9, 1.0])
        ax.set_yticks([-1, 0, 1])
        ax.set_ylim(-1.08, 1.08)
    fig.legend(handles=[Line2D([0], [0], linestyle="", marker="o", markersize=3, color=battery_colors[index], label="Battery " + name.rsplit("-", 1)[-1])
                        for index, name in enumerate(sorted(clean_frames))], loc="outside lower center", ncol=8)
    save_figure(fig, output, "figure3a_released_features", qa)
    for aggregation in ["raw_pooled", "raw_per_battery_minmax_pooled", "author_cleaned_pooled", "author_cleaned_per_battery_minmax_pooled"]:
        matrix = matrices[aggregation]
        fig, ax = plt.subplots(figsize=(10.5, 4.5), layout="constrained")
        image = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(16), [str(index) for index in range(1, 17)])
        ax.set_yticks(range(11), order)
        ax.set(xlabel="Feature Number", title="Pearson Correlation With SOH: " + aggregation.replace("_", " "))
        for row in range(11):
            for col in range(16):
                value = matrix.iloc[row, col]
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(value) > .6 else "black")
        fig.colorbar(image, ax=ax, label="Pearson r", ticks=[-1, -.5, 0, .5, 1])
        save_figure(fig, output, "figure3b_" + aggregation, qa)
    reference_path = Path(__file__).with_name("paper_fig3b_rounded.csv")
    printed = pd.read_csv(reference_path, index_col=0)
    printed.columns = FEATURES
    comparison = []
    for aggregation, matrix in matrices.items():
        differences = np.abs(matrix.to_numpy() - printed.to_numpy())
        comparison.append(dict(aggregation=aggregation, entries=176, rounded_two_decimal_matches=int((np.round(matrix.to_numpy(), 2) == printed.to_numpy()).sum()),
                               maximum_absolute_difference_from_printed=float(differences.max()),
                               entries_within_rounding_tolerance=int((differences <= .0050000001).sum())))
    pd.DataFrame(comparison).to_csv(output / "fig3b_paper_rounded_comparison.csv", index=False)
    write = dict(schema="pinn4soh-feature-csv-audit-v1", batteries=len(sources), paper_claim_batteries=387,
                 raw_rows=int(counts["raw_rows"].sum()), paper_claim_samples=310705,
                 raw_row_difference_from_paper=int(counts["raw_rows"].sum()) - 310705,
                 finite_rows=int(counts["finite_rows"].sum()), author_cleaned_rows=int(counts["author_cleaned_rows"].sum()),
                 author_adjacent_pairs=int(counts["author_adjacent_pairs"].sum()),
                 negative_voltage_kurtosis=int(counts["negative_voltage_kurtosis"].sum()),
                 negative_current_kurtosis=int(counts["negative_current_kurtosis"].sum()),
                 constant_columns=counts.loc[counts["constant_input_columns"] != "", ["battery", "constant_input_columns"]].to_dict("records"),
                 sources_unchanged=all(sha(repo / name) == checksum for name, checksum in hashes.items()),
                 pearson_aggregation="Cycles pooled within each paper group; alternative per-battery minmax and equal-battery coefficient mean exported separately",
                 figure3a_normalization="Author-cleaned features separately scaled to [-1,1] within each of 8 XJTU Batch-1 cells",
                 author_preprocessing="Insert cycle index; drop NaN/Inf; union 3-sigma row exclusion across all features, capacity and cycle index; scale each whole battery including held-out test battery",
                 interpretation="Descriptive repeated-cycle correlations; no independent-cycle inference or causal conclusion; no raw extraction claim",
                 versions=dict(numpy=np.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__),
                 printed_fig3b_reference=dict(source="Manually transcribed displayed two-decimal values from paper PDF p5 Fig3b", sha256=sha(reference_path),
                                             independent_transcription_review="Independent agent visually checked all 176 entries against the original PDF image without viewing computed coefficients; all matched"),
                 script_sha256=sha(Path(__file__).resolve()), figure_style_sha256=sha(style))
    (output / "processed_feature_audit.json").write_text(json.dumps(write, indent=2, allow_nan=False), encoding="utf-8")
    (output / "figure_layout_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps(write, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("PINN4SOH"))
    parser.add_argument("--output", type=Path, default=Path("outputs/full_reproduction/feature_audit/processed_figures"))
    parser.add_argument("--style", type=Path, default=Path("C:/Users/Administrator/.codex/skills/figure-style/kernel.py"))
    options = parser.parse_args()
    run(options.repo.resolve(), options.output.resolve(), options.style.resolve())
