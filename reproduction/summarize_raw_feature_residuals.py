"""Report every fixed candidate residual after unambiguous raw-cycle alignment."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


QUANTITIES = {
    "mean": ["mean"], "std": ["std_ddof0", "std_ddof1"],
    "kurtosis": ["kurtosis_paper", "kurtosis_scipy_biased", "kurtosis_unbiased"],
    "skewness": ["skewness_paper", "skewness_scipy_biased", "skewness_unbiased"],
    "entropy": ["entropy_probability", "entropy_minmax_probability", "entropy_minmax_unscaled"],
    "slope": ["slope_endpoint_per_second", "slope_fit_per_second", "slope_mean_local_per_second",
              "slope_median_local_per_second", "slope_first_local_per_second", "slope_last_local_per_second"],
}


def compare(manifest, task_dir, repo):
    processed = pd.read_csv(repo / "data" / manifest["processed_csv"])
    raw = pd.read_csv(task_dir / "candidate_features.csv")
    current = raw[raw["curve"] == "current"].drop_duplicates("raw_cycle")
    signature = ["mean", "std_ddof0", "kurtosis_scipy_biased", "skewness_scipy_biased"]
    target = ["current mean", "current std", "current kurtosis", "current skewness"]
    source_values = current[signature].to_numpy()
    mapped, mapping = [], []
    for index, row in processed.iterrows():
        difference = np.abs(source_values - row[target].to_numpy(dtype=float))
        hits = current.loc[np.all(difference <= 1e-9, axis=1), "raw_cycle"].astype(int).tolist()
        mapped.append(hits[0] if len(hits) == 1 else None)
        mapping.append(dict(processed_row=index, raw_cycle=hits[0] if len(hits) == 1 else None,
                            candidate_count=len(hits), candidates="|".join(map(str, hits))))
    pd.DataFrame(mapping).to_csv(task_dir / "cycle_alignment.csv", index=False)
    unique = [value for value in mapped if value is not None]
    ordered = all(np.diff(unique) > 0)
    state = dict(dataset=manifest["dataset"], processed_csv=manifest["processed_csv"], published_rows=len(processed),
                 raw_cycles=int(raw["raw_cycle"].nunique()), mapped_rows=len(unique), ordered_unique=bool(ordered),
                 signature="current mean + population std + biased Fisher kurtosis + biased skewness, absolute tolerance 1e-9",
                 signature_statistics_not_independent_validation=True)
    if not ordered:
        return state, []
    indices = [index for index, value in enumerate(mapped) if value is not None]
    cycles = [mapped[index] for index in indices]
    residuals = []
    for (curve, window), group in raw.groupby(["curve", "window"], sort=False):
        aligned = group.set_index("raw_cycle").reindex(cycles)
        mapping = dict(QUANTITIES)
        mapping["CC Q" if curve == "voltage" else "CV Q"] = ["charge_capacity_difference_Ah", "charge_trapezoid_Ah"]
        mapping["CC charge time" if curve == "voltage" else "CV charge time"] = ["duration_seconds"]
        mapping["capacity"] = ["capacity_Ah", "interpolated_capacity_Ah"]
        for quantity, names in mapping.items():
            feature = quantity if quantity in {"capacity", "CC Q", "CV Q", "CC charge time", "CV charge time"} else curve + " " + quantity
            for name in names:
                difference = np.abs(aligned[name].to_numpy() - processed.iloc[indices][feature].to_numpy())
                valid = np.isfinite(difference)
                residuals.append(dict(dataset=manifest["dataset"], processed_csv=manifest["processed_csv"], feature=feature,
                                      window=window, formula_candidate=name, mapped_rows=len(indices),
                                      finite_pairs=int(valid.sum()), matches_at_1e_8=int((difference <= 1e-8).sum()),
                                      absolute_error_sum=float(difference[valid].sum()),
                                      mae=float(difference[valid].mean()) if valid.any() else None,
                                      maximum_absolute_difference=float(difference[valid].max()) if valid.any() else None))
    pd.DataFrame(residuals).to_csv(task_dir / "aligned_formula_residuals.csv", index=False)
    return state, residuals


def main(options):
    root = options.output.resolve()
    alignment, rows = [], []
    for path in sorted((root / "raw_extraction").rglob("manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["status"] != "complete":
            alignment.append(dict(dataset=manifest["dataset"], processed_csv=manifest["processed_csv"], status=manifest["status"], error=manifest.get("error")))
            continue
        item, residuals = compare(manifest, path.parent, options.repo.resolve())
        alignment.append(item)
        rows.extend(residuals)
    pd.DataFrame(alignment).to_csv(root / "raw_cycle_alignment_summary.csv", index=False)
    if rows:
        frame = pd.DataFrame(rows)
        frame.to_csv(root / "raw_candidate_residuals_all_cells.csv", index=False)
        summary = frame.groupby(["dataset", "feature", "window", "formula_candidate"], sort=False, dropna=False).agg(
            cells=("processed_csv", "nunique"), mapped_rows=("mapped_rows", "sum"), finite_pairs=("finite_pairs", "sum"),
            matches_at_1e_8=("matches_at_1e_8", "sum"), absolute_error_sum=("absolute_error_sum", "sum"),
            maximum_absolute_difference=("maximum_absolute_difference", "max")).reset_index()
        summary["mae"] = summary["absolute_error_sum"] / summary["finite_pairs"].replace(0, np.nan)
        summary.to_csv(root / "raw_candidate_residuals_summary.csv", index=False)
    print(json.dumps(dict(extraction_manifests=len(alignment), completed_aligned_cells=len(set(row["processed_csv"] for row in rows)),
                          candidate_cell_feature_comparisons=len(rows)), indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("PINN4SOH"))
    parser.add_argument("--output", type=Path, default=Path("outputs/full_reproduction/feature_audit"))
    main(parser.parse_args())
