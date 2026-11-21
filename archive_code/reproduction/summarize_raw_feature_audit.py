"""Summarize fixed raw-feature candidates with explicit finite/mapping coverage."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from summarize_raw_feature_residuals import compare


FORMULAS = {
    "mean": "mean", "std": "std_ddof0", "kurtosis": "kurtosis_scipy_biased",
    "skewness": "skewness_scipy_biased", "entropy": "entropy_minmax_probability",
    "slope": "slope_endpoint_per_second", "Q": "charge_capacity_difference_Ah",
    "charge time": "duration_seconds",
}
SIGNATURE_FEATURES = {"current mean", "current std", "current kurtosis", "current skewness"}


def canonical_formula(feature):
    for ending, formula in FORMULAS.items():
        if feature.endswith(ending):
            return formula
    return "interpolated_capacity_Ah" if feature == "capacity" else None


def main(options):
    root, repo = options.root.resolve(), options.repo.resolve()
    dataset_expected = {"XJTU": 55, "TJU": 130, "HUST": 77}
    alignments, every_candidate, canonical, unfinished = [], [], [], []
    for manifest_path in sorted((root / "raw_extraction").rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["status"] != "complete":
            unfinished.append(dict(dataset=manifest["dataset"], processed_csv=manifest["processed_csv"], status=manifest["status"]))
            continue
        frame = pd.read_csv(repo / "data" / manifest["processed_csv"])
        alignment, rows = compare(manifest, manifest_path.parent, repo)
        alignment["scope"] = "primary"
        alignments.append(alignment)
        for row in rows:
            row["scope"] = "primary"
        every_candidate.extend(rows)
        selected_rows = rows
        if manifest["dataset"] in {"XJTU", "HUST"}:
            boundary_dir = root / "boundary_extraction" / manifest["dataset"] / manifest_path.parent.name
            boundary_manifest = boundary_dir / "manifest.json"
            if not boundary_manifest.exists() or json.loads(boundary_manifest.read_text(encoding="utf-8"))["status"] != "complete":
                unfinished.append(dict(dataset=manifest["dataset"], processed_csv=manifest["processed_csv"], status="boundary_pending"))
                continue
            supplemental_alignment, selected_rows = compare(manifest, boundary_dir, repo)
            supplemental_alignment["scope"] = "fixed_boundary_supplement"
            alignments.append(supplemental_alignment)
            for row in selected_rows:
                row["scope"] = "fixed_boundary_supplement"
            every_candidate.extend(selected_rows)
        for row in selected_rows:
            feature = row["feature"]
            if feature == "capacity":
                continue
            is_voltage = feature.startswith(("voltage ", "CC "))
            window = ("note2_fixed_cc" if manifest["dataset"] == "TJU" else "fixed_5mV_boundary_allcharge") if is_voltage else "note2_current_stage"
            if row["window"] != window or row["formula_candidate"] != canonical_formula(feature):
                continue
            finite_targets = int(np.isfinite(frame[feature].to_numpy()).sum())
            canonical.append(dict(**row, published_rows=len(frame), published_finite_rows=finite_targets,
                                  published_nonfinite_rows=len(frame) - finite_targets,
                                  mapped_rows_with_missing_candidate=max(0, finite_targets - row["finite_pairs"]),
                                  alignment_signature_feature=feature in SIGNATURE_FEATURES,
                                  source_candidate="Paper/public-reader-derived fixed convention; exploratory 4.195V for XJTU stated separately"))
    pd.DataFrame(alignments).to_csv(root / "raw_cycle_alignment_summary.csv", index=False)
    pd.DataFrame(every_candidate).to_csv(root / "raw_candidate_residuals_all_cells.csv", index=False)
    frame = pd.DataFrame(canonical)
    if not frame.empty:
        frame.to_csv(root / "raw_fixed_convention_residuals_per_cell.csv", index=False)
        summary = frame.groupby(["dataset", "feature", "window", "formula_candidate", "alignment_signature_feature"], sort=False).agg(
            cells=("processed_csv", "nunique"), published_rows=("published_rows", "sum"),
            published_finite_rows=("published_finite_rows", "sum"), published_nonfinite_rows=("published_nonfinite_rows", "sum"),
            mapped_rows=("mapped_rows", "sum"), finite_pairs=("finite_pairs", "sum"),
            matches_at_1e_8=("matches_at_1e_8", "sum"), absolute_error_sum=("absolute_error_sum", "sum"),
            maximum_absolute_difference=("maximum_absolute_difference", "max")).reset_index()
        summary["finite_pair_match_rate"] = summary["matches_at_1e_8"] / summary["finite_pairs"].replace(0, np.nan)
        summary["finite_target_coverage"] = summary["finite_pairs"] / summary["published_finite_rows"].replace(0, np.nan)
        summary["mae"] = summary["absolute_error_sum"] / summary["finite_pairs"].replace(0, np.nan)
        summary.to_csv(root / "raw_fixed_convention_feature_summary.csv", index=False)
    state = dict(updated_at_unix=time.time(), expected_cells=dataset_expected,
                 completed_primary={dataset: sum(item["dataset"] == dataset and item["scope"] == "primary" for item in alignments) for dataset in dataset_expected},
                 completed_fixed_convention={dataset: len(set(row["processed_csv"] for row in canonical if row["dataset"] == dataset)) for dataset in dataset_expected},
                 unfinished=unfinished,
                 all_262_primary_complete=sum(item["scope"] == "primary" for item in alignments) == 262,
                 all_262_fixed_conventions_summarized=len(set(row["processed_csv"] for row in canonical)) == 262,
                 canonical_slopes="Paper endpoint slopes in physical seconds; other frozen candidates retained in raw_candidate_residuals_all_cells.csv",
                 alignment_caveat="The four current signature statistics used for row alignment are not independent validation of alignment",
                 source_scripts_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                        for name in ["summarize_raw_feature_audit.py", "summarize_raw_feature_residuals.py"]})
    (root / "RAW_FEATURE_SUMMARY_STATUS.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in state.items() if key != "unfinished"}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/full_reproduction/feature_audit"))
    parser.add_argument("--repo", type=Path, default=Path("PINN4SOH"))
    main(parser.parse_args())
