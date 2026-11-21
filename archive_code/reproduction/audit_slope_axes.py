"""Test declared slope axes/units without fitting constants to released slopes."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("outputs/full_reproduction/feature_audit")
REPO = Path("PINN4SOH")
ROWS = []

for manifest_path in sorted((ROOT / "raw_extraction").rglob("manifest.json")):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        continue
    source_dir = manifest_path.parent
    if manifest["dataset"] in {"XJTU", "HUST"}:
        source_dir = ROOT / "boundary_extraction" / manifest["dataset"] / source_dir.name
        if not (source_dir / "manifest.json").exists():
            continue
        if json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))["status"] != "complete":
            continue
    alignment_path = source_dir / "cycle_alignment.csv"
    if not alignment_path.exists():
        continue
    align = pd.read_csv(alignment_path).dropna(subset=["raw_cycle"])
    if align.empty:
        continue
    processed = pd.read_csv(REPO / "data" / manifest["processed_csv"])
    raw = pd.read_csv(source_dir / "candidate_features.csv")
    for curve in ("voltage", "current"):
        window = ("note2_fixed_cc" if manifest["dataset"] == "TJU" else "fixed_5mV_boundary_allcharge") if curve == "voltage" else "note2_current_stage"
        selected = raw[(raw["curve"] == curve) & (raw["window"] == window)].set_index("raw_cycle")
        selected = selected.reindex(align["raw_cycle"].astype(int).tolist())
        target = processed.iloc[align["processed_row"].astype(int).tolist()][curve + " slope"].to_numpy()
        amplitude = .2 if curve == "voltage" else -.4
        fixed_span = (.195 if curve == "voltage" and manifest["dataset"] != "TJU" else amplitude)
        candidates = {
            "observed_endpoint_per_second": selected["slope_endpoint_per_second"].to_numpy(),
            "observed_endpoint_per_minute": selected["slope_endpoint_per_second"].to_numpy() * 60,
            "observed_endpoint_per_sample_index": (selected["slope_endpoint_per_second"] * selected["duration_seconds"] / (selected["n"] - 1)).to_numpy(),
            "paper_nominal_boundary_displacement_per_second": (amplitude / selected["duration_seconds"]).to_numpy(),
            "fixed_window_boundary_displacement_per_second": (fixed_span / selected["duration_seconds"]).to_numpy(),
            "observed_endpoint_per_normalized_time": (selected["slope_endpoint_per_second"] * selected["duration_seconds"]).to_numpy(),
        }
        for name, values in candidates.items():
            error = np.abs(values - target)
            finite = np.isfinite(error)
            ROWS.append(dict(dataset=manifest["dataset"], processed_csv=manifest["processed_csv"],
                             feature=curve + " slope", window=window, candidate=name,
                             finite_pairs=int(finite.sum()), matches_at_1e_12=int((error <= 1e-12).sum()),
                             matches_at_1e_8=int((error <= 1e-8).sum()), absolute_error_sum=float(error[finite].sum()),
                             maximum_absolute_difference=float(error[finite].max()) if finite.any() else None))

frame = pd.DataFrame(ROWS)
frame.to_csv(ROOT / "slope_axis_candidates_per_cell.csv", index=False)
summary = frame.groupby(["dataset", "feature", "window", "candidate"], sort=False).agg(
    cells=("processed_csv", "nunique"), finite_pairs=("finite_pairs", "sum"),
    matches_at_1e_12=("matches_at_1e_12", "sum"), matches_at_1e_8=("matches_at_1e_8", "sum"),
    absolute_error_sum=("absolute_error_sum", "sum"), maximum_absolute_difference=("maximum_absolute_difference", "max")).reset_index()
summary["mae"] = summary["absolute_error_sum"] / summary["finite_pairs"].replace(0, np.nan)
summary.to_csv(ROOT / "slope_axis_candidates_summary.csv", index=False)
print(summary.to_string(index=False))
