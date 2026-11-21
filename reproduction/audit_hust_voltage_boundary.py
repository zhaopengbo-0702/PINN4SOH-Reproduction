"""Compare the public HUST reader's 3.595 V boundary without altering frozen runs."""

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from extract_raw_feature_candidates import RestrictedDataUnpickler
from audit_xjtu_feature_sample import curve_candidates


root = Path("outputs/full_reproduction/feature_audit").resolve()
raw = RestrictedDataUnpickler(io.BytesIO((root / "raw_sample" / "HUST_2-5.pkl").read_bytes())).load()["2-5"]
alignment = pd.read_csv(root / "HUST_ranged_member_feature_check" / "cycle_alignment.csv")
published = pd.read_csv("PINN4SOH/data/HUST data/2-5.csv")
rows = []
for record in alignment.itertuples(index=False):
    group = raw["data"][int(record.raw_cycle)]
    charge = group[group["Status"] == "Constant current-constant voltage charge"]
    voltage = charge["Voltage (V)"].to_numpy()
    current = charge["Current (mA)"].to_numpy() / 1000
    time = charge["Time (s)"].to_numpy()
    q = charge["Capacity (mAh)"].to_numpy() / 1000
    for upper in [3.59, 3.595, 3.599, 3.6]:
        mask = (voltage >= 3.4) & (voltage <= upper)
        values = curve_candidates(voltage[mask], time[mask] / 60, q[mask])
        if values:
            rows.append(dict(processed_row=record.processed_row, raw_cycle=record.raw_cycle, upper_V=upper, **values))
frame = pd.DataFrame(rows)
frame.to_csv(root / "HUST_ranged_member_feature_check" / "voltage_boundary_candidates.csv", index=False)
target_mapping = {"voltage mean": "mean", "voltage std": "std_ddof0", "voltage kurtosis": "kurtosis_scipy_biased",
                  "voltage skewness": "skewness_scipy_biased", "voltage entropy": "entropy_minmax_probability",
                  "CC Q": "charge_capacity_difference_Ah", "CC charge time": "duration_seconds",
                  "voltage slope": "slope_endpoint_per_second"}
summary = []
for upper, group in frame.groupby("upper_V"):
    group = group.set_index("processed_row").reindex(range(len(published)))
    for feature, candidate in target_mapping.items():
        difference = np.abs(group[candidate].to_numpy() - published[feature].to_numpy())
        valid = np.isfinite(difference)
        summary.append(dict(upper_V=upper, feature=feature, formula_candidate=candidate,
                            finite_pairs=int(valid.sum()), matches_at_1e_8=int((difference <= 1e-8).sum()),
                            mae=float(difference[valid].mean()), maximum_absolute_difference=float(difference[valid].max())))
pd.DataFrame(summary).to_csv(root / "HUST_ranged_member_feature_check" / "voltage_boundary_residuals.csv", index=False)
print(pd.DataFrame(summary).to_string(index=False), flush=True)
