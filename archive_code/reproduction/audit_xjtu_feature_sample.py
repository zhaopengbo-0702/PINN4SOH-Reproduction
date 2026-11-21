"""Fetch one bounded official XJTU MAT member and compare explicit feature conventions."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from scipy import stats

from audit_raw_data_inventory import RemoteZip, SESSION


MEMBER = "Battery Dataset/Batch-1/2C_battery-1.mat"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def sample_download(root):
    records = json.loads((root / "official_raw_inventory.json").read_text(encoding="utf-8"))
    record = next(item for item in records if item["dataset"] == "XJTU")
    selected = next(item for item in record["archive_members"] if item["name"] == MEMBER)
    if selected["compressed_bytes"] > 64 * 1024 * 1024 or selected["uncompressed_bytes"] > 64 * 1024 * 1024:
        raise ValueError("Selected member exceeds the 64 MiB sample cap")
    target = root / "raw_sample" / "2C_battery-1.mat"
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        remote = RemoteZip(record["download_url"], record["compressed_bytes"], limit=64 * 1024 * 1024)
        with zipfile.ZipFile(remote) as archive:
            content = archive.read(MEMBER)
        if len(content) != selected["uncompressed_bytes"]:
            raise ValueError("Sample size mismatch")
        target.write_bytes(content)
        downloaded = remote.downloaded
    else:
        downloaded = 0
    evidence = dict(source_url=record["download_url"], member=MEMBER, member_crc32=selected["crc32"],
                    compressed_bytes=selected["compressed_bytes"], uncompressed_bytes=target.stat().st_size,
                    sha256=hashlib.sha256(target.read_bytes()).hexdigest(), downloaded_this_run_bytes=downloaded)
    write_json(root / "raw_sample_provenance.json", evidence)
    return target


def curve_candidates(curve, time, charge):
    if len(curve) < 4 or len(curve) != len(time) or time[-1] == time[0]:
        return None
    duration = float(time[-1] - time[0]) * 60
    centered = curve - curve.mean()
    sigma = np.std(curve, ddof=1)
    minimum_maximum = (curve - curve.min()) / (curve.max() - curve.min())
    positive = minimum_maximum[minimum_maximum > 0]
    delta_time = np.diff(time) * 60
    differences = np.diff(curve)[delta_time > 0] / delta_time[delta_time > 0]
    return dict(n=len(curve), mean=float(curve.mean()), std_ddof1=float(sigma), std_ddof0=float(np.std(curve)),
                kurtosis_paper=float(np.sum(centered ** 4) / ((len(curve) - 1) * sigma ** 4)),
                kurtosis_scipy_biased=float(stats.kurtosis(curve, bias=True)),
                kurtosis_unbiased=float(stats.kurtosis(curve, bias=False)),
                skewness_paper=float(np.sum(centered ** 3) / ((len(curve) - 1) * sigma ** 3)),
                skewness_scipy_biased=float(stats.skew(curve, bias=True)),
                skewness_unbiased=float(stats.skew(curve, bias=False)),
                entropy_probability=float(stats.entropy(curve)),
                entropy_minmax_probability=float(stats.entropy(minimum_maximum)),
                entropy_minmax_unscaled=float(-np.sum(positive * np.log(positive))),
                duration_seconds=duration, charge_capacity_difference_Ah=float(charge[-1] - charge[0]),
                slope_endpoint_per_second=float((curve[-1] - curve[0]) / duration),
                slope_fit_per_second=float(stats.linregress(time * 60, curve).slope),
                slope_mean_local_per_second=float(differences.mean()),
                slope_median_local_per_second=float(np.median(differences)),
                slope_first_local_per_second=float(differences[0]),
                slope_last_local_per_second=float(differences[-1]))


def audit(root, repo, target):
    spec = importlib.util.spec_from_file_location("audited_official_xjtu_reader", root / "preprocessing_source" / "XJTUBatteryClass.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    battery = module.Battery(target.as_posix())
    published = pd.read_csv(repo / "data" / "XJTU data" / "2C_battery-1.csv")
    rows, voltage_alternatives = [], []
    for cycle in range(1, int(battery.cycle_life) + 1):
        voltage = battery.get_partial_value(cycle, "voltage_V")
        current = battery.get_partial_value(cycle, "current_A")
        time = battery.get_partial_value(cycle, "relative_time_min")
        charge = battery.get_partial_value(cycle, "capacity_Ah")
        cc = (voltage >= 4.0) & (voltage <= 4.2) & (current >= 3.9)
        cv = (current >= .1) & (current <= .5)
        row = dict(raw_cycle=cycle, raw_capacity_Ah=float(battery.get_capacity()[cycle - 1]))
        if len(voltage) and np.any(current >= 3.9):
            charge_max, cc_max = voltage.max(), voltage[current >= 3.9].max()
            window_specs = [
                ("note2_fixed_cc", 4.0, 4.2, True),
                ("note2_fixed_allcharge", 4.0, 4.2, False),
                ("author_reader_cc_4_199", 4.0, 4.199, True),
                ("fixed_cc_4_195_hypothesis", 4.0, 4.195, True),
                ("fixed_cc_4_19", 4.0, 4.19, True),
                ("actual_charge_max_cc", charge_max - .2, charge_max, True),
                ("actual_cc_max_cc", cc_max - .2, cc_max, True),
            ]
            for label, lower, upper, cc_only in window_specs:
                mask = (voltage >= lower) & (voltage <= upper)
                if cc_only:
                    mask &= current >= 3.9
                values = curve_candidates(voltage[mask], time[mask], charge[mask])
                if values:
                    voltage_alternatives.append(dict(raw_cycle=cycle, window=label, lower_V=float(lower), upper_V=float(upper), **values))
        for label, mask, curve in [("voltage", cc, voltage), ("current", cv, current)]:
            values = curve_candidates(curve[mask], time[mask], charge[mask])
            if values is not None:
                row.update({label + "_" + key: value for key, value in values.items()})
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "raw_sample_candidate_features.csv", index=False)
    alternatives = pd.DataFrame(voltage_alternatives)
    alternatives.to_csv(root / "raw_sample_voltage_window_candidates.csv", index=False)
    capacity_mapping = []
    for shift in range(len(frame) - len(published) + 1):
        raw_values = frame["raw_capacity_Ah"].iloc[shift:shift + len(published)].to_numpy()
        differences = np.abs(raw_values - published["capacity"].to_numpy())
        capacity_mapping.append(dict(first_raw_cycle=shift + 1, rows=len(published),
                                     maximum_absolute_difference_Ah=float(differences.max()),
                                     matching_rows_at_1e_12=int((differences <= 1e-12).sum())))
    pd.DataFrame(capacity_mapping).to_csv(root / "raw_sample_capacity_alignment.csv", index=False)
    exact_mappings = [item for item in capacity_mapping if item["maximum_absolute_difference_Ah"] <= 1e-12]
    write_json(root / "raw_sample_exact_capacity_mapping.json", dict(exact_contiguous_mappings=exact_mappings))
    raw_signature_columns = ["raw_capacity_Ah", "current_mean", "current_std_ddof0", "current_kurtosis_scipy_biased", "current_skewness_scipy_biased"]
    target_signature_columns = ["capacity", "current mean", "current std", "current kurtosis", "current skewness"]
    mapping_rows, mapped_cycles = [], []
    for index, target_row in published[target_signature_columns].iterrows():
        differences = np.abs(frame[raw_signature_columns].to_numpy() - target_row.to_numpy())
        matches = frame.loc[np.all(differences <= 1e-10, axis=1), "raw_cycle"].astype(int).tolist()
        mapping_rows.append(dict(published_row=index, candidate_count=len(matches), candidate_raw_cycles="|".join(map(str, matches))))
        mapped_cycles.append(matches[0] if len(matches) == 1 else -1)
    pd.DataFrame(mapping_rows).to_csv(root / "raw_sample_cycle_alignment.csv", index=False)
    unambiguous = all(value > 0 for value in mapped_cycles) and all(np.diff(mapped_cycles) > 0)
    write_json(root / "raw_sample_exact_signature_mapping.json", dict(
        method="Exact match within 1e-10 of raw capacity plus current mean, population std, biased Fisher kurtosis and biased skewness",
        unambiguous_ordered_mapping=bool(unambiguous), mapped_rows=sum(value > 0 for value in mapped_cycles),
        total_rows=len(published), raw_cycles=mapped_cycles,
        matching_fingerprint_is_not_independent_validation_of_these_four_statistics=True))
    if unambiguous:
        comparisons = []
        quantity_map = {
            "mean": ["mean"], "std": ["std_ddof0", "std_ddof1"],
            "kurtosis": ["kurtosis_paper", "kurtosis_scipy_biased", "kurtosis_unbiased"],
            "skewness": ["skewness_paper", "skewness_scipy_biased", "skewness_unbiased"],
            "entropy": ["entropy_probability", "entropy_minmax_probability", "entropy_minmax_unscaled"],
            "slope": ["slope_endpoint_per_second", "slope_fit_per_second", "slope_mean_local_per_second",
                      "slope_median_local_per_second", "slope_first_local_per_second", "slope_last_local_per_second"],
        }
        for prefix in ["voltage", "current"]:
            windows = alternatives.groupby("window", sort=False) if prefix == "voltage" else [("note2_current", frame.rename(columns=lambda name: name.removeprefix("current_")))]
            for window, candidates in windows:
                aligned = candidates.set_index("raw_cycle").reindex(mapped_cycles)
                mapping = dict(quantity_map)
                mapping["CC Q" if prefix == "voltage" else "CV Q"] = ["charge_capacity_difference_Ah"]
                mapping["CC charge time" if prefix == "voltage" else "CV charge time"] = ["duration_seconds"]
                for quantity, names in mapping.items():
                    target_name = quantity if quantity.startswith(("CC ", "CV ")) else prefix + " " + quantity
                    for name in names:
                        differences = np.abs(aligned[name].to_numpy() - published[target_name].to_numpy())
                        valid = np.isfinite(differences)
                        comparisons.append(dict(feature=target_name, window=window, formula_candidate=name,
                                                n_aligned=len(published), finite_pairs=int(valid.sum()),
                                                matches_at_1e_10=int((differences <= 1e-10).sum()),
                                                mae=float(differences[valid].mean()), maximum_absolute_difference=float(differences[valid].max())))
        pd.DataFrame(comparisons).to_csv(root / "raw_sample_aligned_formula_residuals.csv", index=False)
    comparisons = []
    for prefix in ["voltage", "current"]:
        for quantity, suffixes in {
            "mean": ["mean"], "std": ["std_ddof0", "std_ddof1"],
            "kurtosis": ["kurtosis_paper", "kurtosis_scipy_biased", "kurtosis_unbiased"],
            "skewness": ["skewness_paper", "skewness_scipy_biased", "skewness_unbiased"],
            "entropy": ["entropy_probability", "entropy_minmax_probability", "entropy_minmax_unscaled"],
            "slope": ["slope_endpoint_per_second", "slope_fit_per_second"],
        }.items():
            for suffix in suffixes:
                candidate = frame[prefix + "_" + suffix]
                finite = candidate.dropna()
                closest = (finite - published.iloc[0][prefix + " " + quantity]).abs().idxmin()
                comparisons.append(dict(feature=prefix + " " + quantity, candidate=suffix,
                                        nearest_raw_cycle=int(frame.loc[closest, "raw_cycle"]),
                                        raw_value=float(frame.loc[closest, prefix + "_" + suffix]),
                                        published_first_value=float(published.iloc[0][prefix + " " + quantity]),
                                        absolute_difference=float(abs(frame.loc[closest, prefix + "_" + suffix] - published.iloc[0][prefix + " " + quantity]))))
    pd.DataFrame(comparisons).to_csv(root / "first_row_formula_diagnostics.csv", index=False)
    summary = dict(raw_cycles=int(battery.cycle_life), published_rows=len(published),
                   valid_voltage_windows=int(frame["voltage_mean"].notna().sum()),
                   valid_current_windows=int(frame["current_mean"].notna().sum()),
                   sample_scope="One official XJTU Batch-1 cell only; not evidence for all 387 cells",
                   voltage_window="4.0 <= V <= 4.2 and I >= 3.9 A during charge stage",
                   current_window="0.1 <= I <= 0.5 A during charge stage",
                   first_raw_cycle_candidates=rows[0],
                   formula_search_limit="Diagnostic candidates are compared separately; nearest cycles do not establish a shared row mapping")
    write_json(root / "raw_sample_audit.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/full_reproduction/feature_audit"))
    parser.add_argument("--repo", type=Path, default=Path("PINN4SOH"))
    parser.add_argument("--direct", action="store_true")
    options = parser.parse_args()
    if options.direct:
        SESSION.trust_env = False
    root = options.output.resolve()
    audit(root, options.repo.resolve(), sample_download(root))
