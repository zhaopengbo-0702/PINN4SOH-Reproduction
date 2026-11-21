"""Export 16 transparent features without changing any released training input.

The default reuses the previously audited 14-feature conventions. Two endpoint
slopes are declared replacements: they implement Note 2's endpoint/time formula
in seconds, but do NOT claim to reconstruct the released slope columns.
"""

import argparse
import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import sys
import warnings
import zipfile

import numpy as np
import pandas as pd
import scipy
from scipy.integrate import trapezoid

from audit_xjtu_feature_sample import curve_candidates


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["voltage mean", "voltage std", "voltage kurtosis", "voltage skewness",
            "CC Q", "CC charge time", "voltage slope", "voltage entropy",
            "current mean", "current std", "current kurtosis", "current skewness",
            "CV Q", "CV charge time", "current slope", "current entropy"]
SLOPES = {"voltage slope", "current slope"}
PRESET = {
    "slope": "(last observed curve value - first observed curve value) / elapsed physical seconds",
    "slope_status": "prespecified replacement; released slope columns remain unreconstructed",
    "replacement_slope_selected_by_released_feature_error": False,
    "prior_XJTU_boundary_history": "4.195 V was an exploratory pilot diagnostic; it is fixed here, not a new prespecified paper threshold",
    "entropy": "min-max curve values, divide by their sum, then -sum(p*ln(p)); explicit assumption",
    "released_audit": {
        "std": "population ddof=0", "kurtosis": "biased Fisher excess", "skewness": "biased third standardized moment",
        "charge": "last minus first recorded charge capacity in Ah",
        "voltage_windows": {"XJTU": "[4.0,4.195] V, all charge points (prior exploratory diagnostic fixed across cells)",
                            "TJU": "[4.0,4.2] V with CC stage mask",
                            "HUST": "[3.4,3.595] V, all charge points (public-reader boundary)",
                            "MIT": "unchanged author MIT reader CC stage, strict 3.4<V<3.595"},
        "current_window": "0.1<=I<=0.5 and supplied CV stage mask; MIT uses unchanged strict author mask",
    },
    "paper_note2": {
        "std": "sample ddof=1", "kurtosis": "sum(centered^4)/((n-1)*sample_SD^4)",
        "skewness": "sum(centered^3)/((n-1)*sample_SD^3)",
        "charge": "trapezoid integral of current over physical seconds / 3600",
        "voltage_window": "XJTU/TJU [4.0,4.2], HUST/MIT [3.4,3.6], supplied CC mask",
        "entropy_caveat": "paper does not uniquely define normalization; same declared min-max probability assumption",
    },
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def extract_window(curve, time_seconds, charge_Ah, current, prefix, profile="released_audit"):
    """Extract eight features; preserve undefined results as None, with reasons.

    Inputs retain acquisition order. No smoothing, interpolation, sorting,
    imputation, fitted multiplier or normalization to released columns occurs.
    Strictly increasing time is required; malformed windows are not repaired.
    """
    if profile not in ("released_audit", "paper_note2") or prefix not in ("voltage", "current"):
        raise ValueError("Unknown profile or curve")
    stage = "CC" if prefix == "voltage" else "CV"
    names = [prefix + " " + name for name in ("mean", "std", "kurtosis", "skewness")]
    names += [stage + " Q", stage + " charge time", prefix + " slope", prefix + " entropy"]
    arrays = [np.asarray(a, dtype=np.float64) for a in (curve, time_seconds, charge_Ah, current)]
    result = dict.fromkeys(names)
    report = {"n": len(arrays[0]), "issues": [], "undefined_features": []}
    invalid = None
    if any(a.ndim != 1 for a in arrays) or len({len(a) for a in arrays}) != 1:
        invalid = "shape_mismatch"
    elif len(arrays[0]) < 4:
        invalid = "fewer_than_four_points"
    elif not all(np.isfinite(a).all() for a in arrays):
        invalid = "nonfinite_input"
    elif not np.all(np.diff(arrays[1]) > 0):
        invalid = "time_not_strictly_increasing"
    if invalid:
        report.update(status="invalid_window", issues=[invalid], undefined_features=names)
        return result, report
    x, seconds, charge, amps = arrays
    # Reuse the frozen helper for the 14 audited conventions. Time is converted
    # here only because that helper's documented API takes minutes.
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        values = curve_candidates(x, seconds / 60.0, charge)
    report["issues"] = sorted({str(item.message) for item in observed})
    keys = ["mean", "std_ddof0", "kurtosis_scipy_biased", "skewness_scipy_biased",
            "charge_capacity_difference_Ah", "duration_seconds", "slope_endpoint_per_second", "entropy_minmax_probability"]
    if profile == "paper_note2":
        keys[1:4] = ["std_ddof1", "kurtosis_paper", "skewness_paper"]
        values["charge_capacity_difference_Ah"] = float(trapezoid(amps, seconds) / 3600)
    for name, key in zip(names, keys):
        value = float(values[key])
        result[name] = value if np.isfinite(value) else None
    report["undefined_features"] = [name for name in names if result[name] is None]
    report["status"] = "computed_with_undefined_features" if report["undefined_features"] else "computed"
    report.update(time_start_seconds=float(seconds[0]), time_end_seconds=float(seconds[-1]),
                  observed_first=float(x[0]), observed_last=float(x[-1]))
    return result, report


def extract_cycle(cycle, dataset, profile="released_audit"):
    """Public API for canonical arrays V, I, t_seconds, Q_Ah and stage masks."""
    voltage, current = np.asarray(cycle["V"]), np.asarray(cycle["I"])
    low, high = (4.0, 4.2) if dataset in ("XJTU", "TJU") else (3.4, 3.6)
    cc = np.asarray(cycle["cc_mask"], dtype=bool)
    cv = np.asarray(cycle["cv_mask"], dtype=bool)
    if not all(len(np.asarray(cycle[key])) == len(voltage) for key in ("I", "t_seconds", "Q_Ah", "cc_mask", "cv_mask")):
        raise ValueError("Canonical cycle arrays/masks must have equal length")
    if profile == "released_audit" and dataset in ("XJTU", "HUST"):
        high -= .005
        cc = np.ones(len(voltage), dtype=bool)
    voltage_mask = (voltage >= low) & (voltage <= high) & cc
    current_mask = (current >= .1) & (current <= .5) & cv
    if dataset == "MIT" and profile == "released_audit":
        # Caller still supplies the stage masks; use the public reader's strict
        # numerical boundaries. Full author stage derivation is mit-mirror only.
        voltage_mask = (voltage > 3.4) & (voltage < 3.595) & cc
        current_mask = (current > .1) & (current < .5) & cv
    features, reports = {}, {}
    for prefix, mask, values in (("voltage", voltage_mask, voltage), ("current", current_mask, current)):
        row, report = extract_window(values[mask], np.asarray(cycle["t_seconds"])[mask],
                                     np.asarray(cycle["Q_Ah"])[mask], current[mask], prefix, profile)
        features.update(row)
        report["source_indices"] = np.flatnonzero(mask).tolist()
        reports[prefix] = report
    return features, reports


def canonical_csv(path):
    frame = pd.read_csv(path)
    required = {"raw_cycle", "voltage_V", "current_A", "t_seconds", "charge_Ah", "cc_stage", "cv_stage"}
    if not required.issubset(frame.columns):
        raise ValueError("Canonical CSV requires " + ", ".join(sorted(required)))
    if not frame[["cc_stage", "cv_stage"]].isin([0, 1]).all().all():
        raise ValueError("Stage flags must be 0 or 1")
    for cycle, group in frame.groupby("raw_cycle", sort=False):
        yield dict(raw_cycle=int(cycle), V=group.voltage_V.to_numpy(), I=group.current_A.to_numpy(),
                   t_seconds=group.t_seconds.to_numpy(), Q_Ah=group.charge_Ah.to_numpy(),
                   cc_mask=group.cc_stage.to_numpy().astype(bool), cv_mask=group.cv_stage.to_numpy().astype(bool))


def xjtu_mat(path, source_root):
    reader = source_root / "XJTUBatteryClass.py"
    spec = importlib.util.spec_from_file_location("replacement_author_xjtu_reader", reader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with contextlib.redirect_stdout(io.StringIO()):
        battery = module.Battery(path.as_posix())
    for number in range(1, int(battery.cycle_life) + 1):
        v, i, t, q = [battery.get_partial_value(number, name) for name in
                       ["voltage_V", "current_A", "relative_time_min", "capacity_Ah"]]
        yield dict(raw_cycle=number, V=v, I=i, t_seconds=t * 60, Q_Ah=q,
                   cc_mask=i >= 3.9, cv_mask=np.ones(len(i), dtype=bool))


def mit_mirror(path, mapping_path, csv_name, source_root, limit, wanted):
    """Yield already selected CC/CV windows using unchanged author reader.

    The native-current ambiguity remains explicit. This adapter never converts
    numeric C-rate to amperes and never opens a remote source.
    """
    import h5py
    from extract_mit_mirror_features import author_adapter
    with mapping_path.open(encoding="utf-8-sig", newline="") as stream:
        mappings = [row for row in csv.DictReader(stream) if Path(row["csv"]).name == csv_name]
    mappings.sort(key=lambda row: int(row["csv_row_zero_based"]))
    mappings = [row for row in mappings if not wanted or int(row["cycle"]) in wanted][:limit]
    if not mappings:
        raise ValueError("MIT --csv-name/--cycles selection is empty")
    dates = {row["source_batch_date"] for row in mappings}
    if len(dates) != 1 or path.name != "within_cycle_" + next(iter(dates)) + ".h5":
        raise ValueError("Selected MIT rows do not belong to supplied HDF5 date")
    adapter, _ = author_adapter(source_root)
    with h5py.File(path, "r") as store:
        battery = adapter({next(iter(dates)): store})
        for row in mappings:
            windows = {}
            for prefix in ("voltage", "current"):
                frame = (battery.get_one_battery_one_cycle_CC_stage(None, row) if prefix == "voltage" else
                         battery.get_one_battery_one_cycle_CV_stage(None, row))
                windows[prefix] = (frame[prefix + (" (V)" if prefix == "voltage" else " (A)")].to_numpy(),
                                   frame["time (min)"].to_numpy() * 60, frame["charge Q (Ah)"].to_numpy(),
                                   frame["current (A)"].to_numpy())
            yield {"raw_cycle": int(row["cycle"]), "windows": windows, "physical_mapping": row}


def feature_schema(current_unit, profile):
    rows = []
    for name in FEATURES:
        unit = ("Ah" if name.endswith(" Q") else "s" if name.endswith("charge time") else
                "V/s" if name == "voltage slope" else current_unit + "/s" if name == "current slope" else
                "V" if name in ("voltage mean", "voltage std") else
                current_unit if name in ("current mean", "current std") else "dimensionless")
        rows.append({"feature": name, "unit": unit, "status": "replacement" if name in SLOPES else
                     "prior_audited_convention" if profile == "released_audit" else "paper_formula_with_declared_assumptions"})
    return rows


def compare_released(records, reference, alignment, output, profile="released_audit"):
    """Post-export diagnostic only; cannot alter extraction parameters."""
    truth = pd.read_csv(reference)
    mapping = pd.read_csv(alignment)
    if not {"processed_row", "raw_cycle"}.issubset(mapping):
        raise ValueError("Alignment CSV requires processed_row and raw_cycle")
    mapping = mapping.dropna(subset=["raw_cycle"])
    if mapping.raw_cycle.duplicated().any() or mapping.processed_row.duplicated().any():
        raise ValueError("Alignment is not one-to-one")
    by_cycle = {int(row.raw_cycle): int(row.processed_row) for row in mapping.itertuples()}
    details = []
    for record in records:
        cycle = record["raw_cycle"]
        if cycle not in by_cycle:
            continue
        index = by_cycle[cycle]
        if not 0 <= index < len(truth):
            raise ValueError("Alignment row outside reference")
        for name in FEATURES:
            actual, expected = record[name], float(truth.iloc[index][name])
            finite = actual is not None and np.isfinite(expected)
            details.append({"raw_cycle": cycle, "processed_row": index, "feature": name,
                            "feature_status": "replacement" if name in SLOPES else "prior_audited_convention" if profile == "released_audit" else "paper_formula_with_declared_assumptions",
                            "exported": actual, "released": expected if np.isfinite(expected) else None,
                            "absolute_error": abs(actual - expected) if finite else None})
    pd.DataFrame(details).to_csv(output / "released_comparison.csv", index=False)
    summary = []
    for name in FEATURES:
        rows = [row for row in details if row["feature"] == name]
        errors = [row["absolute_error"] for row in rows if row["absolute_error"] is not None]
        summary.append({"feature": name, "status": "replacement" if name in SLOPES else "prior_audited_convention" if profile == "released_audit" else "paper_formula_with_declared_assumptions",
                        "rows": len(rows), "finite_pairs": len(errors), "matches_at_1e_8": sum(error <= 1e-8 for error in errors),
                        "maximum_absolute_error": max(errors) if errors else None})
    write_json(output / "released_comparison_summary.json", {
        "post_export_only": True, "threshold_descriptive_only": 1e-8,
        "alignment_caveat": "Pre-existing signature mapping uses four current statistics; these are not independent validation of row identity.",
        "reference_sha256": sha(reference), "alignment_sha256": sha(alignment), "features": summary})
    return summary


def run(options):
    output = options.output.resolve()
    if output.exists():
        raise ValueError("Use a new output directory; existing output is never overwritten")
    if options.limit < 1:
        raise ValueError("--limit must be positive; explicit bounded scope is required")
    if bool(options.reference_csv) != bool(options.alignment_csv):
        raise ValueError("Reference and alignment must be supplied together")
    if options.dataset == "MIT" and options.current_unit != "native_unit_unresolved":
        raise ValueError("MIT requires --current-unit native_unit_unresolved; no silent unit conversion")
    if options.profile == "paper_note2" and options.current_unit != "A":
        raise ValueError("Current integration to Ah requires verified current unit A")
    source = options.input.resolve()
    source_hash = sha(source)
    if options.expected_sha256 and source_hash != options.expected_sha256:
        raise ValueError("Source SHA256 mismatch")
    readers = {"export_replacement_features.py", "audit_xjtu_feature_sample.py"}
    provenance = {"input_path": str(source), "input_bytes": source.stat().st_size, "input_sha256": source_hash,
                  "expected_sha256_verified": bool(options.expected_sha256), "kind": options.kind}
    archive = None
    if options.kind == "xjtu-mat":
        if options.dataset != "XJTU":
            raise ValueError("xjtu-mat requires dataset XJTU")
        cycles = xjtu_mat(source, options.source_root)
        provenance["reader_sha256"] = sha(options.source_root / "XJTUBatteryClass.py")
    elif options.kind == "canonical-csv":
        cycles = canonical_csv(source)
        provenance["stage_selection"] = "Caller-supplied explicit stage masks; not independently inferred by exporter"
    elif options.kind == "mit-mirror":
        if options.dataset != "MIT" or options.profile != "released_audit":
            raise ValueError("mit-mirror requires MIT/released_audit; author stage adapter does not implement paper windows")
        if not options.expected_sha256 or not options.csv_name:
            raise ValueError("mit-mirror requires --expected-sha256 and --csv-name")
        provenance.update(mapping_sha256=sha(options.mit_mapping),
                          reader_sha256=sha(options.source_root / "MITBatteryClass.py"),
                          stage_selection="Unchanged author MITBatteryClass via existing audited HDF5 adapter",
                          source_type="Third-party derived float32 HDF5 mirror; not original MAT byte chain")
        readers.add("extract_mit_mirror_features.py")
        cycles = mit_mirror(source, options.mit_mapping, options.csv_name, options.source_root,
                            options.limit, set(options.cycles or []))
    else:
        from extract_raw_feature_candidates import load_xjtu, load_tju, load_hust
        readers.add("extract_raw_feature_candidates.py")
        if not options.member or options.dataset == "MIT":
            raise ValueError("official-archive requires exact --member and dataset XJTU/TJU/HUST")
        if not options.expected_sha256:
            raise ValueError("official-archive requires --expected-sha256")
        archive = zipfile.ZipFile(source)
        provenance["member"] = options.member
        info = archive.getinfo(options.member)
        provenance.update(member_crc32=format(info.CRC, "08x"), member_bytes=info.file_size)
        cycles = {"XJTU": lambda: load_xjtu(archive, options.member, options.source_root),
                  "TJU": lambda: load_tju(archive, options.member),
                  "HUST": lambda: load_hust(archive, options.member)}[options.dataset]()
        if options.dataset == "XJTU":
            provenance["reader_sha256"] = sha(options.source_root / "XJTUBatteryClass.py")
    records, diagnostics = [], []
    wanted = set(options.cycles or [])
    try:
        for cycle in cycles:
            if wanted and cycle["raw_cycle"] not in wanted:
                continue
            if "windows" in cycle:
                features, reports = {}, {}
                for prefix, arrays in cycle["windows"].items():
                    values, report = extract_window(*arrays, prefix, options.profile)
                    features.update(values)
                    report["stage_selection"] = "unchanged_author_MIT_reader"
                    reports[prefix] = report
            else:
                features, reports = extract_cycle(cycle, options.dataset, options.profile)
            records.append({"raw_cycle": cycle["raw_cycle"], **features})
            diagnostics.append({"raw_cycle": cycle["raw_cycle"], "curves": reports,
                                "physical_mapping": cycle.get("physical_mapping")})
            if len(records) >= options.limit or wanted and len(records) == len(wanted):
                break
    finally:
        if archive is not None:
            archive.close()
    if not records:
        raise ValueError("No selected cycles")
    if wanted - {r["raw_cycle"] for r in records}:
        raise ValueError("Some requested cycles missing or outside --limit")
    output.mkdir(parents=True)
    pd.DataFrame(records, columns=["raw_cycle"] + FEATURES).to_csv(output / "features_16.csv", index=False)
    write_json(output / "window_diagnostics.json", diagnostics)
    comparison = compare_released(records, options.reference_csv, options.alignment_csv, output, options.profile) if options.reference_csv else None
    write_json(output / "manifest.json", {
        "status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": options.dataset, "profile": options.profile, "selected_cycles": [r["raw_cycle"] for r in records],
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
        "invocation": sys.argv,
        "selection": "Explicit --cycles in source order or first --limit raw cycles; no feature-error selection",
        "provenance": provenance, "preset_protocol": PRESET, "feature_schema": feature_schema(options.current_unit, options.profile),
        "feature_count": 16, "replacement_feature_count": 2, "missing_values": "Empty CSV fields/JSON null; no imputation",
        "current_unit": options.current_unit,
        "MIT_caveat": "Canonical MIT inputs must retain author-selected windows or explicitly supply stage masks; derived H5 native current units remain uncertain. No silent C-rate-to-A conversion.",
        "training_inputs_changed": False, "all_released_slopes_reconstructed": False,
        "code_sha256": {name: sha(Path(__file__).with_name(name)) for name in sorted(readers)},
        "artifacts_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
        "comparison_available": comparison is not None})
    print(json.dumps({"output": str(output), "cycles": len(records), "features": 16, "replacement_slopes": 2}))


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--kind", choices=["canonical-csv", "xjtu-mat", "official-archive", "mit-mirror"], required=True)
    value.add_argument("--dataset", choices=["XJTU", "TJU", "HUST", "MIT"], required=True)
    value.add_argument("--input", type=Path, required=True)
    value.add_argument("--expected-sha256")
    value.add_argument("--member")
    value.add_argument("--mit-mapping", type=Path, default=ROOT / "outputs/full_reproduction/feature_audit/MIT_mirror_audit/PINN_CSV_to_mirror_cycles.csv")
    value.add_argument("--csv-name", help="Exact released CSV basename for a bounded MIT mirror selection")
    value.add_argument("--source-root", type=Path, default=ROOT / "outputs/full_reproduction/feature_audit/preprocessing_source")
    value.add_argument("--profile", choices=["released_audit", "paper_note2"], default="released_audit")
    value.add_argument("--current-unit", default="A", choices=["A", "native_unit_unresolved"])
    value.add_argument("--cycles", type=int, nargs="+")
    value.add_argument("--limit", type=int, default=3)
    value.add_argument("--reference-csv", type=Path)
    value.add_argument("--alignment-csv", type=Path)
    value.add_argument("--output", type=Path, required=True)
    return value


if __name__ == "__main__":
    run(parser().parse_args())
