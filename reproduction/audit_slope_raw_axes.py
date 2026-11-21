"""Apply fixed sample-index and min-max slope definitions to every raw cell.

This is a separately labelled diagnostic. No per-row fitting of constants or
window bounds is performed, and released training CSVs are never modified.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd

from extract_raw_feature_candidates import load_xjtu, load_tju, load_hust, write_json
from resume_boundary_memory_guard import available_gib


ROOT = Path("outputs/full_reproduction/feature_audit").resolve()
REPO = Path("PINN4SOH").resolve()


def rules(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 4 or x[-1] <= x[0] or y.max() <= y.min():
        return {}
    index = np.arange(len(y), dtype=float)
    values = {}
    for label, a in (("physical_seconds", x), ("sample_index", index),
                     ("minmax_physical_time", (x - x.min()) / (x.max() - x.min())),
                     ("minmax_sample_index", index / index[-1])):
        for scale, b in (("raw_curve", y), ("minmax_curve", (y - y.min()) / (y.max() - y.min()))):
            differences, dx = np.diff(b), np.diff(a)
            positive = dx > 0
            centered = a - a.mean()
            prefix = scale + "__" + label + "__"
            values[prefix + "endpoint"] = (b[-1] - b[0]) / (a[-1] - a[0])
            values[prefix + "OLS"] = float(np.sum(centered * (b - b.mean())) / np.sum(centered ** 2))
            values[prefix + "mean_difference"] = float(np.mean(differences[positive] / dx[positive]))
            if np.all(positive):
                values[prefix + "mean_gradient"] = float(np.gradient(b, a).mean())
    return values


def process_cell(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    dataset = manifest["dataset"]
    destination = ROOT / "slope_additional_axes" / dataset / Path(path).parent.name
    destination.mkdir(parents=True, exist_ok=True)
    output_manifest = destination / "manifest.json"
    implementation = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                      for name in ["audit_slope_raw_axes.py", "extract_raw_feature_candidates.py"]}
    if output_manifest.exists():
        previous = json.loads(output_manifest.read_text(encoding="utf-8"))
        if previous["status"] == "complete":
            if previous["implementation_sha256"] != implementation:
                raise ValueError("Completed diagnostic uses another implementation")
            return previous
    inventory = json.loads((ROOT / "official_raw_inventory.json").read_text(encoding="utf-8"))
    selected = next(record for record in inventory if record["dataset"] == dataset and
                    any(member["name"] == manifest["raw_member"] for member in record["archive_members"]))
    archive_path = ROOT.parent / "raw_data" / dataset / selected["name"]
    start = time.time()
    state = dict(status="running", dataset=dataset, processed_csv=manifest["processed_csv"],
                 implementation_sha256=implementation, selected_by_slope_residual=False)
    write_json(output_manifest, state)
    alignment = pd.read_csv(Path(path).parent / "cycle_alignment.csv").dropna(subset=["raw_cycle"])
    mapping = dict(zip(alignment.raw_cycle.astype(int), alignment.processed_row.astype(int)))
    target = pd.read_csv(REPO / "data" / manifest["processed_csv"])
    aggregates = {}
    try:
        with zipfile.ZipFile(archive_path) as archive:
            if dataset == "XJTU":
                budget = max(3.0, min(4.5, 2 + archive.getinfo(manifest["raw_member"]).file_size * 18 / 1024 ** 3))
                while available_gib() < budget:
                    time.sleep(10)
                source = load_xjtu(archive, manifest["raw_member"], ROOT / "preprocessing_source")
            elif dataset == "TJU":
                source = load_tju(archive, manifest["raw_member"])
            else:
                source = load_hust(archive, manifest["raw_member"])
            for cycle in source:
                if cycle["raw_cycle"] not in mapping:
                    continue
                row = target.iloc[mapping[cycle["raw_cycle"]]]
                voltage = cycle["V"]
                masks = {
                    "voltage": (voltage >= (3.4 if dataset == "HUST" else 4.0)) &
                               (voltage <= (3.595 if dataset == "HUST" else 4.195 if dataset == "XJTU" else 4.2)),
                    "current": (cycle["I"] >= .1) & (cycle["I"] <= .5) & cycle["cv_mask"],
                }
                if dataset == "TJU":
                    masks["voltage"] &= cycle["cc_mask"]
                for curve, mask in masks.items():
                    for definition, candidate in rules(cycle["t_seconds"][mask], cycle["V" if curve == "voltage" else "I"][mask]).items():
                        feature = curve + " slope"
                        error = abs(candidate - row[feature])
                        if not np.isfinite(error):
                            continue
                        aggregate = aggregates.setdefault((feature, definition), [0, 0, 0, 0., 0.])
                        aggregate[0] += 1
                        aggregate[1] += error <= 1e-12
                        aggregate[2] += error <= 1e-8
                        aggregate[3] += error
                        aggregate[4] = max(aggregate[4], error)
        rows = [dict(dataset=dataset, processed_csv=manifest["processed_csv"], feature=feature, definition=definition,
                     finite_pairs=int(values[0]), matches_at_1e_12=int(values[1]), matches_at_1e_8=int(values[2]),
                     absolute_error_sum=values[3], maximum_absolute_difference=values[4])
                for (feature, definition), values in aggregates.items()]
        result = destination / "residuals.csv"
        pd.DataFrame(rows).to_csv(result, index=False)
        state.update(status="complete", seconds=time.time() - start,
                     residuals_sha256=hashlib.sha256(result.read_bytes()).hexdigest(),
                     aligned_rows=len(mapping), definitions=len(rows), capacity_modified=False)
    except Exception as error:
        import traceback
        state.update(status="failed", error=repr(error))
        (destination / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    write_json(output_manifest, state)
    print(json.dumps(state), flush=True)
    gc.collect()
    return state


def main(options):
    if options.cell_manifest:
        process_cell(options.cell_manifest)
        return
    manifests = list((ROOT / "raw_extraction").rglob("manifest.json"))
    small = [path for path in manifests if json.loads(path.read_text())["dataset"] != "XJTU"]
    large = [path for path in manifests if json.loads(path.read_text())["dataset"] == "XJTU"]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(process_cell, str(path)) for path in small]):
            future.result()
    # A fresh process per MAT returns the entire scipy MATLAB object heap to Windows.
    import subprocess
    import sys
    for path in large:
        result = subprocess.run([sys.executable, "-X", "utf8", __file__, "--cell-manifest", str(path)])
        if result.returncode:
            raise RuntimeError("Slope diagnostic child process failed")
    files = list((ROOT / "slope_additional_axes").rglob("residuals.csv"))
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    frame.to_csv(ROOT / "slope_additional_axes_per_cell.csv", index=False)
    summary = frame.groupby(["dataset", "feature", "definition"], sort=False).agg(
        cells=("processed_csv", "nunique"), finite_pairs=("finite_pairs", "sum"), matches_at_1e_12=("matches_at_1e_12", "sum"),
        matches_at_1e_8=("matches_at_1e_8", "sum"), absolute_error_sum=("absolute_error_sum", "sum"),
        maximum_absolute_difference=("maximum_absolute_difference", "max")).reset_index()
    summary["mae"] = summary["absolute_error_sum"] / summary["finite_pairs"]
    summary.to_csv(ROOT / "slope_additional_axes_summary.csv", index=False)
    print(json.dumps(dict(status="complete", summarized_cells=frame.processed_csv.nunique(),
                          any_definition_reconstructs_entire_dataset=bool((summary.matches_at_1e_12 == summary.finite_pairs).any()))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-manifest")
    main(parser.parse_args())
