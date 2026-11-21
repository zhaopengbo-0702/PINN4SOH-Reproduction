"""Add fixed -5mV voltage-boundary candidates without changing frozen extraction."""

import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import pandas as pd

from audit_xjtu_feature_sample import curve_candidates
from extract_raw_feature_candidates import load_hust, load_xjtu, write_json
from summarize_raw_feature_residuals import compare


def main(watch):
    root = Path("outputs/full_reproduction/feature_audit").resolve()
    raw_root = root.parent / "raw_data"
    inventory = json.loads((root / "official_raw_inventory.json").read_text(encoding="utf-8"))
    implementation = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                      for name in ["supplement_voltage_boundaries.py", "extract_raw_feature_candidates.py", "audit_xjtu_feature_sample.py"]}
    finished = set()
    while True:
        for manifest_path in sorted((root / "raw_extraction").rglob("manifest.json")):
            primary = json.loads(manifest_path.read_text(encoding="utf-8"))
            if primary["dataset"] not in {"XJTU", "HUST"} or primary["status"] != "complete":
                continue
            key = primary["processed_csv"]
            if key in finished:
                continue
            task_dir = root / "boundary_extraction" / primary["dataset"] / manifest_path.parent.name
            task_dir.mkdir(parents=True, exist_ok=True)
            output_manifest = task_dir / "manifest.json"
            if output_manifest.exists():
                previous = json.loads(output_manifest.read_text(encoding="utf-8"))
                if previous.get("status") == "complete":
                    if previous.get("implementation_sha256") != implementation:
                        raise ValueError("Completed boundary task has a different implementation")
                    finished.add(key)
                    continue
            state = dict(status="running", dataset=primary["dataset"], processed_csv=key,
                         raw_member=primary["raw_member"], implementation_sha256=implementation,
                         primary_candidate_sha256=primary["candidate_csv_sha256"],
                         boundary_scope="One fixed upper voltage bound for every cell/cycle; no selection by residual",
                         XJTU_upper_voltage_V=4.195, HUST_upper_voltage_V=3.595,
                         evidence="HUST public reader3.595; XJTU pilot exploratory4.195; both held fixed for remaining cells")
            write_json(output_manifest, state)
            started = time.time()
            try:
                archive_record = next(item for item in inventory if item["dataset"] == primary["dataset"])
                archive_path = raw_root / primary["dataset"] / archive_record["name"]
                upper = 4.195 if primary["dataset"] == "XJTU" else 3.595
                lower = 4.0 if primary["dataset"] == "XJTU" else 3.4
                rows = []
                with zipfile.ZipFile(archive_path) as archive:
                    source = (load_xjtu(archive, primary["raw_member"], root / "preprocessing_source")
                              if primary["dataset"] == "XJTU" else load_hust(archive, primary["raw_member"]))
                    for cycle in source:
                        mask = (cycle["V"] >= lower) & (cycle["V"] <= upper)
                        values = curve_candidates(cycle["V"][mask], cycle["t_seconds"][mask] / 60, cycle["Q_Ah"][mask])
                        if values is None:
                            continue
                        from scipy.integrate import trapezoid
                        values["charge_trapezoid_Ah"] = float(trapezoid(cycle["I"][mask], cycle["t_seconds"][mask]) / 3600)
                        rows.append(dict(raw_cycle=cycle["raw_cycle"], capacity_Ah=cycle["capacity_Ah"],
                                         interpolated_capacity_Ah=cycle["interpolated_capacity_Ah"], curve="voltage",
                                         window="fixed_5mV_boundary_allcharge", **values))
                original = pd.read_csv(manifest_path.parent / "candidate_features.csv")
                current = original[original["curve"] == "current"]
                pd.concat([current, pd.DataFrame(rows)], ignore_index=True).to_csv(task_dir / "candidate_features.csv", index=False)
                alignment, residuals = compare(primary, task_dir, Path("PINN4SOH").resolve())
                state.update(status="complete", voltage_candidate_cycles=len(rows), alignment=alignment,
                             seconds=time.time() - started,
                             candidate_sha256=hashlib.sha256((task_dir / "candidate_features.csv").read_bytes()).hexdigest())
            except Exception as error:
                import traceback
                state.update(status="failed", error=repr(error), seconds=time.time() - started)
                (task_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            write_json(output_manifest, state)
            finished.add(key)
            print(json.dumps(state), flush=True)
        if len(finished) == 132 or not watch:
            break
        time.sleep(20)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    main(parser.parse_args().watch)
