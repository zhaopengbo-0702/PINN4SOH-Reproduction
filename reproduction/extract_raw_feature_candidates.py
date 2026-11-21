"""Extract explicit feature candidates from checksum-verified official raw archives.

No candidate is selected by error against the released CSV. All raw cycles and
window definitions are retained; this script does not modify training inputs.
"""

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import pickle
import tempfile
import time
import traceback
import zipfile

import numpy as np
import pandas as pd
from scipy import integrate

from audit_xjtu_feature_sample import curve_candidates
from download_official_raw import digest_file


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class RestrictedDataUnpickler(pickle.Unpickler):
    ALLOWED = {
        ("builtins", "slice"), ("builtins", "set"), ("builtins", "frozenset"),
        ("collections", "OrderedDict"), ("numpy", "ndarray"), ("numpy", "dtype"),
        ("numpy.core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "scalar"),
        ("pandas.core.frame", "DataFrame"), ("pandas.core.series", "Series"),
        ("pandas.core.indexes.base", "_new_Index"), ("pandas.core.indexes.base", "Index"),
        ("pandas.core.indexes.range", "RangeIndex"), ("pandas.core.indexes.numeric", "Int64Index"),
        ("pandas.core.internals.managers", "BlockManager"), ("pandas._libs.internals", "_unpickle_block"),
    }

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError("Unreviewed HUST data constructor: " + module + "." + name)
        if (module, name) == ("pandas.core.indexes.numeric", "Int64Index"):
            return pd.Index
        return super().find_class(module, name)


def raw_requirements(repo, inventory, output):
    records = []
    for path in sorted((repo / "data").rglob("*.csv")):
        relative = path.relative_to(repo / "data")
        dataset = relative.parts[0].split()[0]
        if dataset == "XJTU":
            groups = ["2C", "3C", "R2.5", "R3", "RW", "satellite"]
            batch = next(index + 1 for index, name in enumerate(groups) if name in path.name)
            archive_name, member = "Battery Dataset.zip", f"Battery Dataset/Batch-{batch}/{path.stem}.mat"
        elif dataset == "TJU":
            archive_name, member = relative.parts[1] + ".zip", "/".join(relative.parts[1:])
        elif dataset == "HUST":
            archive_name, member = "our_data.zip", "our_data/" + path.stem + ".pkl"
        else:
            archive_name, member = relative.parts[1] + "_batchdata_updated_struct_errorcorrect.mat", "Unverified source-cell index for " + path.stem
        archive = next((record for record in inventory if record["dataset"] == dataset and record["name"] == archive_name), None)
        raw = next((record for record in archive["archive_members"] if record["name"] == member), None) if archive else None
        records.append(dict(dataset=dataset, processed_csv=relative.as_posix(), archive=archive_name,
                            raw_member=member, official_directory_match=raw is not None,
                            raw_compressed_bytes=raw["compressed_bytes"] if raw else None,
                            raw_uncompressed_bytes=raw["uncompressed_bytes"] if raw else None,
                            source_url=archive["download_url"] if archive else "https://data.matr.io/1/projects/5c48dd2bc625d700019f3204"))
    with (output / "required_raw_files.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return records


def load_xjtu(archive, member, source_root):
    with tempfile.TemporaryDirectory(prefix="pinn_raw_xjtu_") as temporary:
        target = Path(temporary) / Path(member).name
        with target.open("wb") as stream, archive.open(member) as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                stream.write(chunk)
        spec = importlib.util.spec_from_file_location("official_xjtu_raw_reader", source_root / "XJTUBatteryClass.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with contextlib.redirect_stdout(io.StringIO()):
            battery = module.Battery(target.as_posix())
        capacity = battery.get_capacity()
        try:
            trajectory = battery.get_degradation_trajectory()
        except (IndexError, ValueError):
            trajectory = capacity
        for cycle in range(1, int(battery.cycle_life) + 1):
            values = [battery.get_partial_value(cycle, name) for name in ["voltage_V", "current_A", "relative_time_min", "capacity_Ah"]]
            yield dict(raw_cycle=cycle, capacity_Ah=float(capacity[cycle - 1]),
                       interpolated_capacity_Ah=float(trajectory[cycle - 1]),
                       V=values[0], I=values[1], t_seconds=values[2] * 60, Q_Ah=values[3],
                       cc_mask=values[1] >= 3.9, cv_mask=np.ones(len(values[1]), dtype=bool), cutoff=4.2)


def load_tju(archive, member):
    with archive.open(member) as source:
        frame = pd.read_csv(source)
    for cycle, group in frame.groupby("cycle number", sort=True):
        charge = group[group["control/V/mA"] > 0]
        yield dict(raw_cycle=int(cycle), capacity_Ah=float(group["Q discharge/mA.h"].max() / 1000),
                   interpolated_capacity_Ah=None, V=charge["Ecell/V"].to_numpy(), I=charge["<I>/mA"].to_numpy() / 1000,
                   t_seconds=charge["time/s"].to_numpy(), Q_Ah=charge["Q charge/mA.h"].to_numpy() / 1000,
                   cc_mask=(charge["control/mA"] > 0).to_numpy(), cv_mask=(charge["control/V"] > 0).to_numpy(), cutoff=4.2)


def load_hust(archive, member):
    with archive.open(member) as source:
        raw = RestrictedDataUnpickler(source).load()
    cell = raw[Path(member).stem]
    for cycle, group in sorted(cell["data"].items()):
        charge = group[group["Status"] == "Constant current-constant voltage charge"]
        voltage = charge["Voltage (V)"].to_numpy()
        cross = np.where(voltage >= 3.595)[0]
        index = int(cross[0]) if len(cross) else len(voltage)
        positions = np.arange(len(voltage))
        yield dict(raw_cycle=int(cycle), capacity_Ah=float(group["Capacity (mAh)"].max() / 1000),
                   interpolated_capacity_Ah=None, V=voltage, I=charge["Current (mA)"].to_numpy() / 1000,
                   t_seconds=charge["Time (s)"].to_numpy(), Q_Ah=charge["Capacity (mAh)"].to_numpy() / 1000,
                   cc_mask=positions <= index, cv_mask=positions >= index, cutoff=3.6)


def candidates(cycle):
    voltage, current = cycle["V"], cycle["I"]
    if not len(voltage):
        return []
    cutoff = cycle["cutoff"]
    charge_max = float(voltage.max())
    specs = [("note2_fixed_cc", cutoff - .2, cutoff, cycle["cc_mask"]),
             ("note2_fixed_allcharge", cutoff - .2, cutoff, np.ones(len(voltage), dtype=bool)),
             ("actual_charge_max_cc", charge_max - .2, charge_max, cycle["cc_mask"])]
    if np.any(cycle["cc_mask"]):
        cc_max = float(voltage[cycle["cc_mask"]].max())
        specs.append(("actual_cc_max_cc", cc_max - .2, cc_max, cycle["cc_mask"]))
    if cutoff == 4.2:
        specs.extend([("tolerance_1mV_cc", 4.0, 4.199, cycle["cc_mask"]),
                      ("tolerance_5mV_cc_hypothesis", 4.0, 4.195, cycle["cc_mask"]),
                      ("tolerance_10mV_cc_hypothesis", 4.0, 4.19, cycle["cc_mask"])])
    cv = (current >= .1) & (current <= .5) & cycle["cv_mask"]
    selectors = [("current", "note2_current_stage", cv, current),
                 ("current", "note2_current_allcharge", (current >= .1) & (current <= .5), current)]
    selectors.extend(("voltage", label, (voltage >= low) & (voltage <= high) & mask, voltage) for label, low, high, mask in specs)
    records, cache = [], {}
    for curve_name, window, mask, curve in selectors:
        identity = (curve_name, mask.tobytes())
        if identity not in cache:
            cache[identity] = curve_candidates(curve[mask], cycle["t_seconds"][mask] / 60, cycle["Q_Ah"][mask])
        original_values = cache[identity]
        values = dict(original_values) if original_values is not None else None
        if values is None:
            continue
        values["charge_trapezoid_Ah"] = float(integrate.trapezoid(current[mask], cycle["t_seconds"][mask]) / 3600)
        clean = {key: value if np.isfinite(value) else None for key, value in values.items()}
        records.append(dict(raw_cycle=cycle["raw_cycle"], capacity_Ah=cycle["capacity_Ah"],
                            interpolated_capacity_Ah=cycle["interpolated_capacity_Ah"], curve=curve_name, window=window, **clean))
    return records


def process(record, requirement, archive_path, output, source_root, implementation):
    identity = hashlib.sha256(requirement["processed_csv"].encode()).hexdigest()[:12]
    task_dir = output / "raw_extraction" / record["dataset"] / (Path(requirement["raw_member"]).stem + "_" + identity)
    task_dir.mkdir(parents=True, exist_ok=True)
    result_path = task_dir / "manifest.json"
    if result_path.exists():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if previous.get("status") == "complete":
            if previous["implementation_sha256"] != implementation:
                raise ValueError("Completed extraction has different implementation; use another output root")
            return previous
    started = time.time()
    state = dict(status="running", dataset=record["dataset"], raw_member=requirement["raw_member"],
                 processed_csv=requirement["processed_csv"], implementation_sha256=implementation,
                 official_checksum=record["checksum"], selected_by_csv_residual=False)
    write_json(result_path, state)
    try:
        with zipfile.ZipFile(archive_path) as archive, (task_dir / "candidate_features.csv").open("w", encoding="utf-8", newline="") as stream:
            loader = {"XJTU": lambda: load_xjtu(archive, requirement["raw_member"], source_root),
                      "TJU": lambda: load_tju(archive, requirement["raw_member"]),
                      "HUST": lambda: load_hust(archive, requirement["raw_member"])}[record["dataset"]]
            writer, cycles, rows = None, 0, 0
            for cycle in loader():
                cycles += 1
                for row in candidates(cycle):
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    rows += 1
        state.update(status="complete", raw_cycles=cycles, candidate_records=rows,
                     seconds=time.time() - started, candidate_csv_sha256=digest_file(task_dir / "candidate_features.csv", "sha256"))
    except Exception as error:
        state.update(status="failed", error=repr(error), seconds=time.time() - started)
        (task_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    write_json(result_path, state)
    print(json.dumps(state), flush=True)
    return state


def run(options):
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inventory = json.loads(options.inventory.read_text(encoding="utf-8"))
    requirements = raw_requirements(options.repo.resolve(), inventory, output)
    implementation = {name: digest_file(Path(__file__).with_name(name), "sha256")
                      for name in ["extract_raw_feature_candidates.py", "audit_xjtu_feature_sample.py"]}
    finished, verified = {}, set()
    progress_path = output / ("raw_extraction_progress_" + "_".join(options.datasets) + ".json")
    while True:
        for record in inventory:
            if record["dataset"] not in options.datasets:
                continue
            archive_path = options.raw_root.resolve() / record["dataset"] / record["name"]
            status_path = archive_path.with_suffix(".status.json")
            if not archive_path.is_file() or not status_path.is_file():
                continue
            validation = json.loads(status_path.read_text(encoding="utf-8"))
            if validation.get("status") != "verified" or validation.get("official_checksum") != record["checksum"]:
                continue
            if record["name"] not in verified:
                algorithm, expected = record["checksum"].split(":", 1)
                if archive_path.stat().st_size != record["compressed_bytes"] or digest_file(archive_path, algorithm) != expected:
                    raise ValueError("Archive verification failed")
                verified.add(record["name"])
            for requirement in requirements:
                if requirement["archive"] != record["name"] or requirement["processed_csv"] in finished:
                    continue
                finished[requirement["processed_csv"]] = process(record, requirement, archive_path, output,
                                                                output / "preprocessing_source", implementation)
                if options.max_cells and len(finished) >= options.max_cells:
                    write_json(progress_path, list(finished.values()))
                    return
                write_json(progress_path, list(finished.values()))
        expected = sum(item["dataset"] in options.datasets for item in requirements)
        if len(finished) == expected or not options.watch:
            break
        time.sleep(20)
    write_json(progress_path, list(finished.values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("PINN4SOH"))
    parser.add_argument("--inventory", type=Path, default=Path("outputs/full_reproduction/feature_audit/official_raw_inventory.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/full_reproduction/feature_audit"))
    parser.add_argument("--raw-root", type=Path, default=Path("outputs/full_reproduction/raw_data"))
    parser.add_argument("--datasets", nargs="+", choices=["XJTU", "TJU", "HUST"], default=["XJTU", "TJU", "HUST"])
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--watch", action="store_true")
    run(parser.parse_args())
