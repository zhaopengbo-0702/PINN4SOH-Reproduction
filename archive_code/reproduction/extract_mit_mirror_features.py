"""Audit prespecified MIT curve signatures from an explicitly derived HDF5 mirror.

The adapter calls the author's unchanged stage-selection methods. Native mirror
values retain the source numeric units: current is not silently rescaled. The
mirror labels I as C-rate while the PINN reader labels it amperes; that unresolved
unit interpretation is kept explicit. This is a source audit, not a new split.
"""

import argparse
from collections import OrderedDict, defaultdict
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests

from audit_mit_mirror import ROOT, OUTPUT, sha
from audit_xjtu_feature_sample import curve_candidates


SIZES = {"2017-05-12": 797659714, "2017-06-30": 581592476, "2018-04-12": 823392641}
HASHES = {"2017-05-12": "8a789f4a83409ad0c1d6c214c836f2de94b63ef2e73a50722fe057c3d8a89d77",
          "2017-06-30": "5f3f752d9d9a0276442fbb853941d9b212e7fffd152995b207ae6fa4a4503fae",
          "2018-04-12": "bdd195323719986b3d77d4bd610c26e5e4e7c8d8ddde558b03b4696a2bcc6107"}


class RemoteH5(io.RawIOBase):
    def __init__(self, date, block_size=256 * 1024, cap=64 * 1024 * 1024):
        self.url = "https://zenodo.org/records/23984188/files/within_cycle_" + date + ".h5?download=1"
        self.size, self.position, self.block_size, self.cap = SIZES[date], 0, block_size, cap
        self.downloaded, self.cache = 0, OrderedDict()
        self.session = requests.Session()
        self.session.trust_env = False
        self.ranges = []

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (0 if whence == 0 else self.position if whence == 1 else self.size) + offset
        if not 0 <= self.position <= self.size:
            raise ValueError("Seek outside published HDF5 file")
        return self.position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size == 0:
            return b""
        chunks = []
        while size:
            block = self.position // self.block_size
            start, stop = block * self.block_size, min(self.size - 1, (block + 1) * self.block_size - 1)
            if block not in self.cache:
                if self.downloaded + stop - start + 1 > self.cap:
                    raise ValueError("Remote signature sample exceeds explicit download cap")
                with self.session.get(self.url, headers={"Range": f"bytes={start}-{stop}"}, timeout=45) as response:
                    response.raise_for_status()
                    data = response.content
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{stop}/{self.size}" or len(data) != stop - start + 1:
                        raise ValueError("Server did not honor bounded HDF5 Range")
                self.cache[block] = data
                self.downloaded += len(data)
                self.ranges.append({"start": start, "stop_inclusive": stop,
                                    "sha256": hashlib.sha256(data).hexdigest()})
            data = self.cache[block]
            offset, count = self.position - start, min(size, stop - self.position + 1)
            chunks.append(data[offset:offset + count])
            self.position += count
            size -= count
        return b"".join(chunks)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def author_adapter(source_root):
    path = source_root / "MITBatteryClass.py"
    spec = importlib.util.spec_from_file_location("audited_author_mit_reader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class MirrorBattery(module.BatchBattery):
        def __init__(self, stores):
            self.stores = stores

        def get_one_battery_one_cycle(self, cell_num, cycle_num):
            row = cycle_num
            store = self.stores[row["source_batch_date"]]
            index = int(row["h5_row"])
            a, b = store["native/offset"][index:index + 2]
            values = {name: store["native/" + name][int(a):int(b)].astype(np.float64)
                      for name in ["I", "V", "Qc", "Qd", "t"]}
            uid = store["row/cell_uid"][index]
            uid = uid.decode() if isinstance(uid, bytes) else str(uid)
            if uid != row["cell_uid"] or int(store["row/cycle"][index]) != int(row["cycle"]):
                raise ValueError("HDF5 row barcode/cycle disagrees with the capacity source index")
            return pd.DataFrame({"current (A)": values["I"], "voltage (V)": values["V"],
                                 "charge Q (Ah)": values["Qc"], "discharge Q (Ah)": values["Qd"], "time (min)": values["t"]})

    return MirrorBattery, path


def run(options):
    root = options.mirror.resolve()
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (root / "PINN_CSV_to_mirror_cycles.csv").open(encoding="utf-8-sig", newline="") as stream:
        mappings = list(csv.DictReader(stream))
    by_csv = defaultdict(list)
    for row in mappings:
        by_csv[row["csv"]].append(row)
    selected = []
    for csv_path, rows in by_csv.items():
        if options.files and Path(csv_path).name not in options.files:
            continue
        rows.sort(key=lambda row: int(row["csv_row_zero_based"]))
        wanted = rows[:options.first_rows_per_csv]
        different = [row for row in rows if row["match_type"] != "ordered_expected_physical_cell"]
        wanted.extend(different[:options.first_rows_per_csv])
        if options.all_rows:
            wanted = rows
        selected.extend({int(row["csv_row_zero_based"]): row for row in wanted}.values())
    dates = sorted({row["source_batch_date"] for row in selected})
    stores, remote, checksum = {}, {}, {}
    try:
        for date in dates:
            path = root / ("within_cycle_" + date + ".h5")
            if path.is_file() and path.stat().st_size == SIZES[date]:
                actual = sha(path)
                if actual != HASHES[date]:
                    raise ValueError("Mirror HDF5 SHA256 differs from published hash")
                stores[date], checksum[date] = h5py.File(path, "r"), {"verified_full_sha256": actual, "local_path": str(path)}
            elif options.remote:
                remote[date] = RemoteH5(date)
                stores[date], checksum[date] = h5py.File(remote[date], "r"), {
                    "full_file_sha256_not_yet_verified": True, "published_sha256": HASHES[date], "url": remote[date].url}
            else:
                raise ValueError("Verified HDF5 file is missing; use --remote for a bounded provisional sample")
        adapter, reader_path = author_adapter(ROOT / "outputs/full_reproduction/feature_audit/preprocessing_source")
        battery = adapter(stores)
        published, result = {}, []
        for row in selected:
            csv_path = row["csv"]
            if csv_path not in published:
                published[csv_path] = pd.read_csv(ROOT / csv_path)
            truth = published[csv_path].iloc[int(row["csv_row_zero_based"])]
            for curve in ["voltage", "current"]:
                frame = (battery.get_one_battery_one_cycle_CC_stage(None, row) if curve == "voltage" else
                         battery.get_one_battery_one_cycle_CV_stage(None, row))
                values = curve_candidates(frame[curve + (" (V)" if curve == "voltage" else " (A)")].to_numpy(),
                                          frame["time (min)"].to_numpy(), frame["charge Q (Ah)"].to_numpy())
                if values is None:
                    result.append({**row, "curve": curve, "status": "window_has_fewer_than_four_points"})
                    continue
                columns = {"mean": curve + " mean", "std_ddof0": curve + " std",
                           "kurtosis_scipy_biased": curve + " kurtosis", "skewness_scipy_biased": curve + " skewness",
                           "entropy_minmax_probability": curve + " entropy",
                           "duration_seconds": "CC charge time" if curve == "voltage" else "CV charge time",
                           "charge_capacity_difference_Ah": "CC Q" if curve == "voltage" else "CV Q"}
                record = {**row, "curve": curve, "status": "computed", "window_protocol": "unchanged_author_MITBatteryClass_stage_selection",
                          "mirror_precision": "I/V/Q float32, t float64", **values}
                for name, column in columns.items():
                    record["released_" + name] = float(truth[column])
                    record["delta_" + name] = values[name] - float(truth[column])
                result.append({key: value if not isinstance(value, (float, np.floating)) or np.isfinite(value) else None
                               for key, value in record.items()})
            print("SIGNATURE " + Path(csv_path).name + " row=" + row["csv_row_zero_based"], flush=True)
        summary = {"scope": "prespecified source signature audit using a third-party derived HDF5 mirror",
                   "selected_rows": len(selected), "selected_CSVs": len(published), "selection_uses_feature_error": False,
                   "selection": "first rows per CSV and first uniquely mapped other-physical-cell segment rows, or explicit all_rows",
                   "reader_sha256": sha(reader_path), "extractor_sha256": sha(Path(__file__)),
                   "candidate_helper_sha256": sha(Path(__file__).with_name("audit_xjtu_feature_sample.py")),
                   "source_checksum": checksum,
                   "native_current_unit_conflict": "Mirror documents I as C-rate; author reader labels A and uses unscaled numeric I. This audit preserves numeric values and does not settle the physical unit.",
                   "slope_status": "All candidate slopes retained; no slope definition chosen by feature residual",
                   "records": result}
        for date, reader in remote.items():
            summary["source_checksum"][date].update(downloaded_range_bytes=reader.downloaded, ranges=reader.ranges)
        (output / "MIT_mirror_feature_signatures.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        with (output / "MIT_mirror_feature_signatures.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, list(dict.fromkeys(key for row in result for key in row)))
            writer.writeheader()
            writer.writerows(result)
        print(json.dumps({"selected_rows": len(selected), "feature_records": len(result), "output": str(output)}))
    finally:
        for store in stores.values():
            store.close()
        for reader in remote.values():
            reader.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirror", type=Path, default=OUTPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT / "signatures")
    parser.add_argument("--files", nargs="*")
    parser.add_argument("--first-rows-per-csv", type=int, default=3)
    parser.add_argument("--all-rows", action="store_true")
    parser.add_argument("--remote", action="store_true")
    run(parser.parse_args())
