"""Read-only provenance/mapping audit of a third-party MIT within-cycle mirror."""

import argparse
from collections import Counter, defaultdict
import bisect
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import http.client
import json
from pathlib import Path
import re
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/full_reproduction/feature_audit/MIT_mirror_audit"
GITHUB = "https://raw.githubusercontent.com/tayebelleuch-ops/deep-analysis-of-MIT-TOYOTA-dataset/main/"
ZENODO = "https://zenodo.org/api/records/23984188"
FILES = {"zenodo_record.json": ZENODO,
         "github_README.md": GITHUB + "README.md", "github_Data_README.md": GITHUB + "Data/README.md",
         "github_MANIFEST_v1.1.md": GITHUB + "exploration/MANIFEST_v1.1.md",
         "github_MANIFEST.md": GITHUB + "exploration/MANIFEST.md",
         "github_matr.py": GITHUB + "exploration/matr.py",
         "github_extract_within_cycle.py": GITHUB + "exploration/extract_within_cycle.py",
         "github_cell_inventory.csv": GITHUB + "exploration/cell_inventory.csv",
         "github_analysis_set_cells.csv": GITHUB + "exploration/analysis_set_cells.csv"}
for name in ["EXTRACTION.md", "LICENSE.txt", "SHA256SUMS.txt", "within_cycle_cells.csv", "within_cycle_index.csv"]:
    FILES[name] = "https://zenodo.org/records/23984188/files/" + name + "?download=1"


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def download(url, path):
    if path.is_file():
        return
    request = urllib.request.Request(url, headers={"User-Agent": "PINN4SOH-reproduction-provenance-audit"})
    with urllib.request.urlopen(request, timeout=90) as source, path.with_suffix(path.suffix + ".part").open("wb") as output:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            output.write(block)
    path.with_suffix(path.suffix + ".part").replace(path)


def download_ranges(url, path, size, checksum, chunk_bytes=8 * 1024 * 1024, workers=1):
    """Resume public Zenodo range downloads; validate every Content-Range."""
    if path.is_file() and path.stat().st_size == size and sha(path) == checksum:
        return
    temporary = path.with_suffix(path.suffix + ".ranged.part")
    position = temporary.stat().st_size if temporary.exists() else 0
    if position > size:
        raise ValueError("Partial download is larger than published size")
    def fetch(start):
        stop = min(size - 1, start + chunk_bytes - 1)
        for attempt in range(4):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "PINN4SOH-reproduction-audit",
                    "Range": "bytes=" + str(start) + "-" + str(stop)})
                with urllib.request.urlopen(request, timeout=60) as source:
                    if source.status != 206 or source.headers.get("Content-Range") != f"bytes {start}-{stop}/{size}":
                        raise ValueError("Server range response disagrees with requested bytes")
                    data = source.read()
                if len(data) != stop - start + 1:
                    raise ValueError("Incomplete range response")
                return data
            except (OSError, ValueError, http.client.HTTPException) as error:
                print(f"RETRY {path.name} byte={start} attempt={attempt + 1} {type(error).__name__}: {error}", flush=True)
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))

    with temporary.open("ab") as output, ThreadPoolExecutor(max_workers=workers) as executor:
        while position < size:
            starts = list(range(position, min(size, position + workers * chunk_bytes), chunk_bytes))
            # map yields in requested order; only the owner thread appends verified blocks.
            for data in executor.map(fetch, starts):
                output.write(data)
                output.flush()
                position += len(data)
                print(f"RANGE {path.name} {position}/{size}", flush=True)
    if sha(temporary) != checksum:
        raise ValueError("Published SHA256 mismatch after range download")
    temporary.replace(path)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--download-h5", action="store_true")
    parser.add_argument("--h5-date", action="append", choices=["2017-05-12", "2017-06-30", "2018-04-12"])
    parser.add_argument("--range-mib", type=int, choices=[1, 2, 4, 8], default=8)
    parser.add_argument("--range-workers", type=int, choices=[1, 2, 3, 4], default=1)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not args.skip_download:
        for name, url in FILES.items():
            print("FETCH " + name, flush=True)
            if name == "within_cycle_index.csv":
                download_ranges(url, out / name, 72846386, "93176963bbaaaea60a46591a3e7d1e5c3f1a04a250107eec1b754a3311b207dc")
            else:
                download(url, out / name)
    record = json.loads((out / "zenodo_record.json").read_text(encoding="utf-8"))
    expected = {}
    for line in (out / "SHA256SUMS.txt").read_text().splitlines():
        checksum, size, name = line.split(maxsplit=2)
        expected[Path(name.strip().lstrip("*")).name] = checksum
    if args.download_h5:
        def fetch_h5(entry):
            name = entry["key"]
            download_ranges("https://zenodo.org/records/23984188/files/" + name + "?download=1",
                            out / name, entry["size"], expected[name], chunk_bytes=args.range_mib * 1024 * 1024,
                            workers=args.range_workers)
        with ThreadPoolExecutor(max_workers=3) as executor:
            list(executor.map(fetch_h5, [entry for entry in record["files"] if entry["key"].endswith(".h5")
                and (not args.h5_date or any(date in entry["key"] for date in args.h5_date))]))
    provenance = []
    for name, url in FILES.items():
        path = out / name
        checksum = sha(path)
        if name in expected and checksum != expected[name]:
            raise ValueError("Published SHA256 mismatch: " + name)
        provenance.append({"name": name, "url": url, "sha256": checksum, "bytes": path.stat().st_size,
                           "published_sha256": expected.get(name), "published_hash_verified": name in expected})
    h5_provenance = []
    for entry in record["files"]:
        name = entry["key"]
        if not name.endswith(".h5"):
            continue
        path = out / name
        checksum = sha(path) if path.is_file() else None
        if checksum is not None and (checksum != expected[name] or path.stat().st_size != entry["size"]):
            raise ValueError("Complete HDF5 differs from published size/hash: " + name)
        h5_provenance.append({"name": name, "published_bytes": entry["size"], "published_sha256": expected[name],
            "local_complete_file": path.is_file(), "local_sha256": checksum,
            "published_hash_verified": checksum is not None,
            "url": "https://zenodo.org/records/23984188/files/" + name + "?download=1"})
    source_rows, by_struct = [], defaultdict(list)
    with (out / "within_cycle_index.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        for row in reader:
            key = row["source_batch_date"], int(row["source_struct_index"])
            by_struct[key].append({key: row[key] for key in ["cell_uid", "cycle", "segment", "source_batch_date",
                "source_channel", "source_struct_index", "source_cycle_index", "h5_row", "QDischarge_Ah"]})
    import numpy as np
    by_uid = defaultdict(list)
    for values in by_struct.values():
        by_uid[values[0]["cell_uid"]].extend(values)
    struct_capacity = {}
    for key, values in by_struct.items():
        values = by_uid[values[0]["cell_uid"]] if min(int(row["segment"]) for row in values) == 1 else values
        source = sorted(values, key=lambda row: (int(row["segment"]), int(row["source_cycle_index"])))
        buckets = defaultdict(list)
        for index, row in enumerate(source):
            buckets[round(float(row["QDischarge_Ah"]) * 1e6)].append(index)
        struct_capacity[key] = source, buckets
    global_capacity = defaultdict(list)
    for values in by_struct.values():
        for row in values:
            global_capacity[round(float(row["QDischarge_Ah"]) * 1e6)].append(row)
    mapping, cycle_mapping, unresolved = [], [], []
    for path in sorted((ROOT / "PINN4SOH/data/MIT data").rglob("*.csv")):
        batch, position = path.parent.name, int(re.search(r"battery-(\d+)\.csv$", path.name)[1])
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        capacity = np.array([float(row["capacity"]) for row in rows])
        candidates = []
        candidate_maps = {}
        for (source_batch, zero_based), (source, buckets) in struct_capacity.items():
            if source_batch != batch:
                continue
            matched, last, errors = [], -1, []
            for csv_row, value in enumerate(capacity):
                value = float(value)
                key = round(value * 1e6)
                possible = []
                for neighbor in [key - 1, key, key + 1]:
                    indices = buckets.get(neighbor, [])
                    start = bisect.bisect_right(indices, last)
                    possible.extend(index for index in indices[start:]
                                    if abs(float(source[index]["QDischarge_Ah"]) - value) <= 1e-8)
                if possible:
                    index = min(possible)
                    errors.append(abs(float(source[index]["QDischarge_Ah"]) - value))
                    matched.append((csv_row, source[index]))
                    last = index
            candidate_maps[zero_based] = matched
            candidates.append({"struct_index_zero_based": zero_based,
                               "capacity_max_abs_error": max(errors) if errors else None,
                               "capacity_mean_abs_error": float(np.mean(errors)) if errors else None,
                               "n_compared": len(capacity), "n_matched_ordered": len(matched), "n_source": len(source),
                               "source_cell_uid": source[0]["cell_uid"], "source_channel": int(source[0]["source_channel"]),
                               "source_cycle_index_first": int(matched[0][1]["source_cycle_index"]) if matched else None,
                               "analysis_cycle_first": int(matched[0][1]["cycle"]) if matched else None,
                               "source_cycle_index_last": int(matched[-1][1]["source_cycle_index"]) if matched else None,
                               "source_batch_last": matched[-1][1]["source_batch_date"] if matched else None,
                               "n_source_segments": len({row["segment"] for row in source})})
        candidates.sort(key=lambda item: (-item["n_matched_ordered"], item["capacity_max_abs_error"] or 0))
        best = candidates[0] if candidates else None
        current_mapping, local_unresolved = [], []
        if best:
            matched = candidate_maps[best["struct_index_zero_based"]]
            for csv_row, source in matched:
                current_mapping.append({"csv": str(path.relative_to(ROOT)), "csv_row_zero_based": csv_row,
                                        "capacity": float(capacity[csv_row]), "match_type": "ordered_expected_physical_cell", **source})
            matched_indices = {row for row, _ in matched}
            for csv_row, value in enumerate(capacity):
                if csv_row in matched_indices:
                    continue
                key = round(float(value) * 1e6)
                possible = [row for neighbor in [key - 1, key, key + 1] for row in global_capacity.get(neighbor, [])
                            if abs(float(row["QDischarge_Ah"]) - float(value)) <= 1e-12]
                if len(possible) == 1:
                    current_mapping.append({"csv": str(path.relative_to(ROOT)), "csv_row_zero_based": csv_row,
                                            "capacity": float(value), "match_type": "unique_exact_capacity_other_source", **possible[0]})
                else:
                    local_unresolved.append({"csv": str(path.relative_to(ROOT)), "csv_row_zero_based": csv_row,
                                             "capacity": float(value), "n_exact_source_candidates": len(possible),
                                             "exact_source_candidates": possible})
        current_mapping.sort(key=lambda row: row["csv_row_zero_based"])
        for issue in local_unresolved:
            before = [row for row in current_mapping if row["csv_row_zero_based"] < issue["csv_row_zero_based"]]
            after = [row for row in current_mapping if row["csv_row_zero_based"] > issue["csv_row_zero_based"]]
            if before and after:
                left, right = before[-1], after[0]
                same_segment = lambda row: all(row[key] == left[key] == right[key]
                    for key in ["cell_uid", "source_batch_date", "source_struct_index"])
                possible = [row for row in issue["exact_source_candidates"] if same_segment(row) and
                            int(left["source_cycle_index"]) < int(row["source_cycle_index"]) < int(right["source_cycle_index"])]
                if len(possible) == 1:
                    current_mapping.append({"csv": issue["csv"], "csv_row_zero_based": issue["csv_row_zero_based"],
                                            "capacity": issue["capacity"], "match_type": "exact_capacity_and_neighbor_constrained_segment", **possible[0]})
                    continue
            unresolved.append(issue)
        current_mapping.sort(key=lambda row: row["csv_row_zero_based"])
        cycle_mapping.extend(current_mapping)
        uid_counts = Counter(row["cell_uid"] for row in current_mapping)
        mapping.append({"csv": str(path.relative_to(ROOT)), "csv_sha256": sha(path), "csv_battery_number": position,
                        "csv_rows": len(rows), "batch": batch, "best_candidate": best,
                        "capacity_ordered_match_1e-12": bool(best is not None and best["n_matched_ordered"] == len(capacity) and best["capacity_max_abs_error"] <= 1e-12),
                        "capacity_ordered_match_1e-8": bool(best is not None and best["n_matched_ordered"] == len(capacity) and best["capacity_max_abs_error"] <= 1e-8),
                        "all_rows_capacity_sourced": len(current_mapping) == len(capacity),
                        "physical_cell_uid_counts": dict(uid_counts), "contains_multiple_physical_cells": len(uid_counts) > 1,
                        "runner_up_matched": candidates[1]["n_matched_ordered"] if len(candidates) > 1 else None,
                        "tested_candidates": candidates})
    protocol = json.loads((ROOT / "outputs/full_reproduction/paper_protocol.json").read_text(encoding="utf-8"))
    test_ids = protocol["supplement_protocol"]["test_batteries"]
    uid_roles = defaultdict(lambda: defaultdict(set))
    for row in cycle_mapping:
        path = Path(row["csv"])
        date = path.parent.name
        position = int(re.search(r"battery-(\d+)\.csv$", path.name)[1])
        role = "test" if position in test_ids["MIT/" + date] else "train_validation"
        uid_roles[row["cell_uid"]][role].add(row["csv"])
    physical_overlap = [{"cell_uid": uid, "csv_roles": {role: sorted(paths) for role, paths in roles.items()},
                         "evidence_scope": "capacity provenance; feature vectors require raw within-cycle comparison"}
                        for uid, roles in uid_roles.items() if "test" in roles and "train_validation" in roles]
    audit = {"generated_at": datetime.now(timezone.utc).isoformat(), "record_doi": record["doi"],
             "record_license": record["metadata"]["license"], "provenance": provenance,
             "within_cycle_h5_files": h5_provenance,
             "index_columns": columns, "index_struct_count": len(by_struct),
             "index_cycle_count": sum(len(rows) for rows in by_struct.values()),
             "per_batch_structs": dict(Counter(batch for batch, _ in by_struct)),
             "PINN_csv_count": len(mapping), "PINN_csv_batches": dict(Counter(item["batch"] for item in mapping)),
             "capacity_ordered_exact_1e-12": sum(item["capacity_ordered_match_1e-12"] for item in mapping),
             "capacity_ordered_match_1e-8": sum(item["capacity_ordered_match_1e-8"] for item in mapping),
             "csv_cycle_rows": sum(item["csv_rows"] for item in mapping),
             "mapped_cycle_rows": len(cycle_mapping),
             "all_rows_capacity_sourced_cells": sum(item["all_rows_capacity_sourced"] for item in mapping),
             "multiple_physical_cell_CSVs": [item["csv"] for item in mapping if item["contains_multiple_physical_cells"]],
             "capacity_provenance_physical_train_test_overlap": physical_overlap,
             "unresolved_capacity_rows": unresolved,
             "position_offsets": dict(Counter(str(item["best_candidate"]["struct_index_zero_based"] - item["csv_battery_number"])
                                          for item in mapping if item["best_candidate"])),
             "limitations": ["Third-party derived HDF5 is not the original MAT", "Native I/V/Q fields narrowed to float32; t float64",
                              "Capacity matching verifies identity/order only, not all16 feature extraction", "Raw URL availability remains separately probed"],
             "mapping": mapping}
    write_json(out / "MIT_mirror_mapping_audit.json", audit)
    with (out / "PINN_CSV_to_mirror_struct.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        rows = [{key: value for key, value in item.items() if key not in {"tested_candidates", "best_candidate"}} |
                (item["best_candidate"] or {}) for item in mapping]
        writer = csv.DictWriter(stream, list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    if cycle_mapping:
        with (out / "PINN_CSV_to_mirror_cycles.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, list(cycle_mapping[0]))
            writer.writeheader()
            writer.writerows(cycle_mapping)
    print(json.dumps({key: audit[key] for key in ["index_struct_count", "index_cycle_count", "PINN_csv_count",
        "capacity_ordered_exact_1e-12", "capacity_ordered_match_1e-8", "csv_cycle_rows", "mapped_cycle_rows", "position_offsets"]}), flush=True)


if __name__ == "__main__":
    main()
