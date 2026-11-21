"""Validate HUST's restricted data reader on one official bounded ZIP member."""

import hashlib
import argparse
import io
import json
from pathlib import Path
import time
import traceback
import zipfile

from audit_raw_data_inventory import RemoteZip, SESSION
from extract_raw_feature_candidates import candidates, load_hust
from summarize_raw_feature_residuals import compare


def main(max_cycles):
    root = Path("outputs/full_reproduction/feature_audit").resolve()
    record = next(item for item in json.loads((root / "official_raw_inventory.json").read_text()) if item["dataset"] == "HUST")
    selected = next(item for item in record["archive_members"] if item["name"] == "our_data/2-5.pkl")
    target = root / "raw_sample" / "HUST_2-5.pkl"
    state = dict(source_url=record["download_url"], raw_member=selected["name"], crc32=selected["crc32"],
                 compressed_bytes=selected["compressed_bytes"], uncompressed_bytes=selected["uncompressed_bytes"],
                 scope=(f"First {max_cycles} raw cycles" if max_cycles else "All raw cycles") + " of one HUST battery; full archive checksum not established by a ranged member")
    started = time.time()
    try:
        if not target.exists():
            SESSION.trust_env = False
            remote = RemoteZip(record["download_url"], record["compressed_bytes"], limit=64 * 1024 * 1024)
            with zipfile.ZipFile(remote) as archive:
                content = archive.read(selected["name"])
            target.write_bytes(content)
            state["range_downloaded_bytes"] = remote.downloaded
        content = target.read_bytes()
        state["sha256"] = hashlib.sha256(content).hexdigest()

        class SampleArchive:
            def open(self, name):
                if name != selected["name"]:
                    raise ValueError("Wrong HUST sample member")
                return io.BytesIO(content)

        total, all_rows = 0, []
        for index, cycle in enumerate(load_hust(SampleArchive(), selected["name"])):
            rows = candidates(cycle)
            total += len(rows)
            all_rows.extend(rows)
            if max_cycles and index == max_cycles - 1:
                break
        state.update(status="passed", inspected_cycles=index + 1, candidate_records=total)
        if not max_cycles:
            import pandas as pd
            sample_dir = root / "HUST_ranged_member_feature_check"
            sample_dir.mkdir(exist_ok=True)
            pd.DataFrame(all_rows).to_csv(sample_dir / "candidate_features.csv", index=False)
            alignment, residuals = compare(dict(dataset="HUST", processed_csv="HUST data/2-5.csv"), sample_dir, Path("PINN4SOH").resolve())
            state["alignment"] = alignment
            pd.DataFrame(residuals).to_csv(sample_dir / "aligned_formula_residuals.csv", index=False)
    except Exception as error:
        state.update(status="failed", error=repr(error))
        (root / "HUST_raw_sample_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    state["seconds"] = time.time() - started
    (root / "HUST_raw_sample_audit.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-cycles", type=int, default=20)
    main(parser.parse_args().max_cycles)
