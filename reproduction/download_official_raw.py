"""Resume official raw archives with bounded storage and authoritative checksums."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import requests


def status(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def digest_file(path, algorithm):
    result = hashlib.new(algorithm)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def download(record, output, direct, budget):
    output.mkdir(parents=True, exist_ok=True)
    target = output / record["dataset"] / record["name"]
    target.parent.mkdir(exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    status_path = target.with_suffix(".status.json")
    session = requests.Session()
    session.trust_env = not direct
    started = time.time()
    state = dict(dataset=record["dataset"], name=record["name"], expected_bytes=record["compressed_bytes"],
                 official_checksum=record["checksum"], source_url=record["download_url"], status="downloading",
                 started_at_unix=started)
    algorithm, expected = record["checksum"].split(":", 1)
    if target.exists():
        if target.stat().st_size != record["compressed_bytes"] or digest_file(target, algorithm) != expected:
            raise ValueError("Existing archive does not match official checksum: " + target.name)
    else:
        last_report = 0
        for attempt in range(1, 7):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == record["compressed_bytes"]:
                break
            if offset > record["compressed_bytes"]:
                raise ValueError("Partial archive exceeds official size")
            state.update(attempt=attempt, downloaded_bytes=offset)
            status(status_path, state)
            try:
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                with session.get(record["download_url"], headers=headers, stream=True, timeout=(30, 60)) as response:
                    response.raise_for_status()
                    if offset and (response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                        raise ValueError("Server did not honor resume Range; partial data preserved")
                    with partial.open("ab") as stream:
                        for chunk in response.iter_content(1024 * 1024):
                            stream.write(chunk)
                            offset += len(chunk)
                            if offset > record["compressed_bytes"]:
                                raise ValueError("Download exceeds official size")
                            if time.time() - last_report > 15:
                                used = sum(item.stat().st_size for item in output.rglob("*") if item.is_file())
                                if used > budget:
                                    raise ValueError("Raw-data storage budget exceeded")
                                state.update(downloaded_bytes=offset, elapsed_seconds=time.time() - started,
                                             average_bytes_per_second=(offset - state.get("initial_bytes", 0)) / max(1, time.time() - started))
                                status(status_path, state)
                                print(json.dumps(state), flush=True)
                                last_report = time.time()
                if offset != record["compressed_bytes"]:
                    raise ValueError("Download ended before expected size")
                break
            except (requests.RequestException, OSError, ValueError) as error:
                state.update(last_error=repr(error), downloaded_bytes=partial.stat().st_size if partial.exists() else 0)
                status(status_path, state)
                if attempt == 6 or isinstance(error, ValueError):
                    state["status"] = "failed"
                    status(status_path, state)
                    raise
                time.sleep(min(attempt * 2, 10))
        actual = digest_file(partial, algorithm)
        if actual != expected:
            state.update(status="checksum_failed", actual_checksum=actual)
            status(status_path, state)
            raise ValueError("Official archive checksum mismatch")
        partial.replace(target)
    state.update(status="verified", downloaded_bytes=target.stat().st_size, sha256=digest_file(target, "sha256"),
                 official_checksum_passed=True, elapsed_seconds=time.time() - started)
    status(status_path, state)
    print(json.dumps(state), flush=True)
    return state


def group(records, output, direct, budget):
    for record in records:
        download(record, output, direct, budget)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=Path("outputs/full_reproduction/feature_audit/official_raw_inventory.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/full_reproduction/raw_data"))
    parser.add_argument("--datasets", nargs="+", choices=["XJTU", "TJU", "HUST"], default=["XJTU", "TJU", "HUST"])
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--budget-gb", type=float, default=25)
    options = parser.parse_args()
    records = json.loads(options.inventory.read_text(encoding="utf-8"))
    with ThreadPoolExecutor(max_workers=len(options.datasets)) as pool:
        futures = [pool.submit(group, [record for record in records if record["dataset"] == dataset],
                               options.output.resolve(), options.direct, int(options.budget_gb * 1e9))
                   for dataset in options.datasets]
        for future in futures:
            future.result()
