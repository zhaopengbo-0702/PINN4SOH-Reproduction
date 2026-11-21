"""Parallel scheduling only; use the unchanged frozen feature extractor per cell."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

from extract_raw_feature_candidates import process, raw_requirements, write_json
from download_official_raw import digest_file


def run(options):
    output = options.output.resolve()
    inventory = json.loads((output / "official_raw_inventory.json").read_text(encoding="utf-8"))
    requirements = raw_requirements(options.repo.resolve(), inventory, output)
    implementation = {name: digest_file(Path(__file__).with_name(name), "sha256")
                      for name in ("extract_raw_feature_candidates.py", "audit_xjtu_feature_sample.py")}
    tasks, completed = [], []
    for record in inventory:
        if record["dataset"] not in options.datasets:
            continue
        archive = options.raw_root.resolve() / record["dataset"] / record["name"]
        status = json.loads(archive.with_suffix(".status.json").read_text(encoding="utf-8"))
        algorithm, expected = record["checksum"].split(":", 1)
        if status["status"] != "verified" or digest_file(archive, algorithm) != expected:
            raise ValueError("Official archive validation failed")
        for requirement in requirements:
            if requirement["archive"] != record["name"]:
                continue
            identity = hashlib.sha256(requirement["processed_csv"].encode()).hexdigest()[:12]
            manifest = output / "raw_extraction" / record["dataset"] / (Path(requirement["raw_member"]).stem + "_" + identity) / "manifest.json"
            if manifest.exists():
                previous = json.loads(manifest.read_text(encoding="utf-8"))
                if previous["status"] == "complete":
                    if previous["implementation_sha256"] != implementation:
                        raise ValueError("Existing task has different frozen implementation")
                    completed.append(previous)
                    continue
                if previous["status"] == "running":
                    write_json(manifest.with_name("interrupted_before_partition.json"),
                               dict(status="aborted", interrupted_state=previous,
                                    reason="Previous HUST-only worker stopped after observed cell boundary; incomplete next cell recomputed",
                                    preserved_at_unix=time.time()))
            tasks.append((record, requirement, archive, output, output / "preprocessing_source", implementation))
    started = time.time()
    total = len(tasks) + len(completed)
    scheduling = dict(scheduler="partition-only", datasets=options.datasets, pending=len(tasks),
                      already_complete=len(completed), total=total, workers=options.workers,
                      implementation=implementation,
                      scheduler_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(output / ("raw_partition_scheduler_" + "_".join(options.datasets) + ".json"), scheduling)
    print(json.dumps(scheduling), flush=True)
    with ProcessPoolExecutor(max_workers=options.workers) as executor:
        futures = {executor.submit(process, *task): task[1]["processed_csv"] for task in tasks}
        for future in as_completed(futures):
            completed.append(future.result())
            write_json(output / ("raw_partition_progress_" + "_".join(options.datasets) + ".json"),
                       dict(complete=sum(item["status"] == "complete" for item in completed),
                            failed=sum(item["status"] == "failed" for item in completed),
                            total=total,
                            elapsed_seconds=time.time() - started, cells=completed))
    print(json.dumps(dict(status="complete", cells=len(completed),
                          failures=sum(item["status"] != "complete" for item in completed),
                          elapsed_seconds=time.time() - started)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("PINN4SOH"))
    parser.add_argument("--output", type=Path, default=Path("outputs/full_reproduction/feature_audit"))
    parser.add_argument("--raw-root", type=Path, default=Path("outputs/full_reproduction/raw_data"))
    parser.add_argument("--datasets", nargs="+", choices=("XJTU", "TJU", "HUST"), default=["HUST"])
    parser.add_argument("--workers", type=int, default=3)
    run(parser.parse_args())
