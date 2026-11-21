"""Print progress from a PINN4SOH reproduction manifest without importing torch."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("/workspace/pinn4soh_main_20261109"))
    options = parser.parse_args()
    output = options.output.resolve()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    runs = manifest["runs"]
    statuses = Counter(run["status"] for run in runs.values())
    print("PINN4SOH MAIN RUN | " + datetime.now(timezone.utc).isoformat())
    print("Complete: {complete}/110 | Running: {running} | Failed: {failed}".format(
        complete=statuses["complete"], running=statuses["running"], failed=statuses["failed"]))
    print("Dataset  Complete/Planned  Mean Seconds  Mean MAE (SOH pp)")
    for dataset, total in [("XJTU", 60), ("MIT", 10), ("HUST", 10), ("TJU", 30)]:
        done = [run for run in runs.values()
                if run["dataset"] == dataset and run["status"] == "complete"]
        seconds = sum(run["training_seconds"] for run in done) / len(done) if done else 0
        mae = sum(run["metrics"]["MAE_pp"] for run in done) / len(done) if done else 0
        print(f"{dataset:7} {len(done):3}/{total:<8} {seconds:12.2f} {mae:18.4f}")
    for key, run in runs.items():
        if run["status"] == "failed":
            print("FAILED " + key + ": " + run.get("error", "unknown"))
        if run["status"] == "running":
            log_path = output / run["relative_folder"] / "logging.txt"
            lines = log_path.read_text(encoding="utf-8").splitlines() if log_path.exists() else []
            epochs = [line for line in lines if re.search(r"\[Train\] epoch:", line)]
            print("ACTIVE " + key)
            print(epochs[-1] if epochs else "Loading data / initializing model")
            tests = [line for line in lines if "[Test]" in line]
            if tests:
                print(tests[-1])
    print("Output: " + str(output))


if __name__ == "__main__":
    main()
