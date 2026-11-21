"""Run fixed training stages sequentially; stop on failure and retain all outputs."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--repo", default="/workspace/PINN4SOH-main")
    parser.add_argument("--output-root", type=Path, default=Path("/workspace"))
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    stages = [("smoke_plan.json", "pinn4soh_full_paper_smoke", True)]
    if not args.smoke_only:
        stages += [("stage1_plan.json", "pinn4soh_full_paper_stage1", False),
                   ("stage2_reproduced_transfer_plan.json", "pinn4soh_full_paper_stage2", False)]
    status = dict(started_at=time.time(), stages=[])
    status_file = args.output_root / "pinn4soh_full_paper_pipeline.json"
    for plan, folder, smoke in stages:
        command = [sys.executable, "-u", str(args.tools / "run_full_suite.py"),
                   "--plan", str(args.tools / "plans" / plan), "--repo", args.repo,
                   "--output", str(args.output_root / folder), "--gpus", "0", "1",
                   "--workers-per-gpu", "3", "--threads", "2", "--resume"]
        if smoke:
            command.append("--smoke")
        print("STAGE " + plan, flush=True)
        record = dict(plan=plan, output=str(args.output_root / folder), started_at=time.time(), status="running")
        status["stages"].append(record)
        status_file.write_text(json.dumps(status, indent=2), encoding="utf-8")
        record["returncode"] = subprocess.call(command)
        record["finished_at"] = time.time()
        record["status"] = "complete" if record["returncode"] == 0 else "failed"
        status_file.write_text(json.dumps(status, indent=2), encoding="utf-8")
        if record["returncode"]:
            raise SystemExit(record["returncode"])
    print("ALL REQUESTED TRAINING STAGES COMPLETE", flush=True)


if __name__ == "__main__":
    main()
