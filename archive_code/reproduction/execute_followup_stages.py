"""Wait for the fixed paper pipeline, then run separate diagnostics and audits."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main_ready(root):
    path = root / "pinn4soh_full_paper_pipeline.json"
    if not path.exists():
        return False
    stages = read_json(path).get("stages", [])
    if any(stage.get("status") == "failed" for stage in stages):
        raise RuntimeError("The original pipeline failed; inspect it before running follow-ups")
    expected = ["smoke_plan.json", "stage1_plan.json", "stage2_reproduced_transfer_plan.json"]
    if [stage.get("plan") for stage in stages] != expected:
        return False
    if not all(stage.get("status") == "complete" and stage.get("returncode") == 0 for stage in stages):
        return False
    for folder, count in [("pinn4soh_full_paper_stage1", 1880), ("pinn4soh_full_paper_stage2", 240)]:
        state = read_json(root / folder / "manifest.json")
        counts = Counter(item["status"] for item in state["tasks"].values())
        if state.get("status") != "complete" or counts != {"complete": count}:
            raise RuntimeError("Pipeline and suite manifests disagree for " + folder)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace"))
    parser.add_argument("--tools", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--training-tools", type=Path, default=Path("/workspace/full_tools_v1"))
    parser.add_argument("--report-tools", type=Path, default=Path("/workspace/full_report_tools"))
    parser.add_argument("--repo", type=Path, default=Path("/workspace/PINN4SOH-main"))
    args = parser.parse_args()
    root = args.root
    status_path = root / "pinn4soh_followup_pipeline.json"
    if status_path.exists():
        raise ValueError("Follow-up status already exists; inspect before starting another writer")
    status = {"started_at": datetime.now(timezone.utc).isoformat(), "status": "waiting_for_main", "stages": []}

    def save():
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
        temporary.replace(status_path)

    def run(name, command):
        record = {"name": name, "command": [str(value) for value in command], "status": "running",
                  "started_at": datetime.now(timezone.utc).isoformat()}
        status["stages"].append(record)
        status["status"] = "running"
        save()
        print("STAGE " + name, flush=True)
        record["returncode"] = subprocess.call(record["command"])
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        record["status"] = "complete" if record["returncode"] == 0 else "failed"
        save()
        if record["returncode"]:
            raise RuntimeError("Follow-up failed: " + name)

    save()
    try:
        print("Waiting for all 2120 main tasks to pass", flush=True)
        while not main_ready(root):
            time.sleep(10)
        plans = args.tools / "MIT_physical_purity_sensitivity"
        for filename, folder, smoke in [
            ("mit_physical_purity_smoke_plan.json", "pinn4soh_mit_physical_purity_smoke_20261110", True),
            ("mit_physical_purity_plan.json", "pinn4soh_mit_physical_purity_sensitivity_20261110", False),
        ]:
            command = [sys.executable, "-u", args.training_tools / "run_full_suite.py", "--plan", plans / filename,
                       "--repo", args.repo, "--output", root / folder, "--gpus", "0", "1",
                       "--workers-per-gpu", "3", "--threads", "2", "--resume"]
            if smoke:
                command.append("--smoke")
            run(folder, command)
        run("formal_cpu_efficiency", [sys.executable, args.report_tools / "benchmark_efficiency.py",
            "--repo", args.repo, "--output", root / "efficiency_formal_20261110", "--threads", "1", "2",
            "--batch-size", "1000", "--forwards", "1000", "--warmup", "50", "--repeats", "5"])
        report_folder = root / "pinn4soh_full_report_20261110"
        run("independent_main_report", [sys.executable, args.report_tools / "build_suite_report.py",
            "--suite", root / "pinn4soh_full_paper_stage1", "--suite", root / "pinn4soh_full_paper_stage2",
            "--plan", args.report_tools / "stage1_plan.json", "--plan", args.report_tools / "stage2_reproduced_transfer_plan.json",
            "--protocol", args.report_tools / "paper_protocol.json", "--output", report_folder])
        run("main_evidence_export", [sys.executable, args.report_tools / "export_suite_evidence.py",
            "--report", report_folder, "--output", root / "pinn4soh_full_evidence_20261110"])
        run("main_text_evidence_pack", [sys.executable, args.tools / "collect_suite_evidence.py", "pack",
            "--source", root / "pinn4soh_full_evidence_20261110", "--output", root / "pinn4soh_full_text_evidence_20261110",
            "--max-mb", "28"])
        run("separate_mit_report", [sys.executable, args.report_tools / "build_suite_report.py",
            "--suite", root / "pinn4soh_mit_physical_purity_sensitivity_20261110",
            "--plan", plans / "mit_physical_purity_plan.json", "--protocol", args.report_tools / "paper_protocol.json",
            "--output", root / "pinn4soh_mit_physical_purity_report_20261110"])
        status["status"] = "complete"
        status["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()
        print("ALL FOLLOW-UP STAGES COMPLETE", flush=True)
    except BaseException as error:
        status.update(status="failed", error=repr(error), finished_at=datetime.now(timezone.utc).isoformat())
        save()
        raise


if __name__ == "__main__":
    main()
