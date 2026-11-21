"""Run only the frozen 40-task matched control after the separate 12-task GPU smoke."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from install_legacy_runtime import isolated_environment


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--smoke-evidence", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.gpus != ["0"] or args.workers_per_gpu != 1:
        raise ValueError("This deployment is limited to one allocated GPU (0) and one worker")
    prefix, support, control, repo = [path.resolve() for path in [args.prefix, args.support, args.control, args.repo]]
    compat, plan_path = support / "legacy_compat", control / "legacy_control_plan.json"
    design = json.loads((control / "legacy_control_design.json").read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if digest(plan_path) != design["plan_sha256"] or len(plan["tasks"]) != 40:
        raise ValueError("Formal control differs from frozen design")
    output = Path(plan["expected_suite_output"]).resolve()
    for protected in [prefix, support, control, repo]:
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError("Formal output must be separate from runtime, tools and source")
    manifest = json.loads((support / "legacy_compat_manifest.json").read_text(encoding="utf-8"))
    for name, record in manifest["runner_files"].items():
        if digest(compat / name) != record["compat_sha256"]:
            raise ValueError("Compatibility runner changed")
    python = prefix / "bin/python"
    command = [str(python), "-I", str(compat / "run_full_suite.py"), "--plan", str(plan_path), "--repo", str(repo),
        "--output", str(output), "--gpus", *args.gpus, "--workers-per-gpu", str(args.workers_per_gpu), "--threads", "2"]
    if args.resume:
        command.append("--resume")
    if args.dry_run:
        print(json.dumps(dict(training_started=False, command=command, smoke_evidence_required=True), indent=2))
        return 0
    if sys.platform != "linux":
        raise RuntimeError("Original-runtime formal control must execute on Linux x86_64")
    smoke = json.loads(args.smoke_evidence.read_text(encoding="utf-8"))
    if not smoke.get("smoke_passed") or smoke.get("counts") != {"complete": 12} or smoke.get("returncode") != 0:
        raise ValueError("The 12-task original-runtime GPU smoke has not passed")
    smoke_suite = json.loads((args.smoke_evidence.parent / "suite/manifest.json").read_text(encoding="utf-8"))
    if smoke_suite.get("status") != "complete" or len(smoke_suite.get("tasks", {})) != 12 or any(
            row.get("status") != "complete" for row in smoke_suite["tasks"].values()):
        raise ValueError("Smoke summary and suite manifest disagree")
    evidence = args.smoke_evidence.parent / "formal_control_preflight"
    evidence.mkdir(exist_ok=True)
    environment = isolated_environment(prefix)
    for index, gpu in enumerate(args.gpus):
        subprocess.run([str(python), "-I", str(Path(__file__).with_name("probe_legacy_runtime.py")),
            "--prefix", str(prefix), "--output", str(evidence / ("gpu_" + str(index) + ".json")), "--require-cuda"],
            env=dict(environment, CUDA_VISIBLE_DEVICES=gpu), check=True)
    return subprocess.run(command, env=environment).returncode


if __name__ == "__main__":
    raise SystemExit(main())
