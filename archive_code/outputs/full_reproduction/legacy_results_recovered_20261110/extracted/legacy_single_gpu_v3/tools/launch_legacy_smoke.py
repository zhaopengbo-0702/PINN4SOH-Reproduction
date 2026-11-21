"""Launch the separate 12-task legacy GPU smoke after strict runtime preflights."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from install_legacy_runtime import isolated_environment


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.gpus != ["0"] or args.workers_per_gpu != 1:
        raise ValueError("This deployment is limited to one allocated GPU (0) and one worker")
    prefix, support, repo, output = [path.resolve() for path in [args.prefix, args.support, args.repo, args.output]]
    for protected in [prefix, support, repo]:
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError("Diagnostic output must be separate from runtime, source and support")
    manifest = json.loads((support / "legacy_compat_manifest.json").read_text(encoding="utf-8"))
    compat, plan_path = support / "legacy_compat", support / "legacy_smoke_plan.json"
    if digest(plan_path) != manifest["plan_sha256"]:
        raise ValueError("Smoke plan hash changed")
    for name, record in manifest["runner_files"].items():
        if digest(compat / name) != record["compat_sha256"]:
            raise ValueError("Compatibility runner changed: " + name)
    tasks = json.loads(plan_path.read_text(encoding="utf-8"))["tasks"]
    if len(tasks) != 12 or any(task.get("smoke") is not True or task.get("epochs") != 2 for task in tasks):
        raise ValueError("Only the explicit 12 two-epoch smoke tasks are authorized by this launcher")
    environment = isolated_environment(prefix)
    python = prefix / "bin/python"
    command = [str(python), "-I", str(compat / "run_full_suite.py"), "--plan", str(plan_path),
        "--repo", str(repo), "--output", str(output / "suite"), "--gpus", *args.gpus,
        "--workers-per-gpu", str(args.workers_per_gpu), "--threads", "2", "--smoke"]
    if args.resume:
        command.append("--resume")
    if args.dry_run:
        print(json.dumps(dict(training_started=False, command=command, cleared_pythonpath=True,
                             task_count=12, gpu_preflight_required=True), indent=2))
        return 0
    if sys.platform != "linux":
        raise RuntimeError("Install and run the legacy smoke on Linux x86_64")
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("Use a new diagnostic output directory, or --resume identical inputs")
    output.mkdir(parents=True, exist_ok=True)
    for gpu in args.gpus:
        assigned = dict(environment, CUDA_VISIBLE_DEVICES=gpu)
        preflight = [str(python), "-I", str(Path(__file__).with_name("probe_legacy_runtime.py")),
            "--prefix", str(prefix), "--output", str(output / ("preflight_gpu_" + gpu.replace("/", "_") + ".json")), "--require-cuda"]
        subprocess.run(preflight, env=assigned, check=True)
    launch = dict(command=command, prefix=str(prefix), diagnostic_only=True, smoke=True, task_count=12,
                  output=str(output), plan_sha256=digest(plan_path), launcher_sha256=digest(__file__),
                  interpretation="Two epochs prove executable paths and checkpoint reload; no convergence or accuracy claim")
    (output / "launch.json").write_text(json.dumps(launch, indent=2), encoding="utf-8")
    result = subprocess.run(command, env=environment)
    summary_path = output / "suite/manifest.json"
    state = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    counts = Counter(row["status"] for row in state.get("tasks", {}).values())
    evidence = dict(returncode=result.returncode, status=state.get("status", "missing"), counts=dict(counts),
        smoke_passed=result.returncode == 0 and counts.get("complete") == 12 and len(state.get("tasks", {})) == 12,
        scientific_accuracy_assessed=False, formal_experiment_recommendation="Decide independent matched original-runtime controls only after reviewing this smoke")
    (output / "legacy_smoke_summary.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence))
    return 0 if evidence["smoke_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
