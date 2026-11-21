"""Launch only a reviewed selected-repeat extension, with repeat1 left untouched."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from install_legacy_runtime import isolated_environment
from prepare_legacy_extension import read, digest, require, verify_reused, runtime_signature, verify_composite, stable


def check_probe(current, frozen):
    require(current.get("status") == "passed" and current.get("assigned_visibility") == "0", "Assigned legacy GPU probe did not pass")
    for key, value in runtime_signature(frozen).items():
        actual = current.get("versions", {}).get(key) if key in {"torch", "numpy", "pandas", "scipy", "sklearn"} else current.get(key)
        require(actual == value, "Runtime differs from reused repeat1 probe: " + key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["prefix", "support", "extension", "legacy-control-suite", "repo"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    require(args.gpus == ["0"] and args.workers_per_gpu == 1, "One assigned GPU 0 and one worker required")
    prefix, support, extension, old, repo = [path.resolve() for path in
        [args.prefix, args.support, args.extension, args.legacy_control_suite, args.repo]]
    design = read(extension / "extension_design.json")
    plan_path, manifest_path = extension / "extension_plan.json", extension / "extension_analysis_manifest.json"
    require(digest(plan_path) == design["plan_sha256"] and digest(manifest_path) == design["analysis_manifest_sha256"], "Extension support changed")
    plan, analysis = read(plan_path), read(manifest_path)
    verify_composite(plan, analysis, design)
    require(len(plan["tasks"]) == design["new_task_count"] <= 360 and
            all(2 <= task["repeat"] <= 10 and not task.get("smoke") for task in plan["tasks"]), "Unexpected extension tasks")
    reuse = verify_reused(analysis["tasks"], old)
    require(reuse == analysis["reuse_evidence"], "Original repeat1 evidence changed after preparation")
    output = Path(plan["expected_suite_output"]).resolve()
    for protected in [prefix, support, extension, old, repo]:
        require(output != protected and protected not in output.parents and output not in protected.parents, "Output overlaps protected source")
    compat = support / "legacy_compat"
    compatibility = read(support / "legacy_compat_manifest.json")
    old_codes = {Path(path).name: checksum for path, checksum in reuse["control_frozen"]["implementation_sha256"].items()}
    for name, record in compatibility["runner_files"].items():
        require(digest(compat / name) == record["compat_sha256"] == old_codes.get(name), "Compatibility runner differs from original repeat1")
    python = prefix / "bin/python"
    command = [str(python), "-I", str(compat / "run_full_suite.py"), "--plan", str(plan_path), "--repo", str(repo),
               "--output", str(output), "--gpus", "0", "--workers-per-gpu", "1", "--threads", "2"]
    if args.resume:
        command.append("--resume")
    if args.dry_run:
        print(json.dumps({"training_started": False, "command": command,
                          "new_tasks": len(plan["tasks"]), "repeat1_reused": len(reuse["reused_tasks"])}, indent=2))
        return 0
    require(sys.platform == "linux", "Execute extension on allocated Linux GPU")
    environment = isolated_environment(prefix)
    script = ("import importlib.util,json,sys; s=importlib.util.spec_from_file_location('frozen_source',sys.argv[1]); "
              "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); from pathlib import Path; "
              "print(json.dumps(m.source_inventory(Path(sys.argv[2]))))")
    inventory_probe = subprocess.run([str(python), "-I", "-c", script, str(compat / "run_reproduction.py"), str(repo)],
        env=environment, check=True, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    inventory = json.loads(inventory_probe.stdout.strip().splitlines()[-1])
    require(inventory["fingerprint"] == reuse["control_frozen"]["source_fingerprint"] and
            stable(inventory["data_directory_order"]) == reuse["control_frozen"]["data_directory_order_sha256"],
            "Source/data inventory differs from immutable repeat1 before launch")
    # Every invocation gets a fresh evidence file; a resume never overwrites a
    # preceding failed/passed preflight observation.
    preflight_dir = extension / "runtime_preflight_attempts"
    preflight_dir.mkdir(exist_ok=True)
    preflight = preflight_dir / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    require(not preflight.exists(), "Preflight evidence path exists")
    subprocess.run([str(python), "-I", str(Path(__file__).with_name("probe_legacy_runtime.py")),
                    "--prefix", str(prefix), "--output", str(preflight), "--require-cuda"],
                   env=dict(environment, CUDA_VISIBLE_DEVICES="0"), check=True)
    check_probe(read(preflight), reuse["control_frozen"])
    return subprocess.run(command, env=environment).returncode


if __name__ == "__main__":
    raise SystemExit(main())
