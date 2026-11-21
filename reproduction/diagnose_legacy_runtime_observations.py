"""Record architecture/backend/path evidence without training or changing flags.

Run with isolated legacy interpreter: legacy37/bin/python -I this.py --prefix ...
The file imports no sibling helpers and does not load pickle checkpoints.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def command(command_line):
    try:
        result = subprocess.run(command_line, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True, timeout=30)
        return dict(command=command_line, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except Exception as error:
        return dict(command=command_line, error=repr(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prefix, support, repo, control, output = [path.resolve() for path in
                                             [args.prefix, args.support, args.repo, args.control, args.output]]
    if output.exists():
        raise ValueError("Observation output must be a fresh path")
    if prefix != Path(sys.prefix).resolve() or prefix not in Path(sys.executable).resolve().parents:
        raise RuntimeError("Execute with the requested isolated prefix interpreter")
    import torch
    import numpy
    import pandas
    import scipy
    import sklearn
    import matplotlib
    modules = [torch, numpy, pandas, scipy, sklearn, matplotlib]
    evidence = dict(format="pinn4soh-legacy-runtime-observations-v1",
        generated_at=datetime.now(timezone.utc).isoformat(), python=sys.version, executable=sys.executable,
        prefix=sys.prefix, isolated=bool(sys.flags.isolated), sys_path=sys.path, platform=platform.platform(),
        versions={module.__name__: module.__version__ for module in modules},
        imported_paths={module.__name__: module.__file__ for module in modules},
        environment={key: os.environ.get(key) for key in ["CUDA_VISIBLE_DEVICES", "CUDA_FORCE_PTX_JIT",
            "CUDA_DISABLE_PTX_JIT", "CUDA_LAUNCH_BLOCKING", "NVIDIA_TF32_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG",
            "LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONHOME", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]},
        actual_threads=torch.get_num_threads(),
        actual_interop_threads=torch.get_num_interop_threads() if hasattr(torch, "get_num_interop_threads") else None,
        cuda_runtime=torch.version.cuda, cuda_available=torch.cuda.is_available(),
        arch_list=torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
        gencode_flags=torch.cuda.get_gencode_flags() if torch.cuda.is_available() else None,
        cudnn_version=torch.backends.cudnn.version(), cudnn_available=torch.backends.cudnn.is_available(),
        cudnn_enabled=torch.backends.cudnn.enabled, cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        torch_config=torch.__config__.show(),
        nvidia_smi=command(["nvidia-smi", "--query-gpu=name,uuid,driver_version,pci.bus_id,memory.total", "--format=csv,noheader"]),
        nvidia_processes=command(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"]),
        training_performed=False, flags_changed=False, checkpoint_loaded=False,
        scope="Sidecar process observations; worker thread configuration is separately 2; worker backend defaults presumed only from unchanged imported runtime")
    if torch.cuda.is_available():
        evidence["gpu"] = torch.cuda.get_device_name(0)
        evidence["capability"] = list(torch.cuda.get_device_capability(0))
        evidence["device_properties"] = str(torch.cuda.get_device_properties(0))
    files = {}
    for name in ["legacy_compat_manifest.json", "legacy_smoke_plan.json"]:
        path = support / name
        files[str(path)] = sha(path)
    for path in (support / "legacy_compat").glob("*.py"):
        files[str(path)] = sha(path)
    for name in ["legacy_control_plan.json", "legacy_control_design.json"]:
        files[str(control / name)] = sha(control / name)
    plans = [json.loads((support / "legacy_smoke_plan.json").read_text(encoding="utf-8")),
             json.loads((control / "legacy_control_plan.json").read_text(encoding="utf-8"))]
    dependencies = set()
    for plan in plans:
        for task in plan["tasks"]:
            if task.get("reference_run"):
                dependencies.update([task["reference_run"] + "/config.json", task["reference_run"] + "/splits.json"])
            if task.get("split_json"):
                dependencies.add(task["split_json"])
            if task.get("source_mode") == "supplied_author_checkpoint":
                path = Path(task["source_checkpoint"])
                dependencies.add(str(path if path.is_absolute() else repo / path))
    missing = []
    for path in sorted(dependencies):
        item = Path(path)
        if item.is_file():
            files[path] = sha(item)
        else:
            missing.append(path)
    # The source inventory uses all Python and README/fix/data-CSV files, not only imports.
    source_files = [p for p in repo.rglob("*") if p.is_file() and (p.suffix == ".py" or
                    p.name in {"README.md", "fix.md"} or (p.suffix == ".csv" and "data" in p.relative_to(repo).parts))]
    source_hashes = {p.relative_to(repo).as_posix(): sha(p) for p in sorted(source_files)}
    evidence.update(files=files, missing_immediate_dependencies=missing, source_file_count=len(source_hashes),
                    source_fingerprint=hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest(),
                    script_sha256=sha(Path(__file__)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"output": str(output), "cuda_available": evidence["cuda_available"],
                      "arch_list": evidence["arch_list"], "missing_dependencies": missing,
                      "source_fingerprint": evidence["source_fingerprint"],
                      "matmul_allow_tf32": evidence["matmul_allow_tf32"], "cudnn_allow_tf32": evidence["cudnn_allow_tf32"]}))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
