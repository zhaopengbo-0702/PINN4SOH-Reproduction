"""Host-standard-library sidecar: isolated modern/legacy import snapshots only.

Single upload, Python 3.7+. No training, pickle loading, backend flag writes,
package installation, or file changes under either prefix are performed.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


SNAPSHOT_CODE = r'''
import json,os,platform,sys,warnings
from pathlib import Path
path_before_injection=list(sys.path)
injection=None
if len(sys.argv)>1 and sys.argv[1]:
    site=Path(sys.argv[1])
    if not site.is_absolute() or site.is_symlink() or not site.is_dir() or site.resolve()!=site:
        raise RuntimeError('Explicit dependency directory must be a resolved absolute ordinary directory')
    if not any((site/name).is_dir() for name in ['numpy','pandas','scipy','sklearn','matplotlib']):
        raise RuntimeError('Explicit dependency directory contains no expected scientific package')
    metadata=site.stat()
    injection=dict(path=str(site),parent_requested=sys.argv[1],directory_device=metadata.st_dev,
        directory_inode=metadata.st_ino,immediate_entries=sorted(p.name for p in site.iterdir()),
        method='sys.path.insert(0, explicit modern dependency directory); no site.addsitedir and no .pth execution')
    sys.path.insert(0,str(site))
result = dict(python=sys.version, executable=sys.executable, prefix=sys.prefix,
    sys_path=sys.path,sys_path_before_explicit_dependency_injection=path_before_injection,
    explicit_dependency_injection=injection,isolated=bool(sys.flags.isolated), platform=platform.platform(),
    environment={key:os.environ.get(key) for key in ['CUDA_VISIBLE_DEVICES','CUDA_FORCE_PTX_JIT',
        'CUDA_DISABLE_PTX_JIT','CUDA_LAUNCH_BLOCKING','NVIDIA_TF32_OVERRIDE','CUBLAS_WORKSPACE_CONFIG',
        'LD_LIBRARY_PATH','PYTHONPATH','PYTHONHOME','OMP_NUM_THREADS','MKL_NUM_THREADS',
        'OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']}, training_performed=False, flags_changed=False)
with warnings.catch_warnings(record=True) as captured:
    warnings.simplefilter('always')
    import torch,numpy,pandas,scipy,sklearn,matplotlib
    modules=[torch,numpy,pandas,scipy,sklearn,matplotlib]
    result.update(versions={m.__name__:m.__version__ for m in modules},
        imported_paths={m.__name__:m.__file__ for m in modules},
        module_sources={m.__name__:dict(path=str(Path(m.__file__).resolve()),
            from_explicit_dependency_directory=bool(injection and site in Path(m.__file__).resolve().parents)) for m in modules},
        actual_threads=torch.get_num_threads(),
        actual_interop_threads=torch.get_num_interop_threads() if hasattr(torch,'get_num_interop_threads') else None,
        cuda_runtime=torch.version.cuda, cuda_available=torch.cuda.is_available(),
        cudnn_version=torch.backends.cudnn.version(), cudnn_available=torch.backends.cudnn.is_available(),
        cudnn_enabled=torch.backends.cudnn.enabled,cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled() if hasattr(torch,'are_deterministic_algorithms_enabled') else None,
        torch_config=torch.__config__.show(),
        arch_list=torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
        gencode_flags=torch.cuda.get_gencode_flags() if torch.cuda.is_available() else None)
    if torch.cuda.is_available():
        result.update(gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
            device_properties=str(torch.cuda.get_device_properties(0)))
    result['warnings']=[dict(category=w.category.__name__,message=str(w.message),filename=w.filename,lineno=w.lineno) for w in captured]
print('PINN4SOH_SNAPSHOT_JSON='+json.dumps(result))
'''


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def environment_for(python, gpu, legacy):
    environment = dict(os.environ)
    for key in ["PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_SHLVL"]:
        environment.pop(key, None)
    prefix = python.parent.parent
    environment.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES=gpu,
                       OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2")
    environment["PATH"] = str(python.parent) + ":/usr/bin:/bin"
    if legacy:
        libraries = [prefix / "lib", prefix / "lib/python3.7/site-packages/torch/lib",
                     Path("/usr/local/nvidia/lib"), Path("/usr/local/nvidia/lib64")]
        environment["LD_LIBRARY_PATH"] = ":".join(str(path) for path in libraries if path.is_dir())
    environment["MPLBACKEND"] = "Agg"
    return environment


def validate_explicit_site(path):
    path = Path(os.path.abspath(str(path)))
    if not path.is_dir() or path.is_symlink() or path.resolve() != path:
        raise ValueError("Explicit modern dependency directory must be a resolved ordinary directory")
    for item in [path] + list(path.parents):
        if item.is_symlink():
            raise ValueError("Symlink in explicit modern dependency directory")
    if not any((path / name).is_dir() for name in ["numpy", "pandas", "scipy", "sklearn", "matplotlib"]):
        raise ValueError("Modern dependency directory lacks expected scientific packages")
    return path


def snapshot(python, gpu, legacy, modern_site=None):
    if legacy and modern_site is not None:
        raise ValueError("Explicit modern dependency directory must never enter legacy snapshot")
    modern_site = validate_explicit_site(modern_site) if modern_site is not None else None
    command = [str(python), "-B", "-I", "-c", SNAPSHOT_CODE]
    if modern_site is not None:
        command.append(str(modern_site))
    try:
        result = subprocess.run(command, env=environment_for(python, gpu, legacy),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=55)
        marker = "PINN4SOH_SNAPSHOT_JSON="
        line = next((line for line in reversed(result.stdout.splitlines()) if line.startswith(marker)), None)
        payload = json.loads(line[len(marker):]) if line else None
        return dict(python=str(python), python_binary_sha256=digest(python),
                    isolated_command=[str(python), "-B", "-I", "-c", "<embedded snapshot code>"] + ([str(modern_site)] if modern_site else []),
                    explicit_modern_site=str(modern_site) if modern_site else None,
                    returncode=result.returncode, stdout=result.stdout, stderr=result.stderr, snapshot=payload,
                    passed=result.returncode == 0 and payload is not None,
                    import_isolation="-I, no inherited Python/Conda vars, no user site; explicit modern site may be inserted as recorded, without .pth; legacy uses launcher-equivalent LD_LIBRARY_PATH; modern preserves its inherited native-library path")
    except Exception as error:
        return dict(python=str(python), passed=False, error=repr(error))


def nvidia_snapshot():
    command = ["nvidia-smi", "--query-gpu=name,uuid,driver_version,pci.bus_id,memory.total", "--format=csv,noheader"]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=15)
        return dict(command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except Exception as error:
        return dict(command=command, error=repr(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-python", type=Path, required=True)
    parser.add_argument("--modern-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--modern-site", type=Path, help="Explicit persistent modern dependencies, e.g. /workspace/site-packages; modern only")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.gpu != "0":
        raise ValueError("This deployment is limited to assigned GPU 0")
    output = args.output.resolve()
    if output.exists():
        raise ValueError("Observation output must be fresh")
    legacy_python, modern_python = args.legacy_python.absolute(), args.modern_python.absolute()
    modern_site = validate_explicit_site(args.modern_site) if args.modern_site is not None else None
    result = dict(format="pinn4soh-two-runtime-observations-v2", generated_at=datetime.now(timezone.utc).isoformat(),
                  host_python=sys.executable, script_sha256=digest(Path(__file__)),
                  snapshot_code_sha256=hashlib.sha256(SNAPSHOT_CODE.encode()).hexdigest(),
                  nvidia_smi=nvidia_snapshot(), training_performed=False, flags_changed=False,
                  package_installation_performed=False, checkpoint_loaded=False,
                  explicit_modern_site=str(modern_site) if modern_site else None,
                  original_host_pythonpath=os.environ.get("PYTHONPATH"),
                  scope="Fresh isolated child import snapshots only; existing trained worker process defaults were not measured")
    # Sequential imports avoid competing GPU initialization and give each snapshot its own native libraries.
    result["modern"] = snapshot(modern_python, args.gpu, False, modern_site)
    result["legacy"] = snapshot(legacy_python, args.gpu, True)
    fields = ["cuda_runtime", "gpu", "capability", "cudnn_version", "cudnn_enabled", "cudnn_benchmark",
              "cudnn_deterministic", "matmul_allow_tf32", "cudnn_allow_tf32", "deterministic_algorithms",
              "actual_threads", "actual_interop_threads", "arch_list"]
    if result["modern"]["passed"] and result["legacy"]["passed"]:
        modern, legacy = result["modern"]["snapshot"], result["legacy"]["snapshot"]
        result["differences"] = {key: {"modern": modern.get(key), "legacy": legacy.get(key)} for key in fields
                                  if modern.get(key) != legacy.get(key)}
        result["same_fields"] = {key: modern.get(key) for key in fields if modern.get(key) == legacy.get(key)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"output": str(output), "modern_passed": result["modern"]["passed"],
                      "legacy_passed": result["legacy"]["passed"], "differences": result.get("differences")}))
    return 0 if result["modern"]["passed"] and result["legacy"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
