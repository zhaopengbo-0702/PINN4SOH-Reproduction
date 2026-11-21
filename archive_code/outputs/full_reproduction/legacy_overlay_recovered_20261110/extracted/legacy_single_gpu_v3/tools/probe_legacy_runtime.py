"""Prove exact legacy versions, isolated imports, and required CUDA differentiation APIs."""

import argparse
import json
import os
from pathlib import Path
import platform
import ssl
import subprocess
import sys
import tempfile
import traceback


def probe(prefix, require_cuda):
    if sys.platform != "linux" or sys.version_info[:3] != (3, 7, 10):
        raise RuntimeError("Execute this probe with the isolated Linux CPython 3.7.10 interpreter")
    prefix = Path(prefix).resolve()
    if Path(sys.prefix).resolve() != prefix or prefix not in Path(sys.executable).resolve().parents:
        raise RuntimeError("Interpreter is outside requested legacy prefix")
    if os.environ.get("PYTHONPATH") or os.environ.get("PYTHONHOME"):
        raise RuntimeError("Clear inherited PYTHONPATH and PYTHONHOME before probing")
    import numpy
    import pandas
    import sklearn
    import scipy
    import matplotlib
    import torch
    modules = [numpy, pandas, sklearn, scipy, matplotlib, torch]
    versions = [module.__version__ for module in modules]
    if versions != ["1.20.3", "1.3.5", "0.24.2", "1.7.3", "3.3.4", "1.7.1+cu110"]:
        raise RuntimeError("Package versions differ from explicit legacy target: " + repr(versions))
    if any(prefix not in Path(module.__file__).resolve().parents for module in modules):
        raise RuntimeError("Imported package is outside isolated prefix")
    if torch.version.cuda != "11.0":
        raise RuntimeError("Expected the official CUDA 11.0 PyTorch wheel")
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; a CPU pass cannot substitute for GPU feasibility")
    from sklearn.metrics import mean_absolute_percentage_error
    value = float(mean_absolute_percentage_error(numpy.array([1., .8]), numpy.array([.9, .7])))
    if not numpy.isclose(value, .1125):
        raise RuntimeError("Legacy sklearn MAPE API differs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    torch.manual_seed(20261110)
    linear = torch.nn.Linear(17, 1).to(device)
    inputs = torch.randn(16, 17, device=device, requires_grad=True)
    prediction = torch.sin(linear(inputs))
    derivative = torch.autograd.grad(prediction.sum(), inputs, create_graph=True)[0]
    loss = (derivative ** 2).mean() + (prediction ** 2).mean()
    loss.backward()
    if not all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in linear.parameters()):
        raise RuntimeError("First-derivative loss cannot backpropagate to solution parameters")
    conv = torch.nn.Sequential(torch.nn.Conv1d(1, 4, 3), torch.nn.Dropout(.2)).to(device)
    conv(torch.randn(8, 1, 17, device=device)).pow(2).mean().backward()
    if not all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in conv.parameters()):
        raise RuntimeError("Legacy convolution/dropout backward failed")
    with tempfile.TemporaryDirectory(prefix="pinn-legacy-checkpoint-") as directory:
        path = Path(directory) / "self_generated_state.pth"
        torch.save(linear.state_dict(), path)
        state = torch.load(path, map_location=device)
        if any(not torch.equal(state[name], tensor) for name, tensor in linear.state_dict().items()):
            raise RuntimeError("Legacy checkpoint roundtrip changed state")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    glibc = platform.libc_ver()
    if glibc[0] != "glibc" or tuple(int(value) for value in glibc[1].split(".")[:2]) < (2, 17):
        raise RuntimeError("The official manylinux wheels require host glibc >=2.17")
    try:
        process = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=30)
        nvidia = dict(returncode=process.returncode, stdout=process.stdout, stderr=process.stderr)
    except FileNotFoundError:
        nvidia = dict(status="CLI unavailable; CUDA was independently checked through PyTorch")
    return dict(status="passed", format="pinn4soh-legacy-runtime-probe-v1", python=sys.version,
        executable=sys.executable, prefix=sys.prefix, sys_path=sys.path,
        versions={module.__name__: module.__version__ for module in modules},
        module_paths={module.__name__: module.__file__ for module in modules},
        cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(),
        assigned_visibility=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        capability=list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        glibc=glibc, openssl=ssl.OPENSSL_VERSION, nvidia_smi=nvidia,
        api_checks=dict(measured_denominator_MAPE=value, derivative_loss_backward=True,
                        convolution_dropout_backward=True, self_generated_checkpoint_roundtrip=True),
        training_performed=False, scientific_accuracy_assessed=False,
        limit="Exact README package versions plus explicitly selected SciPy/CUDA wheel; OS/BLAS/driver need not match paper")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    try:
        evidence, code = probe(args.prefix, args.require_cuda), 0
    except Exception as error:
        evidence, code = dict(status="failed", error=repr(error), traceback=traceback.format_exc(),
                              python=sys.version, executable=sys.executable, scientific_accuracy_assessed=False), 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
