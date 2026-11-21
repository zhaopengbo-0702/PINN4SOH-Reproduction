"""Separate Python3.7/Torch1.7 CPU inference benchmark; no training or checkpoint loading."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


EXPECTED_PARAMETERS = {"F": 7781, "MLP": 7781, "CNN": 8465}
PAPER_SECONDS_PER_1000 = {"F": 5.81e-4, "MLP": 5.64e-4, "CNN": 1.29e-2}
AUTHOR_FILES = ["Model/Model.py", "Model/Compare_Models.py", "utils/util.py"]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def cpu_information():
    result = {"platform": platform.platform(), "machine": platform.machine(),
              "processor": platform.processor(), "logical_cpu_count": os.cpu_count(),
              "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
              "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None}
    if os.name == "nt":
        command = ("Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,"
                   "NumberOfLogicalProcessors,MaxClockSpeed | ConvertTo-Json -Compress")
        response = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                                  text=True, capture_output=True, timeout=20, check=True)
        result["processor_details"] = json.loads(response.stdout)
    elif Path("/proc/cpuinfo").is_file():
        records = []
        for block in Path("/proc/cpuinfo").read_text().split("\n\n"):
            values = dict(line.split(":", 1) for line in block.splitlines() if ":" in line)
            values = {key.strip(): value.strip() for key, value in values.items()}
            if values:
                records.append(values)
        result["processor_models"] = sorted({item.get("model name", item.get("Hardware", "unknown")) for item in records})
        result["physical_cores_observed"] = len({(item.get("physical id"), item.get("core id"))
                                                 for item in records if "core id" in item}) or None
        result["cgroup"] = {str(path): path.read_text().strip() for path in
            [Path("/sys/fs/cgroup/cpu.max"), Path("/sys/fs/cgroup/cpuset.cpus.effective")]
            if path.is_file()}
    return result


def source_hashes(repo):
    return {name: digest(repo / name) for name in AUTHOR_FILES}



def exact_legacy_imports():
    if sys.platform != "linux" or sys.version_info[:3] != (3, 7, 10) or not sys.flags.isolated:
        raise RuntimeError("Use isolated Linux Python3.7.10: python -B -I")
    if os.environ.get("PYTHONPATH") or os.environ.get("PYTHONHOME"):
        raise RuntimeError("Inherited PYTHONPATH/PYTHONHOME must be cleared")
    import torch, numpy, pandas, scipy, sklearn, matplotlib
    modules = [torch, numpy, pandas, scipy, sklearn, matplotlib]
    expected = ["1.7.1+cu110", "1.20.3", "1.3.5", "1.7.3", "0.24.2", "3.3.4"]
    if [m.__version__ for m in modules] != expected:
        raise RuntimeError("Exact old-stack package versions required")
    prefix = Path(sys.prefix).resolve()
    if any(prefix not in Path(m.__file__).resolve().parents for m in modules):
        raise RuntimeError("Package import escaped legacy prefix")
    if torch.version.cuda != "11.0":
        raise RuntimeError("Expected original CUDA11.0 official PyTorch wheel")
    return {m.__name__:dict(version=m.__version__, path=str(Path(m.__file__).resolve())) for m in modules}


def dry_run_plan(options):
    return dict(format="pinn4soh-legacy-efficiency-dryrun-v1", training_performed=False,
        checkpoint_loaded=False, torch_imported=False, timing_performed=False,
        expected_python="3.7.10", expected_torch="1.7.1+cu110", device="cpu",
        defaults=dict(batch_size=options.batch_size, forwards=options.forwards, warmup=options.warmup,
                      repeats=options.repeats, threads=options.threads),
        command=["/workspace/legacy37/bin/python", "-B", "-I", str(Path(__file__).resolve()),
                 "--repo", str(options.repo), "--output", str(options.output),
                 "--batch-size", str(options.batch_size), "--forwards", str(options.forwards),
                 "--warmup", str(options.warmup), "--repeats", str(options.repeats),
                 "--seed", str(options.seed), "--threads"]+[str(t) for t in options.threads]
                 +(["--smoke"] if options.smoke else []),
        expected_parameters=EXPECTED_PARAMETERS,
        source_sha256=source_hashes(options.repo.resolve()), benchmark_sha256=digest(__file__),
        run_condition="Run after formal training; no simultaneous CPU contention",
        comparison_limit="Same reported software versions; cloud CPU differs from paper Intel i5-10400F")

def benchmark_worker(options):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[key] = str(options.worker_threads)
    exact_legacy_imports()
    import torch
    torch.set_num_threads(options.worker_threads)
    torch.set_num_interop_threads(1)
    repo = options.repo.resolve()
    sys.path.insert(0, str(repo))
    author = importlib.import_module("Model.Model")
    baseline = importlib.import_module("Model.Compare_Models")
    factories = {"F": author.Solution_u, "MLP": baseline.MLP, "CNN": baseline.CNN}
    order = list(EXPECTED_PARAMETERS)
    offset = (options.worker_repeat - 1) % len(order)
    order = order[offset:] + order[:offset]
    generator = torch.Generator().manual_seed(options.seed + options.worker_repeat)
    inputs = torch.randn(options.batch_size, 17, generator=generator, dtype=torch.float32, device="cpu").contiguous()
    input_sha = hashlib.sha256(inputs.numpy().tobytes()).hexdigest()
    records = []
    for name in order:
        torch.manual_seed(options.seed + options.worker_repeat + list(EXPECTED_PARAMETERS).index(name) * 10000)
        model = factories[name]().to(device="cpu", dtype=torch.float32).eval()
        count = sum(parameter.numel() for parameter in model.parameters())
        trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        if count != EXPECTED_PARAMETERS[name] or trainable_count != count:
            raise RuntimeError("Author architecture parameter count changed: " + name + ": " + str(count))
        buffers = {key: value.clone() for key, value in model.named_buffers()}
        with torch.no_grad():
            for _ in range(options.warmup):
                prediction = model(inputs)
            disabled_autograd = not torch.is_grad_enabled()
            started_cpu = time.process_time_ns()
            started = time.perf_counter_ns()
            for _ in range(options.forwards):
                prediction = model(inputs)
            elapsed_ns = time.perf_counter_ns() - started
            elapsed_cpu_ns = time.process_time_ns() - started_cpu
        unchanged_buffers = all(torch.equal(value, dict(model.named_buffers())[key]) for key, value in buffers.items())
        if (not disabled_autograd or prediction.requires_grad or prediction.device.type != "cpu"
                or any(layer.training for layer in model.modules()) or not unchanged_buffers
                or any(parameter.grad is not None for parameter in model.parameters())
                or tuple(prediction.shape) != (options.batch_size, 1)
                or not bool(torch.isfinite(prediction).all())):
            raise RuntimeError("Inference-only validation failed for " + name)
        seconds_per_batch = elapsed_ns / 1e9 / options.forwards
        record = {"model": name, "implementation": "Model.Model.Solution_u" if name == "F" else "Model.Compare_Models." + name,
                  "parameters": count, "trainable_parameters": trainable_count,
                  "threads": options.worker_threads, "repeat": options.worker_repeat,
                  "batch_size": options.batch_size, "forwards": options.forwards, "warmup_forwards": options.warmup,
                  "samples_timed": options.batch_size * options.forwards, "elapsed_wall_seconds": elapsed_ns / 1e9,
                  "elapsed_process_cpu_seconds": elapsed_cpu_ns / 1e9,
                  "seconds_per_batch": seconds_per_batch,
                  "seconds_per_1000_samples": seconds_per_batch * 1000 / options.batch_size,
                  "microseconds_per_sample": seconds_per_batch * 1e6 / options.batch_size,
                  "input_sha256": input_sha, "output_mean": float(prediction.mean()),
                  "output_l2": float(torch.norm(prediction)), "output_shape": list(prediction.shape),
                  "eval_mode": True, "no_grad": disabled_autograd, "unchanged_buffers": unchanged_buffers,
                  "device": "cpu", "dtype": "float32", "seed": options.seed + options.worker_repeat,
                  "weight_mode": "Seeded author architecture initialization; no trained checkpoint loaded"}
        records.append(record)
        print(json.dumps({"model": name, "threads": options.worker_threads,
                          "repeat": options.worker_repeat, "seconds_per_1000_samples": record["seconds_per_1000_samples"]}), flush=True)
    environment = {"python": sys.version, "python_executable": sys.executable, "torch": torch.__version__,
                   "torch_git_version": torch.version.git_version, "torch_cuda_build": torch.version.cuda,
                   "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "torch_threads": torch.get_num_threads(),
                   "torch_interop_threads": torch.get_num_interop_threads(), "mkldnn_enabled": torch.backends.mkldnn.enabled,
                   "torch_config": torch.__config__.show(), "cpu": cpu_information(),
                   "isolated": bool(sys.flags.isolated), "exact_imports": exact_legacy_imports()}
    write_json(options.worker_output, {"status": "complete", "created_at": utc_now(),
               "source_sha256": source_hashes(repo), "benchmark_sha256": digest(__file__),
               "environment": environment, "model_order": order, "records": records})


def summarize(records):
    rows = []
    for threads in sorted({record["threads"] for record in records}):
        for model in EXPECTED_PARAMETERS:
            group = [record for record in records if record["model"] == model and record["threads"] == threads]
            values = [record["seconds_per_1000_samples"] for record in group]
            if not values:
                continue
            mean = statistics.mean(values)
            rows.append({"model": model, "threads": threads, "parameters": EXPECTED_PARAMETERS[model],
                         "independent_process_repeats": len(values), "mean_seconds_per_1000_samples": mean,
                         "sample_sd_seconds_per_1000_samples": statistics.stdev(values) if len(values) > 1 else None,
                         "median_seconds_per_1000_samples": statistics.median(values),
                         "min_seconds_per_1000_samples": min(values), "max_seconds_per_1000_samples": max(values),
                         "raw_seconds_per_1000_samples": values,
                         "paper_reference_seconds_per_1000_samples": PAPER_SECONDS_PER_1000[model],
                         "measured_to_paper_time_ratio": mean / PAPER_SECONDS_PER_1000[model]})
    return rows


def run_benchmark(options):
    if options.dry_run:
        print(json.dumps(dry_run_plan(options), indent=2))
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[key] = "1"
    exact_legacy_imports()
    repo, output = options.repo.resolve(), options.output.resolve()
    if output == repo or repo in output.parents or output in repo.parents:
        raise ValueError("Benchmark output must be separate from the author repository")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Benchmark output is not empty; use a new output directory")
    if options.smoke:
        options.forwards, options.warmup, options.repeats = 5, 2, 2
    if (options.batch_size < 1 or options.forwards < 1 or options.warmup < 0 or options.repeats < 1
            or not options.threads or len(set(options.threads)) != len(options.threads)
            or any(value < 1 for value in options.threads)):
        raise ValueError("Invalid benchmark configuration")
    output.mkdir(parents=True, exist_ok=True)
    (output / "raw").mkdir()
    (output / "logs").mkdir()
    before = source_hashes(repo)
    config = {"repo": str(repo), "source_sha256": before, "benchmark_sha256": digest(__file__),
              "batch_size": options.batch_size, "forwards": options.forwards, "warmup_forwards": options.warmup,
              "repeats": options.repeats, "threads": options.threads, "interop_threads": 1,
              "dtype": "float32", "device": "cpu", "eval": True, "no_grad": True, "seed": options.seed,
              "smoke": options.smoke, "table_s8_iteration_protocol": options.batch_size == 1000 and options.forwards == 1000,
              "repeat_independence": "Fresh process for each thread-count/repeat, all models warmed up separately",
              "model_order": "Cyclic rotation by repeat; identical order and inputs across thread counts",
              "timed_scope": "Forward model execution only; allocation, initialization, warmup and output checks excluded",
              "paper_protocol": "Supplement Table S.8: batch=1000, 1000 forward passes, mean seconds per batch, CPU",
              "paper_environment": {"cpu": "Intel Core i5-10400F @ 2.90 GHz", "torch": "1.7.1"},
              "paper_unreported_settings": ["warmup", "thread counts", "eval/no_grad flags", "independent-repeat count"],
              "comparison_limit": "Reported software versions match; CPU hardware and unreported paper settings differ; ratios are descriptive",
              "F_scope": "Author Solution_u (F) inference only; training-only dynamical G and automatic derivatives excluded",
              "python": sys.version, "torch_distribution_version": "1.7.1+cu110",
              "exact_imports": exact_legacy_imports(), "isolated": bool(sys.flags.isolated),
              "cpu": cpu_information()}
    write_json(output / "config.json", config)
    manifest = {"format": "pinn4soh-efficiency-v1", "status": "running", "created_at": utc_now(),
                "config_sha256": digest(output / "config.json"), "smoke": options.smoke, "measurements": []}
    write_json(output / "manifest.json", manifest)
    records = []
    try:
        for repeat in range(1, options.repeats + 1):
            for threads in options.threads:
                name = "threads" + str(threads) + "_repeat" + str(repeat).zfill(2)
                raw = output / "raw" / (name + ".json")
                command = [sys.executable, "-B", "-I", "-u", str(Path(__file__).resolve()), "--worker", "--repo", str(repo),
                           "--worker-output", str(raw), "--worker-threads", str(threads), "--worker-repeat", str(repeat),
                           "--batch-size", str(options.batch_size), "--forwards", str(options.forwards),
                           "--warmup", str(options.warmup), "--seed", str(options.seed)]
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
                for variable in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
                    environment[variable] = str(threads)
                log = output / "logs" / (name + ".log")
                with log.open("w", encoding="utf-8") as stream:
                    subprocess.run(command, cwd=repo, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
                value = read_json(raw)
                if (value.get("status") != "complete" or value["source_sha256"] != before
                        or value["benchmark_sha256"] != config["benchmark_sha256"]
                        or value["environment"]["torch_threads"] != threads
                        or value["environment"]["torch_interop_threads"] != 1
                        or value["environment"]["cuda_visible_devices"] != ""
                        or {item["model"] for item in value["records"]} != set(EXPECTED_PARAMETERS)):
                    raise RuntimeError("Benchmark child protocol or source verification failed: " + name)
                for record in value["records"]:
                    if not math.isfinite(record["seconds_per_1000_samples"]) or record["seconds_per_1000_samples"] <= 0:
                        raise RuntimeError("Invalid timing measurement")
                records.extend(value["records"])
                manifest["measurements"].append({"name": name, "raw_sha256": digest(raw), "log_sha256": digest(log)})
                write_json(output / "manifest.json", manifest)
                print("COMPLETE " + name, flush=True)
        if source_hashes(repo) != before:
            raise RuntimeError("Author source files changed during benchmark")
        rows = summarize(records)
        summary = {"status": "complete", "smoke": options.smoke, "config": config,
                   "standard_deviation": "Across independent process measurements, sample SD ddof=1",
                   "rows": rows, "raw_records": records}
        write_json(output / "summary.json", summary)
        with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows({**row, "raw_seconds_per_1000_samples": json.dumps(row["raw_seconds_per_1000_samples"])} for row in rows)
        manifest.update(status="complete", finished_at=utc_now(),
                        artifact_sha256={str(path.relative_to(output)): digest(path) for path in output.rglob("*")
                                         if path.is_file() and path.name != "manifest.json"})
    except BaseException as error:
        manifest.update(status="failed", finished_at=utc_now(), error=repr(error))
        raise
    finally:
        write_json(output / "manifest.json", manifest)
    print(json.dumps({"status": "complete", "smoke": options.smoke, "output": str(output), "rows": rows}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", nargs="+", type=int, default=[1])
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--forwards", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Stdlib-only command preview; no torch import or timing")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-threads", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-repeat", type=int, help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.worker:
        benchmark_worker(options)
    elif options.output is None:
        parser.error("--output is required")
    else:
        run_benchmark(options)


if __name__ == "__main__":
    main()
