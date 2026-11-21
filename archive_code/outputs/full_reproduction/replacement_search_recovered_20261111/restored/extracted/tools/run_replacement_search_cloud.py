"""Single-GPU pipeline for the frozen replacement search; no installation or remote access."""
import argparse
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from search_replacement_grid import read, sha, write, identity
from run_extended import task_lock


def now():
    return datetime.now(timezone.utc).isoformat()


def preflight(bundle, repo, gpu, check_gpu=True):
    expected = read(bundle / "TOOLS_SHA256.json")
    for name, record in expected["files"].items():
        path = bundle / name
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
            raise ValueError("Tool package differs: " + name)
    source = read(bundle / "EXPECTED_AUTHOR_INPUTS.json")
    actual_py = {p.relative_to(repo).as_posix() for p in repo.rglob("*.py")}
    if actual_py != set(source["author_python_sha256"]):
        raise ValueError("Author Python member set differs; inspect before executing")
    for name, checksum in {**source["author_python_sha256"], **source["data_sha256"]}.items():
        if sha(repo / name) != checksum:
            raise ValueError("Author source/data differs: " + name)
    if check_gpu:
        probe = subprocess.run([sys.executable, "-c", "import inspect,json,sys,torch,numpy,pandas,scipy,sklearn; assert 'weights_only' in inspect.signature(torch.load).parameters, 'weights_only requires modern torch'; assert torch.cuda.is_available() and torch.cuda.device_count()==1, 'exactly one visible GPU required'; print(json.dumps(dict(python=sys.version,torch=torch.__version__,numpy=numpy.__version__,pandas=pandas.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,visible_gpus=torch.cuda.device_count())))"],
            capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu))
        if probe.returncode:
            raise RuntimeError("Modern single-GPU dependency probe failed: " + probe.stderr)
        environment = json.loads(probe.stdout.strip())
    else:
        environment = {"gpu_probe": "skipped_for_local_packaging_check"}
    return dict(status="passed", checked_at=now(), package_manifest_sha256=sha(bundle / "TOOLS_SHA256.json"),
                expected_author_inputs_sha256=sha(bundle / "EXPECTED_AUTHOR_INPUTS.json"), environment=environment,
                author_python_files=len(source["author_python_sha256"]), input_csvs=len(source["data_sha256"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--prior-pipeline", type=Path)
    parser.add_argument("--skip-gpu-probe", action="store_true", help="Local preflight only; forbidden with --execute")
    options = parser.parse_args()
    if not options.gpu.isdigit() or (options.skip_gpu_probe and options.execute):
        parser.error("Use one GPU index; executing requires the GPU probe")
    bundle = Path(__file__).resolve().parent
    repo, output = options.repo.resolve(), options.output.resolve()
    if output == repo or repo in output.parents or output == bundle or bundle in output.parents:
        raise ValueError("Output must be outside immutable repo/tools")
    state_path = output.with_name(output.name + "_pipeline_status.json")
    request = dict(repo=str(repo), output=str(output), gpu=options.gpu, evaluate_test=options.evaluate_test,
                   bundle=str(bundle), bundle_manifest_sha256=sha(bundle / "TOOLS_SHA256.json"),
                   prior_pipeline=str(options.prior_pipeline.resolve()) if options.prior_pipeline else None)
    if options.prior_pipeline and read(options.prior_pipeline).get("status") != "complete":
        raise ValueError("Prior training pipeline must be complete; do not overlap tasks")
    check = preflight(bundle, repo, options.gpu, not options.skip_gpu_probe)
    if not options.execute:
        print(json.dumps(dict(status="preflight_only", preflight=check, request=request), indent=2))
        return
    with task_lock(output.with_name("." + output.name + ".pipeline.lock")):
        previous = read(state_path) if state_path.exists() else None
        if previous:
            if not options.resume:
                raise ValueError("Pipeline already exists; inspect its state before explicit --resume")
            if previous["request"] != request:
                raise ValueError("Resume request changed; use new output")
            if previous.get("status") == "complete":
                print("PIPELINE_ALREADY_COMPLETE " + str(state_path))
                return
        elif output.exists():
            raise ValueError("New pipeline requires a fresh output path")
        state = dict(format="pinn4soh-replacement-search-cloud-pipeline-v1", status="running", request=request,
                     pid=os.getpid(), started_at=now(), steps=[], prior_attempt=previous)
        write(state_path, state)

        def step(name, function):
            record = dict(name=name, status="running", started_at=now())
            state["steps"].append(record)
            state["updated_at"] = now()
            write(state_path, state)
            print("START " + name, flush=True)
            result = function()
            record.update(status="complete", finished_at=now(), result=result)
            state["updated_at"] = now()
            write(state_path, state)
            print("COMPLETE " + name, flush=True)
            return result

        def command(argv):
            completed = subprocess.call(argv, env=dict(os.environ, CUDA_VISIBLE_DEVICES=options.gpu,
                PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2"))
            if completed:
                raise RuntimeError("Command failed with exit " + str(completed))
            return dict(returncode=completed, argv=argv)

        try:
            step("preflight", lambda: check)
            smoke_spec = read(bundle / "replacement_search_example.json")
            smoke_spec["label"] = "MODERN_GPU_CLOUD_TWO_EPOCH_FUNCTIONAL_SMOKE_ONLY"
            smoke_spec["seeds"] = smoke_spec["seeds"][:1]
            smoke_spec["fixed"].update(epochs=2, warmup_epochs=1, early_stop=1)
            smoke_spec["grid"] = {key: values[:1] for key, values in smoke_spec["grid"].items()}
            smoke_spec_path = output.with_name(output.name + "_smoke_spec.json")
            if smoke_spec_path.exists() and read(smoke_spec_path) != smoke_spec:
                raise ValueError("Prior smoke spec changed")
            write(smoke_spec_path, smoke_spec)
            smoke_output = output.with_name(output.name + "_gpu_smoke")
            base = [sys.executable, str(bundle / "search_replacement_grid.py"), "--repo", str(repo), "--device", options.gpu]
            step("gpu_smoke", lambda: command(base + ["--spec", str(smoke_spec_path), "--output", str(smoke_output), "--execute", "--resume", "--evaluate-test"]))
            step("formal_plan", lambda: command(base + ["--spec", str(bundle / "replacement_search_example.json"), "--output", str(output)]))
            search = base + ["--spec", str(bundle / "replacement_search_example.json"), "--output", str(output), "--execute", "--resume"]
            if options.evaluate_test:
                search += ["--evaluate-test"]
            step("formal_search", lambda: command(search))
            collection = output.with_name(output.name + "_backup")
            if collection.exists():
                if not options.resume:
                    raise ValueError("Collection already exists")
                collection.rename(collection.with_name(collection.name + ".incomplete." + str(time.time_ns())))
            collector = [sys.executable, str(bundle / "collect_replacement_search.py"), "--search", str(output),
                "--bundle", str(bundle), "--smoke", str(smoke_output), "--smoke-spec", str(smoke_spec_path),
                "--pipeline-status", str(state_path), "--output", str(collection)]
            if options.evaluate_test:
                collector.append("--require-test")
            step("audit_and_collection", lambda: command(collector))
            state.update(status="complete", finished_at=now(), collection=str(collection))
            print("REPLACEMENT_SEARCH_PIPELINE_COMPLETE", flush=True)
        except BaseException as error:
            state.update(status="failed", error=repr(error), failed_at=now())
            if state["steps"] and state["steps"][-1]["status"] == "running":
                state["steps"][-1].update(status="failed", error=repr(error))
            raise
        finally:
            state["updated_at"] = now()
            write(state_path, state)


if __name__ == "__main__":
    main()
