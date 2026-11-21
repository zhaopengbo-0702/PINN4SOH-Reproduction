"""Resume PINN4SOH with isolated same-GPU workers and one manifest writer."""

import argparse
from collections import deque
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid


def pid_alive(pid):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5
        try:
            status = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(status)):
                return True
            return status.value == 259
        finally:
            kernel.CloseHandle(handle)
    if sys.platform == "linux":
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[0]
            if state == "Z":
                return False
        except FileNotFoundError:
            return False
        except (PermissionError, IndexError):
            pass
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def previous_process_alive(pid, output):
    if not pid or not pid_alive(pid):
        return False
    if sys.platform == "linux":
        try:
            arguments = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").split("\0")
        except FileNotFoundError:
            return False
        if str(output) not in arguments:
            raise RuntimeError(f"PID {pid} is alive with a different command (possible PID reuse); "
                               "no workers launched: " + " ".join(arguments))
    return True


def load_helpers(path):
    spec = importlib.util.spec_from_file_location("parallel_runner_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def acquire_lock(path, helpers):
    stream = path.open("a+", encoding="utf-8")
    try:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise RuntimeError("Another parallel scheduler is running") from None
    stream.seek(0)
    stream.truncate()
    json.dump({"pid": os.getpid(), "started_at": helpers.utc_now()}, stream)
    stream.flush()
    return stream


def task_keys(options, helpers):
    if options.batches and len(set(options.batches)) != len(options.batches):
        raise ValueError("Duplicate batch names would schedule the same output twice")
    datasets = list(helpers.BATCHES) if options.dataset == "all" else [options.dataset]
    for dataset in datasets:
        batches = options.batches or helpers.BATCHES[dataset]
        if any(batch not in helpers.BATCHES[dataset] for batch in batches):
            raise ValueError("Invalid batches for " + dataset)
        for batch in batches:
            for repeat in range(1, options.repeats + 1):
                yield {"key": f"{dataset}/{batch}/Experiment{repeat:02d}",
                       "dataset": dataset, "batch": batch, "repeat": repeat,
                       "seed": options.seed + list(helpers.BATCHES).index(dataset) * 10000
                       + helpers.BATCHES[dataset].index(batch) * 100 + repeat - 1}


def validate_run_config(task, record, options, inventory):
    folder = options.output / task["key"]
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    actual_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    parameters = config["parameters"]
    expected = {"source_fingerprint": inventory["fingerprint"], "seed": task["seed"],
                "threads": options.threads, "smoke": False, "checkpoint_snapshot_deepcopy": True}
    if actual_hash != record["config_hash"] or any(config.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Completed task configuration changed: " + task["key"])
    if parameters["epochs"] != options.epochs or parameters["save_folder"] != str(folder):
        raise RuntimeError("Completed task parameters/output changed: " + task["key"])


def completed_valid(task, record, options, inventory, helpers):
    if not record or record["status"] != "complete":
        return False
    folder = options.output / task["key"]
    validate_run_config(task, record, options, inventory)
    if not all((folder / name).is_file() for name in ["true_label.npy", "pred_label.npy", "model.pth"]):
        return False
    import numpy as np
    helpers.calculate_metrics(np.load(folder / "true_label.npy", allow_pickle=False),
                              np.load(folder / "pred_label.npy", allow_pickle=False))
    return True


def run_queue(options, helpers, manifest, tasks, inventory):
    pending, active, failures = deque(tasks), {}, []
    manifest_path = options.output / "manifest.json"
    stop_signal = []
    old_handlers = {}

    def request_stop(signum, frame):
        stop_signal.append(signum)

    for signum in [signal.SIGINT, signal.SIGTERM]:
        old_handlers[signum] = signal.signal(signum, request_stop)

    def save():
        helpers.write_json(manifest_path, manifest)

    def merge_worker(item):
        path = options.output / ".workers" / item["task"]["key"] / "manifest.json"
        if not path.exists():
            return False
        try:
            state = json.loads(path.read_text(encoding="utf-8"))["runs"][item["task"]["key"]]
        except (OSError, ValueError, KeyError):
            return False
        if state.get("worker_attempt") != item["attempt"]:
            return False
        if state.get("seed") != item["task"]["seed"]:
            raise RuntimeError("Worker seed does not match task")
        validate_run_config(item["task"], state, options, inventory)
        previous = manifest["runs"].get(item["task"]["key"])
        if state != previous:
            manifest["runs"][item["task"]["key"]] = state
            save()
        return True

    try:
        while pending or active:
            if stop_signal:
                break
            while pending and len(active) < options.workers and not stop_signal:
                task = pending.popleft()
                attempt = uuid.uuid4().hex
                worker_dir = options.output / ".workers" / task["key"]
                worker_dir.mkdir(parents=True, exist_ok=True)
                log = (worker_dir / ("process." + attempt + ".log")).open("wb")
                command = [sys.executable, "-u", str(options.runner), "--repo", str(options.repo),
                           "--output", str(options.output), "--dataset", task["dataset"],
                           "--batches", task["batch"], "--repeat-index", str(task["repeat"]),
                           "--worker", "--worker-attempt", attempt, "--epochs", str(options.epochs),
                           "--seed", str(options.seed), "--threads", str(options.threads), "--resume"]
                if options.allow_cpu:
                    command.append("--allow-cpu")
                record = dict(task, relative_folder=task["key"], status="running",
                              started_at=helpers.utc_now(), worker_attempt=attempt)
                record.pop("key")
                manifest["runs"][task["key"]] = record
                save()
                try:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                               cwd=str(options.repo))
                except BaseException:
                    log.close()
                    manifest["runs"][task["key"]].update(status="failed", error="Worker launch failed")
                    save()
                    raise
                active[process.pid] = {"process": process, "task": task,
                                       "attempt": attempt, "log": log}
                print(f"DISPATCH {task['key']} pid={process.pid} active={len(active)}", flush=True)
            changed = False
            for pid, item in list(active.items()):
                merged = merge_worker(item)
                code = item["process"].poll()
                if code is None:
                    continue
                item["log"].close()
                merged = merge_worker(item) or merged
                record = manifest["runs"][item["task"]["key"]]
                if code != 0 or not merged or record["status"] != "complete":
                    record.update(status="failed", finished_at=helpers.utc_now(),
                                  error=record.get("error", f"Worker exited {code} without a complete result"),
                                  worker_exit_code=code)
                    failures.append(item["task"]["key"])
                else:
                    folder = options.output / item["task"]["key"]
                    if not all((folder / name).is_file() for name in ["true_label.npy", "pred_label.npy", "model.pth"]):
                        record.update(status="failed", error="Worker omitted required result files")
                        failures.append(item["task"]["key"])
                print(f"FINISH {item['task']['key']} status={record['status']} exit={code}", flush=True)
                del active[pid]
                changed = True
                save()
            if changed:
                helpers.summarize(options.output, preserve_run_outputs=True)
            if active and not stop_signal:
                time.sleep(options.poll_seconds)
        if stop_signal:
            print("Stopping this scheduler's workers", flush=True)
    finally:
        for item in active.values():
            if item["process"].poll() is None:
                item["process"].terminate()
        for item in active.values():
            try:
                item["process"].wait(timeout=10)
            except subprocess.TimeoutExpired:
                item["process"].kill()
                item["process"].wait(timeout=10)
            item["log"].close()
            try:
                merge_worker(item)
            except Exception as error:
                print("Worker state rejected during shutdown: " + repr(error), flush=True)
            record = manifest["runs"][item["task"]["key"]]
            if record["status"] != "complete":
                record.update(status="failed", interrupted=True,
                              error="Interrupted by scheduler shutdown", finished_at=helpers.utc_now())
        save()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    helpers.summarize(options.output, options.plots, preserve_run_outputs=True)
    if stop_signal:
        return 128 + stop_signal[-1]
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runner", type=Path, default=Path(__file__).with_name("run_reproduction.py"))
    parser.add_argument("--dataset", choices=["all", "XJTU", "MIT", "HUST", "TJU"], default="all")
    parser.add_argument("--batches", nargs="+")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--threads", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--previous-pid", type=int, help="Refuse dispatch while the old serial PID still exists")
    parser.add_argument("--poll-seconds", type=float, default=1)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args()
    if options.repeats < 1 or options.epochs < 1 or options.poll_seconds <= 0:
        parser.error("repeats, epochs, and poll-seconds must be positive")
    if options.batches and options.dataset == "all":
        parser.error("--batches requires one dataset")
    options.repo, options.output, options.runner = options.repo.resolve(), options.output.resolve(), options.runner.resolve()
    helpers = load_helpers(options.runner)
    manifest_path = options.output / "manifest.json"
    inventory = json.loads((options.output / "source_inventory.json").read_text(encoding="utf-8"))
    if options.dry_run:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tasks = [task for task in task_keys(options, helpers)
                 if not completed_valid(task, manifest["runs"].get(task["key"]), options, inventory, helpers)]
        print(json.dumps({"pending": len(tasks), "workers": options.workers,
                          "tasks": [task["key"] for task in tasks]}, indent=2))
        return
    if previous_process_alive(options.previous_pid, options.output):
        raise RuntimeError(f"Old serial process {options.previous_pid} is still alive; no workers launched")
    lock_path = options.output / ".parallel.lock"
    lock = acquire_lock(lock_path, helpers)
    try:
        command = [sys.executable, "-u", str(options.runner), "--repo", str(options.repo),
                   "--output", str(options.output), "--threads", str(options.threads), "--validate-only"]
        if options.allow_cpu:
            command.append("--allow-cpu")
        subprocess.run(command, check=True, cwd=str(options.repo))
        if previous_process_alive(options.previous_pid, options.output):
            raise RuntimeError("Old serial process is still alive")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tasks = [task for task in task_keys(options, helpers)
                 if not completed_valid(task, manifest["runs"].get(task["key"]), options, inventory, helpers)]
        print(f"PLAN remaining={len(tasks)} workers={options.workers}", flush=True)
        result = run_queue(options, helpers, manifest, tasks, inventory)
    finally:
        lock.close()
    raise SystemExit(result)


if __name__ == "__main__":
    main()
