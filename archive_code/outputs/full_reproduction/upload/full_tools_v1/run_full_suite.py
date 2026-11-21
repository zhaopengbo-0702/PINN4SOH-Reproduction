"""Single-writer, dependency-aware GPU coordinator for audited task runners."""

import argparse
from collections import Counter
import contextlib
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time
import uuid


FORMAT = "pinn4soh-full-suite-v1"
CORE_ARTIFACTS = {"config.json", "splits.json", "model.pth", "true_label.npy", "pred_label.npy"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def canonical_id(value):
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value
            or str(PurePosixPath(value)) != value or PurePosixPath(value).is_absolute()
            or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", part)
                   or part in {".", ".."} or part.endswith(".")
                   or part.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL",
                       *["COM" + str(i) for i in range(1, 10)], *["LPT" + str(i) for i in range(1, 10)]}
                   for part in value.split("/"))):
        raise ValueError("task_id must be a canonical relative path: " + repr(value))
    return value


def prepare_plan(plan, threads):
    if not isinstance(plan, dict) or not isinstance(plan.get("tasks"), list) or not plan["tasks"]:
        raise ValueError("Plan must contain a nonempty tasks list")
    tasks, lookup, folded = [], {}, set()
    for raw in plan["tasks"]:
        if not isinstance(raw, dict):
            raise ValueError("Every task must be an object")
        task = dict(raw)
        task_id = canonical_id(task.get("task_id"))
        if task_id.casefold() in folded:
            raise ValueError("Duplicate task_id: " + task_id)
        folded.add(task_id.casefold())
        if task.get("runner") not in {"extended", "transfer"}:
            raise ValueError("Task runner must be extended or transfer: " + task_id)
        if "threads" in task and task["threads"] != threads:
            raise ValueError("Task threads differ from --threads: " + task_id)
        task.setdefault("threads", threads)
        deps = task.get("depends_on", task.get("dependencies", []))
        if not isinstance(deps, list) or any(not isinstance(item, str) for item in deps) or len(set(deps)) != len(deps):
            raise ValueError("Invalid depends_on list: " + task_id)
        if "depends_on" in task and "dependencies" in task and task["depends_on"] != task["dependencies"]:
            raise ValueError("Conflicting dependency fields: " + task_id)
        tasks.append(task)
        lookup[task_id] = task
    for task in tasks:
        task_id = task["task_id"]
        for parent in PurePosixPath(task_id).parents:
            if str(parent).casefold() in folded:
                raise ValueError("Task output directories cannot contain one another: " + task_id)
        for dependency in dependencies(task):
            if dependency not in lookup:
                raise ValueError("Unknown dependency " + dependency + " for " + task_id)
    visiting, visited, ordered = set(), set(), []

    def visit(task_id):
        if task_id in visiting:
            raise ValueError("Dependency cycle at " + task_id)
        if task_id in visited:
            return
        visiting.add(task_id)
        for dependency in dependencies(lookup[task_id]):
            visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)
        ordered.append(lookup[task_id])

    for task in tasks:
        visit(task["task_id"])
    return ordered


def dependencies(task):
    return task.get("depends_on", task.get("dependencies", []))


@contextlib.contextmanager
def coordinator_lock(output):
    path = output / ".full_suite.lock"
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another coordinator owns this output directory") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def process_identity(pid):
    if not isinstance(pid, int) or pid < 1:
        return None
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *[ctypes.POINTER(wintypes.FILETIME)] * 4]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return None
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != 259:
                return None
            created, exited, kerneltime, usertime = (wintypes.FILETIME() for _ in range(4))
            if not kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                          ctypes.byref(kerneltime), ctypes.byref(usertime)):
                raise RuntimeError("Cannot determine worker process creation time")
            return {"pid": pid, "created": (created.dwHighDateTime << 32) | created.dwLowDateTime}
        finally:
            kernel.CloseHandle(handle)
    try:
        fields = Path("/proc", str(pid), "stat").read_text().rsplit(") ", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": pid, "created": fields[19]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def load_inventory(helper, repo):
    spec = importlib.util.spec_from_file_location("suite_inventory_helpers", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.source_inventory(repo)


def normalize_environment(environment):
    aliases = {"scikit_learn": "sklearn", "cuda_runtime": "cuda"}
    return {aliases.get(key, key): value for key, value in environment.items()
            if key in {"python", "torch", "numpy", "pandas", "scipy", "sklearn", "scikit_learn",
                       "cuda", "cuda_runtime", "cuda_available", "gpu"}}


def probe_environment(gpus, allow_cpu=False):
    script = ("import json,sys; import torch,numpy,pandas,scipy,sklearn; "
              "print(json.dumps(dict(python=sys.version,torch=torch.__version__,numpy=numpy.__version__,"
              "pandas=pandas.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,"
              "cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),"
              "gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)))")
    probes = {}
    for gpu in gpus:
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        result = subprocess.run([sys.executable, "-c", script], env=environment, text=True,
                                capture_output=True, timeout=60, check=True)
        probes[gpu] = normalize_environment(json.loads(result.stdout.strip().splitlines()[-1]))
        if not probes[gpu]["cuda_available"] and not allow_cpu:
            raise RuntimeError("CUDA unavailable in assigned GPU slot " + gpu)
    return probes


def input_fingerprints(task, repo, inventory):
    records = {}

    def add(value, split=False):
        path = Path(value)
        path = path.resolve() if path.is_absolute() else (repo / path).resolve()
        if not path.is_file():
            raise ValueError("Required task input is missing: " + str(path))
        try:
            cached = inventory["sha256"].get(path.relative_to(repo).as_posix())
        except ValueError:
            cached = None
        records[str(path)] = cached or digest(path)
        if split:
            for entry in read_json(path):
                for battery in entry["paths"]:
                    add(battery)
        return path

    if task.get("split_json"):
        add(task["split_json"], split=True)
    if task.get("reference_run"):
        run = Path(task["reference_run"])
        if not run.is_absolute():
            run = repo / run
        add(run / "config.json")
        add(run / "splits.json", split=True)
    if task.get("source_checkpoint"):
        add(task["source_checkpoint"])
    if task.get("source_run"):
        run = Path(task["source_run"])
        if not run.is_absolute():
            run = repo / run
        for name in ["config.json", "splits.json", "logging.txt"]:
            add(run / name, split=name == "splits.json")
        source_manifest = task.get("source_manifest") or run / "manifest.json"
        if not Path(source_manifest).is_absolute():
            source_manifest = repo / source_manifest
        if not Path(source_manifest).is_file() and len(run.parents) > 2:
            source_manifest = run.parents[2] / "manifest.json"
        add(source_manifest)
        parameters = run / "parameters.json"
        if parameters.is_file():
            add(parameters)
    elif task.get("source_manifest"):
        add(task["source_manifest"])
    return records


class Coordinator:
    def __init__(self, options, plan, inventory, environments):
        self.options, self.plan = options, plan
        self.repo, self.output = options.repo.resolve(), options.output.resolve()
        self.tasks = prepare_plan(plan, options.threads)
        self.inventory, self.environments = inventory, environments
        self.engines = {"extended": options.extended_runner.resolve(), "transfer": options.transfer_runner.resolve()}
        self.helper = options.helpers.resolve()
        self.manifest_path = self.output / "manifest.json"
        self.active = {}
        self.codes = {str(Path(__file__).resolve()): digest(__file__), str(self.helper): digest(self.helper)}
        for task in self.tasks:
            engine = self.engines[task["runner"]]
            self.codes[str(engine)] = digest(engine)
            if task["runner"] == "extended" and task.get("model") == "PINN" and task.get("loss_variant", "author_current") != "author_current":
                loss_helper = engine.with_name("run_transfer.py")
                self.codes[str(loss_helper)] = digest(loss_helper)
                common = engine.with_name("run_reproduction.py")
                self.codes[str(common)] = digest(common)
            if task["runner"] == "transfer":
                common = engine.with_name("run_reproduction.py")
                self.codes[str(common)] = digest(common)
                if task.get("validation_protocol") == "explicit_battery_split":
                    extended = engine.with_name("run_extended.py")
                    self.codes[str(extended)] = digest(extended)
        signatures = {stable_hash(value): value for value in environments.values()}
        self.frozen = {"plan_sha256": stable_hash(plan), "repo": str(self.repo),
                       "source_fingerprint": inventory["fingerprint"],
                       "data_directory_order_sha256": stable_hash(inventory["data_directory_order"]),
                       "implementation_sha256": self.codes, "environment_signatures": signatures,
                       "threads": options.threads, "smoke": options.smoke, "allow_cpu": options.allow_cpu}

    def task_output(self, task):
        destination = (self.output / "tasks" / task["task_id"]).resolve()
        if self.output / "tasks" not in destination.parents:
            raise ValueError("Task output escaped the task root")
        return destination

    def expected_implementation(self, task):
        engine = self.engines[task["runner"]]
        if task["runner"] == "extended":
            result = {"run_extended.py": self.codes[str(engine)], "helpers": self.codes[str(self.helper)]}
            if task.get("model") == "PINN" and task.get("loss_variant", "author_current") != "author_current":
                result["run_transfer.py"] = self.codes[str(engine.with_name("run_transfer.py"))]
            return result
        result = {"run_transfer.py": self.codes[str(engine)],
                  "run_reproduction.py": self.codes[str(engine.with_name("run_reproduction.py"))]}
        if task.get("validation_protocol") == "explicit_battery_split":
            result["run_extended.py"] = self.codes[str(engine.with_name("run_extended.py"))]
        return result

    def validate_complete(self, task, expected_inputs, gpu=None):
        output = self.task_output(task)
        record = read_json(output / "manifest.json")
        task_id = task["task_id"]
        if record.get("status") != "complete" or record.get("task_id") != task_id:
            raise ValueError("Task manifest does not confirm completion: " + task_id)
        if record.get("task_sha256") != stable_hash(task):
            raise ValueError("Completed task specification changed: " + task_id)
        if record.get("source_fingerprint") != self.inventory["fingerprint"]:
            raise ValueError("Completed task source/data changed: " + task_id)
        if record.get("implementation_sha256") != self.expected_implementation(task):
            raise ValueError("Completed task runner implementation changed: " + task_id)
        artifacts = record.get("artifact_sha256")
        required = set(CORE_ARTIFACTS)
        if task.get("family") == "transfer":
            required.update(["source_only_true_label.npy", "source_only_pred_label.npy"])
        if not isinstance(artifacts, dict) or not required.issubset(artifacts):
            raise ValueError("Completed task lacks required artifact hashes: " + task_id)
        for name, expected in artifacts.items():
            if not isinstance(name, str) or not isinstance(expected, str) or not re.fullmatch("[a-fA-F0-9]{64}", expected):
                raise ValueError("Malformed artifact hash: " + task_id)
            path = (output / name).resolve()
            if output not in path.parents or not path.is_file() or digest(path) != expected.lower():
                raise ValueError("Completed artifact is missing, changed, or outside task output: " + task_id + "/" + name)
        config = read_json(output / "config.json")
        if config.get("source_fingerprint") != self.inventory["fingerprint"]:
            raise ValueError("Completed task configuration source/data changed: " + task_id)
        order = (config.get("data_directory_order_sha256") if task["runner"] == "extended"
                 else stable_hash(config.get("source_directory_order")))
        if order != stable_hash(self.inventory["data_directory_order"]):
            raise ValueError("Completed task data directory enumeration changed: " + task_id)
        for path, checksum in config.get("reference_data_sha256", {}).items():
            if expected_inputs.get(str(Path(path).resolve())) != checksum:
                raise ValueError("Worker observed different referenced battery data: " + task_id)
        if config.get("reference_split_sha256"):
            split = task.get("split_json") or str(Path(task["reference_run"]) / "splits.json")
            split = Path(split)
            split = split.resolve() if split.is_absolute() else (self.repo / split).resolve()
            if expected_inputs.get(str(split)) != config["reference_split_sha256"]:
                raise ValueError("Worker observed a different reference split: " + task_id)
        lineage = config.get("source_checkpoint_lineage")
        if lineage:
            checkpoint = Path(lineage["path"]).resolve()
            if expected_inputs.get(str(checkpoint)) != lineage["sha256"]:
                raise ValueError("Worker observed a different source checkpoint: " + task_id)
        effective = record.get("effective_task", config.get("task", {}))
        actual_smoke = bool(config.get("smoke", False) or effective.get("smoke", False))
        expected_smoke = bool(self.options.smoke or task.get("smoke", False))
        if actual_smoke != expected_smoke:
            raise ValueError("Smoke/formal task configuration mismatch: " + task_id)
        actual = normalize_environment(config.get("environment", {}))
        essential = {"python", "torch", "numpy", "pandas", "sklearn", "cuda", "cuda_available", "gpu"}
        if not essential.issubset(actual):
            raise ValueError("Completed task environment is incomplete: " + task_id)
        candidates = [self.environments[gpu]] if gpu else self.environments.values()
        if not any(all(expected.get(key) == value for key, value in actual.items()) for expected in candidates):
            raise ValueError("Completed task runtime environment changed: " + task_id)
        if input_fingerprints(task, self.repo, self.inventory) != expected_inputs:
            raise ValueError("Completed task reference inputs changed: " + task_id)
        return record

    def save(self):
        self.state["updated_at"] = utc_now()
        self.state["counts"] = dict(Counter(item["status"] for item in self.state["tasks"].values()))
        atomic_json(self.manifest_path, self.state)

    def output_hashes(self, task):
        output = self.task_output(task)
        files = {}
        for path in output.rglob("*"):
            if path.is_file():
                if output not in path.resolve().parents:
                    raise ValueError("Task output contains an external file link")
                files[path.relative_to(output).as_posix()] = digest(path)
        return files

    def initialize(self):
        if self.manifest_path.exists():
            state = read_json(self.manifest_path)
            if state.get("format") != FORMAT:
                raise ValueError("Output belongs to a different writer; use a new output directory")
            if not self.options.resume:
                raise ValueError("Existing suite requires --resume")
            if state.get("frozen") != self.frozen:
                raise ValueError("Plan/source/code/environment/configuration changed; use a new output directory")
            self.state = state
            if set(state["tasks"]) != {task["task_id"] for task in self.tasks}:
                raise ValueError("Common manifest task set changed")
            for task in self.tasks:
                record = state["tasks"][task["task_id"]]
                if record["status"] == "launching":
                    raise RuntimeError("A previous launch stopped before worker ownership was recorded; inspect it before reusing output: " + task["task_id"])
                identity = record.get("process_identity")
                if record["status"] == "running" and identity and process_identity(identity["pid"]) == identity:
                    raise RuntimeError("A previous worker is still active; refusing duplicate task: " + task["task_id"])
                if record["status"] == "complete":
                    self.validate_complete(task, record.get("input_sha256", {}))
                    if not record.get("verified_file_sha256") or self.output_hashes(task) != record["verified_file_sha256"]:
                        raise ValueError("Completed task file tree changed: " + task["task_id"])
                else:
                    record.update(status="pending", resumed_at=utc_now())
        else:
            unexpected = [path for path in self.output.iterdir() if path.name != ".full_suite.lock"]
            if unexpected:
                raise ValueError("Output is not empty and has no recognized suite manifest")
            self.state = {"format": FORMAT, "created_at": utc_now(), "frozen": self.frozen,
                          "tasks": {task["task_id"]: {"status": "pending", "task_sha256": stable_hash(task),
                              "runner": task["runner"], "attempts": 0} for task in self.tasks}}
            atomic_json(self.output / "plan.json", self.plan)
            atomic_json(self.output / "source_inventory.json", self.inventory)
        self.state.update(status="running", coordinator_identity=process_identity(os.getpid()))
        self.save()

    def command(self, task, task_json):
        command = [sys.executable, "-u", str(self.engines[task["runner"]]), "--task-json", str(task_json),
                   "--repo", str(self.repo), "--output", str(self.task_output(task)), "--resume"]
        if task["runner"] == "extended":
            command.extend(["--helpers", str(self.helper)])
        if self.options.smoke or task.get("smoke", False):
            command.append("--smoke")
        if self.options.allow_cpu:
            command.append("--allow-cpu")
        return command

    def launch(self, task, gpu):
        task_id = task["task_id"]
        record = self.state["tasks"][task_id]
        for name, expected in self.codes.items():
            if digest(name) != expected:
                raise RuntimeError("Frozen implementation changed during scheduling: " + name)
        inputs = input_fingerprints(task, self.repo, self.inventory)
        metadata = self.output / ".coordinator"
        metadata.mkdir(exist_ok=True)
        key = hashlib.sha256(task_id.encode()).hexdigest()
        task_json = metadata / (key + ".json")
        atomic_json(task_json, task)
        attempt = record["attempts"] + 1
        log_path = metadata / (key + ".attempt" + str(attempt) + ".log")
        stream = log_path.open("w", encoding="utf-8", buffering=1)
        command = self.command(task, task_json)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS=str(self.options.threads),
                           MKL_NUM_THREADS=str(self.options.threads), OPENBLAS_NUM_THREADS=str(self.options.threads),
                           NUMEXPR_NUM_THREADS=str(self.options.threads), PYTHONUNBUFFERED="1")
        record.update(status="launching", input_sha256=inputs, command=command,
                      process_log=str(log_path.relative_to(self.output)), launch_intent_at=utc_now())
        self.save()
        try:
            process = subprocess.Popen(command, env=environment, cwd=self.repo, stdout=stream,
                                       stderr=subprocess.STDOUT)
        except BaseException:
            stream.close()
            raise
        self.active[task_id] = {"process": process, "stream": stream, "gpu": gpu, "task": task}
        for name in ["error", "blocked_by", "finished_at", "returncode"]:
            record.pop(name, None)
        record.update(status="running", attempts=attempt, assigned_gpu=gpu, started_at=utc_now(),
                      process_identity=process_identity(process.pid), subprocess_pid=process.pid,
                      process_log=str(log_path.relative_to(self.output)), command=command, input_sha256=inputs)
        self.save()
        print("START " + task_id + " GPU=" + gpu, flush=True)

    def collect(self):
        for task_id, active in list(self.active.items()):
            returncode = active["process"].poll()
            if returncode is None:
                continue
            active["stream"].close()
            record = self.state["tasks"][task_id]
            try:
                if returncode != 0:
                    raise RuntimeError("Worker exited with code " + str(returncode))
                result = self.validate_complete(active["task"], record["input_sha256"], active["gpu"])
                record.update(status="complete", worker_manifest_sha256=digest(self.task_output(active["task"]) / "manifest.json"),
                              artifact_sha256=result["artifact_sha256"], metrics=result.get("metrics"),
                              source_fingerprint=result["source_fingerprint"],
                              verified_file_sha256=self.output_hashes(active["task"]))
                record.pop("error", None)
            except Exception as error:
                record.update(status="failed", error=repr(error))
            record.update(returncode=returncode, finished_at=utc_now())
            del self.active[task_id]
            self.save()
            print(record["status"].upper() + " " + task_id + (" " + record["error"] if record.get("error") else ""), flush=True)

    def stop_own_workers(self):
        for active in self.active.values():
            if active["process"].poll() is None:
                active["process"].terminate()
        deadline = time.monotonic() + 5
        for task_id, active in self.active.items():
            try:
                active["process"].wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                active["process"].kill()
                active["process"].wait()
            active["stream"].close()
            self.state["tasks"][task_id].update(status="interrupted", finished_at=utc_now(),
                                                returncode=active["process"].returncode)
        self.active.clear()

    def pause(self):
        time.sleep(0.1)

    def run(self):
        self.output.mkdir(parents=True, exist_ok=True)
        if self.output == self.repo or self.repo in self.output.parents or self.output in self.repo.parents:
            raise ValueError("Suite output must be separate from the author repository")
        with coordinator_lock(self.output):
            self.initialize()
            try:
                while True:
                    self.collect()
                    occupied = Counter(active["gpu"] for active in self.active.values())
                    for task in self.tasks:
                        record = self.state["tasks"][task["task_id"]]
                        if record["status"] != "pending":
                            continue
                        states = {parent: self.state["tasks"][parent]["status"] for parent in dependencies(task)}
                        failed = [parent for parent, status in states.items() if status in {"failed", "blocked", "interrupted"}]
                        if failed:
                            record.update(status="blocked", blocked_by=failed, finished_at=utc_now())
                            self.save()
                            continue
                        if any(status != "complete" for status in states.values()):
                            continue
                        available = [gpu for gpu in self.options.gpus if occupied[gpu] < self.options.workers_per_gpu]
                        if not available:
                            break
                        gpu = min(available, key=lambda candidate: occupied[candidate])
                        try:
                            self.launch(task, gpu)
                            occupied[gpu] += 1
                        except Exception as error:
                            record.update(status="failed", error=repr(error), finished_at=utc_now())
                            self.save()
                            print("FAILED " + task["task_id"] + " " + repr(error), flush=True)
                    if not self.active and not any(record["status"] == "pending" for record in self.state["tasks"].values()):
                        break
                    self.pause()
                self.state.update(status="complete" if all(record["status"] == "complete" for record in self.state["tasks"].values()) else "failed",
                                  finished_at=utc_now())
                self.save()
                return 0 if self.state["status"] == "complete" else 1
            except BaseException:
                self.stop_own_workers()
                self.state.update(status="interrupted", finished_at=utc_now())
                self.save()
                raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan", type=Path)
    result.add_argument("--repo", type=Path)
    result.add_argument("--output", "--output-root", dest="output", type=Path, required=True)
    result.add_argument("--gpus", nargs="+", default=["0"])
    result.add_argument("--workers-per-gpu", type=int, default=3)
    result.add_argument("--threads", type=int, default=2)
    result.add_argument("--extended-runner", type=Path, default=Path(__file__).with_name("run_extended.py"))
    result.add_argument("--transfer-runner", type=Path, default=Path(__file__).with_name("run_transfer.py"))
    result.add_argument("--helpers", type=Path, default=Path(__file__).with_name("run_reproduction.py"))
    result.add_argument("--resume", action="store_true")
    result.add_argument("--smoke", action="store_true")
    result.add_argument("--allow-cpu", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--summary", action="store_true")
    return result


def main(argv=None):
    options = parser().parse_args(argv)
    if options.summary:
        state = read_json(options.output / "manifest.json")
        if state.get("format") != FORMAT:
            raise ValueError("Not a recognized full-suite manifest")
        print(json.dumps({"status": state["status"], "updated_at": state.get("updated_at"),
                          "counts": dict(Counter(item["status"] for item in state["tasks"].values())),
                          "failures": {key: value.get("error", value.get("blocked_by"))
                              for key, value in state["tasks"].items() if value["status"] in {"failed", "blocked"}}}, indent=2))
        return 0
    if options.plan is None or options.repo is None:
        raise ValueError("--plan and --repo are required unless --summary is used")
    if options.workers_per_gpu < 1 or options.threads not in {1, 2, 3, 4}:
        raise ValueError("Invalid worker/thread limit")
    if not options.gpus or len(set(options.gpus)) != len(options.gpus) or any(not re.fullmatch(r"[0-9]+|GPU-[A-Za-z0-9-]+|MIG-[A-Za-z0-9/-]+", gpu) for gpu in options.gpus):
        raise ValueError("GPU slots must be unique numeric indices or NVIDIA GPU/MIG identifiers")
    plan = read_json(options.plan)
    tasks = prepare_plan(plan, options.threads)
    if options.dry_run:
        print(json.dumps({"plan_sha256": stable_hash(plan), "task_count": len(tasks),
                          "runners": dict(Counter(task["runner"] for task in tasks)),
                          "gpus": options.gpus, "maximum_workers": len(options.gpus) * options.workers_per_gpu,
                          "topological_order": [task["task_id"] for task in tasks],
                          "smoke": options.smoke, "training_started": False}, indent=2))
        return 0
    inventory = load_inventory(options.helpers.resolve(), options.repo.resolve())
    environments = probe_environment(options.gpus, options.allow_cpu)
    coordinator = Coordinator(options, plan, inventory, environments)
    return coordinator.run()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("INTERRUPTED: only workers started by this coordinator were terminated", file=sys.stderr)
        sys.exit(130)
