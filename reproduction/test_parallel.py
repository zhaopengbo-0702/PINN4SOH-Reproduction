"""Integration checks with fake training; no optimization or GPU work is performed."""

import argparse
import _thread
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "run_reproduction.py"
PARALLEL = ROOT / "run_parallel.py"

FAKE_DATA = '''import torch
from torch.utils.data import DataLoader, TensorDataset
class DF:
    def load_one_battery(self, path, nominal_capacity=None):
        return ((torch.zeros(4, 17).numpy(), torch.ones(4).numpy()), (torch.zeros(4, 17).numpy(), torch.ones(4).numpy()))
    def load_all_battery(self, path_list, nominal_capacity):
        for path in path_list:
            self.load_one_battery(path, nominal_capacity)
        loader = DataLoader(TensorDataset(torch.zeros(4, 17), torch.zeros(4, 17), torch.ones(4, 1), torch.ones(4, 1)), batch_size=2)
        return {'train_2': loader, 'valid_2': loader, 'test_3': loader}
'''

FAKE_MAIN = r'''import argparse, json, logging, os, time
from pathlib import Path
import numpy as np
import torch
from dataloader.dataloader import DF
def get_args():
    return argparse.Namespace(data=__name__.split('_')[-1], batch='2C', epochs=200, warmup_epochs=30, batch_size=256, log_dir='logging.txt', save_folder='unused')
def load_data(args):
    data = DF()
    train = data.load_all_battery(['train.csv'], 1.0)
    test = data.load_all_battery(['test.csv'], 1.0)
    return {'train': train['train_2'], 'valid': train['valid_2'], 'test': test['test_3']}
load_MIT_data = load_HUST_data = load_TJU_data = load_data
class PINN(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.logger = logging.getLogger('mylogger')
        self.logger.addHandler(logging.FileHandler(Path(args.save_folder) / args.log_dir))
    def Train(self, trainloader, validloader, testloader):
        folder = Path(self.args.save_folder)
        trace = Path(os.environ['FAKE_TRACE']) / (str(os.getpid()) + '.json')
        event = {'run': str(folder), 'start': time.time()}
        trace.write_text(json.dumps(event))
        time.sleep(float(os.environ.get('FAKE_DELAY', '1.2')))
        if os.environ.get('FAKE_FAIL') and os.environ['FAKE_FAIL'] in folder.as_posix():
            raise RuntimeError('Expected fake failure')
        self.best_model = {'solution_u': {'fake': torch.ones(1)}, 'dynamical_F': {'fake': torch.ones(1)}}
        np.save(folder / 'true_label.npy', np.ones((4, 1)))
        np.save(folder / 'pred_label.npy', np.full((4, 1), 0.99))
        torch.save(self.best_model, folder / 'model.pth')
        (folder / 'logging.txt').write_text('[Train] epoch:1, lr:0.01\n')
        event['end'] = time.time()
        trace.write_text(json.dumps(event))
        self.logger.handlers.clear()
'''


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def invoke(command, environment, expect=0):
    result = subprocess.run([sys.executable, "-u", *map(str, command)], env=environment,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != expect:
        raise AssertionError(f"Exit {result.returncode}, expected {expect}\n{result.stdout}")
    return result.stdout


def snapshots(folder):
    return {path.relative_to(folder).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in folder.rglob("*") if path.is_file()}


def main():
    helpers = load_module("tested_runner", RUNNER)
    scheduler = load_module("tested_scheduler", PARALLEL)
    with tempfile.TemporaryDirectory(prefix="pinn4soh_parallel_test_") as temporary:
        root = Path(temporary)
        repo, trace = root / "repo", root / "trace"
        trace.mkdir()
        for name in ["Model", "dataloader", "data"]:
            (repo / name).mkdir(parents=True)
        (repo / "Model/Model.py").write_text("# Fake model marker\n")
        (repo / "dataloader/__init__.py").write_text("")
        (repo / "dataloader/dataloader.py").write_text(FAKE_DATA)
        for dataset in helpers.BATCHES:
            (repo / ("main_" + dataset + ".py")).write_text(FAKE_MAIN)
        environment = dict(os.environ, FAKE_TRACE=str(trace), FAKE_DELAY="1.2")

        def initialize(output):
            invoke([RUNNER, "--repo", repo, "--output", output, "--dataset", "XJTU",
                    "--batches", "2C", "--repeats", "1", "--allow-cpu"], environment)

        output = root / "success"
        initialize(output)
        first_run = output / "XJTU/2C/Experiment01"
        before = snapshots(first_run)
        metadata = {name: (output / name).read_bytes() for name in ["environment.json", "source_inventory.json", "manifest.json"]}
        invoke([RUNNER, "--repo", repo, "--output", output, "--dataset", "XJTU", "--batches", "2C",
                "--worker", "--repeat-index", "1", "--worker-attempt", "compat", "--resume", "--allow-cpu"], environment)
        assert all((output / name).read_bytes() == contents for name, contents in metadata.items())
        assert snapshots(first_run) == before
        sidecar = json.loads((output / ".workers/XJTU/2C/Experiment01/manifest.json").read_text())
        legacy = json.loads(metadata["manifest.json"])["runs"]["XJTU/2C/Experiment01"]
        assert sidecar["runs"]["XJTU/2C/Experiment01"]["config_hash"] == legacy["config_hash"]
        print("PASS legacy completed config hash, worker skip, metadata and artifacts unchanged", flush=True)
        command = [PARALLEL, "--repo", repo, "--output", output, "--dataset", "XJTU",
                   "--batches", "2C", "3C", "--repeats", "2", "--workers", "3", "--allow-cpu", "--poll-seconds", "0.1"]
        invoke(command, environment)
        assert snapshots(first_run) == before
        manifest = json.loads((output / "manifest.json").read_text())
        assert len(manifest["runs"]) == 4 and all(r["status"] == "complete" for r in manifest["runs"].values())
        intervals = [json.loads(path.read_text()) for path in trace.glob("*.json")]
        edges = sorted([(item["start"], 1) for item in intervals] + [(item["end"], -1) for item in intervals])
        current = maximum = 0
        for _, change in edges:
            current += change
            maximum = max(maximum, current)
        assert 1 < maximum <= 3, maximum
        trace_count = len(list(trace.glob("*.json")))
        all_runs = {key: snapshots(output / key) for key in manifest["runs"]}
        invoke(command, environment)
        assert len(list(trace.glob("*.json"))) == trace_count
        assert all(snapshots(output / key) == value for key, value in all_runs.items())
        assert all((output / name).read_bytes() == metadata[name] for name in ["environment.json", "source_inventory.json"])
        print(f"PASS three-worker concurrency (max={maximum}), all-complete resume launches zero tasks", flush=True)

        failure_output = root / "failure"
        initialize(failure_output)
        failure_command = [failure_output if value == output else value for value in command]
        invoke(failure_command, dict(environment, FAKE_FAIL="3C/Experiment02"), expect=1)
        failed_manifest = json.loads((failure_output / "manifest.json").read_text())
        assert sum(r["status"] == "failed" for r in failed_manifest["runs"].values()) == 1
        invoke(failure_command, environment)
        assert all(r["status"] == "complete" for r in json.loads((failure_output / "manifest.json").read_text())["runs"].values())
        print("PASS failed worker exits nonzero; successful siblings persist and failed task resumes", flush=True)

        interrupted_output = root / "interrupted"
        initialize(interrupted_output)
        options = argparse.Namespace(repo=repo, output=interrupted_output, runner=RUNNER, dataset="XJTU",
                                     batches=["2C", "3C"], repeats=2, workers=3, epochs=200,
                                     seed=20261109, threads=2, allow_cpu=True, poll_seconds=0.1, plots=False)
        manifest = json.loads((interrupted_output / "manifest.json").read_text())
        inventory = json.loads((interrupted_output / "source_inventory.json").read_text())
        tasks = [task for task in scheduler.task_keys(options, helpers) if task["key"] not in manifest["runs"]]
        sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        original_environment = dict(os.environ)
        os.environ.update(environment)
        os.environ["FAKE_DELAY"] = "10"
        timer = threading.Timer(1.5, _thread.interrupt_main)
        timer.start()
        try:
            code = scheduler.run_queue(options, helpers, manifest, tasks, inventory)
            assert code == 130 and sentinel.poll() is None
        finally:
            timer.cancel()
            os.environ.clear()
            os.environ.update(original_environment)
            sentinel.terminate()
            sentinel.wait(timeout=10)
        state = json.loads((interrupted_output / "manifest.json").read_text())
        assert sum(r.get("interrupted", False) for r in state["runs"].values()) == 3
        print("PASS SIGINT terminates only owned workers and marks three interrupted tasks", flush=True)
    print("All parallel checks passed. No neural-network optimization was performed.", flush=True)


if __name__ == "__main__":
    main()
