"""Standard-library integration tests; fake workers never import CUDA or train."""

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

import run_full_suite as suite


ENVIRONMENT = {"python": sys.version, "torch": "fake", "numpy": "fake", "pandas": "fake",
               "scipy": "fake", "sklearn": "fake", "cuda": "fake", "cuda_available": True, "gpu": "Fake GPU"}
FAKE_WORKER = r'''
import argparse, hashlib, json, os
from pathlib import Path
import sys, time
p=argparse.ArgumentParser()
p.add_argument('--task-json',type=Path);p.add_argument('--repo',type=Path);p.add_argument('--output',type=Path)
p.add_argument('--helpers',type=Path,default=Path(__file__).with_name('run_reproduction.py'))
p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--allow-cpu',action='store_true')
a=p.parse_args();t=json.loads(a.task_json.read_text());o=a.output;o.mkdir(parents=True,exist_ok=True)
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def j(p,v):Path(p).write_text(json.dumps(v,sort_keys=True))
key=hashlib.sha256(t['task_id'].encode()).hexdigest();events=a.repo/'events';events.mkdir(exist_ok=True)
event=dict(task_id=t['task_id'],gpu=os.environ['CUDA_VISIBLE_DEVICES'],start=time.monotonic(),pid=os.getpid())
time.sleep(t.get('seconds',.2));event['end']=time.monotonic()
j(events/(key+'.'+str(time.time_ns())+'.json'),event)
marker=events/(key+'.failed')
if t.get('fail_once') and not marker.exists():marker.touch();sys.exit(7)
if t['runner']=='extended':
 implementation={'run_extended.py':h(__file__),'helpers':h(a.helpers)}
 if t.get('model')=='PINN' and t.get('loss_variant','author_current')!='author_current':implementation['run_transfer.py']=h(Path(__file__).with_name('run_transfer.py'))
else:
 implementation={'run_transfer.py':h(__file__),'run_reproduction.py':h(Path(__file__).with_name('run_reproduction.py'))}
 if t.get('validation_protocol')=='explicit_battery_split':implementation['run_extended.py']=h(Path(__file__).with_name('run_extended.py'))
environment=dict(t['test_environment']);environment['CUDA_VISIBLE_DEVICES']=os.environ['CUDA_VISIBLE_DEVICES']
order={'data': {'files': [], 'directories': []}}
config={'task':t,'smoke':a.smoke,'environment':environment,'source_fingerprint':t['test_source_fingerprint'],
        'data_directory_order_sha256':hashlib.sha256(json.dumps(order,sort_keys=True).encode()).hexdigest(),
        'source_directory_order':order}
j(o/'config.json',config)
for name in ['model.pth','true_label.npy','pred_label.npy','splits.json']:(o/name).write_bytes(('fake '+name).encode())
names=['config.json','model.pth','true_label.npy','pred_label.npy','splits.json']
if t.get('family')=='transfer':
 for name in ['source_only_true_label.npy','source_only_pred_label.npy']:(o/name).write_bytes(b'fake');names.append(name)
record=dict(task_id=t['task_id'],status='complete',task_sha256=hashlib.sha256(json.dumps(t,sort_keys=True).encode()).hexdigest(),source_fingerprint=t['test_source_fingerprint'],implementation_sha256=implementation,artifact_sha256={name:h(o/name) for name in names},metrics={'MSE':.001})
j(o/'manifest.json',record)
'''


class FullSuiteTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pinn4soh_suite_test_")
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        (self.repo / "author.py").write_text("# Fake source.\n")
        self.engine = self.root / "fake_runner.py"
        self.engine.write_text(FAKE_WORKER)
        self.helper = self.root / "run_reproduction.py"
        self.helper.write_text("# Fake dependency.\n")
        (self.root / "run_extended.py").write_text("# Explicit split dependency.\n")
        (self.root / "run_transfer.py").write_text("# Loss implementation dependency.\n")
        self.inventory = {"fingerprint": suite.digest(self.repo / "author.py"),
                          "sha256": {}, "data_directory_order": {"data": {"files": [], "directories": []}}}
        self.output = self.root / "results"

    def tearDown(self):
        self.temporary.cleanup()

    def task(self, task_id, **kwargs):
        return dict(task_id=task_id, runner="extended", threads=2,
                    test_environment=ENVIRONMENT, test_source_fingerprint=self.inventory["fingerprint"], **kwargs)

    def coordinator(self, tasks, resume=False, inventory=None, environments=None, **kwargs):
        options = suite.parser().parse_args(["--repo", str(self.repo), "--output", str(self.output),
            "--gpus", "0", "1", "--workers-per-gpu", "2", "--extended-runner", str(self.engine),
            "--transfer-runner", str(self.engine), "--helpers", str(self.helper)])
        options.resume = resume
        for key, value in kwargs.items():
            setattr(options, key, value)
        return suite.Coordinator(options, {"tasks": tasks}, inventory or self.inventory,
                                 environments or {"0": ENVIRONMENT, "1": ENVIRONMENT})

    def run_quietly(self, coordinator):
        with contextlib.redirect_stdout(io.StringIO()):
            return coordinator.run()

    def state(self):
        return suite.read_json(self.output / "manifest.json")

    def test_parallel_failure_dependency_and_immutable_resume(self):
        tasks = [self.task("source/ok"), self.task("source/fail", fail_once=True),
                 self.task("dependent/ok", depends_on=["source/ok"]),
                 self.task("dependent/retry", depends_on=["source/fail"]),
                 *[self.task("independent/" + str(i)) for i in range(6)]]
        self.assertEqual(self.run_quietly(self.coordinator(tasks)), 1)
        first = self.state()
        self.assertEqual(first["counts"], {"complete": 8, "failed": 1, "blocked": 1})
        completed = {task_id: (self.output / "tasks" / task_id / "model.pth").stat().st_mtime_ns
                     for task_id, record in first["tasks"].items() if record["status"] == "complete"}
        events = [suite.read_json(path) for path in (self.repo / "events").glob("*.json")]
        maximum_total = 0
        for event in events:
            concurrent = [other for other in events if other["start"] <= event["start"] < other["end"]]
            maximum_total = max(maximum_total, len(concurrent))
            self.assertLessEqual(sum(other["gpu"] == event["gpu"] for other in concurrent), 2)
        self.assertGreater(maximum_total, 1)
        by_id = {event["task_id"]: event for event in events}
        self.assertGreaterEqual(by_id["dependent/ok"]["start"], by_id["source/ok"]["end"])
        self.assertNotIn("dependent/retry", by_id)
        self.assertEqual(self.run_quietly(self.coordinator(tasks, resume=True)), 0)
        second = self.state()
        self.assertEqual(second["counts"], {"complete": len(tasks)})
        self.assertEqual(second["tasks"]["source/fail"]["attempts"], 2)
        self.assertEqual(second["tasks"]["dependent/retry"]["attempts"], 1)
        for task_id, modified in completed.items():
            self.assertEqual((self.output / "tasks" / task_id / "model.pth").stat().st_mtime_ns, modified)
        self.assertEqual(self.run_quietly(self.coordinator(tasks, resume=True)), 0)
        self.assertEqual(self.state()["tasks"], second["tasks"])

    def test_transfer_command_and_checkpoint_inputs(self):
        checkpoint = self.root / "supplied.pth"
        checkpoint.write_bytes(b"source one")
        task = self.task("transfer/one", family="transfer", source_checkpoint=str(checkpoint))
        task["runner"] = "transfer"
        self.assertEqual(self.run_quietly(self.coordinator([task])), 0)
        record = self.state()["tasks"][task["task_id"]]
        self.assertNotIn("--helpers", record["command"])
        self.assertEqual(self.run_quietly(self.coordinator([task], resume=True)), 0)
        checkpoint.write_bytes(b"source changed")
        with self.assertRaisesRegex(ValueError, "reference inputs changed"):
            self.run_quietly(self.coordinator([task], resume=True))

    def test_completed_artifact_tampering_and_worker_manifest_tampering(self):
        task = self.task("integrity/one")
        self.assertEqual(self.run_quietly(self.coordinator([task])), 0)
        model = self.output / "tasks" / task["task_id"] / "model.pth"
        model.write_bytes(b"altered")
        with self.assertRaisesRegex(ValueError, "artifact is missing, changed"):
            self.run_quietly(self.coordinator([task], resume=True))
        manifest_path = model.with_name("manifest.json")
        manifest = suite.read_json(manifest_path)
        manifest["artifact_sha256"]["model.pth"] = suite.digest(model)
        suite.atomic_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "file tree changed"):
            self.run_quietly(self.coordinator([task], resume=True))

    def test_plan_code_source_and_environment_freeze(self):
        task = self.task("freeze/one")
        self.assertEqual(self.run_quietly(self.coordinator([task])), 0)
        changes = [dict(tasks=[dict(task, seed=7)]),
                   dict(tasks=[task], inventory=dict(self.inventory, fingerprint="changed")),
                   dict(tasks=[task], environments={"0": dict(ENVIRONMENT, torch="changed"), "1": ENVIRONMENT})]
        for change in changes:
            with self.subTest(change=change):
                with self.assertRaisesRegex(ValueError, "changed; use a new output"):
                    self.run_quietly(self.coordinator(resume=True, **change))
        self.engine.write_text(FAKE_WORKER + "\n# Changed.\n")
        with self.assertRaisesRegex(ValueError, "changed; use a new output"):
            self.run_quietly(self.coordinator([task], resume=True))

    def test_ids_cycles_and_nested_outputs(self):
        for task_id in ["../bad", "/absolute", "a/../b", "a//b", "a\\b", "C:bad", "a/.", "a/CON", "a/ends."]:
            with self.subTest(task_id=task_id):
                with self.assertRaises(ValueError):
                    suite.prepare_plan({"tasks": [self.task(task_id)]}, 2)
        invalid = [[self.task("one"), self.task("ONE")],
                   [self.task("a"), self.task("a/b")],
                   [self.task("a", depends_on=["b"]), self.task("b", depends_on=["a"])],
                   [self.task("a", depends_on=["missing"])]]
        for tasks in invalid:
            with self.assertRaises(ValueError):
                suite.prepare_plan({"tasks": tasks}, 2)

    def test_lock_old_writer_and_active_previous_pid(self):
        self.output.mkdir()
        with suite.coordinator_lock(self.output):
            with self.assertRaisesRegex(RuntimeError, "Another coordinator"):
                self.run_quietly(self.coordinator([self.task("lock/one")]))
        suite.atomic_json(self.output / "manifest.json", {"runs": {}})
        with self.assertRaisesRegex(ValueError, "different writer"):
            self.run_quietly(self.coordinator([self.task("lock/one")], resume=True))
        (self.output / "manifest.json").unlink()
        task = self.task("lock/one")
        self.assertEqual(self.run_quietly(self.coordinator([task])), 0)
        state = self.state()
        state["tasks"][task["task_id"]].update(status="running", process_identity=suite.process_identity(os.getpid()))
        suite.atomic_json(self.output / "manifest.json", state)
        with self.assertRaisesRegex(RuntimeError, "previous worker is still active"):
            self.run_quietly(self.coordinator([task], resume=True))
        state["tasks"][task["task_id"]].update(status="launching", process_identity=None)
        suite.atomic_json(self.output / "manifest.json", state)
        with self.assertRaisesRegex(RuntimeError, "launch stopped before worker ownership"):
            self.run_quietly(self.coordinator([task], resume=True))

    def test_interrupt_terminates_owned_workers(self):
        coordinator = self.coordinator([self.task("interrupt/one", seconds=10)])
        with mock.patch.object(coordinator, "pause", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_quietly(coordinator)
        state = self.state()
        self.assertEqual(state["status"], "interrupted")
        record = state["tasks"]["interrupt/one"]
        self.assertEqual(record["status"], "interrupted")
        self.assertIsNone(suite.process_identity(record["subprocess_pid"]))
        self.assertFalse(coordinator.active)

    def test_smoke_configuration_isolation(self):
        task = self.task("smoke/one")
        self.assertEqual(self.run_quietly(self.coordinator([task], smoke=True)), 0)
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            self.run_quietly(self.coordinator([task], resume=True))
        self.assertEqual(self.run_quietly(self.coordinator([task], resume=True, smoke=True)), 0)

    def test_alternate_loss_helper_is_frozen_and_verified(self):
        task = self.task("loss/one", model="PINN", loss_variant="paper_prediction_monotonicity")
        self.assertEqual(self.run_quietly(self.coordinator([task])), 0)
        manifest = suite.read_json(self.output / "tasks" / task["task_id"] / "manifest.json")
        self.assertIn("run_transfer.py", manifest["implementation_sha256"])
        (self.root / "run_transfer.py").write_text("# Changed loss implementation.\n")
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            self.run_quietly(self.coordinator([task], resume=True))

    def test_dry_run_and_summary_do_not_probe_training_runtime(self):
        plan = self.root / "plan.json"
        suite.atomic_json(plan, {"tasks": [self.task("dry/one")]})
        with mock.patch.object(suite, "probe_environment", side_effect=AssertionError("Unexpected runtime probe")):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(suite.main(["--plan", str(plan), "--repo", str(self.repo),
                                            "--output", str(self.output), "--dry-run"]), 0)
            self.assertFalse(json.loads(output.getvalue())["training_started"])
            self.assertFalse(self.output.exists())
            self.assertEqual(self.run_quietly(self.coordinator([self.task("dry/one")])), 0)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(suite.main(["--output", str(self.output), "--summary"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
