"""Bounded local safety/compatibility checks; no legacy execution or training."""

import argparse
import ast
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import install_legacy_runtime as install
import launch_legacy_smoke as launch
import launch_legacy_control as launch_control
import prepare_legacy_smoke as prepare
import prepare_legacy_control as control
import probe_legacy_runtime as probe


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "outputs/full_reproduction/plans/stage1_plan.json"


class LegacyTools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="pinn-legacy-tools-")
        cls.base = Path(cls.temp.name)
        cls.support = cls.base / "support"
        cls.prepared = prepare.prepare(ROOT / "reproduction", PLAN, cls.support)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def prefixer(self):
        with patch.object(sys, "platform", "linux"):
            return install.load_prefixer()

    def test_deployment_rejects_multiple_gpus_and_workers(self):
        smoke_base = ["--prefix", str(self.base / "prefix"), "--support", str(self.support),
                      "--repo", str(self.base / "repo"), "--output", str(self.base / "out"), "--dry-run"]
        control_base = ["--prefix", str(self.base / "prefix"), "--support", str(self.support),
                        "--control", str(self.base / "control"), "--smoke-evidence", str(self.base / "smoke.json"),
                        "--repo", str(self.base / "repo"), "--dry-run"]
        for extra in [["--gpus", "0", "1"], ["--workers-per-gpu", "2"], ["--gpus", "1"]]:
            with self.assertRaisesRegex(ValueError, "one allocated GPU"):
                launch.main(smoke_base + extra)
            with patch.object(sys, "argv", ["launch_legacy_control.py"] + control_base + extra):
                with self.assertRaisesRegex(ValueError, "one allocated GPU"):
                    launch_control.main()

    def test_archive_traversal_rejected(self):
        for name in ["../outside", "/absolute", "C:/absolute", "a\\..\\outside", "a/../../outside"]:
            with self.assertRaises(ValueError):
                install.inside(self.base, name)

    def test_binary_relocation_preserves_suffixes_and_length(self):
        prefixer = self.prefixer()
        old, new = "/old/" + "x" * 40, "/workspace/legacy"
        raw = ("start\0" + old + "/lib:" + old + "/bin\0end").encode()
        output = prefixer.replace_prefix(raw, "binary", old, new)
        self.assertEqual(len(output), len(raw))
        self.assertIn((new + "/lib:" + new + "/bin\0").encode(), output)
        self.assertNotIn(old.encode(), output)

    def test_binary_longer_prefix_is_rejected(self):
        with self.assertRaises(ValueError):
            self.prefixer().replace_prefix(b"/old/lib\0", "binary", "/old", "/too/long/new/prefix")

    def test_text_shebang_and_sysconfig_relocation(self):
        prefixer = self.prefixer()
        raw = b"#!/old/prefix/bin/python3.7\nPREFIX='/old/prefix'\n"
        self.assertEqual(prefixer.replace_prefix(raw, "text", "/old/prefix", "/workspace/legacy"),
                         b"#!/workspace/legacy/bin/python3.7\nPREFIX='/workspace/legacy'\n")

    def test_has_prefix_metadata_parsing(self):
        data = io.BytesIO()
        content = b'"/old/prefix" binary "bin/python3.7"\n/old/prefix text lib/config.py\n'
        with tarfile.open(fileobj=data, mode="w") as archive:
            item = tarfile.TarInfo("info/has_prefix")
            item.size = len(content)
            archive.addfile(item, io.BytesIO(content))
        data.seek(0)
        with tarfile.open(fileobj=data) as archive:
            rows = install.prefix_records(archive)
        self.assertEqual([row["mode"] for row in rows], ["binary", "text"])

    def test_frozen_runners_unchanged(self):
        for name, record in self.prepared["runner_files"].items():
            self.assertEqual(prepare.digest(ROOT / "reproduction" / name), record["original_sha256"])
            if name in ["run_reproduction.py", "run_extended.py"]:
                self.assertEqual(record["original_sha256"], record["compat_sha256"])

    def test_only_required_api_patches(self):
        transfer = (self.support / "legacy_compat/run_transfer.py").read_text(encoding="utf-8")
        coordinator = (self.support / "legacy_compat/run_full_suite.py").read_text(encoding="utf-8")
        self.assertNotIn("weights_only", transfer)
        self.assertNotIn("missing_ok", coordinator)
        self.assertEqual(transfer.count("checkpoint = torch.load("), 2)

    def test_python37_syntax(self):
        paths = list((self.support / "legacy_compat").glob("*.py"))
        paths += [ROOT / "reproduction" / name for name in ["install_legacy_runtime.py", "probe_legacy_runtime.py", "launch_legacy_smoke.py"]]
        for path in paths:
            ast.parse(path.read_text(encoding="utf-8"), feature_version=7)

    def test_twelve_tasks_keep_scientific_configuration(self):
        original = {row["task_id"]: row for row in json.loads(PLAN.read_text(encoding="utf-8"))["tasks"]}
        tasks = json.loads((self.support / "legacy_smoke_plan.json").read_text(encoding="utf-8"))["tasks"]
        self.assertEqual(len(tasks), 12)
        self.assertEqual(sum(row["family"] == "transfer" for row in tasks), 2)
        self.assertEqual({row["dataset"] for row in tasks if row["model"] == "PINN" and row["family"] == "full"},
                         {"XJTU", "MIT", "HUST", "TJU"})
        for row in tasks:
            reference = original[row["original_task_id"]]
            for name, value in reference.items():
                if name not in {"task_id", "epochs", "adaptation_epochs"}:
                    self.assertEqual(row[name], value)
            self.assertTrue(row["smoke"])
            self.assertEqual(row["epochs"], 2)

    def test_inherited_environment_is_removed(self):
        with patch.dict(os.environ, {"PYTHONPATH": "/host/cp310", "PYTHONHOME": "/host", "CONDA_PREFIX": "/host/env"}):
            environment = install.isolated_environment(self.base)
        self.assertNotIn("PYTHONPATH", environment)
        self.assertNotIn("PYTHONHOME", environment)
        self.assertNotIn("CONDA_PREFIX", environment)
        self.assertEqual(environment["PYTHONNOUSERSITE"], "1")
        self.assertNotIn("/host", environment["LD_LIBRARY_PATH"])

    def test_dry_run_never_executes_runtime(self):
        code = launch.main(["--prefix", str(self.base / "prefix"), "--support", str(self.support),
            "--repo", str(ROOT / "PINN4SOH"), "--output", str(self.base / "results"), "--dry-run"])
        self.assertEqual(code, 0)
        self.assertFalse((self.base / "results").exists())

    def test_current_interpreter_cannot_masquerade_as_legacy(self):
        if sys.platform != "linux" or sys.version_info[:3] != (3, 7, 10):
            with self.assertRaisesRegex(RuntimeError, "CPython 3.7.10"):
                probe.probe(self.base, False)

    def test_formal_control_has_all_directions_and_matched_sources(self):
        with tempfile.TemporaryDirectory(dir=self.base) as directory:
            output = Path(directory) / "control"
            design = control.prepare(PLAN, PLAN.with_name("stage2_reproduced_transfer_plan.json"), output,
                "/workspace/pinn4soh_legacy_formal_control_20261110")
            plan = json.loads((output / "legacy_control_plan.json").read_text(encoding="utf-8"))
            self.assertEqual(design["task_count"], 40)
            spec = importlib.util.spec_from_file_location("control_coordinator_test", self.support / "legacy_compat/run_full_suite.py")
            coordinator = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(coordinator)
            ordered = coordinator.prepare_plan(plan, 2)
            self.assertEqual(len(ordered), 40)
            index = {row["task_id"]: position for position, row in enumerate(ordered)}
            for row in ordered:
                self.assertEqual(row["repeat"], 1)
                self.assertEqual(row["epochs"], 200)
                if row["family"] == "transfer":
                    self.assertEqual(len(row["depends_on"]), 1)
                    source_id = row["depends_on"][0]
                    self.assertLess(index[source_id], index[row["task_id"]])
                    self.assertEqual(row["source_run"], "/workspace/pinn4soh_legacy_formal_control_20261110/tasks/" + source_id)
                    source = ordered[index[source_id]]
                    self.assertEqual(source["loss_variant"], row["loss_variant"])
                    self.assertEqual(source["repeat"], row["repeat"])
            transfer = [row for row in ordered if row["family"] == "transfer"]
            self.assertEqual(len({(row["source"], row["target"]) for row in transfer}), 12)
            self.assertEqual(set(row["loss_variant"] for row in transfer), set(control.LOSSES))


if __name__ == "__main__":
    unittest.main()
