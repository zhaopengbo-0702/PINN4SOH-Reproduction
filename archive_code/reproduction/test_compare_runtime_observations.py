import ast
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import compare_runtime_observations as observation


class ObservationTests(unittest.TestCase):
    def test_environment_does_not_mutate_parent_and_keeps_backend_flags(self):
        original = {"PYTHONPATH": "bad", "PYTHONHOME": "bad", "CONDA_PREFIX": "/wrong",
                    "NVIDIA_TF32_OVERRIDE": "0", "CUDA_FORCE_PTX_JIT": "1", "LD_LIBRARY_PATH": "/modernlib"}
        with patch.dict(observation.os.environ, original, clear=True):
            result = observation.environment_for(Path("/usr/bin/python"), "0", False)
            self.assertEqual(dict(observation.os.environ), original)
            self.assertNotIn("PYTHONPATH", result)
            self.assertNotIn("PYTHONHOME", result)
            self.assertNotIn("CONDA_PREFIX", result)
            self.assertEqual(result["LD_LIBRARY_PATH"], "/modernlib")
            self.assertEqual(result["NVIDIA_TF32_OVERRIDE"], "0")
            self.assertEqual(result["CUDA_FORCE_PTX_JIT"], "1")
            self.assertEqual(result["OMP_NUM_THREADS"], "2")

    def test_snapshot_success_isolated_without_backend_assignments(self):
        payload = {"matmul_allow_tf32": True, "actual_threads": 2}
        completed = subprocess.CompletedProcess([], 0, "library banner\nPINN4SOH_SNAPSHOT_JSON=" + json.dumps(payload) + "\n", "warning")
        with tempfile.TemporaryDirectory() as directory, patch.object(observation.subprocess, "run", return_value=completed) as run:
            binary = Path(directory) / "python"
            binary.write_bytes(b"fixture")
            result = observation.snapshot(binary, "0", True)
            self.assertTrue(result["passed"])
            self.assertEqual(result["snapshot"], payload)
            self.assertEqual(run.call_args[0][0][1:4], ["-B", "-I", "-c"])
            self.assertEqual(result["stderr"], "warning")
        parsed = ast.parse(observation.SNAPSHOT_CODE)
        for node in ast.walk(parsed):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        self.assertNotIn(target.attr, ["allow_tf32", "benchmark", "deterministic", "enabled"])

    def test_failed_import_preserves_full_stderr(self):
        completed = subprocess.CompletedProcess([], 1, "", "full trace\nImportError: libcudnn")
        with tempfile.TemporaryDirectory() as directory, patch.object(observation.subprocess, "run", return_value=completed):
            binary = Path(directory) / "python"
            binary.write_bytes(b"fixture")
            result = observation.snapshot(binary, "0", True)
            self.assertFalse(result["passed"])
            self.assertEqual(result["stderr"], completed.stderr)
            self.assertIsNone(result["snapshot"])

    def test_timeout_preserves_failure_in_result(self):
        with patch.object(observation.subprocess, "run", side_effect=subprocess.TimeoutExpired("snapshot", 55)):
            result = observation.snapshot(Path("/not/executed"), "0", True)
            self.assertFalse(result["passed"])
            self.assertIn("TimeoutExpired", result["error"])

    def test_explicit_modern_site_only_enters_modern_child_argument(self):
        completed = subprocess.CompletedProcess([], 0, 'PINN4SOH_SNAPSHOT_JSON={}\n', '')
        with tempfile.TemporaryDirectory() as directory, patch.object(observation.subprocess, "run", return_value=completed) as run:
            root = Path(directory).resolve()
            site, binary = root / "site-packages", root / "python"
            (site / "pandas").mkdir(parents=True)
            binary.write_bytes(b"fixture")
            modern = observation.snapshot(binary, "0", False, site)
            self.assertTrue(modern["passed"])
            self.assertEqual(run.call_args[0][0][-1], str(site))
            self.assertEqual(modern["explicit_modern_site"], str(site))
            with self.assertRaisesRegex(ValueError, "never enter legacy"):
                observation.snapshot(binary, "0", True, site)

    def test_missing_or_empty_explicit_site_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                observation.validate_explicit_site(root / "missing")
            with self.assertRaises(ValueError):
                observation.validate_explicit_site(root)

    def test_child_injection_does_not_process_pth_files(self):
        parsed = ast.parse(observation.SNAPSHOT_CODE)
        self.assertNotIn("site.addsitedir(", observation.SNAPSHOT_CODE)
        self.assertIn("sys.path.insert(0,str(site))", observation.SNAPSHOT_CODE)
        self.assertIn("from_explicit_dependency_directory", observation.SNAPSHOT_CODE)


if __name__ == "__main__":
    unittest.main()
