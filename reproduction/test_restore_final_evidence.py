"""Verify exact restoration and rejection of corrupt or escaping evidence bundles."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

SCRIPT = Path(__file__).resolve().with_name("restore_final_evidence.py")


def main():
    spec = importlib.util.spec_from_file_location("restore_helper", SCRIPT)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    with tempfile.TemporaryDirectory(prefix="pinn4soh_restore_test_") as temporary:
        root = Path(temporary)
        files = {name: "\ufeff原始证据\r\n" + name + "\r\n" for name in sorted(helper.EXPECTED_FILES)}
        bundle = {"format": "pinn4soh-final-evidence-v1",
                  "audit": {"passed": True, "expected_runs": 110, "status_counts": {"complete": 110}},
                  "files": files, "sha256": {name: hashlib.sha256(text.encode("utf-8")).hexdigest()
                                             for name, text in files.items()}}
        source = root / "bundle.json"

        def save(value):
            source.write_bytes(json.dumps(value, ensure_ascii=False).encode("utf-8"))

        def run(output, success):
            result = subprocess.run([sys.executable, "-S", str(SCRIPT), str(source), "--output-dir", str(output)],
                                    capture_output=True, text=True)
            assert (result.returncode == 0) == success, result.stdout + result.stderr
            return result

        save(bundle)
        output = root / "restored"
        run(output, True)
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in output.rglob("*") if path.is_file()}
        assert len(before) == 9
        for relative, text in files.items():
            assert (output / relative).read_bytes() == text.encode("utf-8")
        result = run(output, True)
        assert json.loads(result.stdout)["identical_files_preserved"] == 9
        assert all((path.read_bytes(), path.stat().st_mtime_ns) == value for path, value in before.items())
        print("PASS stdlib-only exact UTF-8 BOM/CRLF restoration; identical originals retain their mtimes")

        corrupt = json.loads(json.dumps(bundle))
        corrupt["files"]["manifest.json"] += "tampered"
        save(corrupt)
        rejected = root / "corrupt_output"
        result = run(rejected, False)
        assert "SHA256 mismatch" in result.stderr and not rejected.exists()
        print("PASS corrupt hash rejected before any file is written")

        for bad_path in ["../escape.json", "/absolute.json", "XJTU\\escape.json", "C:/drive.json"]:
            escaping = json.loads(json.dumps(bundle))
            text = escaping["files"].pop("manifest.json")
            sha = escaping["sha256"].pop("manifest.json")
            escaping["files"][bad_path], escaping["sha256"][bad_path] = text, sha
            save(escaping)
            rejected = root / "escaping_output"
            run(rejected, False)
            assert not rejected.exists() and not (root / "escape.json").exists()
        print("PASS traversal, absolute, backslash and Windows-drive paths rejected")

        save(bundle)
        conflict = root / "conflict"
        conflict.mkdir()
        (conflict / "manifest.json").write_bytes(b"existing original")
        run(conflict, False)
        assert list(conflict.iterdir()) == [conflict / "manifest.json"]
        assert (conflict / "manifest.json").read_bytes() == b"existing original"
        print("PASS differing originals refused before restoring any other files")


if __name__ == "__main__":
    main()
