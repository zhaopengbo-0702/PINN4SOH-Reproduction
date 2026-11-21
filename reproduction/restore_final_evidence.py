"""Restore the nine exact evidence files from a verified final JSON bundle."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

EXPECTED_FILES = {"metrics_per_run.csv", "metrics_summary.csv", "manifest.json",
                  "environment.json", "source_inventory.json",
                  "XJTU/2C/Experiment01/verification_checkpoint.json",
                  "MIT/all/Experiment01/verification_checkpoint.json",
                  "HUST/all/Experiment01/verification_checkpoint.json",
                  "TJU/0/Experiment01/verification_checkpoint.json"}


def restore(bundle_path, output):
    bundle = json.loads(bundle_path.read_bytes().decode("utf-8"))
    audit = bundle.get("audit", {})
    if (bundle.get("format") != "pinn4soh-final-evidence-v1" or audit.get("passed") is not True
            or audit.get("expected_runs") != 110 or audit.get("status_counts") != {"complete": 110}):
        raise ValueError("Bundle is not a passed final audit of exactly 110 complete runs")
    files, hashes = bundle["files"], bundle["sha256"]
    if not isinstance(files, dict) or not isinstance(hashes, dict) or set(files) != set(hashes):
        raise ValueError("File and SHA256 mappings must have identical keys")
    output = output.resolve()
    pending = []
    for relative, text in files.items():
        if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
            raise ValueError("Invalid evidence path: " + repr(relative))
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
            raise ValueError("Evidence path must be a confined canonical relative path: " + relative)
        destination = output.joinpath(*path.parts)
        try:
            destination.resolve().relative_to(output)
        except ValueError:
            raise ValueError("Evidence path escapes output directory: " + relative) from None
        if not isinstance(text, str):
            raise ValueError("Evidence text must be a string: " + relative)
        data = text.encode("utf-8")
        if hashlib.sha256(data).hexdigest() != hashes[relative]:
            raise ValueError("SHA256 mismatch: " + relative)
        if destination.exists() and (not destination.is_file() or destination.read_bytes() != data):
            raise FileExistsError("Refusing to overwrite a different original: " + str(destination))
        for parent in destination.parents:
            if parent == output.parent:
                break
            if parent.exists() and not parent.is_dir():
                raise FileExistsError("Output parent is not a directory: " + str(parent))
        pending.append((destination, data))
    if set(files) != EXPECTED_FILES:
        raise ValueError("Bundle must contain exactly the expected nine evidence files")
    written = unchanged = 0
    for destination, data in pending:
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with destination.open("xb") as stream:
                stream.write(data)
            written += 1
        except FileExistsError:
            if destination.read_bytes() != data:
                raise FileExistsError("Refusing to overwrite a different original: " + str(destination)) from None
            unchanged += 1
        if destination.read_bytes() != data:
            raise IOError("Restored bytes do not match the bundle: " + str(destination))
    return {"passed": True, "complete_runs": 110, "files_written": written,
            "identical_files_preserved": unchanged, "output_dir": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    options = parser.parse_args()
    print(json.dumps(restore(options.bundle, options.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"passed": False, "error": repr(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
