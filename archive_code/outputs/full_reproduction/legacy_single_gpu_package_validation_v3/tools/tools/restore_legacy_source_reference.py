"""Restore exact optional source/reference support without overwriting a path."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

from reassemble_legacy_upload import check_ancestors, require, safe_extract_archive, sha256, write_json


SOURCE_SHA = "28ded058f73c3c74acda043a94d555e3c9cda34ddee9e84d22588e30caf2f504"
SOURCE_BYTES = 39235820
ROOTS = {"PINN4SOH-main", "pinn4soh_main_20261109"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True, help="Fresh staging directory")
    parser.add_argument("--workspace", type=Path, required=True, help="Existing /workspace; both restored directories must be absent")
    args = parser.parse_args()
    archive, stage, workspace = [Path(os.path.abspath(str(path))) for path in [args.archive, args.stage, args.workspace]]
    for path in [archive, stage, workspace]:
        check_ancestors(path)
    require(archive.is_file() and archive.stat().st_size == SOURCE_BYTES and sha256(archive) == SOURCE_SHA,
            "Optional source/reference archive identity differs")
    require(workspace.is_dir() and not os.path.lexists(str(stage)) and stage.parent.is_dir(), "Invalid fresh staging/workspace path")
    require(stage != workspace and workspace not in stage.parents and stage not in workspace.parents,
            "Staging must be outside destination workspace")
    evidence_path = workspace / "LEGACY_SOURCE_REFERENCE_RESTORED.json"
    require(not os.path.lexists(str(evidence_path)), "Existing restore evidence; no overwrite permitted")
    for name in ROOTS:
        require(not os.path.lexists(str(workspace / name)), "Existing source/reference path; verify existing data instead of restoring")
    stage.mkdir()
    try:
        safe_extract_archive(archive, stage / "payload", 427, 200 * 1024 * 1024)
        payload = stage / "payload"
        metadata = json.loads((payload / "LEGACY_SOURCE_REFERENCE_SHA256.json").read_text(encoding="utf-8"))
        expected = metadata["files"]
        require(len(expected) == 426 and {Path(name).parts[0] for name in expected} == ROOTS, "Unexpected support file inventory")
        require(metadata["source_fingerprint"] == metadata["inventory"]["fingerprint"] ==
                hashlib.sha256(json.dumps(metadata["inventory"]["sha256"], sort_keys=True).encode()).hexdigest(),
                "Source fingerprint is inconsistent")
        for name, record in expected.items():
            path = payload / name
            require(path.is_file() and path.stat().st_size == record["bytes"] and sha256(path) == record["sha256"],
                    "Extracted support member identity differs: " + name)
        actual = {path.relative_to(payload).as_posix() for path in payload.rglob("*") if path.is_file()}
        require(actual == set(expected) | {"LEGACY_SOURCE_REFERENCE_SHA256.json"}, "Unexpected extracted support content")
        # Claim both destination roots before writing and use xb on every member.
        for name in sorted(ROOTS):
            (workspace / name).mkdir(exist_ok=False)
        for name, record in expected.items():
            destination = workspace / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            check_ancestors(destination.parent)
            with (payload / name).open("rb") as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target, 1024 * 1024)
            destination.chmod(0o644)
            require(sha256(destination) == record["sha256"], "Restored disk hash differs: " + name)
        evidence = {"format": "pinn4soh-legacy-source-reference-restoration-v1", "status": "passed",
                    "generated_at": datetime.now(timezone.utc).isoformat(), "archive_sha256": SOURCE_SHA,
                    "source_fingerprint": metadata["source_fingerprint"], "files_verified": 426,
                    "workspace": str(workspace), "stage": str(stage), "files": expected,
                    "no_existing_path_overwritten": True, "training_performed": False,
                    "script_sha256": sha256(Path(__file__))}
        write_json(evidence_path, evidence)
        print(json.dumps({"status": "passed", "files_verified": 426, "workspace": str(workspace)}))
        return 0
    except Exception as error:
        write_json(stage / "RESTORE_FAILED.json", {"status": "failed", "error": repr(error),
                   "partial_output_preserved": True, "training_performed": False})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
