"""Archive selected ten-repeat evidence with exact modern/source counterparts.

Standard library only; independent of the frozen 12+40 collector. Reads source
files, never loads checkpoint pickle or starts a training process.
"""

import argparse
import base64
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import tarfile

from prepare_legacy_extension import (read, digest, require, verify_reused,
    verify_completed, verify_modern_frozen, same_legacy_cohort, stable, verify_composite)


def collect(workspace, extension, legacy_control_suite, legacy_extension_suite,
            current_suites, comparison, screening_report, output_prefix, chunk_mib=64):
    workspace, extension, old, new, comparison, screen, output = [Path(path).resolve() for path in
        [workspace, extension, legacy_control_suite, legacy_extension_suite, comparison, screening_report, output_prefix]]
    require(not output.exists() and output.parent.is_dir() and re.fullmatch(r"[A-Za-z0-9_-]+", output.name), "Use a fresh output prefix with simple filename")
    require(1 <= chunk_mib <= 64, "Chunk size must be 1..64 MiB")
    design, manifest = read(extension / "extension_design.json"), read(extension / "extension_analysis_manifest.json")
    require(digest(extension / "extension_plan.json") == design["plan_sha256"] and
            digest(extension / "extension_analysis_manifest.json") == design["analysis_manifest_sha256"], "Prepared extension support changed")
    require(verify_reused(manifest["tasks"], old) == manifest["reuse_evidence"], "Original repeat1 changed")
    new_state = read(new / "manifest.json")
    new_plan = read(extension / "extension_plan.json")
    verify_composite(new_plan, manifest, design)
    require(new_state.get("status") == "complete" and set(new_state["tasks"]) == {task["task_id"] for task in new_plan["tasks"]} and
            new_state["frozen"]["plan_sha256"] == stable(new_plan), "New suite is incomplete or differs from selected plan")
    same_legacy_cohort(manifest["reuse_evidence"]["control_frozen"], new_state["frozen"])
    result = read(comparison / "summary.json")
    require(result.get("assessment") == "complete_selected_ten_repeat_sensitivity" and
            result.get("task_counts") == {"matched": len(manifest["tasks"])} and
            result.get("analysis_manifest_sha256") == design["analysis_manifest_sha256"], "Need complete matched selected comparison")
    require(digest(screen) == design["inputs"]["screening_report"]["sha256"], "40-task screening report changed")
    paths, expected, modern = {}, {}, {}

    def add(target):
        target = Path(target).resolve()
        require(target.exists() and target != output and output not in target.parents and target != workspace and workspace in target.parents,
                "Collected source must exist inside workspace and outside output")
        for path in target.rglob("*") if target.is_dir() else [target]:
            require(not path.is_symlink(), "Symlink in evidence archive")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                paths[path.relative_to(workspace).as_posix()] = path

    suites = [old, new] + [Path(path).resolve() for path in current_suites]
    for suite in suites:
        require(result["input_suite_manifest_sha256"].get(str(suite)) == digest(suite / "manifest.json"), "Suite changed since comparison")
        for name in ["manifest.json", "plan.json", "source_inventory.json"]:
            if (suite / name).is_file():
                add(suite / name)
    for suite in suites[2:]:
        state = read(suite / "manifest.json")
        verify_modern_frozen(state["frozen"])
        needed = {item["modern_task_id"] for item in manifest["tasks"]}
        for task_id in needed & set(state["tasks"]):
            require(task_id not in modern, "Duplicate selected modern task")
            modern[task_id] = suite
    for item in manifest["tasks"]:
        for is_modern in [False, True]:
            task = item["modern_task" if is_modern else "legacy_task"]
            suite = modern[task["task_id"]] if is_modern else old if item["repeat"] == 1 else new
            folder, record, _, _ = verify_completed(suite, task)
            add(folder)
            for name, checksum in record["verified_file_sha256"].items():
                expected[(folder / name).relative_to(workspace).as_posix()] = checksum
    add(extension)
    add(comparison)
    add(screen.parent)
    for name, record in design["inputs"].items():
        if name.startswith("screening_table:"):
            require(digest(screen.parent / name.split(":", 1)[1]) == record["sha256"], "Screening table changed")
    # Bind all selected source files before streaming; any later modification is
    # rejected, including support/report files not hashed by a task coordinator.
    frozen_bytes = {name: digest(path) for name, path in paths.items()}
    for name, checksum in expected.items():
        require(frozen_bytes[name] == checksum, "Selected completion-time artifact changed")
    output.mkdir()
    # Keep the established archive-name grammar accepted by the strict restore
    # utility; the extension scope is explicit in its collection manifest.
    archive = output / ("PINN4SOH_LEGACY_EXTENSION_" + output.name + ".tar.gz")
    inventory = []
    with archive.open("wb") as compressed:
        with gzip.GzipFile(fileobj=compressed, mode="wb", compresslevel=1, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w|") as tar:
                for name, path in sorted(paths.items()):
                    before = path.stat()
                    hasher = hashlib.sha256()
                    with path.open("rb") as stream:
                        class Reader:
                            def read(self, size):
                                block = stream.read(size)
                                hasher.update(block)
                                return block
                        tar.addfile(tar.gettarinfo(str(path), arcname=name), Reader())
                    after = path.stat()
                    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns) and hasher.hexdigest() == frozen_bytes[name], "Source changed during archiving")
                    inventory.append({"path": name, "bytes": before.st_size, "sha256": hasher.hexdigest()})
    by_name = {row["path"]: row for row in inventory}
    verified = set()
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            require(member.isfile() and member.name in by_name and member.name not in verified, "Unexpected archive member")
            stream, checksum, size = tar.extractfile(member), hashlib.sha256(), 0
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                checksum.update(block)
                size += len(block)
            require(size == by_name[member.name]["bytes"] and checksum.hexdigest() == by_name[member.name]["sha256"], "Archive member bytes changed")
            verified.add(member.name)
    require(verified == set(by_name), "Archive member set differs")
    archive_hash, archive_size = digest(archive), archive.stat().st_size
    chunk_bytes = chunk_mib * 1024 * 1024
    count = max(1, (archive_size + chunk_bytes - 1) // chunk_bytes)
    wrappers = []
    with archive.open("rb") as stream:
        for index in range(count):
            data = stream.read(chunk_bytes)
            metadata = {"format": "pinn4soh-backup-json-base64-v1", "archive_type": "tar.gz", "original_zip_name": archive.name,
                "original_zip_bytes": archive_size, "original_zip_sha256": archive_hash, "chunk_index": index,
                "chunk_count": count, "chunk_offset": index * chunk_bytes, "chunk_bytes": len(data),
                "chunk_sha256": hashlib.sha256(data).hexdigest(), "encoding": "base64"}
            wrapper = output / ("results.extension." + output.name + ".chunk%03d.json" % (index + 1))
            wrapper.write_text(json.dumps(dict(metadata, data_base64=base64.b64encode(data).decode("ascii")), separators=(",", ":")) + "\n", encoding="ascii")
            require(base64.b64decode(read(wrapper)["data_base64"], validate=True) == data, "JSON chunk roundtrip failed")
            wrappers.append(dict(metadata, wrapper_name=wrapper.name, wrapper_bytes=wrapper.stat().st_size,
                                 wrapper_sha256=digest(wrapper), roundtrip_verified=True))
    evidence = {"format": "pinn4soh-selected-extension-collection-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "collection_complete": True, "verified_without_errors": True, "archive": str(archive), "archive_sha256": archive_hash,
        "archive_bytes": archive_size, "selected_modern_counterparts": len(modern), "selected_legacy_analysis_tasks": len(manifest["tasks"]),
        "original_repeat1_reused": design["reused_repeat1_count"], "new_tasks": len(new_plan["tasks"]),
        "all_archived_member_hashes_verified": True, "completion_time_hash_checks": len(expected), "inventory": inventory,
        "training_performed": False, "full_modern2120_or_all40_task_artifacts_included": False}
    (output / "EXTENSION_COLLECTION_VERIFIED.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    (output / "JSON_TRANSFER_MANIFEST.json").write_text(json.dumps({"format": "pinn4soh-backup-json-base64-v1", "archive_type": "tar.gz",
        "verified_without_errors": True, "original_archives": 1, "wrapper_files": len(wrappers), "records": wrappers}, indent=2), encoding="utf-8")
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["workspace", "extension", "legacy-control-suite", "legacy-extension-suite", "comparison", "screening-report", "output-prefix"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--current-suites", type=Path, nargs="+", required=True)
    parser.add_argument("--chunk-mib", type=int, default=64)
    result = collect(**vars(parser.parse_args()))
    print(json.dumps({key: value for key, value in result.items() if key != "inventory"}, indent=2))
