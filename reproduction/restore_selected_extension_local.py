"""Strictly restore selected-extension wrappers and verify external inventory.

Uses the established chunk restore tool, then streams regular tar members into
a new confined directory. It never extracts archive paths through tar.extract.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile

from restore_backup_json import restore


RESERVED = re.compile(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)", re.I)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as reader:
        for block in iter(lambda: reader.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extended(path):
    result = Path(path).resolve()
    if os.name == "nt" and not str(result).startswith("\\\\?\\"):
        result = Path("\\\\?\\" + str(result))
    return result


def safe_name(name):
    require(isinstance(name, str) and name and "\\" not in name and ":" not in name and
            "\x00" not in name, "Unsafe tar member spelling")
    path = PurePosixPath(name)
    require(not path.is_absolute() and path.as_posix() == name and
            not any(part in {".", "..", ""} for part in path.parts), "Noncanonical relative member")
    for part in path.parts:
        require(not part.endswith((" ", ".")) and not RESERVED.match(part) and
                not any(char in part for char in '<>"|?*') and
                not any(ord(char) < 32 for char in part), "Windows-unsafe tar member")
    return path.parts


def run(transfer, output):
    transfer, output = Path(transfer).resolve(), Path(output).resolve()
    require(not output.exists() and output.parent.is_dir(), "Use a fresh output recovery directory")
    inventory_path = transfer / "EXTENSION_COLLECTION_VERIFIED.json"
    external = read(inventory_path)
    require(external.get("format") == "pinn4soh-selected-extension-collection-v1" and
            external.get("collection_complete") is True and external.get("verified_without_errors") is True and
            external.get("all_archived_member_hashes_verified") is True and
            external.get("selected_modern_counterparts") == external.get("selected_legacy_analysis_tasks") == 110 and
            external.get("original_repeat1_reused") == 11 and external.get("new_tasks") == 99,
            "External collector status or scope differs")
    inventory = {}
    casefolded = set()
    for row in external["inventory"]:
        safe_name(row["path"])
        require(row["path"] not in inventory and row["path"].casefold() not in casefolded,
                "Duplicate or Windows case-colliding inventory member")
        require(isinstance(row["bytes"], int) and row["bytes"] >= 0 and
                re.fullmatch(r"[0-9a-f]{64}", row["sha256"]), "Invalid member metadata")
        inventory[row["path"]] = row
        casefolded.add(row["path"].casefold())
    output.mkdir()
    archives = output / "archives"
    archives.mkdir()
    results = restore(transfer, archives, settle_seconds=0, manifest=transfer / "JSON_TRANSFER_MANIFEST.json")
    require(len(results) == 1 and read(archives / "LOCAL_BACKUP_RESTORE_VERIFIED.json")["all_expected_archives_verified"] is True,
            "Requires exactly one fully verified extension archive")
    archive = Path(results[0]["path"])
    require(archive.name == PurePosixPath(external["archive"]).name and archive.stat().st_size == external["archive_bytes"] and
            sha(archive) == external["archive_sha256"], "Archive differs from external inventory")
    extracted = extended(output / "extracted")
    extracted.mkdir()
    seen, members = set(), []
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            parts = safe_name(member.name)
            require(member.isfile() and member.name in inventory and member.name not in seen,
                    "Unexpected/nonregular/duplicate member")
            expected = inventory[member.name]
            require(member.size == expected["bytes"], "Tar member size differs from collector inventory")
            destination = extracted.joinpath(*parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            require(not destination.exists() and not any(parent.is_symlink() for parent in destination.parents),
                    "Destination exists or symlink detected")
            reader = stream.extractfile(member)
            require(reader is not None, "Missing member payload")
            size, digest = 0, hashlib.sha256()
            with destination.open("xb") as writer:
                for block in iter(lambda: reader.read(4 * 1024 * 1024), b""):
                    writer.write(block)
                    digest.update(block)
                    size += len(block)
            require(size == member.size and digest.hexdigest() == expected["sha256"], "Member SHA/size differs during extraction")
            require(destination.stat().st_size == size and sha(destination) == expected["sha256"], "Saved member SHA/size differs")
            seen.add(member.name)
            members.append(dict(path=member.name, bytes=size, sha256=digest.hexdigest()))
    require(seen == set(inventory), "Collector inventory not fully recovered")
    document = dict(format="pinn4soh-selected-extension-local-recovery-v1", checked_at_utc=datetime.now(timezone.utc).isoformat(),
                    archive=str(archive), archive_bytes=archive.stat().st_size, archive_sha256=sha(archive),
                    external_inventory=str(inventory_path), external_inventory_sha256=sha(inventory_path),
                    archive_files=len(members), source_bytes=sum(row["bytes"] for row in members),
                    every_saved_member_sha_and_size_verified=True, member_set_exactly_matches_external_inventory=True,
                    regular_files_only=True, no_absolute_parent_link_duplicate_or_windows_unsafe_members=True,
                    extracted=str(extracted), verified_without_errors=True,
                    selected_legacy_analysis_tasks=110, selected_modern_counterparts=110, new_training_tasks=99, reused_repeat1=11,
                    completion_time_hash_checks_expected=external["completion_time_hash_checks"],
                    scientific_array_audit_not_yet_performed=True, script_sha256=sha(__file__))
    (output / "EXTENSION_LOCAL_EXTRACTION_VERIFIED.json").write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "EXTENSION_RECOVERED_MEMBER_SHA256.json").write_text(json.dumps(dict(records=members), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: document[key] for key in ["archive_files", "source_bytes", "archive_bytes", "archive_sha256", "verified_without_errors"]}))
    return document


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(**vars(args))
