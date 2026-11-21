"""Restore replacement-search wrappers into a fresh, confined local directory.

Verifies the transfer manifest and the collector's complete ordinary-file
inventory. Does not load checkpoints, import archived tools or alter evidence.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tarfile

from restore_backup_json import restore
from restore_selected_extension_local import extended, read, require, safe_name, sha


def run(transfer, output, expected_runs=192):
    transfer, output = Path(transfer).resolve(), Path(output).resolve()
    require(not output.exists() and output.parent.is_dir(), "Use a fresh output directory with an existing parent")
    collection_path = transfer / "SEARCH_COLLECTION_VERIFIED.json"
    manifest_path = transfer / "JSON_TRANSFER_MANIFEST.json"
    external, manifest = read(collection_path), read(manifest_path)
    require(external.get("format") == "pinn4soh-replacement-search-collection-v1" and
            external.get("status") == "complete" and external.get("all_archived_member_hashes_verified") is True,
            "Collector is incomplete or has an unexpected format")
    formal = external["formal_audit"]
    require(formal.get("status") == "passed" and formal["candidate_runs"] == expected_runs and
            formal["test_evaluations"] == len(formal["seeds"]) and formal["seeds"] and
            len(formal["seeds"]) == len(set(formal["seeds"])) and
            formal["candidates"] * len(formal["seeds"]) == expected_runs,
            "Collector does not cover the expected complete formal search and selected tests")
    require(manifest.get("format") == "pinn4soh-backup-json-base64-v1" and
            manifest.get("verified_without_errors") is True and manifest.get("original_archives") == 1 and
            manifest.get("records"), "Invalid or missing transfer manifest")
    archive_names = {row["original_zip_name"] for row in manifest["records"]}
    require(len(archive_names) == 1, "Exactly one archive is required")
    archive_name = next(iter(archive_names))
    require(re.fullmatch(r"PINN4SOH_LEGACY_SEARCH_[A-Za-z0-9_-]+\.tar\.gz", archive_name), "Unexpected search archive name")
    require(all(row["original_zip_bytes"] == external["archive_bytes"] and
                row["original_zip_sha256"] == external["archive_sha256"] for row in manifest["records"]),
            "Transfer archive metadata differs from collector evidence")
    inventory, folded = {}, set()
    for row in external["inventory"]:
        safe_name(row["path"])
        require(row["path"] not in inventory and row["path"].casefold() not in folded and
                isinstance(row["bytes"], int) and not isinstance(row["bytes"], bool) and row["bytes"] >= 0 and
                re.fullmatch(r"[0-9a-f]{64}", row["sha256"]), "Duplicate/unsafe/invalid inventory entry")
        inventory[row["path"]] = row
        folded.add(row["path"].casefold())
    require(inventory and "search/plan.json" in inventory and "search/selection.json" in inventory and
            "search/status.json" in inventory and "tools/TOOLS_SHA256.json" in inventory,
            "Required archived search/tool files are missing")
    output.mkdir()
    archives = output / "archives"
    results = restore(transfer, archives, settle_seconds=0, manifest=manifest_path)
    require(len(results) == 1 and read(archives / "LOCAL_BACKUP_RESTORE_VERIFIED.json")["all_expected_archives_verified"] is True,
            "Transfer verification did not recover exactly one complete archive")
    archive = Path(results[0]["path"])
    require(archive.name == archive_name and archive.stat().st_size == external["archive_bytes"] and
            sha(archive) == external["archive_sha256"], "Recovered archive differs from collector")
    extracted = extended(output / "extracted")
    extracted.mkdir()
    seen, members = set(), []
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            parts = safe_name(member.name)
            require(member.isfile() and member.name in inventory and member.name not in seen,
                    "Unexpected/nonregular/duplicate archive member")
            record = inventory[member.name]
            require(member.size == record["bytes"], "Archive member size differs from inventory")
            destination = extracted.joinpath(*parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            require(not destination.exists() and not any(p.is_symlink() for p in destination.parents),
                    "Destination already exists or a symlink is present")
            reader = stream.extractfile(member)
            require(reader is not None, "Missing member payload")
            digest, size = hashlib.sha256(), 0
            with destination.open("xb") as writer:
                for block in iter(lambda: reader.read(4 * 1024 * 1024), b""):
                    writer.write(block)
                    digest.update(block)
                    size += len(block)
            require(size == member.size and digest.hexdigest() == record["sha256"] and
                    destination.stat().st_size == size and sha(destination) == record["sha256"],
                    "Member size or SHA differs during writing or after saving")
            seen.add(member.name)
            members.append(dict(path=member.name, bytes=size, sha256=digest.hexdigest()))
    require(seen == set(inventory), "Not every collector inventory member was recovered")
    report = dict(format="pinn4soh-replacement-search-local-recovery-v1", checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  status="passed", expected_candidate_runs=expected_runs, archive=str(archive), archive_bytes=archive.stat().st_size,
                  archive_sha256=sha(archive), extracted=str(extracted), archive_files=len(members),
                  source_bytes=sum(row["bytes"] for row in members), all_saved_member_hashes_verified=True,
                  inventory_set_exact=True, ordinary_relative_files_only=True, scientific_array_audit_not_yet_performed=True,
                  transfer_manifest_sha256=sha(manifest_path), collection_sha256=sha(collection_path), script_sha256=sha(__file__))
    for name, value in (("SEARCH_LOCAL_EXTRACTION_VERIFIED.json", report),
                        ("SEARCH_RECOVERED_MEMBER_SHA256.json", dict(records=members))):
        (output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "archive_files", "source_bytes", "archive_bytes", "archive_sha256")}))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-runs", type=int, default=192)
    run(**vars(parser.parse_args()))
