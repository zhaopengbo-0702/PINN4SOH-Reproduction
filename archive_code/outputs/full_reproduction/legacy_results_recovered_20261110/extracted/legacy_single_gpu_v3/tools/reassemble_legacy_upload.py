"""Verify 14 pinned ZIP parts, reassemble, CRC-check, and safely extract fresh.

Standard library only, Python 3.7+. This does not install, execute, or train.
The command line deliberately has no option to weaken the pinned evidence.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import zipfile


PRODUCTION_PINS = {
    "index_sha256": "9344dbbb49686d10d9a28a625f11ec1669efbe6c8c51c55bc3c84da1f3a316eb",
    "original": "legacy-offline-20261110-v1.zip",
    "bytes": 1338319780,
    "sha256": "e422cc82b6b95062809bf72cea8727beb4cb2a360ab09244dcd00f633948e87f",
    "part_count": 14,
    "part_bytes": 96 * 1024 * 1024,
    "payload_entries": 45,
    "payload_bytes": 1338312424,
}
INDEX_NAME = "legacy_archive_parts.json"
BLOCK = 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(BLOCK), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def no_duplicate_json_keys(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "Duplicate JSON key: " + key)
        value[key] = item
    return value


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      object_pairs_hook=no_duplicate_json_keys)


def is_link(path):
    details = path.lstat()
    return stat.S_ISLNK(details.st_mode) or bool(getattr(details, "st_file_attributes", 0) & 0x400)


def regular_file(path):
    require(path.exists() and not is_link(path) and stat.S_ISREG(path.lstat().st_mode),
            "Expected ordinary file without links: " + str(path))


def check_ancestors(path):
    for item in [path] + list(path.parents):
        if item.exists() or item.is_symlink():
            require(not is_link(item), "Symlink/reparse path is forbidden: " + str(item))


def fresh_output(parts, output):
    parts = Path(os.path.abspath(str(parts)))
    output = Path(os.path.abspath(str(output)))
    check_ancestors(parts)
    check_ancestors(output)
    require(parts.is_dir(), "Parts directory is missing")
    require(not os.path.lexists(str(output)), "Output must be a fresh nonexistent path")
    require(output.parent.is_dir(), "Output parent must already exist")
    require(parts != output and parts not in output.parents and output not in parts.parents,
            "Output and uploaded parts must not overlap")
    return parts, output


def validate_index(parts, pins):
    index_path = parts / INDEX_NAME
    regular_file(index_path)
    require(sha256(index_path) == pins["index_sha256"], "Pinned index SHA256 mismatch")
    index = read_json(index_path)
    require(set(index) == {"format", "original", "bytes", "sha256", "parts"}, "Unexpected index schema")
    require(index["format"] == "pinn4soh-offline-archive-parts-v1", "Unsupported index format")
    for key in ["original", "bytes", "sha256"]:
        require(index[key] == pins[key], "Pinned original archive identity differs: " + key)
    require(isinstance(index["parts"], list) and len(index["parts"]) == pins["part_count"],
            "Exact ordered part count required")
    total = 0
    for number, record in enumerate(index["parts"], 1):
        member = "legacy_archive.part%03d" % number
        require(set(record) == {"archive", "member", "bytes", "sha256", "archive_sha256"},
                "Unexpected part schema")
        require(record["member"] == member and record["archive"] == member + ".zip", "Wrong part identity/order")
        expected = min(pins["part_bytes"], pins["bytes"] - total)
        require(type(record["bytes"]) is int and record["bytes"] == expected and expected > 0,
                "Wrong part size: " + member)
        for key in ["sha256", "archive_sha256"]:
            require(isinstance(record[key], str) and re.fullmatch(r"[a-f0-9]{64}", record[key]), "Bad part SHA256")
        total += record["bytes"]
    require(total == pins["bytes"], "Part sizes do not equal original size")
    expected_archives = {item["archive"] for item in index["parts"]}
    actual_archives = {path.name for path in parts.glob("legacy_archive.part*")}
    require(actual_archives == expected_archives, "Missing, extra, or auto-extracted part files")
    return index


def ordinary_zip_info(info):
    require(not info.flag_bits & 1, "Encrypted ZIP member is forbidden")
    require(info.orig_filename == info.filename, "Truncated/noncanonical ZIP member name")
    mode = info.external_attr >> 16
    kind = stat.S_IFMT(mode)
    require(kind in {0, stat.S_IFREG, stat.S_IFDIR}, "ZIP link/device member is forbidden")
    require(not mode & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX), "Privileged ZIP mode is forbidden")
    require(info.compress_type in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}, "Unsupported ZIP compression")


def verify_part_archive(path, record):
    regular_file(path)
    # ZIP_STORED, one exact ASCII member, no metadata: frozen splitter overhead 142.
    require(path.stat().st_size == record["bytes"] + 142, "Part ZIP size differs: " + path.name)
    require(sha256(path) == record["archive_sha256"], "Part archive SHA256 mismatch: " + path.name)
    with zipfile.ZipFile(path) as archive:
        require(not archive.comment, "Unexpected part ZIP comment")
        infos = archive.infolist()
        require(len(infos) == 1, "Part ZIP must contain exactly one member")
        item = infos[0]
        ordinary_zip_info(item)
        require(item.filename == record["member"] and not item.is_dir(), "Part ZIP member name differs")
        require(item.file_size == record["bytes"] and item.compress_size == record["bytes"], "Part member size differs")
        require(item.compress_type == zipfile.ZIP_STORED and not item.extra and not item.comment,
                "Part ZIP metadata/compression differs")


def safe_relative(name):
    require(isinstance(name, str) and name and not any(char in name for char in '\\:<>"|?*'),
            "Unsafe ZIP path: " + repr(name))
    require(not any(ord(char) < 32 or ord(char) == 127 for char in name), "Control character in ZIP path")
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and relative.parts and all(part not in {".", ".."} for part in relative.parts),
            "ZIP path escapes destination: " + name)
    require(not name.endswith("//") and str(relative) == name.rstrip("/"), "Noncanonical ZIP path: " + name)
    reserved = {"CON", "PRN", "AUX", "NUL"} | {"COM%d" % i for i in range(1, 10)} | {"LPT%d" % i for i in range(1, 10)}
    for part in relative.parts:
        require(not part.endswith((".", " ")) and part.split(".")[0].upper() not in reserved,
                "Unsafe cross-platform ZIP path component: " + part)
    return relative


def safe_extract_archive(archive_path, destination, max_entries, max_bytes):
    """Prevalidate every path/member, CRC-read all bytes, then extract with xb."""
    destination = Path(destination)
    require(not os.path.lexists(str(destination)), "Extraction destination must be fresh")
    check_ancestors(destination)
    records = []
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        require(0 < len(infos) <= max_entries, "ZIP member count exceeds bound")
        require(sum(item.file_size for item in infos) <= max_bytes, "ZIP uncompressed size exceeds bound")
        identities, files = set(), set()
        for item in infos:
            ordinary_zip_info(item)
            relative = safe_relative(item.filename)
            folded = str(relative).casefold()
            require(folded not in identities, "Duplicate/case-colliding ZIP member")
            identities.add(folded)
            if not item.is_dir():
                files.add(folded)
        for item in infos:
            relative = safe_relative(item.filename)
            require(not any(str(parent).casefold() in files for parent in relative.parents if str(parent) != "."),
                    "File/directory prefix collision")
        require(archive.testzip() is None, "Reassembled ZIP CRC failed")
        destination.mkdir()
        for item in infos:
            relative = safe_relative(item.filename)
            target = destination.joinpath(*relative.parts)
            require(destination.resolve() in target.resolve().parents, "ZIP extraction escapes destination")
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            check_ancestors(target.parent)
            count, hasher = 0, hashlib.sha256()
            with archive.open(item) as source, target.open("xb") as output:
                for block in iter(lambda: source.read(BLOCK), b""):
                    count += len(block)
                    require(count <= item.file_size, "ZIP member exceeds advertised size")
                    hasher.update(block)
                    output.write(block)
            require(count == item.file_size, "Short extracted ZIP member")
            target.chmod(0o644)
            require(sha256(target) == hasher.hexdigest(), "Extracted file disk SHA256 differs")
            records.append({"path": item.filename, "bytes": count, "sha256": hasher.hexdigest(),
                            "crc32": "%08x" % item.CRC})
    return records


def verify_payload_assets(payload):
    manifests = [payload / "assets/legacy_asset_manifest.json", payload / "assets/legacy_torch_manifest.json"]
    entries = read_json(manifests[0]) + [read_json(manifests[1])]
    require(len(entries) == 31 and len({row["filename"] for row in entries}) == 31, "Expected 31 distinct runtime assets")
    expected = {"legacy_asset_manifest.json", "legacy_torch_manifest.json"}
    for row in entries:
        name = row["filename"]
        require(safe_relative(name).name == name, "Unsafe asset manifest filename")
        path = payload / "assets" / name
        regular_file(path)
        require(path.stat().st_size == row["bytes"] and sha256(path) == row["sha256"], "Runtime asset differs: " + name)
        expected.add(name)
    require({path.name for path in (payload / "assets").iterdir()} == expected, "Unexpected runtime asset content")
    return len(entries)


def rebuild(parts, output, pins=None, verify_assets=True):
    # Alternative pins exist only for tiny local test fixtures; CLI always uses constants.
    pins = dict(PRODUCTION_PINS if pins is None else pins)
    parts, output = fresh_output(parts, output)
    index = validate_index(parts, pins)
    for record in index["parts"]:
        verify_part_archive(parts / record["archive"], record)
    output.mkdir()
    evidence = {"format": "pinn4soh-offline-reassembly-evidence-v1", "status": "incomplete",
                "generated_at": datetime.now(timezone.utc).isoformat(), "parts": str(parts),
                "output": str(output), "pins": pins, "script_sha256": sha256(Path(__file__)),
                "training_performed": False, "installation_performed": False, "part_results": []}
    try:
        combined_path = output / pins["original"]
        total, total_hash = 0, hashlib.sha256()
        with combined_path.open("xb") as combined:
            for record in index["parts"]:
                count, member_hash = 0, hashlib.sha256()
                with zipfile.ZipFile(parts / record["archive"]) as archive:
                    with archive.open(record["member"]) as source:
                        for block in iter(lambda: source.read(BLOCK), b""):
                            count += len(block)
                            require(count <= record["bytes"], "Part exceeds advertised size")
                            member_hash.update(block)
                            total_hash.update(block)
                            combined.write(block)
                    # Reading archive.open through EOF verifies the member CRC.
                require(count == record["bytes"] and member_hash.hexdigest() == record["sha256"],
                        "Part member SHA256/size mismatch: " + record["member"])
                total += count
                evidence["part_results"].append(dict(record, crc_passed=True, member_sha256_passed=True,
                                                      archive_sha256_passed=True))
                print("VERIFIED " + record["archive"], flush=True)
            combined.flush()
            os.fsync(combined.fileno())
        require(total == pins["bytes"] and total_hash.hexdigest() == pins["sha256"], "Reassembled stream identity differs")
        require(combined_path.stat().st_size == pins["bytes"] and sha256(combined_path) == pins["sha256"],
                "Reassembled on-disk archive identity differs")
        with zipfile.ZipFile(combined_path) as archive:
            require(len(archive.infolist()) == pins["payload_entries"], "Pinned payload member count differs")
            require(sum(item.file_size for item in archive.infolist()) == pins["payload_bytes"], "Pinned payload size differs")
        payload = output / "payload"
        extracted = safe_extract_archive(combined_path, payload, pins["payload_entries"], pins["payload_bytes"])
        asset_count = verify_payload_assets(payload) if verify_assets else None
        evidence.update(status="passed", archive_sha256=pins["sha256"], archive_bytes=total,
                        zip_crc_passed=True, safe_extraction_passed=True, payload=str(payload),
                        extracted_files=extracted, runtime_assets_verified=asset_count,
                        source_parts_unchanged=True, no_existing_path_overwritten=True)
        write_json(output / "REASSEMBLY_VERIFIED.json", evidence)
        print(json.dumps({"status": "passed", "payload": str(payload), "sha256": pins["sha256"],
                          "runtime_assets_verified": asset_count}))
        return evidence
    except Exception as error:
        evidence.update(status="failed", error=repr(error), partial_output_preserved=True)
        write_json(output / "REASSEMBLY_FAILED.json", evidence)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", type=Path, required=True, help="Directory containing exact 14 part ZIPs and frozen index")
    parser.add_argument("--output", type=Path, required=True, help="New nonexistent directory outside parts")
    args = parser.parse_args()
    rebuild(args.parts, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
