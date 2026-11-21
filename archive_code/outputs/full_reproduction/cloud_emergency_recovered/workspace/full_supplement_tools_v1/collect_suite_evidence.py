"""Collect only textual result evidence into size-bounded, ordinary JSON files."""

import argparse
import hashlib
import json
from pathlib import Path


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def safe_name(value):
    if Path(value).name != value or value in {"", ".", ".."}:
        raise ValueError("Evidence entries must use plain filenames")
    if Path(value).suffix.lower() not in {".json", ".csv", ".md"}:
        raise ValueError("Only textual result evidence is supported")
    return value


def pack(source, output, max_mb):
    index_path = source / "suite_evidence_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    entries = []
    for item in index["files"] + [dict(path=index_path.name, bytes=index_path.stat().st_size,
                                       sha256=sha256(index_path.read_bytes()))]:
        name = safe_name(item["path"])
        data = (source / name).read_bytes()
        if len(data) != item["bytes"] or sha256(data) != item["sha256"]:
            raise ValueError("Source evidence changed: " + name)
        entries.append(dict(name=name, bytes=len(data), sha256=sha256(data), text=data.decode("utf-8-sig"),
                            utf8_bom=data.startswith(b"\xef\xbb\xbf")))
    output.mkdir(parents=True, exist_ok=True)
    limit = int(max_mb * 1024 * 1024)
    parts, batch = [], []

    def flush():
        if not batch:
            return
        data = encoded(dict(format="pinn4soh-text-evidence-part-v1", files=batch))
        name = f"suite_text_evidence.part{len(parts)+1:03d}.json"
        (output / name).write_bytes(data)
        parts.append(dict(path=name, bytes=len(data), sha256=sha256(data), files=len(batch)))
        batch.clear()

    for entry in entries:
        if len(encoded(dict(format="pinn4soh-text-evidence-part-v1", files=[entry]))) > limit:
            raise ValueError("An input text file exceeds the evidence part cap: " + entry["name"])
        if batch and len(encoded(dict(format="pinn4soh-text-evidence-part-v1", files=batch + [entry]))) > limit:
            flush()
        batch.append(entry)
    flush()
    summary = dict(format="pinn4soh-text-evidence-index-v1", source=str(source.resolve()),
                   files=len(entries), parts=parts,
                   contents="Result JSON, CSV and provenance metadata only; model and full arrays remain cloud-side")
    (output / "suite_text_evidence_index.json").write_bytes(encoded(summary))
    return summary


def restore(source, output):
    index = json.loads((source / "suite_text_evidence_index.json").read_text(encoding="utf-8"))
    if index.get("format") != "pinn4soh-text-evidence-index-v1":
        raise ValueError("Unknown text evidence index")
    restored, seen = [], set()
    for part in index["parts"]:
        data = (source / safe_name(part["path"])).read_bytes()
        if len(data) != part["bytes"] or sha256(data) != part["sha256"]:
            raise ValueError("Invalid downloaded evidence part: " + part["path"])
        content = json.loads(data)
        if content.get("format") != "pinn4soh-text-evidence-part-v1":
            raise ValueError("Unknown evidence part")
        for item in content["files"]:
            name = safe_name(item["name"])
            if name in seen:
                raise ValueError("Duplicate evidence filename: " + name)
            seen.add(name)
            restored_data = (b"\xef\xbb\xbf" if item["utf8_bom"] else b"") + item["text"].encode("utf-8")
            if len(restored_data) != item["bytes"] or sha256(restored_data) != item["sha256"]:
                raise ValueError("Invalid restored evidence: " + name)
            restored.append((name, restored_data))
    if len(restored) != index["files"]:
        raise ValueError("Missing evidence files")
    output.mkdir(parents=True, exist_ok=True)
    for name, data in restored:
        path = output / name
        if path.exists() and path.read_bytes() != data:
            raise ValueError("Refusing to overwrite different existing evidence: " + name)
    for name, data in restored:
        (output / name).write_bytes(data)
    return dict(restored_files=len(restored), all_original_bytes_and_sha256_verified=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["pack", "restore"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-mb", type=float, default=18)
    args = parser.parse_args()
    if args.max_mb <= 0:
        parser.error("max-mb must be positive")
    if args.source.resolve() == args.output.resolve():
        parser.error("Source and output must be separate")
    result = pack(args.source, args.output, args.max_mb) if args.mode == "pack" else restore(args.source, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
