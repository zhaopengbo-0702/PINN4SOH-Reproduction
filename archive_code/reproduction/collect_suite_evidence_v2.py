"""Losslessly collect text evidence, splitting oversized files at UTF-8 boundaries."""

import argparse
import hashlib
import json
from pathlib import Path


PART = "pinn4soh-text-evidence-part-v2"
INDEX = "pinn4soh-text-evidence-index-v2"


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def safe_name(value):
    if Path(value).name != value or value in {"", ".", ".."} or "/" in value or "\\" in value:
        raise ValueError("Evidence entries must use plain filenames")
    if Path(value).suffix.lower() not in {".json", ".csv", ".md"}:
        raise ValueError("Only textual result evidence is supported")
    return value


def envelope(chunks):
    return dict(format=PART, chunks=chunks)


def split_file(name, data, limit):
    # Decode the BOM as U+FEFF so re-encoding preserves original bytes exactly.
    text = data.decode("utf-8")
    if text.encode("utf-8") != data:
        raise ValueError("Input cannot round-trip as UTF-8: " + name)
    if not text:
        yield dict(name=name, offset=0, bytes=0, sha256=sha256(b""), text="")
        return
    position, offset = 0, 0
    while position < len(text):
        # A Unicode codepoint needs at most 4 UTF-8 bytes or 6 JSON-escaped
        # bytes; reserve bounded framing space, then verify the exact envelope.
        count = min(len(text) - position, max(1, (limit - 512) // 6))
        while True:
            part = text[position:position + count]
            raw = part.encode("utf-8")
            item = dict(name=name, offset=offset, bytes=len(raw), sha256=sha256(raw), text=part)
            if len(encoded(envelope([item]))) <= limit:
                break
            if count == 1:
                raise ValueError("Evidence size cap cannot hold one character and framing")
            count = max(1, count // 2)
        yield item
        position += count
        offset += len(raw)
    if offset != len(data):
        raise ValueError("Internal UTF-8 chunk offset mismatch")


def pack(source, output, max_mb):
    index_path = source / "suite_evidence_index.json"
    raw_index = index_path.read_bytes()
    index = json.loads(raw_index)
    files = []
    seen = set()
    limit = int(max_mb * 1024 * 1024)
    if limit < 1024:
        raise ValueError("Evidence cap must be at least 1024 bytes")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use an empty output directory for a new evidence pack")
    parts, batch = [], []

    def flush():
        if not batch:
            return
        data = encoded(envelope(batch))
        if len(data) > limit:
            raise ValueError("Internal evidence part exceeds the cap")
        name = f"suite_text_evidence_v2.part{len(parts) + 1:03d}.json"
        (output / name).write_bytes(data)
        parts.append(dict(path=name, bytes=len(data), sha256=sha256(data), chunks=len(batch)))
        batch.clear()

    for metadata in index["files"] + [dict(path=index_path.name, bytes=len(raw_index), sha256=sha256(raw_index))]:
        name = safe_name(metadata["path"])
        if name in seen:
            raise ValueError("Duplicate source evidence: " + name)
        seen.add(name)
        raw = (source / name).read_bytes()
        if len(raw) != metadata["bytes"] or sha256(raw) != metadata["sha256"]:
            raise ValueError("Source evidence changed: " + name)
        files.append(dict(name=name, bytes=len(raw), sha256=sha256(raw)))
        for chunk in split_file(name, raw, limit):
            if batch and len(encoded(envelope(batch + [chunk]))) > limit:
                flush()
            batch.append(chunk)
    flush()
    summary = dict(format=INDEX, source=str(source.resolve()), files=files, parts=parts,
                   max_part_bytes=limit, maximum_actual_part_bytes=max((item["bytes"] for item in parts), default=0),
                   contents="Exact UTF-8 result JSON/CSV/metadata, chunked without changing bytes; model/full arrays remain cloud-side")
    (output / "suite_text_evidence_v2_index.json").write_bytes(encoded(summary))
    return summary


def restore(source, output):
    index = json.loads((source / "suite_text_evidence_v2_index.json").read_text(encoding="utf-8"))
    if index.get("format") != INDEX:
        raise ValueError("Unknown evidence index format")
    metadata = {}
    for file in index["files"]:
        name = safe_name(file["name"])
        if name in metadata:
            raise ValueError("Duplicate file identity")
        metadata[name] = file
    chunks = {name: [] for name in metadata}
    for part in index["parts"]:
        name = safe_name(part["path"])
        data = (source / name).read_bytes()
        if len(data) != part["bytes"] or sha256(data) != part["sha256"]:
            raise ValueError("Downloaded evidence part hash mismatch: " + name)
        document = json.loads(data)
        if document.get("format") != PART or len(document["chunks"]) != part["chunks"]:
            raise ValueError("Evidence part structure mismatch")
        for item in document["chunks"]:
            target = safe_name(item["name"])
            if target not in metadata:
                raise ValueError("Unexpected file chunk")
            raw = item["text"].encode("utf-8")
            if len(raw) != item["bytes"] or sha256(raw) != item["sha256"]:
                raise ValueError("Chunk hash mismatch: " + target)
            chunks[target].append((item["offset"], raw))
    restored = []
    for name, records in chunks.items():
        offset, blocks = 0, []
        if not records:
            raise ValueError("No chunks for source file: " + name)
        for position, data in sorted(records, key=lambda item: item[0]):
            if position != offset:
                raise ValueError("Missing, overlapping or duplicate chunk: " + name)
            blocks.append(data)
            offset += len(data)
        data = b"".join(blocks)
        original = metadata[name]
        if len(data) != original["bytes"] or sha256(data) != original["sha256"]:
            raise ValueError("Restored file hash mismatch: " + name)
        restored.append((name, data))
    output.mkdir(parents=True, exist_ok=True)
    for name, data in restored:
        path = output / name
        if path.exists() and path.read_bytes() != data:
            raise ValueError("Refusing to overwrite different evidence: " + name)
    for name, data in restored:
        (output / name).write_bytes(data)
    return dict(restored_files=len(restored), all_original_bytes_and_sha256_verified=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["pack", "restore"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-mb", type=float, default=28)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("Source and output must be separate")
    result = pack(args.source, args.output, args.max_mb) if args.mode == "pack" else restore(args.source, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
