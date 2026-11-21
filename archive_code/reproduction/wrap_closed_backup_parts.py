"""Verify and wrap already closed ZIP parts without waiting for the full backup.

python wrap_closed_backup_parts.py /workspace/backup_PINN4SOH_<timestamp>
The highest ZIP number is conservatively skipped until VERIFIED_BACKUP is true.
Each wrapper is an ordinary JSON transfer file with at most 64 MiB binary data.
"""
import base64
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

BLOCK = 4 * 1024 * 1024
CHUNK = 64 * 1024 * 1024
FORMAT = 'pinn4soh-backup-json-base64-v1'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as reader:
        for block in iter(lambda: reader.read(BLOCK), b''):
            value.update(block)
    return value.hexdigest()


def run(output):
    output = Path(output)
    parts = sorted(output.glob('results.*.zip'))
    full_verified = False
    try:
        full_verified = bool(json.loads((output / 'VERIFIED_BACKUP.json').read_text())['verified_without_errors'])
    except (FileNotFoundError, ValueError, KeyError):
        pass
    candidates = parts if full_verified else parts[:-1]
    index_path = output / 'CLOSED_PART_TRANSFER_MANIFEST.json'
    existing = json.loads(index_path.read_text()) if index_path.is_file() else {}
    records = {row['wrapper_name']: row for row in existing.get('records', [])}
    checked = []
    for part in candidates:
        before = part.stat()
        if time.time() - before.st_mtime < 2:
            continue
        original_hash = digest(part)
        previous = [row for row in records.values() if row['original_zip_name'] == part.name]
        if previous and all(row['original_zip_bytes'] == before.st_size and row['original_zip_sha256'] == original_hash
                            and (output / row['wrapper_name']).is_file() for row in previous):
            checked.append(part.name)
            continue
        with zipfile.ZipFile(part) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise RuntimeError('ZIP CRC failed: ' + part.name + ' / ' + bad)
        count = max(1, (before.st_size + CHUNK - 1) // CHUNK)
        with part.open('rb') as reader:
            for number in range(count):
                data = reader.read(CHUNK)
                row = dict(format=FORMAT, original_zip_name=part.name,
                           original_zip_bytes=before.st_size, original_zip_sha256=original_hash,
                           chunk_index=number, chunk_count=count, chunk_offset=number * CHUNK,
                           chunk_bytes=len(data), chunk_sha256=hashlib.sha256(data).hexdigest(), encoding='base64')
                name = part.name + ('' if count == 1 else f'.chunk{number + 1:03d}') + '.json'
                target = output / name
                temp = target.with_suffix('.json.tmp')
                with temp.open('w', encoding='ascii', newline='\n') as writer:
                    writer.write(json.dumps(row, separators=(',', ':'))[:-1] + ',"data_base64":"')
                    writer.write(base64.b64encode(data).decode('ascii'))
                    writer.write('"}\n')
                # Verify the actual file before publishing it as completed.
                check = json.loads(temp.read_text(encoding='ascii'))
                decoded = base64.b64decode(check.pop('data_base64'), validate=True)
                if hashlib.sha256(decoded).hexdigest() != row['chunk_sha256'] or len(decoded) != row['chunk_bytes']:
                    raise RuntimeError('JSON roundtrip failed: ' + name)
                temp.replace(target)
                records[name] = dict(row, wrapper_name=name, wrapper_bytes=target.stat().st_size,
                                    wrapper_sha256=digest(target), roundtrip_verified=True)
                print(json.dumps(records[name], separators=(',', ':')), flush=True)
                del data, check, decoded
        after = part.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or digest(part) != original_hash:
            raise RuntimeError('ZIP changed during wrapping: ' + part.name)
        checked.append(part.name)
        state = dict(format=FORMAT, verified_without_errors=True, full_backup_verified=full_verified,
                     scope='closed_parts_only', original_archives=len(checked),
                     records=sorted(records.values(), key=lambda row: row['wrapper_name']))
        temporary = index_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
        temporary.replace(index_path)
    print(json.dumps(dict(closed_parts_checked=checked,
                         skipped_current_part=parts[-1].name if parts and not full_verified else None,
                         transfer_manifest=str(index_path))), flush=True)


if __name__ == '__main__':
    run(sys.argv[1])
