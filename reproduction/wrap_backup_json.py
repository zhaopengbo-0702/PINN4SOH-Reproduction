"""Wrap verified backup ZIPs in JSON for ordinary browser file transfer.

Usage: python wrap_backup_json.py /workspace/backup_PINN4SOH_<timestamp>
Without an argument, use the latest backup directory under /workspace.
Decode each data_base64 value, concatenate chunks in chunk_index order, then
verify original_zip_bytes and original_zip_sha256 before opening the ZIP.
"""
import base64
import hashlib
import json
from pathlib import Path
import sys

MAX_CHUNK = 64 * 1024 * 1024


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def wrap_backup(output):
    output = Path(output)
    archives = sorted(output.glob('results.*.zip'))
    if not archives:
        raise RuntimeError('No results.*.zip files found in ' + str(output))
    verification = output / 'VERIFIED_BACKUP.json'
    if not verification.is_file():
        raise RuntimeError('Backup verification has not completed')
    state = json.loads(verification.read_text(encoding='utf-8'))
    if not state.get('verified_without_errors'):
        raise RuntimeError('Backup reports errors; inspect VERIFIED_BACKUP.json')
    records = []
    for archive in archives:
        original_bytes = archive.stat().st_size
        original_hash = digest(archive)
        chunk_count = max(1, (original_bytes + MAX_CHUNK - 1) // MAX_CHUNK)
        with archive.open('rb') as stream:
            for chunk_index in range(chunk_count):
                payload = stream.read(MAX_CHUNK)
                chunk_hash = hashlib.sha256(payload).hexdigest()
                name = archive.name + ('' if chunk_count == 1 else f'.chunk{chunk_index + 1:03d}') + '.json'
                wrapper = output / name
                metadata = dict(format='pinn4soh-backup-json-base64-v1',
                    original_zip_name=archive.name, original_zip_bytes=original_bytes,
                    original_zip_sha256=original_hash, chunk_index=chunk_index,
                    chunk_count=chunk_count, chunk_offset=chunk_index * MAX_CHUNK,
                    chunk_bytes=len(payload), chunk_sha256=chunk_hash,
                    encoding='base64')
                encoded = base64.b64encode(payload).decode('ascii')
                temp = wrapper.with_suffix(wrapper.suffix + '.tmp')
                with temp.open('w', encoding='ascii', newline='\n') as writer:
                    prefix = json.dumps(metadata, separators=(',', ':'))
                    writer.write(prefix[:-1] + ',"data_base64":"')
                    writer.write(encoded)
                    writer.write('"}\n')
                # Check the written transfer artifact, not only the input payload.
                written = json.loads(temp.read_text(encoding='ascii'))
                decoded = base64.b64decode(written.pop('data_base64'), validate=True)
                if len(decoded) != len(payload) or hashlib.sha256(decoded).hexdigest() != chunk_hash:
                    raise RuntimeError('Wrapper roundtrip failed: ' + name)
                temp.replace(wrapper)
                records.append(dict(metadata, wrapper_name=name,
                    wrapper_bytes=wrapper.stat().st_size, wrapper_sha256=digest(wrapper),
                    roundtrip_verified=True))
                print(json.dumps(records[-1], separators=(',', ':')), flush=True)
                del payload, encoded, decoded, written
        if digest(archive) != original_hash or archive.stat().st_size != original_bytes:
            raise RuntimeError('Source ZIP changed while wrapping: ' + archive.name)
    result = dict(format='pinn4soh-backup-json-base64-v1', output=str(output),
        original_archives=len(archives), wrapper_files=len(records),
        max_decoded_chunk_bytes=MAX_CHUNK, verified_without_errors=True, records=records)
    (output / 'JSON_TRANSFER_MANIFEST.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('JSON_WRAPPERS_VERIFIED ' + str(output), flush=True)
    return result


if __name__ == '__main__':
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(Path('/workspace').glob('backup_PINN4SOH_*'))[-1]
    wrap_backup(destination)
