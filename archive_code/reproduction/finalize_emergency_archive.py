"""Verify a completed emergency tar.gz and publish ordinary JSON transfer chunks.

python finalize_emergency_archive.py /workspace/EMERGENCY_FULL_<stamp>.tar.gz
Run only after the tar process exits. Output is ARCHIVE_NAME_transfer/.
Metadata retains original_zip_* field names for compatibility with the existing
local transfer restore tool; archive_type identifies this as tar.gz.
"""
import base64
import hashlib
import json
from pathlib import Path
import sys
import tarfile

CHUNK = 64 * 1024 * 1024
BLOCK = 8 * 1024 * 1024
FORMAT = 'pinn4soh-backup-json-base64-v1'


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as reader:
        for block in iter(lambda: reader.read(BLOCK), b''):
            hasher.update(block)
    return hasher.hexdigest()


def finalize(archive):
    archive = Path(archive)
    before = archive.stat()
    output = archive.parent / (archive.name + '_transfer')
    output.mkdir(exist_ok=True)
    original_hash = digest(archive)
    inventory, total_bytes = [], 0
    with tarfile.open(archive, 'r|gz') as stream:
        for member in stream:
            row = dict(path=member.name, bytes=member.size, is_file=member.isfile())
            if member.isfile():
                reader = stream.extractfile(member)
                actual = 0
                for block in iter(lambda: reader.read(BLOCK), b''):
                    actual += len(block)
                if actual != member.size:
                    raise RuntimeError('Truncated tar member: ' + member.name)
                total_bytes += actual
            inventory.append(row)
    (output / 'EMERGENCY_ARCHIVE_VERIFIED.json').write_text(json.dumps(dict(
        archive=str(archive), archive_bytes=before.st_size, archive_sha256=original_hash,
        archive_type='tar.gz', members=len(inventory), uncompressed_file_bytes=total_bytes,
        verified_without_errors=True, inventory=inventory), indent=2), encoding='utf-8')
    print(json.dumps(dict(archive=str(archive), bytes=before.st_size, sha256=original_hash,
                          members=len(inventory), uncompressed_file_bytes=total_bytes)), flush=True)
    count = max(1, (before.st_size + CHUNK - 1) // CHUNK)
    records = []
    with archive.open('rb') as reader:
        for index in range(count):
            data = reader.read(CHUNK)
            metadata = dict(format=FORMAT, archive_type='tar.gz', original_zip_name=archive.name,
                original_zip_bytes=before.st_size, original_zip_sha256=original_hash,
                chunk_index=index, chunk_count=count, chunk_offset=index * CHUNK,
                chunk_bytes=len(data), chunk_sha256=hashlib.sha256(data).hexdigest(), encoding='base64')
            name = f'results.emergency.chunk{index + 1:03d}.json'
            target = output / name
            temp = target.with_suffix('.json.tmp')
            with temp.open('w', encoding='ascii', newline='\n') as writer:
                writer.write(json.dumps(metadata, separators=(',', ':'))[:-1] + ',"data_base64":"')
                writer.write(base64.b64encode(data).decode('ascii'))
                writer.write('"}\n')
            check = json.loads(temp.read_text(encoding='ascii'))
            decoded = base64.b64decode(check.pop('data_base64'), validate=True)
            if len(decoded) != metadata['chunk_bytes'] or hashlib.sha256(decoded).hexdigest() != metadata['chunk_sha256']:
                raise RuntimeError('JSON roundtrip failed: ' + name)
            temp.replace(target)
            records.append(dict(metadata, wrapper_name=name, wrapper_bytes=target.stat().st_size,
                                wrapper_sha256=digest(target), roundtrip_verified=True))
            print(json.dumps(records[-1], separators=(',', ':')), flush=True)
            del data, check, decoded
    after = archive.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or digest(archive) != original_hash:
        raise RuntimeError('Archive changed during wrapping; tar has not completed')
    state = dict(format=FORMAT, archive_type='tar.gz', verified_without_errors=True,
                 original_archives=1, wrapper_files=len(records), records=records)
    (output / 'JSON_TRANSFER_MANIFEST.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    print('EMERGENCY_TRANSFER_VERIFIED ' + str(output), flush=True)
    return state


if __name__ == '__main__':
    finalize(sys.argv[1])
