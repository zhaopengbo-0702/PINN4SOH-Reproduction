"""Restore independently complete backup ZIPs while other files download.

python restore_backup_json.py DOWNLOADS OUTPUT --incremental --settle-seconds 2
Add --manifest JSON_TRANSFER_MANIFEST.json to check the full expected set.
Original downloads are retained; ZIP contents are checked but never extracted.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tarfile
import time
import zipfile

FORMAT = 'pinn4soh-backup-json-base64-v1'
DATA_MARKER = b',"data_base64":"'
BLOCK_BYTES = 4 * 1024 * 1024


def _stat(path):
    value = path.stat()
    return dict(bytes=value.st_size, mtime_ns=value.st_mtime_ns)


def _header(path):
    with path.open('rb') as reader:
        prefix = reader.read(64 * 1024)
    offset = prefix.find(DATA_MARKER)
    if offset < 0:
        raise ValueError('Backup wrapper header missing or unsupported')
    metadata = json.loads(prefix[:offset] + b'}')
    if metadata.get('format') != FORMAT:
        raise ValueError('Unexpected backup wrapper format')
    name = metadata['original_zip_name']
    if name != Path(name).name or not (re.fullmatch(r'results\.\d+\.zip', name)
            or re.fullmatch(r'EMERGENCY_FULL_[A-Za-z0-9_]+\.tar\.gz', name)
            or re.fullmatch(r'PINN4SOH_LEGACY_[A-Za-z0-9_-]+\.tar\.gz', name)):
        raise ValueError('Unexpected original ZIP file name')
    for key in ('original_zip_bytes', 'chunk_index', 'chunk_count', 'chunk_offset', 'chunk_bytes'):
        if not isinstance(metadata[key], int) or metadata[key] < 0:
            raise ValueError('Invalid ' + key)
    if metadata['chunk_count'] < 1 or metadata['chunk_index'] >= metadata['chunk_count']:
        raise ValueError('Invalid chunk index/count')
    return metadata, offset + len(DATA_MARKER)


def _verify_zip(path, expected_bytes, expected_hash, archive_type='zip'):
    hasher = hashlib.sha256()
    with path.open('rb') as reader:
        for block in iter(lambda: reader.read(BLOCK_BYTES), b''):
            hasher.update(block)
    if path.stat().st_size != expected_bytes or hasher.hexdigest() != expected_hash:
        raise RuntimeError('Restored ZIP SHA256 or size mismatch')
    if archive_type == 'tar.gz':
        count = 0
        with tarfile.open(path, 'r|gz') as archive:
            for member in archive:
                count += 1
                if member.isfile():
                    actual = 0
                    reader = archive.extractfile(member)
                    for block in iter(lambda: reader.read(BLOCK_BYTES), b''):
                        actual += len(block)
                    if actual != member.size:
                        raise RuntimeError('Truncated tar member: ' + member.name)
        return count
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError('ZIP CRC failure: ' + bad)
        return len(archive.infolist())


def _decode_into(path, metadata, offset, writer, overall_hasher, expected=None):
    before = _stat(path)
    chunk_hasher, wrapper_hasher = hashlib.sha256(), hashlib.sha256()
    encoded_remaining = 4 * ((metadata['chunk_bytes'] + 2) // 3)
    decoded_bytes = 0
    with path.open('rb') as reader:
        wrapper_hasher.update(reader.read(offset))
        while encoded_remaining:
            block = reader.read(min(BLOCK_BYTES, encoded_remaining))
            if not block:
                raise RuntimeError('Incomplete JSON payload: ' + path.name)
            encoded_remaining -= len(block)
            wrapper_hasher.update(block)
            data = base64.b64decode(block, validate=True)
            chunk_hasher.update(data)
            overall_hasher.update(data)
            writer.write(data)
            decoded_bytes += len(data)
        suffix = reader.read(1024)
        wrapper_hasher.update(suffix)
        if suffix.strip() != b'"}' or reader.read(1):
            raise RuntimeError('Invalid JSON payload suffix: ' + path.name)
    if _stat(path) != before:
        raise RuntimeError('Download changed while reading: ' + path.name)
    if decoded_bytes != metadata['chunk_bytes'] or chunk_hasher.hexdigest() != metadata['chunk_sha256']:
        raise RuntimeError('Chunk SHA256 or size mismatch: ' + path.name)
    if expected is not None:
        if before['bytes'] != expected['wrapper_bytes'] or wrapper_hasher.hexdigest() != expected['wrapper_sha256']:
            raise RuntimeError('Wrapper SHA256 or size differs from cloud manifest: ' + path.name)
    return dict(path=str(path), **before, wrapper_sha256=wrapper_hasher.hexdigest(),
                chunk_index=metadata['chunk_index'], chunk_sha256=chunk_hasher.hexdigest())


def restore(source, destination, incremental=False, settle_seconds=0, manifest=None):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    index_path = destination / 'LOCAL_BACKUP_RESTORE_INDEX.json'
    previous = {}
    if index_path.is_file():
        try:
            previous = {row['original_zip_name']: row for row in json.loads(index_path.read_text(encoding='utf-8'))['archives']}
        except (ValueError, KeyError):
            pass
    manifest_path = Path(manifest) if manifest else source / 'JSON_TRANSFER_MANIFEST.json'
    expected_records, expected_archives = {}, set()
    if manifest_path.is_file():
        manifest_data = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest_data.get('format') != FORMAT or not manifest_data.get('verified_without_errors'):
            raise RuntimeError('Cloud transfer manifest is invalid or reports errors')
        for record in manifest_data['records']:
            key = (record['original_zip_name'], record['chunk_index'])
            expected_records[key] = record
            expected_archives.add(key[0])
    groups, pending_files, errors = {}, [], []
    for path in sorted(source.glob('results.*')):
        if not path.is_file():
            continue
        if path.suffix in {'.crdownload', '.tmp', '.part'}:
            pending_files.append(dict(path=str(path), status='downloading', **_stat(path)))
            continue
        if path.suffix != '.json':
            continue
        snapshot = _stat(path)
        if time.time_ns() - snapshot['mtime_ns'] < settle_seconds * 1e9:
            pending_files.append(dict(path=str(path), status='settling', **snapshot))
            continue
        try:
            metadata, offset = _header(path)
            if _stat(path) != snapshot:
                pending_files.append(dict(path=str(path), status='changed_during_scan'))
                continue
            expected = expected_records.get((metadata['original_zip_name'], metadata['chunk_index']))
            if expected_records and expected is None:
                # A browser downloads directory may contain other completed backups.
                # The manifest defines this restore operation's archive set.
                pending_files.append(dict(path=str(path), status='outside_current_manifest'))
                continue
            if expected is not None:
                for key in ('original_zip_bytes', 'original_zip_sha256', 'chunk_count', 'chunk_bytes', 'chunk_offset', 'chunk_sha256'):
                    if metadata[key] != expected[key]:
                        raise RuntimeError('Wrapper metadata differs from cloud transfer manifest: ' + key)
                if snapshot['bytes'] != expected['wrapper_bytes']:
                    pending_files.append(dict(path=str(path), status='size_not_complete',
                                              expected_bytes=expected['wrapper_bytes'], **snapshot))
                    continue
            groups.setdefault(metadata['original_zip_name'], []).append((path, metadata, offset, expected))
        except Exception as error:
            errors.append(dict(path=str(path), error=str(error)))
    results, pending_archives = [], []
    all_names = sorted(set(groups) | expected_archives)
    for name in all_names:
        records = sorted(groups.get(name, []), key=lambda item: (item[1]['chunk_index'], str(item[0])))
        if not records:
            pending_archives.append(dict(original_zip_name=name, status='no_completed_wrappers'))
            continue
        # Browser duplicate downloads are acceptable if every duplicate has identical metadata.
        distinct = {}
        try:
            for item in records:
                chunk_index = item[1]['chunk_index']
                if chunk_index in distinct and distinct[chunk_index][1] != item[1]:
                    raise RuntimeError('Conflicting duplicate chunk ' + str(chunk_index))
                distinct.setdefault(chunk_index, item)
            records = [distinct[key] for key in sorted(distinct)]
            first = records[0][1]
            expected_count = first['chunk_count']
            missing = sorted(set(range(expected_count)) - set(distinct))
            if missing:
                pending_archives.append(dict(original_zip_name=name, status='missing_chunks',
                                              missing_chunk_indices=missing, expected_chunks=expected_count))
                continue
            target = destination / name
            fingerprints = [dict(path=str(item[0]), **_stat(item[0])) for item in records]
            cached = previous.get(name, {})
            if (target.is_file() and cached.get('status') == 'verified'
                    and cached.get('output_stat') == _stat(target)
                    and cached.get('source_fingerprints') == fingerprints
                    and cached.get('sha256') == first['original_zip_sha256']
                    and cached.get('cloud_manifest_checked') == bool(expected_records)):
                results.append(dict(cached, reused_verified=True))
                continue
            temp = destination / (name + '.restoring')
            hasher, written_bytes, sources = hashlib.sha256(), 0, []
            try:
                with temp.open('wb') as writer:
                    for path, metadata, offset, expected in records:
                        for key in ('original_zip_name', 'original_zip_bytes', 'original_zip_sha256', 'chunk_count'):
                            if metadata[key] != first[key]:
                                raise RuntimeError('Inconsistent metadata in ' + path.name)
                        if metadata['chunk_offset'] != written_bytes:
                            raise RuntimeError('Inconsistent chunk offset in ' + path.name)
                        sources.append(_decode_into(path, metadata, offset, writer, hasher, expected))
                        written_bytes += metadata['chunk_bytes']
                if written_bytes != first['original_zip_bytes'] or hasher.hexdigest() != first['original_zip_sha256']:
                    raise RuntimeError('Restored ZIP SHA256 or size mismatch: ' + name)
                members = _verify_zip(temp, first['original_zip_bytes'], first['original_zip_sha256'],
                                      first.get('archive_type', 'zip'))
                temp.replace(target)
            except Exception:
                if temp.exists():
                    temp.unlink()
                raise
            row = dict(original_zip_name=name, path=str(target), status='verified', bytes=written_bytes,
                       sha256=hasher.hexdigest(), chunks=expected_count, members=members,
                       sha256_verified=True, crc_verified=True, cloud_manifest_checked=bool(expected_records),
                       output_stat=_stat(target), source_fingerprints=fingerprints, sources=sources)
            results.append(row)
            print(json.dumps({key: row[key] for key in ('original_zip_name', 'bytes', 'sha256', 'members', 'status')}), flush=True)
        except Exception as error:
            errors.append(dict(original_zip_name=name, error=str(error)))
    index = dict(format=FORMAT, updated_utc=datetime.now(timezone.utc).isoformat(),
                 source=str(source), destination=str(destination),
                 cloud_manifest=str(manifest_path) if expected_records else None,
                 expected_archives=len(expected_archives) if expected_records else None,
                 verified_archives=len(results), verified_bytes=sum(row['bytes'] for row in results),
                 all_expected_archives_verified=bool(expected_records) and len(results) == len(expected_archives) and not errors,
                 verified_without_errors=not errors and not pending_archives and bool(results)
                    and (bool(expected_records) or not pending_files),
                 archives=results, pending_archives=pending_archives, pending_files=pending_files, errors=errors)
    temp_index = index_path.with_suffix('.json.tmp')
    temp_index.write_text(json.dumps(index, indent=2), encoding='utf-8')
    temp_index.replace(index_path)
    (destination / 'LOCAL_BACKUP_RESTORE_VERIFIED.json').write_text(json.dumps(index, indent=2), encoding='utf-8')
    print(json.dumps({key: index[key] for key in ('verified_archives', 'verified_bytes', 'expected_archives',
          'all_expected_archives_verified')} | dict(pending_archives=len(pending_archives),
          pending_files=len(pending_files), errors=len(errors))), flush=True)
    if not incremental and (pending_archives or errors or not results):
        raise RuntimeError('Backup incomplete or invalid; see ' + str(index_path))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--incremental', action='store_true', help='Restore complete ZIPs and report unfinished ones without stopping')
    parser.add_argument('--settle-seconds', type=float, default=2,
                        help='Ignore final JSON files modified more recently than this (default: 2)')
    parser.add_argument('--manifest', type=Path)
    arguments = parser.parse_args()
    restore(arguments.source, arguments.destination, incremental=arguments.incremental,
            settle_seconds=arguments.settle_seconds, manifest=arguments.manifest)
