"""Verify or restore a PINN4SOH content-addressed local archive.

Only the Python standard library is required. Never extracts arbitrary ZIP paths.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile
import zipfile

FORMAT = 'pinn4soh-local-cas-v1'
HASH = re.compile(r'^[0-9a-f]{64}$')
PART = re.compile(r'^full-cas-part-[0-9]+\.zip$')
WINDOWS_RESERVED = re.compile(r'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?$', re.I)
MAX_CHUNK = 64 * 1024 * 1024
MAX_PART = 700 * 1024 * 1024


def require(value, message):
    if not value:
        raise ValueError(message)


def integer(value, message):
    require(type(value) is int and value >= 0, message)
    return value


def checksum(value, message):
    require(isinstance(value, str) and HASH.fullmatch(value), message)
    return value


def safe_relative(value):
    require(isinstance(value, str) and value and '\\' not in value and '\0' not in value,
            'Invalid relative path: %r' % value)
    p = PurePosixPath(value)
    require(not p.is_absolute() and str(p) == value and not any(x in ('', '.', '..') for x in p.parts),
            'Non-canonical relative path: %r' % value)
    for component in p.parts:
        require(not any(c in component for c in ':<>"|?*') and not component.endswith((' ', '.'))
                and not WINDOWS_RESERVED.fullmatch(component), 'Unsafe path component: %r' % component)
    return p


def long_path(path):
    path = Path(os.path.abspath(str(path)))
    if os.name == 'nt' and not str(path).startswith('\\\\?\\'):
        s = str(path)
        return Path('\\\\?\\UNC\\' + s[2:]) if s.startswith('\\\\') else Path('\\\\?\\' + s)
    return path


def reparse(path):
    s = path.lstat()
    return stat.S_ISLNK(s.st_mode) or bool(getattr(s, 'st_file_attributes', 0) & 0x400)


def no_links(path):
    """Reject symlinks and Windows junction/reparse ancestors, including root."""
    p = path
    while True:
        if p.exists() or p.is_symlink():
            require(not reparse(p), 'Symlink or junction is not allowed: ' + str(p))
        if p.parent == p:
            break
        p = p.parent


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def unique_json_pairs(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, 'Duplicate JSON key: ' + key)
        value[key] = item
    return value


def load_index(index_path):
    require(index_path.is_file(), 'Index is missing: ' + str(index_path))
    no_links(index_path)
    with index_path.open('r', encoding='utf-8-sig') as stream:
        idx = json.load(stream, object_pairs_hook=unique_json_pairs)
    require(idx.get('format') == FORMAT, 'Unsupported archive index format')
    require(not idx.get('external_archives'), 'External archive references are not supported by this version')
    require(isinstance(idx.get('files'), list) and isinstance(idx.get('blobs'), dict)
            and isinstance(idx.get('archive_parts'), list), 'Missing index collections')
    files, blobs, parts = idx['files'], idx['blobs'], idx['archive_parts']
    part_map = {}
    for row in parts:
        name = row.get('name')
        require(isinstance(name, str) and PART.fullmatch(name), 'Unsafe archive filename')
        require(name not in part_map, 'Duplicate archive part: ' + name)
        require(0 < integer(row.get('bytes'), 'Invalid part size') <= MAX_PART, 'Part exceeds limit')
        checksum(row.get('sha256'), 'Invalid archive SHA256')
        part_map[name] = row
    members = {}
    for digest, row in blobs.items():
        checksum(digest, 'Invalid blob identifier')
        size = integer(row.get('bytes'), 'Invalid blob size')
        require(0 < size <= MAX_CHUNK, 'Invalid blob chunk size')
        require(row.get('archive') in part_map, 'Unknown archive for blob')
        require(row.get('member') == 'blobs/' + digest, 'Invalid blob member path')
        pair = (row['archive'], row['member'])
        require(pair not in members, 'Duplicate blob member')
        members[pair] = digest
    paths, folded, refs = set(), set(), Counter()
    for row in files:
        path = safe_relative(row.get('path'))
        key = str(path)
        require(key not in paths and key.casefold() not in folded, 'Duplicate or case-colliding file path: ' + key)
        paths.add(key); folded.add(key.casefold())
        integer(row.get('bytes'), 'Invalid source file size')
        checksum(row.get('sha256'), 'Invalid source file SHA256')
        require(isinstance(row.get('chunks'), list), 'Missing file chunks')
        total = 0
        for chunk in row['chunks']:
            digest = checksum(chunk.get('sha256'), 'Invalid file chunk SHA256')
            require(digest in blobs and chunk.get('bytes') == blobs[digest]['bytes'], 'File/blob size mismatch')
            total += chunk['bytes']; refs[digest] += 1
        require(total == row['bytes'], 'File chunk total differs: ' + key)
        if not row['chunks']:
            require(row['bytes'] == 0 and row['sha256'] == hashlib.sha256(b'').hexdigest(), 'Invalid empty file')
    for key in paths:
        parent = PurePosixPath(key).parent
        while str(parent) != '.':
            require(str(parent).casefold() not in folded, 'File/directory path collision: ' + key)
            parent = parent.parent
    require(set(refs) == set(blobs), 'Unreferenced blob or missing reference in index')
    require({b['archive'] for b in blobs.values()} == set(part_map), 'Unused archive part')
    return idx, part_map, members


def verify_or_restore(index_path, archives_dir, output=None):
    index_path, archives_dir = long_path(index_path), long_path(archives_dir)
    no_links(archives_dir)
    idx, parts, members = load_index(index_path)
    dest = long_path(output) if output else None
    if dest:
        no_links(dest)
        if dest.exists():
            require(dest.is_dir(), 'Output is not a directory')
        else:
            dest.mkdir(parents=True)
        # Check every destination before writing any file, including case-sensitive hosts.
        for row in idx['files']:
            path = dest.joinpath(*PurePosixPath(row['path']).parts)
            no_links(path)
            if path.exists():
                require(path.is_file() and path.stat().st_size == row['bytes']
                        and sha_file(path) == row['sha256'], 'Refusing to overwrite different file: ' + row['path'])
    opened, restored, skipped, verified_blobs = {}, 0, 0, set()
    try:
        # Verify compressed file bytes before reading any blob.
        for name, row in parts.items():
            path = archives_dir / name
            no_links(path)
            require(path.is_file() and path.stat().st_size == row['bytes'], 'Archive size/missing error: ' + name)
            require(sha_file(path) == row['sha256'], 'Archive SHA256 error: ' + name)
            z = zipfile.ZipFile(path, 'r'); opened[name] = z
            seen = set()
            for item in z.infolist():
                pair = (name, item.filename)
                require(pair in members and item.filename not in seen and not item.is_dir(), 'Unexpected ZIP member')
                require(not item.flag_bits & 1, 'Encrypted ZIP member is unsupported')
                require(not stat.S_ISLNK(item.external_attr >> 16), 'Symlink ZIP member is forbidden')
                digest = members[pair]
                require(item.file_size == idx['blobs'][digest]['bytes'], 'ZIP declared blob size differs')
                seen.add(item.filename)
            require(seen == {m for a, m in members if a == name}, 'Missing ZIP blob members')
        # Iterate original files and hash both complete byte stream and every chunk.
        for row in idx['files']:
            path = dest.joinpath(*PurePosixPath(row['path']).parts) if dest else None
            skip = bool(path and path.exists())
            tmp_path, stream = None, None
            if path and not skip:
                no_links(path.parent)
                path.parent.mkdir(parents=True, exist_ok=True)
                no_links(path.parent)
                fd, temp_name = tempfile.mkstemp(prefix='.cas-restore-', suffix='.tmp', dir=str(path.parent))
                tmp_path = Path(temp_name); stream = os.fdopen(fd, 'wb')
            try:
                full = hashlib.sha256(); count = 0
                for chunk in row['chunks']:
                    digest = chunk['sha256']; b = idx['blobs'][digest]
                    part_hash = hashlib.sha256(); chunk_count = 0
                    with opened[b['archive']].open(b['member'], 'r') as source:
                        for data in iter(lambda: source.read(1024 * 1024), b''):
                            part_hash.update(data); full.update(data)
                            count += len(data); chunk_count += len(data)
                            if stream:
                                stream.write(data)
                    require(chunk_count == chunk['bytes'] and part_hash.hexdigest() == digest,
                            'Blob SHA256/size mismatch: ' + digest)
                    verified_blobs.add(digest)
                require(count == row['bytes'] and full.hexdigest() == row['sha256'],
                        'Reassembled file SHA256/size mismatch: ' + row['path'])
                if stream:
                    stream.flush(); os.fsync(stream.fileno()); stream.close(); stream = None
                    no_links(path)
                    require(not path.exists(), 'Destination appeared during restore: ' + row['path'])
                    # Hard-link publishes without replacing any unexpectedly existing path.
                    os.link(tmp_path, path)
                    tmp_path.unlink(); tmp_path = None
                    restored += 1
                elif skip:
                    skipped += 1
            finally:
                if stream:
                    stream.close()
                if tmp_path and tmp_path.exists():
                    tmp_path.unlink()
        require(verified_blobs == set(idx['blobs']), 'Some blobs were not verified')
        return {'status': 'passed', 'format': FORMAT, 'mode': 'restore' if dest else 'verify-only',
                'index_sha256': sha_file(index_path), 'archives': len(parts), 'unique_blobs': len(verified_blobs),
                'files': len(idx['files']), 'original_bytes': sum(r['bytes'] for r in idx['files']),
                'restored_files': restored, 'existing_identical_files_skipped': skipped,
                'all_archive_and_blob_and_reassembled_file_sha256_verified': True}
    finally:
        for z in opened.values():
            z.close()


def main():
    parser = argparse.ArgumentParser(description='Verify or safely restore the complete local PINN4SOH archive.')
    parser.add_argument('--index', default='index.json', help='Path to the downloaded archive index')
    parser.add_argument('--archives-dir', help='Directory containing all full-cas-part ZIP assets; defaults to index directory')
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--verify-only', action='store_true', help='Verify all bytes without writing restored files')
    modes.add_argument('--output', help='Restore into this directory; refuses different existing files')
    args = parser.parse_args()
    index = Path(args.index)
    result = verify_or_restore(index, args.archives_dir or index.parent, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError, zipfile.BadZipFile) as error:
        print('Archive verification/restoration failed: ' + str(error), file=sys.stderr)
        raise SystemExit(1)
