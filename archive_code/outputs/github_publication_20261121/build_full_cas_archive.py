"""Read-only, deduplicated full-workspace archival for GitHub Release assets."""
from pathlib import Path
import os
import json
import hashlib
import zipfile
from datetime import datetime, timezone
import re

ROOT = Path('\\\\?\\' + os.getcwd())
PUB_REL = 'outputs/github_publication_20261121'
OUT = ROOT / PUB_REL / 'full_archive'
OUT.mkdir(parents=True, exist_ok=True)
CHUNK_BYTES = 64 * 1024**2
MAX_PART_BYTES = 650 * 1024**2
EXTRAS = ('stage_a', 'stage-a', 'stagea', 'robustness', 'deadline', 'tuning_diagnosis')
TEXT_EXT = {'.py', '.ps1', '.md', '.json', '.jsonl', '.txt', '.log', '.csv', '.html', '.js', '.yaml', '.yml', '.ini', '.cfg', '.url'}
SECRETS = [
    ('github_token', re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,})\b')),
    ('private_key', re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')),
    ('api_key', re.compile(rb'\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}\b')),
    ('credential_assignment', re.compile(rb'(?i)(?:["\x27]?(?:password|passwd|access_token|refresh_token|authorization|cookie)["\x27]?\s*[:=]\s*["\x27])[^"\x27\r\n]{8,}')),
]


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(data)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def excluded_reason(rel, is_dir=False):
    lower = rel.lower()
    name = rel.rsplit('/', 1)[-1]
    if lower == PUB_REL or lower.startswith(PUB_REL + '/'):
        return 'publication_work_directory_not_source'
    if name in {'.git', '__pycache__', 'node_modules', '.pytest_cache'}:
        return 'tool_cache_or_git_metadata'
    if any(t in lower for t in EXTRAS):
        return 'user_removed_optional_stage_a_or_robustness_experiment'
    if not is_dir and (name.endswith(('.pyc', '.pyo', '.tmp', '.lock', '.part')) or name in {'.DS_Store', 'Thumbs.db'}):
        return 'transient_or_incomplete_runtime_artifact'
    return None


def run():
    assert not (OUT / 'index.json').exists(), 'Full index exists; verify instead of overwriting'
    assert not list(OUT.glob('full-cas-part-*.zip')), 'Partial archives already exist; inspect first'
    index = {'format': 'pinn4soh-local-cas-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
             'source_root': str(ROOT).replace('\\\\?\\', ''), 'date': '2026-11-21',
             'scope': '完整保留本实验本地源文件字节，以SHA256内容寻址去重。用户撤出的追加实验与临时缓存按excluded清单排除。',
             'license_note': '原始数据、第三方源码、论文、依赖软件等许可归各自权利人；本档案不创设覆盖它们的统一许可证。',
             'chunk_bytes': CHUNK_BYTES, 'max_part_bytes': MAX_PART_BYTES,
             'files': [], 'blobs': {}, 'excluded': [], 'archive_parts': [],
             'sensitive_text_scan': {'matched': [], 'file_count': 0, 'public_network_metadata_preserved': True},
             'source_files_modified': False}
    source_files = []
    for parent, dirs, names in os.walk(ROOT):
        kept = []
        for name in sorted(dirs):
            path = Path(parent) / name
            rel = path.relative_to(ROOT).as_posix()
            reason = excluded_reason(rel, True)
            if reason:
                index['excluded'].append({'path': rel + '/', 'reason': reason})
            else:
                assert not path.is_symlink(), 'Source symlink not supported: ' + rel
                kept.append(name)
        dirs[:] = kept
        for name in sorted(names):
            path = Path(parent) / name
            rel = path.relative_to(ROOT).as_posix()
            reason = excluded_reason(rel)
            if reason:
                index['excluded'].append({'path': rel, 'reason': reason})
            else:
                assert not path.is_symlink(), 'Source symlink not supported: ' + rel
                source_files.append((rel, path))
    source_files.sort(key=lambda item: item[0])
    total_bytes = sum(p.stat().st_size for rel, p in source_files)
    print('SOURCE_INVENTORY', len(source_files), total_bytes, flush=True)
    write_json(OUT / 'BUILD_STATUS.json', {'state': 'packing', 'source_files': len(source_files), 'source_bytes': total_bytes})
    archive = None
    part_path = None
    part_n = 0
    part_payload = 0
    processed_bytes = 0
    text_cache = set()

    def close_part():
        nonlocal archive, part_path, part_payload
        if archive is None:
            return
        archive.close()
        size = part_path.stat().st_size
        assert size <= MAX_PART_BYTES, (part_path.name, size)
        row = {'name': part_path.name, 'bytes': size, 'sha256': sha_file(part_path)}
        index['archive_parts'].append(row)
        write_json(OUT / 'BUILD_PARTS_PROGRESS.json', {'parts': index['archive_parts']})
        print('PART_COMPLETE', json.dumps(row), flush=True)
        archive = None
        part_payload = 0

    def add_blob(data, digest):
        nonlocal archive, part_path, part_n, part_payload
        if digest in index['blobs']:
            assert index['blobs'][digest]['bytes'] == len(data)
            return
        name = 'blobs/' + digest
        # Leave ample room for ZIP central-directory records and headers.
        estimated_entry = len(data) + 1024
        if archive is None or part_payload + estimated_entry > MAX_PART_BYTES - 1024**2:
            close_part()
            part_n += 1
            part_path = OUT / ('full-cas-part-%03d.zip' % part_n)
            assert not part_path.exists()
            archive = zipfile.ZipFile(part_path, 'w', compression=zipfile.ZIP_STORED, allowZip64=True)
        archive.writestr(name, data)
        part_payload += estimated_entry
        index['blobs'][digest] = {'bytes': len(data), 'archive': part_path.name, 'member': name}

    for file_n, (rel, path) in enumerate(source_files, 1):
        before = path.stat()
        size = before.st_size
        checksum = hashlib.sha256()
        chunks = []
        scan_text = path.suffix.lower() in TEXT_EXT and size <= 32 * 1024**2
        text_data = bytearray() if scan_text else None
        with path.open('rb') as stream:
            for data in iter(lambda: stream.read(CHUNK_BYTES), b''):
                digest = hashlib.sha256(data).hexdigest()
                checksum.update(data)
                chunks.append({'sha256': digest, 'bytes': len(data)})
                add_blob(data, digest)
                if scan_text:
                    text_data.extend(data)
        file_sha = checksum.hexdigest()
        after = path.stat()
        assert before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns, 'Source changed: ' + rel
        if scan_text and file_sha not in text_cache:
            index['sensitive_text_scan']['file_count'] += 1
            for label, pattern in SECRETS:
                if pattern.search(text_data):
                    index['sensitive_text_scan']['matched'].append({'path': rel, 'pattern': label})
            text_cache.add(file_sha)
        index['files'].append({'path': rel, 'bytes': size, 'sha256': file_sha, 'chunks': chunks})
        processed_bytes += size
        if file_n % 1000 == 0 or file_n == len(source_files):
            status = {'state': 'packing', 'processed_files': file_n, 'source_files': len(source_files),
                      'processed_bytes': processed_bytes, 'source_bytes': total_bytes,
                      'unique_blob_count': len(index['blobs']), 'completed_parts': len(index['archive_parts'])}
            write_json(OUT / 'BUILD_STATUS.json', status)
            print('PROGRESS', json.dumps(status), flush=True)
    close_part()
    index['source_file_count'] = len(index['files'])
    index['source_file_bytes'] = sum(row['bytes'] for row in index['files'])
    index['unique_blob_bytes'] = sum(row['bytes'] for row in index['blobs'].values())
    index['dedup_saved_bytes'] = index['source_file_bytes'] - index['unique_blob_bytes']
    index['sensitive_text_scan']['passed'] = not index['sensitive_text_scan']['matched']
    index['status'] = 'packed_pending_independent_verification'
    write_json(OUT / 'index.json', index)
    (OUT / 'SHA256SUMS.txt').write_text(''.join(p['sha256'] + '  ' + p['name'] + '\n' for p in index['archive_parts'])
                                           + sha_file(OUT / 'index.json') + '  index.json\n', encoding='ascii')
    write_json(OUT / 'BUILD_STATUS.json', {'state': index['status'], 'source_files': index['source_file_count'],
                                         'source_bytes': index['source_file_bytes'], 'unique_blob_bytes': index['unique_blob_bytes'],
                                         'parts': len(index['archive_parts']), 'index_sha256': sha_file(OUT / 'index.json'),
                                         'sensitive_text_scan': index['sensitive_text_scan']})
    print('FULL_CAS_PACKED', index['source_file_count'], index['source_file_bytes'], index['unique_blob_bytes'], len(index['archive_parts']), flush=True)


if __name__ == '__main__':
    run()
