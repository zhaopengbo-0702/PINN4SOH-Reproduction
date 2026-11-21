"""Small I/O fixtures for archive verification and safe restoration."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cas_restore', HERE / 'restore_full_archive.py')
restore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(restore)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def put_index(folder, value):
    path = folder / 'index.json'
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    return path


def fixture(folder):
    chunks = [b'hello ', '中文\n'.encode('utf-8')]
    blob = {sha(c): c for c in chunks}
    part = folder / 'full-cas-part-001.zip'
    with zipfile.ZipFile(part, 'w', zipfile.ZIP_DEFLATED) as z:
        for digest, data in blob.items():
            z.writestr('blobs/' + digest, data)
    idx = {'format': restore.FORMAT, 'files': [
        {'path': '子目录/结果.txt', 'bytes': sum(map(len, chunks)), 'sha256': sha(b''.join(chunks)),
         'chunks': [{'sha256': sha(c), 'bytes': len(c)} for c in chunks]},
        {'path': '同内容副本.txt', 'bytes': sum(map(len, chunks)), 'sha256': sha(b''.join(chunks)),
         'chunks': [{'sha256': sha(c), 'bytes': len(c)} for c in chunks]},
        {'path': 'empty.txt', 'bytes': 0, 'sha256': sha(b''), 'chunks': []}],
        'blobs': {h: {'bytes': len(c), 'archive': part.name, 'member': 'blobs/' + h} for h, c in blob.items()},
        'archive_parts': [{'name': part.name, 'bytes': part.stat().st_size, 'sha256': sha(part.read_bytes())}],
        'excluded': []}
    return idx, chunks


def run():
    passed = []
    with tempfile.TemporaryDirectory(prefix='restore-fixture-', dir=str(HERE)) as temp:
        folder = Path(temp); idx, chunks = fixture(folder); original = copy.deepcopy(idx)
        index = put_index(folder, idx)
        r = restore.verify_or_restore(index, folder)
        assert r['files'] == 3 and r['unique_blobs'] == 2 and not (folder / 'output').exists()
        passed.append('verify_only_reassembles_all_files_without_output')
        output = folder / 'output'
        r = restore.verify_or_restore(index, folder, output)
        assert r['restored_files'] == 3 and (output / '子目录/结果.txt').read_bytes() == b''.join(chunks)
        passed.append('restore_unicode_multichunk_dedup_and_empty')
        r = restore.verify_or_restore(index, folder, output)
        assert r['restored_files'] == 0 and r['existing_identical_files_skipped'] == 3
        passed.append('existing_identical_files_verified_and_skipped')
        (output / 'empty.txt').write_bytes(b'different')
        try:
            restore.verify_or_restore(index, folder, output)
        except ValueError:
            pass
        else:
            raise AssertionError('Different existing output was overwritten')
        assert (output / 'empty.txt').read_bytes() == b'different'
        passed.append('existing_different_file_rejected_unchanged')
        changes = [
            ('path_traversal_rejected', lambda x: x['files'][0].update(path='../escape.txt')),
            ('absolute_path_rejected', lambda x: x['files'][0].update(path='/escape.txt')),
            ('windows_drive_path_rejected', lambda x: x['files'][0].update(path='C:/escape.txt')),
            ('backslash_path_rejected', lambda x: x['files'][0].update(path='a\\escape.txt')),
            ('reserved_filename_rejected', lambda x: x['files'][0].update(path='CON.txt')),
            ('case_collision_rejected', lambda x: x['files'].append(dict(x['files'][2], path='EMPTY.TXT'))),
            ('parent_file_collision_rejected', lambda x: x['files'].append(dict(x['files'][2], path='empty.txt/child'))),
            ('wrong_reassembled_file_hash_rejected', lambda x: x['files'][0].update(sha256='0' * 64)),
            ('wrong_chunk_size_rejected', lambda x: x['files'][0]['chunks'][0].update(bytes=1)),
            ('missing_blob_rejected', lambda x: x['blobs'].pop(next(iter(x['blobs'])))),
            ('external_reference_rejected', lambda x: x.update(external_archives=[{'name': 'unknown.tar.gz'}])),
            ('wrong_archive_hash_rejected', lambda x: x['archive_parts'][0].update(sha256='0' * 64)),
        ]
        for name, mutate in changes:
            idx = copy.deepcopy(original); mutate(idx); put_index(folder, idx)
            try:
                restore.verify_or_restore(index, folder)
            except (ValueError, KeyError):
                passed.append(name)
            else:
                raise AssertionError(name)
        idx = copy.deepcopy(original); part = folder / idx['archive_parts'][0]['name']
        with zipfile.ZipFile(part, 'a') as z:
            z.writestr('unexpected.txt', b'not indexed')
        idx['archive_parts'][0].update(bytes=part.stat().st_size, sha256=sha(part.read_bytes()))
        put_index(folder, idx)
        try:
            restore.verify_or_restore(index, folder)
        except ValueError:
            passed.append('extra_zip_member_rejected_even_when_archive_hash_matches')
        else:
            raise AssertionError('Extra ZIP member accepted')
        # A malicious symlink ZIP entry is rejected even if index and outer SHA match.
        part.unlink(); idx, chunks = fixture(folder)
        digest = next(iter(idx['blobs']))
        with zipfile.ZipFile(part, 'w') as z:
            for h, row in idx['blobs'].items():
                info = zipfile.ZipInfo('blobs/' + h)
                info.create_system = 3
                info.external_attr = ((0o120777 if h == digest else 0o100600) << 16)
                z.writestr(info, next(c for c in chunks if sha(c) == h))
        idx['archive_parts'][0].update(bytes=part.stat().st_size, sha256=sha(part.read_bytes()))
        put_index(folder, idx)
        try:
            restore.verify_or_restore(index, folder)
        except ValueError:
            passed.append('zip_symlink_member_rejected')
        else:
            raise AssertionError('ZIP symlink accepted')
    receipt = {'status': 'passed', 'test_count': len(passed), 'tests': passed,
               'script_sha256': sha((HERE / 'restore_full_archive.py').read_bytes()),
               'limitations': ['These are small local I/O fixtures, not the complete archive verification.',
                               'Filesystem symlink/junction ancestor rejection is implemented; this fixture tests malicious ZIP symlinks.']}
    (HERE / 'RESTORE_FIXTURE_ACCEPTANCE.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
