"""Recheck original workspace bytes and code coverage against the completed CAS index."""
from pathlib import Path
import os
import json
import hashlib
import importlib.util
from datetime import datetime, timezone

ROOT = Path('\\\\?\\' + os.getcwd())
PUB = ROOT / 'outputs/github_publication_20261121'
INDEX = PUB / 'full_archive/index.json'
spec = importlib.util.spec_from_file_location('cas_builder', str(PUB / 'build_full_cas_archive.py'))
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def run():
    index = json.loads(INDEX.read_text(encoding='utf-8'))
    expected = {row['path']: row for row in index['files']}
    found = set()
    problems = []
    code_files = []
    verified_bytes = 0
    for parent, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in sorted(dirs) if not builder.excluded_reason((Path(parent) / d).relative_to(ROOT).as_posix(), True)]
        for name in sorted(files):
            path = Path(parent) / name
            rel = path.relative_to(ROOT).as_posix()
            if builder.excluded_reason(rel):
                continue
            if rel not in expected:
                problems.append({'path': rel, 'reason': 'new_unarchived_file'})
                continue
            row = expected[rel]
            found.add(rel)
            current = path.stat().st_size
            checksum = digest(path)
            if current != row['bytes'] or checksum != row['sha256']:
                problems.append({'path': rel, 'reason': 'source_bytes_changed_since_archive'})
            verified_bytes += current
            if path.suffix.lower() in {'.py', '.ps1', '.sh', '.bat', '.js', '.ipynb'}:
                code_files.append({'path': rel, 'bytes': current, 'sha256': checksum, 'present_in_full_cas': checksum == row['sha256']})
            if len(found) % 5000 == 0:
                print('SOURCE_VERIFY', len(found), verified_bytes, flush=True)
    for rel in sorted(set(expected) - found):
        problems.append({'path': rel, 'reason': 'source_file_missing'})
    result = {'format': 'pinn4soh-full-cas-source-coverage-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'index_sha256': digest(INDEX), 'source_file_count': len(expected), 'source_files_rechecked': len(found),
              'source_bytes_rechecked': verified_bytes, 'problems': problems, 'all_source_bytes_stable_and_archived': not problems,
              'code_file_count': len(code_files), 'code_files': code_files,
              'not_a_deletion_receipt': True, 'source_files_modified_by_this_script': False,
              'exclusion_policy': list(builder.EXTRAS), 'excluded_entries': index['excluded']}
    (PUB / 'full_archive/SOURCE_COVERAGE_RECHECK.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('SOURCE_COVERAGE', json.dumps({k: v for k, v in result.items() if k not in {'code_files', 'excluded_entries'}}, ensure_ascii=True), flush=True)
    if problems:
        raise SystemExit(1)


if __name__ == '__main__':
    run()
