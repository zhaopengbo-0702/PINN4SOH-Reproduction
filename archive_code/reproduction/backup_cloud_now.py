"""Conservatively snapshot existing cloud results with original-file and ZIP checks."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

root = Path('/workspace')
output = root / ('backup_PINN4SOH_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
output.mkdir()
patterns = ['pinn4soh_*', 'efficiency*',
            'full_tools*', 'full_report_tools*', 'full_supplement_tools*', 'PINN4SOH-main',
            '*pipeline*', '*.log', '*.py', '*.md', '*.txt', '*.json', '*tools*.zip']
excluded_parts = {'site-packages', 'offline', 'offline_wheels', 'legacy_runtime',
                  'legacy_env', '__pycache__', '.git', '.ipynb_checkpoints'}
files = sorted({path for pattern in patterns for target in root.glob(pattern)
                for path in (target.rglob('*') if target.is_dir() else [target])
                if path.is_file() and not path.is_symlink()
                and not any(part in excluded_parts for part in path.relative_to(root).parts)},
               key=lambda p: (0 if p.suffix in {'.pth', '.npy', '.json'} else 1, str(p)))
required = sum(path.stat().st_size for path in files)
if shutil.disk_usage(root).free < required + 100 * 1024 ** 2:
    raise RuntimeError('Insufficient free workspace for a complete backup; original sources remain untouched')
records, errors, archive, size, number = [], [], None, 0, 0

def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

for path in files:
    try:
        before = path.stat()
        original_hash = digest(path)
        if archive is None or size + before.st_size > 64 * 1024 ** 2:
            if archive:
                archive.close()
            number += 1
            archive = zipfile.ZipFile(output / f'results.{number:03d}.zip', 'w',
                                      zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True)
            size = 0
        name = path.relative_to(root).as_posix()
        archive.write(path, name)
        after = path.stat()
        stable = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns) and digest(path) == original_hash
        record = dict(path=name, bytes=before.st_size, sha256=original_hash, source_stable=stable,
                      archive=Path(archive.filename).name,
                      source_exceeds_64mib=before.st_size > 64 * 1024 ** 2)
        records.append(record)
        if not stable:
            errors.append(dict(path=name, error='Source changed while copying; retain original and inspect/re-copy this file'))
        size += before.st_size
    except Exception as error:
        errors.append(dict(path=str(path), error=repr(error)))
    (output / 'manifest.json').write_text(json.dumps(dict(files=records, errors=errors), indent=2))
if archive:
    archive.close()
for part in sorted(output.glob('results.*.zip')):
    with zipfile.ZipFile(part) as value:
        bad = value.testzip()
        if bad:
            errors.append(dict(path=bad, error='ZIP CRC check failed'))
        for record in [r for r in records if r['archive'] == part.name]:
            hasher = hashlib.sha256()
            with value.open(record['path']) as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    hasher.update(block)
            if hasher.hexdigest() != record['sha256']:
                errors.append(dict(path=record['path'], error='Archived SHA256 differs from source snapshot'))
    print(part.name, part.stat().st_size, digest(part), flush=True)
state = dict(output=str(output), files=len(records), bytes=sum(r['bytes'] for r in records),
             source_stable_files=sum(r['source_stable'] for r in records), errors=errors,
             verified_without_errors=not errors, patterns=patterns,
             excluded_directory_names=sorted(excluded_parts),
             compression='ZIP_DEFLATED level=1', records=records)
(output / 'VERIFIED_BACKUP.json').write_text(json.dumps(state, indent=2))
(output / 'archive_sha256.json').write_text(json.dumps([
    dict(path=p.name, bytes=p.stat().st_size, sha256=digest(p)) for p in sorted(output.glob('results.*.zip'))], indent=2))
print(json.dumps({k: v for k, v in state.items() if k != 'records'}, indent=2), flush=True)
