"""Create one fast immutable tar snapshot without per-file manifest rewrites.

python emergency_cloud_snapshot.py
Only generated results, report/tool folders, and workspace-level scripts/logs
are included. Offline runtimes and original runtime upload parts are excluded.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

root = Path('/workspace')
output = root / ('emergency_PINN4SOH_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
output.mkdir()
patterns = ['pinn4soh_*', 'efficiency*', 'full_tools*', 'full_report_tools*',
            'full_supplement_tools*', 'PINN4SOH-main', '*pipeline*',
            '*.log', '*.py', '*.md', '*.txt', '*.json', '*tools*.zip']
selected = sorted({path.name for pattern in patterns for path in root.glob(pattern)
                   if not path.is_symlink() and path != output})
paths_file = output / 'paths.nul'
paths_file.write_bytes(b''.join(name.encode('utf-8') + b'\0' for name in selected))
archive = output / 'emergency_results.tar.gz'
command = ['tar', '--create', '--file', str(archive), '--use-compress-program=gzip -1',
           '--directory', str(root), '--null', '--files-from', str(paths_file)]
# All excludes precede the selected paths through the files-from argument.
for excluded in ('site-packages', 'offline', 'offline_wheels', 'legacy_runtime',
                 'legacy_env', '__pycache__', '.git', '.ipynb_checkpoints'):
    command.insert(1, '--exclude=' + excluded)
print(json.dumps(dict(output=str(output), selected=selected, command=command)), flush=True)
with (output / 'tar.log').open('w', encoding='utf-8') as log:
    completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
if completed.returncode != 0:
    raise RuntimeError('tar reported nonzero status ' + str(completed.returncode) + '; inspect ' + str(output / 'tar.log'))
hasher = hashlib.sha256()
with archive.open('rb') as reader:
    for block in iter(lambda: reader.read(8 * 1024 * 1024), b''):
        hasher.update(block)
members, source_bytes, errors = 0, 0, []
with tarfile.open(archive, 'r|gz') as stream:
    for member in stream:
        members += 1
        if member.isfile():
            reader = stream.extractfile(member)
            actual = 0
            for block in iter(lambda: reader.read(8 * 1024 * 1024), b''):
                actual += len(block)
            source_bytes += actual
            if actual != member.size:
                errors.append(dict(path=member.name, expected=member.size, actual=actual))
state = dict(output=str(output), archive=str(archive), archive_bytes=archive.stat().st_size,
             sha256=hasher.hexdigest(), members=members, uncompressed_file_bytes=source_bytes,
             selected=selected, tar_returncode=completed.returncode, errors=errors,
             verified_without_errors=not errors)
(output / 'EMERGENCY_SNAPSHOT_VERIFIED.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
print(json.dumps(state, indent=2), flush=True)
