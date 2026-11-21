"""Collect the completed 12+40 legacy controls into audited JSON transfer chunks.

Run with the modern host Python, outside the isolated legacy interpreter:
python3 collect_legacy_results.py --output-prefix /workspace/legacy_results_20261110
Standard library only. Sources are read without modification. Runtime binaries,
offline package archives, and uploaded 14-part archives are not copied.
"""
import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile

FORMAT = 'pinn4soh-backup-json-base64-v1'
BLOCK = 4 * 1024 * 1024


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as reader:
        for block in iter(lambda: reader.read(BLOCK), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def collect(workspace, output_prefix, runtime_prefix=None, smoke=None, control=None,
            comparison=None, include=None, allow_partial=False,
            without_modern_pairs=False, chunk_mib=64):
    workspace, output = Path(workspace).resolve(), Path(output_prefix).resolve()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', output.name):
        raise ValueError('Use letters, digits, underscore or hyphen in output prefix')
    if output.exists():
        raise ValueError('Use a new output prefix; existing data will not be overwritten')
    if not output.parent.is_dir() or not 1 <= chunk_mib <= 64:
        raise ValueError('Output parent must exist; chunk size must be 1..64 MiB')
    runtime = Path(runtime_prefix or workspace / 'legacy37').resolve()
    smoke = Path(smoke or workspace / 'pinn4soh_legacy_smoke_20261110').resolve()
    control = Path(control or workspace / 'pinn4soh_legacy_formal_control_20261110').resolve()
    comparison = Path(comparison or workspace / 'pinn4soh_legacy_control_comparison_20261110').resolve()
    tool = workspace / 'legacy_single_gpu_v3'
    recovered = workspace / 'legacy_offline_recovered_v1'
    paths, expected_artifacts, scope, errors, pairs = {}, {}, [], [], []

    def add(target, required=False):
        target = Path(target)
        if not target.exists():
            if required:
                errors.append('Missing required source: ' + str(target))
            return
        resolved = target.resolve()
        if resolved == output or output in resolved.parents:
            raise ValueError('Output directory cannot be a source')
        if resolved != workspace and workspace not in resolved.parents:
            raise ValueError('All collected sources must be inside workspace: ' + str(target))
        scope.append(str(target.relative_to(workspace)))
        candidates = target.rglob('*') if target.is_dir() else [target]
        for path in candidates:
            if '__pycache__' in path.parts or path.suffix == '.pyc':
                continue
            if path.is_symlink():
                raise ValueError('Symlink in result sources: ' + str(path))
            if path.is_file():
                paths[path.relative_to(workspace).as_posix()] = path

    statuses = {}
    for label, folder, manifest_path, expected in [
            ('smoke', smoke, smoke / 'suite/manifest.json', 12),
            ('formal_control', control, control / 'manifest.json', 40)]:
        add(folder, required=True)
        if manifest_path.is_file():
            manifest = read_json(manifest_path)
            tasks = manifest.get('tasks', {})
            counts = dict(Counter(row.get('status') for row in tasks.values()))
            statuses[label] = dict(expected=expected, tasks=len(tasks), counts=counts,
                                   status=manifest.get('status'))
            if len(tasks) != expected or counts != {'complete': expected} or manifest.get('status') != 'complete':
                errors.append(label + ' suite is not fully complete')
            suite_root = manifest_path.parent
            for task_id, row in tasks.items():
                for relative, digest in row.get('verified_file_sha256', row.get('artifact_sha256', {})).items():
                    name = (suite_root / 'tasks' / task_id / relative).relative_to(workspace).as_posix()
                    expected_artifacts[name] = digest
        else:
            errors.append(label + ' manifest missing')
    summary_path = smoke / 'legacy_smoke_summary.json'
    smoke_summary = read_json(summary_path) if summary_path.is_file() else {}
    if not (smoke_summary.get('smoke_passed') and smoke_summary.get('counts') == {'complete': 12}
            and smoke_summary.get('returncode') == 0):
        errors.append('12-task smoke summary is not passed')
    add(comparison, required=True)
    comparison_summary = read_json(comparison / 'summary.json') if (comparison / 'summary.json').is_file() else {}
    if sum(comparison_summary.get('task_counts', {}).values()) != 40:
        errors.append('Comparison is absent or does not account for all 40 tasks')

    add(runtime / 'installation_evidence.json', required=True)
    add(recovered / 'REASSEMBLY_VERIFIED.json', required=True)
    add(recovered / 'payload/support', required=True)
    add(recovered / 'payload/tools', required=True)
    for name in ('legacy_asset_manifest.json', 'legacy_torch_manifest.json'):
        add(recovered / 'payload/assets' / name, required=True)
    add(tool, required=True)
    for pattern in ('*.log', '*legacy*.json', '*legacy*.py', '*legacy*.md',
                    'NOTEBOOK_SCAN*.json', '*post_restart*.json',
                    'restart_verification_*.json', 'two_runtime_observations*.json', '*TF32*', '*tf32*'):
        for path in workspace.glob(pattern):
            if path.is_file() or ('tf32' in path.name.lower() and path.is_dir()):
                add(path)
    for extra in include or []:
        add(Path(extra).resolve(), required=True)

    plan_path = tool / 'control/legacy_control_plan.json'
    if not without_modern_pairs and plan_path.is_file():
        needed = {task['original_task_id'] for task in read_json(plan_path)['tasks']}
        for name in ('pinn4soh_full_paper_stage1', 'pinn4soh_full_paper_stage2'):
            suite = workspace / name
            manifest_path = suite / 'manifest.json'
            if not manifest_path.is_file():
                continue
            manifest = read_json(manifest_path)
            add(manifest_path)
            for metadata in ('plan.json', 'source_inventory.json'):
                add(suite / metadata)
            for task_id, row in manifest.get('tasks', {}).items():
                if task_id not in needed:
                    continue
                directory = suite / 'tasks' / task_id
                add(directory, required=True)
                pairs.append(task_id)
                for relative, digest in row.get('verified_file_sha256', row.get('artifact_sha256', {})).items():
                    expected_artifacts[(directory / relative).relative_to(workspace).as_posix()] = digest
        if set(pairs) != needed or len(pairs) != 40:
            errors.append('Expected exactly 40 unique selected modern task counterparts')
    elif not without_modern_pairs:
        errors.append('Missing frozen control plan for modern counterpart selection')
    if errors and not allow_partial:
        raise RuntimeError('Refusing to label incomplete evidence as a completed collection: ' + '; '.join(errors))
    if not paths:
        raise RuntimeError('No evidence files selected')
    output.mkdir()
    inventory, unstable = [], []
    archive = output / ('PINN4SOH_LEGACY_' + output.name + '.tar.gz')
    with archive.open('wb') as compressed:
        with gzip.GzipFile(fileobj=compressed, mode='wb', compresslevel=1, mtime=0) as gzip_stream:
            with tarfile.open(fileobj=gzip_stream, mode='w|') as tar:
                for name, path in sorted(paths.items()):
                    before = path.stat()
                    data = path.open('rb')
                    info = tar.gettarinfo(str(path), arcname=name)
                    hasher = hashlib.sha256()

                    class Reader:
                        def read(self, size):
                            value = data.read(size)
                            hasher.update(value)
                            return value

                    try:
                        tar.addfile(info, Reader())
                    finally:
                        data.close()
                    after = path.stat()
                    stable = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
                    row = dict(path=name, bytes=before.st_size, sha256=hasher.hexdigest(), source_stable=stable)
                    inventory.append(row)
                    if not stable:
                        unstable.append(name)
    records_by_name = {row['path']: row for row in inventory}
    artifact_errors = [name for name, digest in expected_artifacts.items()
                       if name not in records_by_name or records_by_name[name]['sha256'] != digest]
    if unstable or artifact_errors:
        raise RuntimeError('Source changed or suite artifact SHA mismatch; archive retained for diagnosis: '
                           + repr(dict(unstable=unstable, artifact_errors=artifact_errors)))
    with tarfile.open(archive, 'r|gz') as tar:
        verified = 0
        for member in tar:
            expected = records_by_name[member.name]
            hasher, total = hashlib.sha256(), 0
            reader = tar.extractfile(member)
            for block in iter(lambda: reader.read(BLOCK), b''):
                hasher.update(block)
                total += len(block)
            if total != expected['bytes'] or hasher.hexdigest() != expected['sha256']:
                raise RuntimeError('Tar member SHA mismatch: ' + member.name)
            verified += 1
    if verified != len(inventory):
        raise RuntimeError('Tar member count mismatch')
    archive_hash, archive_bytes = sha256(archive), archive.stat().st_size
    state = dict(format='pinn4soh-legacy-results-collection-v1',
        created_utc=datetime.now(timezone.utc).isoformat(), output=str(output),
        archive=str(archive), archive_sha256=archive_hash, archive_bytes=archive_bytes,
        statuses=statuses, comparison_counts=comparison_summary.get('task_counts'),
        comparison_assessment=comparison_summary.get('assessment'),
        comparison_analysis_errors=comparison_summary.get('errors', []),
        modern_task_pairs=len(pairs), modern_pairs_included=not without_modern_pairs,
        files=len(inventory), source_bytes=sum(row['bytes'] for row in inventory),
        artifact_sha256_checks=len(expected_artifacts), source_files_stable=True,
        every_archived_member_sha256_verified=True, scope=sorted(set(scope)),
        excluded='Runtime binaries, offline runtime assets, uploaded archive parts; old modern-suite tasks except 40 selected pairs',
        scientific_completion_errors=errors, collection_complete=not errors,
        verified_without_errors=True, inventory=inventory)
    write_json(output / 'LEGACY_COLLECTION_VERIFIED.json', state)
    chunk_bytes = chunk_mib * 1024 * 1024
    count = max(1, (archive_bytes + chunk_bytes - 1) // chunk_bytes)
    wrappers = []
    with archive.open('rb') as reader:
        for index in range(count):
            data = reader.read(chunk_bytes)
            metadata = dict(format=FORMAT, archive_type='tar.gz', original_zip_name=archive.name,
                original_zip_bytes=archive_bytes, original_zip_sha256=archive_hash,
                chunk_index=index, chunk_count=count, chunk_offset=index * chunk_bytes,
                chunk_bytes=len(data), chunk_sha256=hashlib.sha256(data).hexdigest(), encoding='base64')
            name = 'results.legacy.' + output.name + '.chunk%03d.json' % (index + 1)
            target = output / name
            temporary = target.with_suffix('.json.tmp')
            with temporary.open('w', encoding='ascii', newline='\n') as writer:
                writer.write(json.dumps(metadata, separators=(',', ':'))[:-1] + ',"data_base64":"')
                writer.write(base64.b64encode(data).decode('ascii'))
                writer.write('"}\n')
            decoded_document = read_json(temporary)
            decoded = base64.b64decode(decoded_document.pop('data_base64'), validate=True)
            if len(decoded) != len(data) or hashlib.sha256(decoded).hexdigest() != metadata['chunk_sha256']:
                raise RuntimeError('JSON wrapper roundtrip failed: ' + name)
            temporary.replace(target)
            wrappers.append(dict(metadata, wrapper_name=name, wrapper_bytes=target.stat().st_size,
                wrapper_sha256=sha256(target), roundtrip_verified=True))
            print(json.dumps(wrappers[-1], separators=(',', ':')), flush=True)
    write_json(output / 'JSON_TRANSFER_MANIFEST.json', dict(format=FORMAT, archive_type='tar.gz',
        verified_without_errors=True, original_archives=1, wrapper_files=len(wrappers), records=wrappers))
    print(json.dumps({key: state[key] for key in ('output', 'archive_bytes', 'archive_sha256',
          'files', 'source_bytes', 'artifact_sha256_checks', 'collection_complete')}, indent=2), flush=True)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path('/workspace'))
    parser.add_argument('--output-prefix', type=Path, required=True)
    parser.add_argument('--runtime-prefix', type=Path)
    parser.add_argument('--smoke', type=Path)
    parser.add_argument('--control', type=Path)
    parser.add_argument('--comparison', type=Path)
    parser.add_argument('--include', type=Path, action='append')
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--without-modern-pairs', action='store_true')
    parser.add_argument('--chunk-mib', type=int, default=64)
    options = parser.parse_args()
    collect(**vars(options))
