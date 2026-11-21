"""Preserve stopped, possibly incomplete Stage A work without declaring it complete.

Reads source trees only. Writes a fresh audit/archive directory and a verified
second copy. This is a deadline snapshot, not the formal Stage A collector.
"""
import argparse
import base64
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tarfile


GROUPS = tuple(d + '_' + m for d in ('mit', 'hust') for m in ('pinn', 'mlp', 'cnn'))
TRAIN_ARTIFACTS = {'best_model.pth', 'valid_truth.npy', 'valid_prediction.npy',
                   'validation_metrics.json', 'data_identity.json', 'history.json',
                   'gradient_diagnostics.json', 'diagnostic_batch.json'}
TEST_ARTIFACTS = {'test_truth.npy', 'test_prediction.npy', 'test_metrics.json', 'data_identity.json'}
RUNNERS = {'search_stage_a.py', 'run_stage_a_cloud.py', 'run_stage_a_confirmation.py',
           'run_transfer_robustness_cloud.py', 'run_transfer_robustness.py'}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def process_snapshot():
    require(Path('/proc').is_dir(), 'Live collector requires Linux /proc')
    rows = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            args = p.joinpath('cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0')
            fields = p.joinpath('stat').read_text().rsplit(') ', 1)[1].split()
            if any(Path(a).name in RUNNERS for a in args):
                rows.append(dict(pid=int(p.name), ppid=int(fields[1]), state=fields[0],
                                 start_ticks=fields[19], argv=args))
        except (OSError, ValueError, IndexError):
            continue
    return sorted(rows, key=lambda r: r['pid'])


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_tree(v) for v in value)
    return not isinstance(value, (int, float)) or math.isfinite(value)


def objective(truth, prediction, batteries):
    import numpy as np
    truth, prediction = np.asarray(truth).reshape(-1).astype('float64'), np.asarray(prediction).reshape(-1).astype('float64')
    require(truth.shape == prediction.shape and truth.size and np.isfinite(truth).all() and np.isfinite(prediction).all(), 'Invalid arrays')
    grouped, start = {}, 0
    for row in batteries:
        require(isinstance(row['n_pairs'], int) and row['n_pairs'] > 0, 'Invalid battery length')
        end = start + row['n_pairs']
        grouped.setdefault(row['physical_id'], []).extend((prediction[start:end] - truth[start:end]).tolist())
        start = end
    require(start == len(truth), 'Array/battery boundary mismatch')
    return sum(math.sqrt(sum(v * v for v in vs) / len(vs)) for vs in grouped.values()) / len(grouped)


def audit_run(folder, plan, candidate, seed, mode='train'):
    import numpy as np
    worker, result = read(folder / 'worker.json'), read(folder / 'result.json')
    spec = plan['specification']
    roles = ('train', 'valid') if mode == 'train' else ('test',)
    splits = {r: plan['resolved_splits'][r] for r in roles}
    require(worker['mode'] == mode and worker['model'] == spec['model'] and worker['repo'] == plan['repo']
            and Path(worker['output']).resolve() == folder.resolve() and worker['candidate_id'] == candidate['candidate_id']
            and worker['parameters'] == candidate['parameters'] and worker['seed'] == seed
            and worker['fixed'] == spec['fixed'] and worker['splits'] == splits
            and worker['device'] == plan['device'] and worker['source_sha256'] == plan['source_sha256']
            and worker['input_sha256'] == {p: plan['input_sha256'][p] for r in roles for p in splits[r]['paths']}, 'Worker differs from plan')
    require(result['status'] == 'complete' and result['worker_identity'] == identity(worker), 'Result/worker identity mismatch')
    require(set(result['artifact_sha256']) == (TRAIN_ARTIFACTS if mode == 'train' else TEST_ARTIFACTS), 'Artifact set mismatch')
    for name, checksum in result['artifact_sha256'].items():
        require(sha(folder / name) == checksum, 'Artifact changed: ' + str(folder / name))
    data = read(folder / 'data_identity.json')
    require(set(data) == set(roles), 'Loaded roles differ')
    for role in roles:
        rows = data[role]['batteries']
        require([r['path'] for r in rows] == splits[role]['paths'] and
                [r['physical_id'] for r in rows] == splits[role]['physical_ids'], 'Physical identity differs')
    prefix = 'valid' if mode == 'train' else 'test'
    score = objective(np.load(folder / (prefix + '_truth.npy'), allow_pickle=False),
                      np.load(folder / (prefix + '_prediction.npy'), allow_pickle=False), data[prefix]['batteries'])
    key = 'objective' if mode == 'train' else 'test_battery_macro_RMSE'
    require(math.isclose(score, result[key], rel_tol=1e-14, abs_tol=1e-15), 'Independent metric differs')
    if mode == 'train':
        require(result['test_loaded'] is False and result['checkpoint_reload_bitwise_prediction_equal'] is True, 'Training/test or reload mismatch')
        history = read(folder / 'history.json')
        require(history and len(history) <= spec['fixed']['epochs'] and finite_tree(history)
                and [r['epoch'] for r in history] == list(range(1, len(history) + 1)), 'History sequence/value differs')
        require(all(r['gradients_finite'] and r['mono_branch_mode'] == 'eval_with_grad'
                    and r['data_branch_mode'] == 'train'
                    and 0 < r['batch_min'] <= r['batch_max'] <= spec['fixed']['batch_size'] for r in history), 'Invalid history flags/batch bounds')
        best = min(history, key=lambda r: (r['validation']['battery_macro_RMSE'], r['epoch']))
        require(len(history) == result['epochs_completed'] and best['epoch'] == result['best_epoch']
                and best['validation']['battery_macro_RMSE'] == result['objective'], 'Best epoch differs')
        require(spec['smoke'] or len(history) >= spec['fixed']['min_epochs'], 'Stopped before minimum epochs')
        diagnostics = read(folder / 'gradient_diagnostics.json')
        require([r['epoch'] for r in diagnostics] == [r['epoch'] for r in history] and finite_tree(diagnostics)
                and all(r['mono_requires_grad'] and r['mono_dropout'] == 'eval_with_grad'
                        and r['data_dropout'] == 'train'
                        and r['pde_computed'] == (spec['model'] == 'PINN' and candidate['parameters']['alpha'] > 0)
                        for r in diagnostics), 'Invalid gradient diagnostics')
    else:
        require(result['test_used_for_selection'] is False, 'Test used for selection')
        suite = folder.parent.parent
        require(Path(worker['checkpoint']).resolve() == (suite / candidate['candidate_id'] / ('seed_%d' % seed) / 'best_model.pth').resolve()
                and Path(worker['selection']).resolve() == (suite / 'selection.json').resolve(), 'Test path binding differs')
        require(sha(worker['checkpoint']) == worker['checkpoint_sha256'] and
                sha(worker['selection']) == worker['selection_sha256'] == result['selection_sha256'], 'Test binding differs')
    return dict(candidate_id=candidate['candidate_id'], seed=seed, objective=result[key],
                independently_recomputed_objective=score,
                result_sha256=sha(folder / 'result.json'), artifact_hashes_verified=len(result['artifact_sha256']))


def audit_group(folder, spec_path, expected_plan=None):
    spec = read(spec_path)
    report = dict(expected_training=len(spec['seeds']) * len(spec['candidates']),
                  audited_complete_training=0, expected_test=len(spec['seeds']), audited_complete_test=0,
                  absent_training=[], incomplete_training=[], invalid_training=[], invalid_test=[],
                  absent_test=[], incomplete_test=[], rows=[])
    if not (folder / 'plan.json').exists():
        require(not folder.exists() or not any(folder.iterdir()), 'Nonempty group exists without plan')
        report['absent_training'] = ['candidate_%02d/seed_%d' % (i + 1, seed)
                                      for i, _ in enumerate(spec['candidates']) for seed in spec['seeds']]
        report['absent_test'] = ['selected_test/seed_%d' % seed for seed in spec['seeds']]
        report.update(status='not_started', selection_present=False, original_status=None)
        return report
    plan = read(folder / 'plan.json')
    require(plan['specification'] == spec, 'Plan differs from registration: ' + str(folder))
    require(plan['candidates'] == [dict(candidate_id='candidate_%02d' % (i + 1), parameters=p)
                                  for i, p in enumerate(spec['candidates'])]
            and plan['candidate_runs'] == report['expected_training'], 'Derived candidate plan differs')
    if expected_plan is not None:
        require(plan == expected_plan, 'Derived plan differs from trusted bundle/registration')
    state = read(folder / 'status.json') if (folder / 'status.json').is_file() else None
    report['original_status'] = state
    if state:
        require(state['plan_sha256'] == identity(plan), 'Status plan hash mismatch')
    expected = set()
    for candidate in plan['candidates']:
        for seed in spec['seeds']:
            run = folder / candidate['candidate_id'] / ('seed_%d' % seed)
            relative = run.relative_to(folder).as_posix()
            expected.add(relative)
            if not run.exists():
                report['absent_training'].append(relative)
            elif not (run / 'result.json').is_file():
                report['incomplete_training'].append(dict(path=relative, files=sorted(p.name for p in run.iterdir() if p.is_file())))
            else:
                try:
                    row = audit_run(run, plan, candidate, seed)
                    report['rows'].append(row)
                    report['audited_complete_training'] += 1
                except Exception as error:
                    report['invalid_training'].append(dict(path=relative, error=str(error)))
    actual = {p.relative_to(folder).as_posix() for p in folder.glob('candidate_*/seed_*') if p.is_dir()}
    report['unexpected_training_dirs'] = sorted(actual - expected)
    report['selection_present'] = (folder / 'selection.json').is_file()
    report['selection_audited'] = False
    if report['selection_present']:
        require(report['audited_complete_training'] == report['expected_training'], 'Selection exists before all candidate results validate')
        selection, ranking = read(folder / 'selection.json'), []
        for candidate in plan['candidates']:
            values = [r['objective'] for r in report['rows'] if r['candidate_id'] == candidate['candidate_id']]
            ranking.append(dict(**candidate, mean_validation_battery_macro_RMSE=sum(values) / len(values), per_seed_objectives=values))
        ranking.sort(key=lambda r: (r['mean_validation_battery_macro_RMSE'], r['candidate_id']))
        selection_rows = [{k: row[k] for k in ('candidate_id', 'seed', 'objective', 'result_sha256')} for row in report['rows']]
        require(selection['plan_sha256'] == identity(plan) and selection['test_metrics_used'] is False and
                selection['ranking'] == ranking and selection['selected'] == ranking[0]
                and selection['candidate_results'] == selection_rows, 'Selection differs from independent ranking')
        report['selection_audited'] = True
        expected_tests = {'seed_%d' % seed for seed in spec['seeds']}
        actual_tests = {p.name for p in folder.glob('selected_test/seed_*') if p.is_dir()}
        for name in sorted(actual_tests - expected_tests):
            report['invalid_test'].append(dict(path='selected_test/' + name, error='Unexpected test seed directory'))
        for seed in spec['seeds']:
            run = folder / 'selected_test' / ('seed_%d' % seed)
            if not run.exists():
                report['absent_test'].append(run.relative_to(folder).as_posix())
                continue
            if not (run / 'result.json').is_file():
                report['incomplete_test'].append(dict(path=run.relative_to(folder).as_posix(), files=sorted(p.name for p in run.iterdir() if p.is_file())))
                continue
            try:
                audit_run(run, plan, ranking[0], seed, 'test')
                report['audited_complete_test'] += 1
            except Exception as error:
                report['invalid_test'].append(dict(path=run.relative_to(folder).as_posix(), error=str(error)))
    else:
        require(not (folder / 'selected_test').exists() or not any((folder / 'selected_test').iterdir()), 'Test exists without frozen selection')
        report['absent_test'] = ['selected_test/seed_%d' % seed for seed in spec['seeds']]
    bad = report['invalid_training'] or report['invalid_test'] or report['unexpected_training_dirs']
    report['status'] = 'audit_errors' if bad else ('selection_complete_test_incomplete' if report['selection_audited'] else 'partial_training')
    if not bad and report['selection_audited'] and report['audited_complete_test'] == report['expected_test']:
        report['status'] = 'complete_group'
    return report


def add_tree(paths, label, target):
    require(re.fullmatch('[A-Za-z0-9_-]+', label), 'Unsafe archive label')
    require(target.exists() and not target.is_symlink(), 'Missing or symlink input: ' + str(target))
    for p in sorted(target.rglob('*')) if target.is_dir() else [target]:
        require(not p.is_symlink(), 'Symlink rejected: ' + str(p))
        if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc':
            name = label + '/' + (p.relative_to(target).as_posix() if target.is_dir() else p.name)
            require(name not in paths or paths[name] == p, 'Archive name collision')
            paths[name] = p


def archive_verified(paths, output):
    inventory = {name: dict(bytes=p.stat().st_size, sha256=sha(p)) for name, p in sorted(paths.items())}
    archive = output / ('PINN4SOH_LEGACY_STAGEA_' + output.name + '.tar.gz')
    with archive.open('xb') as compressed:
        with gzip.GzipFile(fileobj=compressed, mode='wb', compresslevel=1, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode='w|') as tar:
                for name, p in sorted(paths.items()):
                    before = p.stat()
                    with p.open('rb') as stream:
                        tar.addfile(tar.gettarinfo(str(p), arcname=name), stream)
                    after = p.stat()
                    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
                            and sha(p) == inventory[name]['sha256'], 'Input changed during archive')
    seen = set()
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            require(member.isfile() and member.name in inventory and member.name not in seen, 'Unexpected archive member')
            h, n, stream = hashlib.sha256(), 0, tar.extractfile(member)
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
                n += len(block)
            require(n == inventory[member.name]['bytes'] and h.hexdigest() == inventory[member.name]['sha256'], 'Archive member differs')
            seen.add(member.name)
    require(seen == set(inventory), 'Archive inventory differs')
    return archive, inventory


def collect(options):
    root, bundle, output, mirror = [Path(v).resolve() for v in (options.root, options.bundle, options.output, options.mirror)]
    require(not output.exists() and not mirror.exists(), 'Fresh output and mirror required')
    require(re.fullmatch('[A-Za-z0-9_-]+', output.name), 'Archive restore name grammar differs')
    require(output not in mirror.parents and mirror not in output.parents and output != mirror, 'Copies must be separate directories')
    inputs = [('stage_a', root), ('tools', bundle), ('pipeline', Path(options.pipeline).resolve()),
              ('stop_record', Path(options.stop_record).resolve()), ('collector_source', Path(__file__).resolve())]
    for item in options.include:
        label, separator, value = item.partition('=')
        require(separator and value, 'Use --include label=/absolute/path')
        inputs.append((label, Path(value).resolve()))
    for label, source in inputs:
        require(output != source and mirror != source and source not in output.parents and source not in mirror.parents
                and output not in source.parents and mirror not in source.parents,
                'Backup output overlaps a source: ' + label)
    processes = process_snapshot()
    require(not [r for r in processes if r['state'] != 'Z'], 'Training/coordinator still exists; let worker finish and stop dispatch before collection')
    registration = bundle / 'registration'
    pipeline_request = read(options.pipeline)['request']
    require(Path(pipeline_request['bundle']).resolve() == bundle and Path(pipeline_request['output']).resolve() == root
            and pipeline_request['bundle_manifest_sha256'] == sha(bundle / 'TOOLS_SHA256.json'), 'Pipeline bundle/output binding differs')
    manifest = read(bundle / 'TOOLS_SHA256.json')
    for name, row in manifest['files'].items():
        p = bundle / name
        require(p.is_file() and p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], 'Tool bundle changed: ' + name)
    repo = Path(read(root / 'mit_pinn' / 'plan.json')['repo']).resolve()
    require(Path(pipeline_request['repo']).resolve() == repo, 'Pipeline repo binding differs')
    expected_author = read(bundle / 'EXPECTED_AUTHOR_INPUTS.json')['author_python_sha256']
    require({p.relative_to(repo).as_posix() for p in repo.rglob('*.py')} == set(expected_author), 'Author source set differs')
    for name, checksum in expected_author.items():
        require(sha(repo / name) == checksum, 'Author source differs: ' + name)
    # Import only the already hash-checked bundle for plan and registration binding.
    sys.path.insert(0, str(bundle))
    from verify_stage_a_registration import verify
    from search_stage_a import build_plan
    checked_registration = verify(registration, repo)
    require(checked_registration['registration_sha256'] == pipeline_request['registration_sha256'],
            'Pipeline registration hash differs')
    groups, errors = {}, []
    for group in GROUPS:
        try:
            expected_plan = build_plan(read(registration / (group + '_stage_a9.json')), repo)
            expected_plan['device'] = pipeline_request['gpu']
            groups[group] = audit_group(root / group, registration / (group + '_stage_a9.json'), expected_plan)
        except Exception as error:
            errors.append(dict(group=group, error=str(error)))
    paths = {}
    for label, source in inputs:
        add_tree(paths, label, source)
    source_checks = 0
    for plan_path in sorted(root.glob('*/plan.json')):
        plan = read(plan_path)
        repo = Path(plan['repo']).resolve()
        for name, checksum in {**plan['source_sha256'], **plan['input_sha256']}.items():
            p = Path(name).resolve()
            require(p.is_file() and not p.is_symlink() and sha(p) == checksum, 'Referenced source/input changed: ' + name)
            source_checks += 1
            if repo in p.parents:
                paths['repo_snapshot/' + p.relative_to(repo).as_posix()] = p
    report = dict(format='pinn4soh-deadline-partial-v1', created_at_utc=datetime.now(timezone.utc).isoformat(),
                  deadline=options.deadline, collection_kind='deadline_partial_not_formal_complete',
                  experiment_status='stopped_for_deadline_incomplete', groups=groups, group_errors=errors,
                  audited_complete_training=sum(g.get('audited_complete_training', 0) for g in groups.values()),
                  planned_training=162, audited_complete_test=sum(g.get('audited_complete_test', 0) for g in groups.values()),
                  planned_test=18, referenced_source_input_hash_checks=source_checks, stopped_process_snapshot=processes,
                  registration_verification=checked_registration, bundle_manifest_sha256=sha(bundle / 'TOOLS_SHA256.json'),
                  all_matching_non_zombie_processes_absent=True,
                  original_status_files_preserved_without_edit=True, stop_record_sha256=sha(options.stop_record),
                  limits=['Incomplete Stage A cannot establish complete two-dataset by three-model comparisons or improvement.',
                          'Queued confirmation and transfer training are not represented as completed.',
                          'Second copy is on the same NFS if both paths are under /workspace; it is not an off-server backup.'])
    bad_groups = any(g.get('status') == 'audit_errors' or g.get('invalid_training') or g.get('invalid_test')
                     or g.get('unexpected_training_dirs') for g in groups.values())
    report['audit_status'] = 'errors_preserved' if errors or bad_groups else 'passed_completed_runs_only'
    output.mkdir(parents=True)
    write(output / 'DEADLINE_PARTIAL_AUDIT.json', report)
    paths['deadline_audit/DEADLINE_PARTIAL_AUDIT.json'] = output / 'DEADLINE_PARTIAL_AUDIT.json'
    archive, inventory = archive_verified(paths, output)
    write(output / 'FILE_SHA256_INVENTORY.json', inventory)
    record = dict(format='pinn4soh-deadline-archive-v1', archive=archive.name, bytes=archive.stat().st_size,
                  sha256=sha(archive), members=len(inventory), all_member_bytes_sha256_verified=True,
                  audit_sha256=sha(output / 'DEADLINE_PARTIAL_AUDIT.json'), audit_status=report['audit_status'])
    write(output / 'ARCHIVE_VERIFIED.json', record)
    if options.json_chunks:
        chunk_size, rows = 48 * 1024 * 1024, []
        count = max(1, (record['bytes'] + chunk_size - 1) // chunk_size)
        with archive.open('rb') as stream:
            for i in range(count):
                data = stream.read(chunk_size)
                meta = dict(format='pinn4soh-backup-json-base64-v1', archive_type='tar.gz', original_zip_name=archive.name,
                            original_zip_bytes=record['bytes'], original_zip_sha256=record['sha256'], chunk_index=i,
                            chunk_count=count, chunk_offset=i * chunk_size, chunk_bytes=len(data),
                            chunk_sha256=hashlib.sha256(data).hexdigest(), encoding='base64')
                p = output / ('results.stagea.' + output.name + '.chunk%03d.json' % (i + 1))
                write(p, dict(meta, data_base64=base64.b64encode(data).decode('ascii')))
                require(base64.b64decode(read(p)['data_base64'], validate=True) == data, 'JSON chunk differs')
                rows.append(dict(meta, wrapper_name=p.name, wrapper_bytes=p.stat().st_size, wrapper_sha256=sha(p), roundtrip_verified=True))
        write(output / 'JSON_TRANSFER_MANIFEST.json', dict(format='pinn4soh-backup-json-base64-v1', archive_type='tar.gz',
              verified_without_errors=True, original_archives=1, wrapper_files=len(rows), records=rows))
    after = process_snapshot()
    require(not [r for r in after if r['state'] != 'Z'], 'Runner appeared during collection; preserve output but do not call backup final')
    shutil.copytree(output, mirror)
    primary_files = sorted(p.name for p in output.iterdir() if p.is_file())
    require(sorted(p.name for p in mirror.iterdir() if p.is_file()) == primary_files, 'Second-copy inventory differs')
    mirror_files = []
    for name in primary_files:
        a, b = output / name, mirror / name
        require(a.stat().st_size == b.stat().st_size and sha(a) == sha(b), 'Second copy differs: ' + name)
        mirror_files.append(dict(path=name, bytes=a.stat().st_size, sha256=sha(a)))
    final_processes = process_snapshot()
    require(not [r for r in final_processes if r['state'] != 'Z'], 'Runner appeared during second copy; do not call backup final')
    receipt = dict(primary=str(output), mirror=str(mirror), files=mirror_files, all_files_bytes_sha256_equal=True,
                   archive_sha256=record['sha256'], same_storage_domain_note='Two directories; do not infer off-server backup',
                   stopped_process_snapshot_after=final_processes, original_results_not_modified=True)
    write(output / 'SECOND_COPY_VERIFIED.json', receipt)
    write(mirror / 'SECOND_COPY_VERIFIED.json', receipt)
    require((output / 'SECOND_COPY_VERIFIED.json').read_bytes() == (mirror / 'SECOND_COPY_VERIFIED.json').read_bytes(), 'Second-copy receipt differs')
    summary = dict(record)
    summary.update(audit_status=report['audit_status'], training=report['audited_complete_training'], planned_training=162,
                   tests=report['audited_complete_test'], planned_tests=18, output=str(output), mirror=str(mirror),
                   all_second_copy_files_equal=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'bundle', 'pipeline', 'stop-record', 'output', 'mirror'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--include', action='append', default=[])
    parser.add_argument('--deadline', default='2026-11-12T09:00:00+08:00')
    parser.add_argument('--json-chunks', action='store_true')
    options = parser.parse_args()
    result = collect(options)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result['audit_status'] != 'passed_completed_runs_only':
        sys.exit(2)


if __name__ == '__main__':
    main()
