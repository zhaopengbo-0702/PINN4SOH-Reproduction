"""Independent audit/portable collection of all fixed Stage A future-seed runs.

This collector never selects a configuration or a lucky confirmation seed.
The reused Stage A worker is audited from its saved arrays and physical IDs.
"""
import argparse
import math
from pathlib import Path
import re
import statistics

from search_replacement_grid import read, write, sha, identity
from collect_stage_a import archive_verified, audit_suite, audit_data_identity, finite_tree, objective
from verify_stage_a_registration import verify


FUTURE_SEEDS = list(range(2026111111, 2026111121))
TRAIN_ARTIFACTS = {'best_model.pth', 'valid_truth.npy', 'valid_prediction.npy', 'validation_metrics.json',
                   'data_identity.json', 'history.json', 'gradient_diagnostics.json', 'diagnostic_batch.json'}
TEST_ARTIFACTS = {'test_truth.npy', 'test_prediction.npy', 'test_metrics.json', 'data_identity.json'}


def require(value, message):
    if not value:
        raise ValueError(message)


def close(actual, expected):
    return math.isclose(actual, expected, rel_tol=1e-13, abs_tol=1e-15)


def summary(values):
    """All-seed mean and uncertainty over random initialization, never batteries."""
    values = list(values)
    require(values and all(math.isfinite(v) for v in values), 'Nonfinite/empty summary')
    n, mean = len(values), statistics.mean(values)
    sd = statistics.stdev(values) if n > 1 else None
    se = sd / math.sqrt(n) if sd is not None else None
    if n == 10:
        critical = 2.2621571628540993  # two-sided Student t, df=9, confidence=.95
    elif n > 1:
        from scipy.stats import t
        critical = float(t.ppf(.975, n - 1))
    else:
        critical = None
    return dict(n=n, mean=mean, sample_std=sd, standard_error=se,
                mean_t95_interval=[mean-critical*se, mean+critical*se] if critical is not None else None,
                t_degrees_freedom=n-1 if n > 1 else None,
                uncertainty_unit='training seed with fixed historical test batteries',
                unit='SOH fraction RMSE')


def physical_metrics(truth, prediction, batteries):
    """Recompute companion errors independently from every saved prediction."""
    import numpy as np
    truth, prediction = np.asarray(truth).reshape(-1).astype(np.float64), np.asarray(prediction).reshape(-1).astype(np.float64)
    require(truth.size and truth.shape == prediction.shape and np.isfinite(truth).all() and np.isfinite(prediction).all() and
            np.all(truth > 0), 'Invalid arrays for companion battery metrics')
    grouped, start = {}, 0
    for battery in batteries:
        end = start+battery['n_pairs']
        grouped.setdefault(battery['physical_id'], []).append((truth[start:end], prediction[start:end]))
        start = end
    require(start == len(truth), 'Metric battery boundaries differ')
    per_battery = []
    for uid, pieces in grouped.items():
        true = np.concatenate([row[0] for row in pieces]); pred = np.concatenate([row[1] for row in pieces]); diff = pred-true
        per_battery.append(dict(physical_id=uid, n_pairs=len(true), MSE=float(np.mean(diff**2)), RMSE=float(np.sqrt(np.mean(diff**2))),
                                MAE=float(np.mean(np.abs(diff))), MAPE_pct=float(np.mean(np.abs(diff)/true)*100)))
    return dict(battery_macro_RMSE=sum(r['RMSE'] for r in per_battery)/len(per_battery),
                battery_macro_MAE=sum(r['MAE'] for r in per_battery)/len(per_battery),
                battery_macro_MAPE_pct=sum(r['MAPE_pct'] for r in per_battery)/len(per_battery),
                pooled_MSE=float(np.mean((prediction-truth)**2)), per_physical_battery=per_battery)


def metric_record_matches(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(metric_record_matches(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(metric_record_matches(a, b) for a, b in zip(actual, expected))
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and close(actual, expected)
    return actual == expected


def seed_folders(parent):
    if not parent.is_dir():
        return {}
    records = {}
    for path in parent.iterdir():
        if not path.is_dir() or not re.fullmatch(r'seed_[0-9]+', path.name):
            continue
        seed = int(path.name[5:])
        require(path.name == 'seed_%d' % seed and seed not in records, 'Noncanonical/duplicate seed directory')
        records[seed] = path
    return records


def expected_rate_ratio(epoch, fixed):
    warm = fixed['warmup_epochs']
    if warm and epoch <= warm:
        return fixed['warmup_ratio'] + (1-fixed['warmup_ratio']) * (epoch-1) / max(1, warm-1)
    progress = (epoch-warm-1) / max(1, fixed['epochs']-warm-1)
    return fixed['final_lr_ratio'] + .5*(1-fixed['final_lr_ratio'])*(1+math.cos(math.pi*progress))


def check_worker(worker, plan, folder, seed, test):
    spec, selected = plan['specification'], plan['selected']
    roles = ('test',) if test else ('train', 'valid')
    expected_splits = {role: plan['resolved_splits'][role] for role in roles}
    require(worker['mode'] == ('test' if test else 'train') and worker['model'] == spec['model'] and
            worker['repo'] == plan['repo'] and Path(worker['output']).resolve() == folder and worker['seed'] == seed and
            worker['candidate_id'] == selected['candidate_id'] and worker['parameters'] == selected['parameters'] and
            worker['fixed'] == spec['fixed'] and worker['splits'] == expected_splits and
            worker['source_sha256'] == plan['source_sha256'] and worker['device'] == plan['device'] and
            worker['input_sha256'] == {name: plan['input_sha256'][name] for role in roles for name in expected_splits[role]['paths']},
            'Fixed future-seed worker differs from frozen plan')


def check_artifacts(folder, worker, result, expected):
    require(result['status'] == 'complete' and result['worker_identity'] == identity(worker) and
            result['model'] == worker['model'] and set(result['artifact_sha256']) == expected,
            'Incomplete/changed worker result')
    for name, checksum in result['artifact_sha256'].items():
        require(sha(folder/name) == checksum, 'Worker artifact SHA differs: '+name)
    environment = result['environment']
    require(environment['workers'] == 1 and environment['threads'] == worker['fixed']['threads'] and
            environment['deterministic_algorithms'] == worker['fixed']['deterministic'] and
            environment['cudnn_benchmark'] is False and environment['cudnn_deterministic'] == worker['fixed']['deterministic'] and
            environment['matmul_tf32'] == worker['fixed']['allow_tf32'] and environment['cudnn_tf32'] == worker['fixed']['allow_tf32'],
            'Worker runtime controls differ')


def check_history(folder, plan, result, score, formal):
    fixed, params = plan['specification']['fixed'], plan['selected']['parameters']
    history, diagnostics = read(folder/'history.json'), read(folder/'gradient_diagnostics.json')
    require(history and finite_tree(history) and [r['epoch'] for r in history] == list(range(1, len(history)+1)) and
            len(history) <= fixed['epochs'] and (not formal or len(history) >= fixed['min_epochs']), 'Invalid history/epoch bound')
    pde = plan['specification']['model'] == 'PINN' and params['alpha'] > 0
    best = min(history, key=lambda r: (r['validation']['battery_macro_RMSE'], r['epoch']))
    require(close(score, result['objective']) and result['objective'] == result['validation_battery_macro_RMSE'] == best['validation']['battery_macro_RMSE'] and
            result['best_epoch'] == best['epoch'] and result['epochs_completed'] == len(history) and
            result['G_optimizer_enabled'] == pde, 'Best epoch/checkpoint objective differs')
    best_so_far, stale = float('inf'), 0
    for row in history:
        current = row['validation']['battery_macro_RMSE']
        stale += 1
        if current < best_so_far:
            best_so_far, stale = current, 0
        ratio = expected_rate_ratio(row['epoch'], fixed)
        require(row['stale_epochs'] == stale and row['gradients_finite'] is True and
                row['mono_branch_mode'] == 'eval_with_grad' and row['data_branch_mode'] == 'train' and row['pde_computed'] == pde and
                0 < row['batch_min'] <= row['batch_max'] <= fixed['batch_size'] and 0 <= row['clip_trigger_batches'] <= row['batch_count'] and
                close(row['lr_u'], params['lr_u']*ratio) and
                (close(row['lr_F'], params['lr_F']*ratio) if pde else row['lr_F'] is None), 'History schedule/control differs')
        for term, weight in [('data', 1), ('pde', params['alpha']), ('mono', params['beta_mean'])]:
            require(close(row['weighted_losses'][term], weight*row['losses'][term]), 'Weighted history loss differs')
        # Torch forms the total in float32; Python-reweighted per-term epoch
        # means can differ by that arithmetic's rounding error.
        require(math.isclose(row['losses']['total'], sum(row['weighted_losses'].values()), rel_tol=1e-6, abs_tol=1e-8),
                'Total loss differs')
    require(len(history) == fixed['epochs'] or (len(history) >= fixed['min_epochs'] and stale >= fixed['patience']),
            'Worker stopped before registered stop rule')
    require(finite_tree(diagnostics) and [r['epoch'] for r in diagnostics] == [r['epoch'] for r in history] and
            all(r['mono_requires_grad'] and r['mono_dropout'] == 'eval_with_grad' and r['data_dropout'] == 'train' and
                r['pde_computed'] == pde for r in diagnostics), 'Gradient/monotonicity diagnostics differ')


def audit_group(root, require_test=True, formal=True):
    import numpy as np
    root = Path(root).resolve()
    plan, state = read(root/'plan.json'), read(root/'status.json')
    spec, selected = plan['specification'], plan['selected']
    seeds, fixed = spec['seeds'], spec['fixed']
    require(seeds and len(seeds) == len(set(seeds)) and plan['workers'] == 1 and
            plan['training_runs'] == plan['candidate_runs'] == len(seeds) and spec['candidates'] == [selected['parameters']] and
            plan['candidates'] == [dict(candidate_id=selected['candidate_id'], parameters=selected['parameters'])],
            'Fixed candidate/seed scope differs')
    if formal:
        require(seeds == FUTURE_SEEDS and spec['smoke'] is False and fixed['min_epochs'] >= 30 and fixed['patience'] >= 30 and
                fixed['warmup_epochs'] == 5 and fixed['epochs'] == 200 and fixed['normalization_method'] == 'min-max',
                'Formal confirmation does not match preregistered ten seeds/schedule')
    require(state['plan_sha256'] == identity(plan) and state['total_training'] == state['completed_training'] == len(seeds) and
            state['total_test'] == len(seeds) and state['status'] in ({'complete'} if require_test else {'running', 'training_complete', 'complete'}),
            'Incomplete confirmation group')
    selection_path = root/'frozen_selection.json'
    require(sha(selection_path) == plan['stage_a_selection_sha256'] and
            sha(plan['stage_a_selection_path']) == plan['stage_a_selection_sha256'] and
            read(selection_path)['selected'] == selected and read(selection_path)['test_metrics_used'] is False,
            'Stage A selection snapshot/source changed')
    source_plan = read(plan['stage_a_plan_path'])
    require(identity(source_plan) == plan['stage_a_plan_sha256'] and
            read(selection_path)['plan_sha256'] == plan['stage_a_plan_sha256'], 'Stage A source plan binding differs')
    for name, checksum in {**plan['source_sha256'], **plan['input_sha256']}.items():
        require(sha(name) == checksum, 'Source/input changed since confirmation plan')
    repo = Path(plan['repo']).resolve()
    original_author = {name: checksum for name, checksum in source_plan['source_sha256'].items() if repo in Path(name).resolve().parents}
    current_author = {name: checksum for name, checksum in plan['source_sha256'].items() if repo in Path(name).resolve().parents}
    original_tools = {Path(name).name: checksum for name, checksum in source_plan['source_sha256'].items() if name not in original_author}
    current_tools = {Path(name).name: checksum for name, checksum in plan['source_sha256'].items() if name not in current_author}
    require(len(current_tools) == len(plan['source_sha256'])-len(current_author) and
            original_author == current_author and set(current_tools) == set(original_tools) | {'run_stage_a_confirmation.py'} and
            all(current_tools[name] == checksum for name, checksum in original_tools.items()) and
            plan['input_sha256'] == source_plan['input_sha256'] and plan['resolved_splits'] == source_plan['resolved_splits'],
            'Stage A author/tool bytes or data splits changed')
    training, tests = seed_folders(root/'training'), seed_folders(root/'fixed_test')
    require(set(training) == set(seeds) and (set(tests) == set(seeds) if require_test else set(tests).issubset(seeds)),
            'Confirmation training/test seed directory set differs')
    require(state['completed_test'] == len(tests), 'Test completion count differs')
    rows, test_rows, checks, tensor_identity, test_tensor_identity = [], [], 0, None, None
    for seed in seeds:
        folder = training[seed]
        worker, result = read(folder/'worker.json'), read(folder/'result.json')
        check_worker(worker, plan, folder, seed, False)
        check_artifacts(folder, worker, result, TRAIN_ARTIFACTS)
        require(not formal or (plan['device'].isdigit() and result['environment']['device'] == 'cuda' and result['environment']['gpu']),
                'Formal confirmation must execute on the allocated single GPU')
        require(result['test_loaded'] is False and result['checkpoint_reload_bitwise_prediction_equal'] is True,
                'Training accessed test or checkpoint did not reload exactly')
        data = read(folder/'data_identity.json'); audit_data_identity(data, worker['splits'])
        require(tensor_identity is None or tensor_identity == data, 'Data tensors differ across confirmation seeds')
        tensor_identity = data
        truth, prediction = np.load(folder/'valid_truth.npy', allow_pickle=False), np.load(folder/'valid_prediction.npy', allow_pickle=False)
        score = objective(truth, prediction, data['valid']['batteries'])
        check_history(folder, plan, result, score, formal)
        metrics = read(folder/'validation_metrics.json')
        require(close(metrics['battery_macro_RMSE'], score) and metric_record_matches(metrics, physical_metrics(truth, prediction, data['valid']['batteries'])),
                'Saved validation metrics differ from arrays')
        rows.append(dict(seed=seed, validation_battery_macro_RMSE=score, best_epoch=result['best_epoch'],
                         epochs_completed=result['epochs_completed'], result_sha256=sha(folder/'result.json')))
        checks += len(TRAIN_ARTIFACTS)
        if seed not in tests:
            continue
        test_folder = tests[seed]
        test_worker, test_result = read(test_folder/'worker.json'), read(test_folder/'result.json')
        check_worker(test_worker, plan, test_folder, seed, True)
        check_artifacts(test_folder, test_worker, test_result, TEST_ARTIFACTS)
        checkpoint = folder/'best_model.pth'
        require(Path(test_worker['checkpoint']).resolve() == checkpoint and sha(checkpoint) == test_worker['checkpoint_sha256'] and
                Path(test_worker['selection']).resolve() == selection_path and test_worker['selection_sha256'] == test_result['selection_sha256'] == sha(selection_path) and
                test_result['test_used_for_selection'] is False and test_result['initial_solution_sha256'] == result['initial_solution_sha256'],
                'Fixed test did not use same-seed checkpoint/frozen Stage A config')
        test_data = read(test_folder/'data_identity.json'); audit_data_identity(test_data, test_worker['splits'])
        require(test_tensor_identity is None or test_tensor_identity == test_data, 'Fixed test tensors differ across confirmation seeds')
        test_tensor_identity = test_data
        truth, prediction = np.load(test_folder/'test_truth.npy', allow_pickle=False), np.load(test_folder/'test_prediction.npy', allow_pickle=False)
        test_score = objective(truth, prediction, test_data['test']['batteries'])
        test_metrics = read(test_folder/'test_metrics.json')
        require(close(test_score, test_result['test_battery_macro_RMSE']) and close(test_score, test_metrics['battery_macro_RMSE']) and
                metric_record_matches(test_metrics, physical_metrics(truth, prediction, test_data['test']['batteries'])),
                'Fixed test macro RMSE differs from arrays')
        test_rows.append(dict(seed=seed, battery_macro_RMSE=test_score, battery_macro_MAE=test_metrics['battery_macro_MAE'],
                              battery_macro_MAPE_pct=test_metrics['battery_macro_MAPE_pct'], result_sha256=sha(test_folder/'result.json')))
        checks += len(TEST_ARTIFACTS)
    return dict(status='passed', dataset=spec['dataset'], model=spec['model'], seeds=seeds, selected=selected,
                training_runs=len(rows), fixed_test_evaluations=len(test_rows), training_per_seed=rows, test_per_seed=test_rows,
                test_summary=summary(r['battery_macro_RMSE'] for r in test_rows) if test_rows else None,
                artifact_sha256_checks=checks, config_selected_from_confirmation=False, seed_selected_from_confirmation=False,
                stage_a_selection_sha256=plan['stage_a_selection_sha256'], same_seed_checkpoint_verified=True if require_test else None,
                training_data_identity_sha256=identity(tensor_identity),
                test_data_identity_sha256=identity(test_tensor_identity) if test_tensor_identity else None,
                formal=formal, historical_test_observed=True)


def paired_comparisons(audits):
    comparisons = {}
    for dataset in ('mit', 'hust'):
        pinn = audits[dataset+'_pinn']
        for model in ('mlp', 'cnn'):
            other = audits[dataset+'_'+model]
            left = {r['seed']: r['battery_macro_RMSE'] for r in pinn['test_per_seed']}
            right = {r['seed']: r['battery_macro_RMSE'] for r in other['test_per_seed']}
            require(set(left) == set(right) == set(FUTURE_SEEDS), 'Paired comparison seed sets differ')
            rows = [dict(seed=seed, PINN=left[seed], baseline=right[seed], PINN_minus_baseline=left[seed]-right[seed]) for seed in FUTURE_SEEDS]
            comparisons[dataset+'_pinn_minus_'+model] = dict(per_seed=rows,
                summary=summary(r['PINN_minus_baseline'] for r in rows), negative_delta_favors='PINN',
                interpretation='Seed variability on the same historical test batteries; exploratory and not multiplicity adjusted')
    return comparisons


def add_tree(paths, label, target):
    target = Path(target).resolve()
    for path in target.rglob('*') if target.is_dir() else [target]:
        require(not path.is_symlink(), 'Symlink rejected')
        if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
            name = label+'/'+(path.relative_to(target).as_posix() if target.is_dir() else path.name)
            require(name not in paths or paths[name] == path, 'Conflicting archive path')
            paths[name] = path


def collect(root, bundle, registration, stage_a_root, output):
    root, bundle, registration, stage_a_root, output = [Path(p).resolve() for p in (root, bundle, registration, stage_a_root, output)]
    require(not output.exists() and re.fullmatch(r'[A-Za-z0-9_-]+', output.name), 'Fresh ASCII backup folder required')
    manifest = read(bundle/'TOOLS_SHA256.json')
    for name, row in manifest['files'].items():
        path = bundle/name
        require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Confirmation tool bundle changed')
    registration_document = read(registration/'REGISTRATION.json')
    require(registration_document['future_confirmation_seeds'] == FUTURE_SEEDS and
            not set(FUTURE_SEEDS).intersection(registration_document['selection_seeds']), 'Registered future seed scope differs')
    repo = Path(read(root/'mit_pinn/plan.json')['repo']).resolve()
    checked = verify(registration, repo)
    audits, prior_audits, source_snapshots = {}, {}, {}
    for dataset in ('mit', 'hust'):
        for model in ('pinn', 'mlp', 'cnn'):
            group = dataset+'_'+model
            folder = root/group
            plan = read(folder/'plan.json')
            spec, source_spec = plan['specification'], read(registration/(group+'_stage_a9.json'))
            require(plan['registration_sha256'] == checked['registration_sha256'] and plan['repo'] == str(repo) and
                    plan['stage_a_selection_path'] == str(stage_a_root/group/'selection.json') and
                    plan['stage_a_plan_path'] == str(stage_a_root/group/'plan.json'), 'Confirmation origin differs')
            require(set(spec) == set(source_spec) and all(spec[key] == source_spec[key] for key in source_spec
                    if key not in {'label', 'seeds', 'candidates', 'protocol_evidence'}) and
                    spec['seeds'] == FUTURE_SEEDS and spec['candidates'] == [plan['selected']['parameters']],
                    'Confirmation alters frozen Stage A settings')
            prior_audits[group] = audit_suite(stage_a_root/group, True)
            prior_plan = read(stage_a_root/group/'plan.json')
            require(prior_plan['specification'] == source_spec and prior_plan['candidates'] == [
                    dict(candidate_id='candidate_%02d' % (i+1), parameters=params) for i, params in enumerate(source_spec['candidates'])],
                    'Source Stage A candidate grid differs from registration')
            require(prior_audits[group]['selected'] == plan['selected'], 'Confirmation config differs from source selection')
            audits[group] = audit_group(folder, True, True)
            for name in ('plan.json', 'selection.json', 'status.json'):
                source_snapshots['stage_a_origin/'+group+'/'+name] = stage_a_root/group/name
    require(sum(r['training_runs'] for r in audits.values()) == 60 and
            sum(r['fixed_test_evaluations'] for r in audits.values()) == 60 and
            sum(r['candidate_runs'] for r in prior_audits.values()) == 162 and
            sum(r['test_evaluations'] for r in prior_audits.values()) == 18, 'Incomplete formal confirmation/origin scope')
    comparisons = paired_comparisons(audits)
    for dataset in ('mit', 'hust'):
        require(len({audits[dataset+'_'+model]['training_data_identity_sha256'] for model in ('pinn', 'mlp', 'cnn')}) == 1 and
                len({audits[dataset+'_'+model]['test_data_identity_sha256'] for model in ('pinn', 'mlp', 'cnn')}) == 1,
                'Paired model comparison does not use identical tensors/batteries')
    output.mkdir(parents=True)
    report = dict(status='passed', training_runs=60, fixed_test_evaluations=60, future_seeds=FUTURE_SEEDS,
                  registration=checked, groups=audits, source_stage_a_audits=prior_audits,
                  paired_comparisons=comparisons, config_and_seed_selection_performed=False,
                  tools_manifest_sha256=sha(bundle/'TOOLS_SHA256.json'),
                  limits=['Historical test batteries already observed; exploratory improvement experiment',
                          'Full-trajectory 3sigma cleaning includes capacity and full-trajectory minmax is transductive',
                          'Ten seeds measure training randomness on fixed datasets, not ten independent battery cohorts',
                          'No old-runtime confirmation or migration robustness experiment is performed here'])
    write(output/'STAGE_A_CONFIRMATION_INDEPENDENT_AUDIT.json', report)
    paths = dict(source_snapshots)
    for label, target in [('results', root), ('tools', bundle), ('registration', registration),
                          ('audit', output/'STAGE_A_CONFIRMATION_INDEPENDENT_AUDIT.json')]:
        add_tree(paths, label, target)
    for group in audits:
        plan = read(root/group/'plan.json')
        for name, checksum in {**plan['source_sha256'], **plan['input_sha256']}.items():
            path = Path(name).resolve()
            if repo in path.parents:
                require(sha(path) == checksum and not path.is_symlink(), 'Original repo/input snapshot changed')
                paths['repo_snapshot/'+path.relative_to(repo).as_posix()] = path
    frozen, archive, records = archive_verified(paths, output)
    evidence = dict(format='pinn4soh-stage-a-confirmation-collection-v1', status='complete',
        archive_bytes=archive.stat().st_size, archive_sha256=sha(archive), all_archived_member_hashes_verified=True,
        training_runs=60, fixed_test_evaluations=60, future_seeds=FUTURE_SEEDS, groups=audits,
        paired_comparisons=comparisons, inventory=[dict(path=name, **row) for name, row in sorted(frozen.items())],
        runtime_note='Modern runtime; LEGACY_STAGEA archive prefix solely preserves strict restore grammar', limits=report['limits'])
    write(output/'STAGE_A_CONFIRMATION_COLLECTION_VERIFIED.json', evidence)
    write(output/'JSON_TRANSFER_MANIFEST.json', dict(format='pinn4soh-backup-json-base64-v1', archive_type='tar.gz',
        verified_without_errors=True, original_archives=1, wrapper_files=len(records), records=records))
    return evidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'bundle', 'registration', 'stage_a_root', 'output'):
        parser.add_argument('--'+name.replace('_', '-'), dest=name, type=Path, required=True)
    evidence = collect(**vars(parser.parse_args()))
    print({key: value for key, value in evidence.items() if key not in {'inventory', 'groups', 'paired_comparisons'}})
