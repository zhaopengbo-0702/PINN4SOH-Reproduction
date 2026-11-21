"""Freeze two-direction exploratory transfer improvement before new target training."""
import argparse
import copy
from pathlib import Path
import re
import os

from search_replacement_grid import read, write, sha, identity
from search_stage_a import require
from verify_stage_a_registration import verify

ESSENTIAL_SOURCE_FILES = ('model.pth', 'config.json', 'manifest.json', 'splits.json',
                          'logging.txt', 'data_identity.json', 'source_inventory.json', 'verification_checkpoint.json')
DIRECTIONS = {'HUST-to-MIT': ('HUST', 'all', .005), 'XJTU-to-MIT': ('XJTU', '2C', .001)}


def long_archive(path):
    path=Path(path).resolve()
    if os.name=='nt' and not str(path).startswith('\\\\?\\'):
        path=Path('\\\\?\\'+str(path))
    return path


def source_record(archive, relative, dataset, batch, repo):
    folder = archive / relative
    manifest, config = read(folder/'manifest.json'), read(folder/'config.json')
    task, parameters = config['task'], config['parameters']
    require(manifest['status'] == 'complete' and config['checkpoint_snapshot_deepcopy'] is True
            and task['loss_variant'] == 'paper_prediction_monotonicity' and task['dataset'] == dataset
            and task['batch'] == batch and parameters['epochs'] == 200 and not config.get('smoke', False),
            'Source must be a full, completed, declared paper-prediction source: ' + relative)
    for name in ESSENTIAL_SOURCE_FILES:
        require((folder/name).is_file(), 'Source evidence missing: ' + name)
        if name in manifest['artifact_sha256']:
            require(sha(folder/name) == manifest['artifact_sha256'][name], 'Source artifact changed: ' + name)
    validations = [(int(e), float(v)) for e, v in re.findall(r'\[Valid\] epoch:(\d+), MSE: ([^\s]+)', (folder/'logging.txt').read_text(encoding='utf-8'))]
    require(validations and min(validations, key=lambda p:p[1])[0] == manifest['best_epoch'], 'Source best-epoch history mismatch')
    data = read(folder/'data_identity.json')
    inventory=read(folder/'source_inventory.json')
    for name,checksum in inventory['sha256'].items():
        if name.endswith('.py'):require(sha(repo/name)==checksum,'Historical source author Python differs: '+name)
    for path, checksum in data['input_file_sha256'].items():
        require(sha(repo/path) == checksum, 'Historical source CSV differs: ' + path)
    splits = read(folder/'splits.json')
    require(all('/'+dataset+' data/' in '/'+p for row in splits for p in row['paths']), 'Unexpected source dataset')
    return dict(relative_folder=relative, dataset=dataset, batch=batch, training_seed=task['seed'],
        best_epoch=manifest['best_epoch'], environment=config['environment'],
        source_fingerprint=config['source_fingerprint'], input_tensor_identity=data,
        evidence_sha256={name:sha(folder/name) for name in ESSENTIAL_SOURCE_FILES},
        source_selection='Prespecified same repeat index, never target score',
        historical_source_seed=True)


def make_protocol(archive, registration, repo):
    archive=long_archive(archive)
    registration,repo=[Path(x).resolve() for x in (registration,repo)]
    checked = verify(registration, repo)
    base = read(registration/'mit_pinn_stage_a9.json')
    target = copy.deepcopy(base['splits'])
    first = 'data/MIT data/2017-05-12/2017-05-12_battery-1.csv'
    i = target['train']['paths'].index(first)
    target['train']['paths'], target['train']['physical_ids'] = [first], [target['train']['physical_ids'][i]]
    physical = read(registration/'mit_physical_identity.json')
    sources = {}
    for direction, (dataset, batch, _) in DIRECTIONS.items():
        sources[direction] = {}
        for stack in ('modern', 'legacy'):
            records = {}
            for repeat in range(1, 11):
                if stack == 'modern':
                    prefix = 'pinn4soh_full_paper_stage1/tasks/full'
                elif repeat == 1:
                    prefix = 'pinn4soh_legacy_formal_control_20261110/tasks/legacy_control/full'
                else:
                    prefix = 'pinn4soh_legacy_selected_extension_20261110/tasks/legacy_extension/full'
                relative = '%s/paper_equation/%s/%s/PINN/Experiment%02d' % (prefix, dataset, batch, repeat)
                records[str(repeat)] = source_record(archive, relative, dataset, batch, repo)
            sources[direction][stack] = records
        for repeat in map(str, range(1, 11)):
            a, b = [sources[direction][stack][repeat] for stack in ('modern', 'legacy')]
            require(a['training_seed'] == b['training_seed'] and a['source_fingerprint'] == b['source_fingerprint']
                    and a['input_tensor_identity'] == b['input_tensor_identity'], 'Historical source pair is not matched')
            require(a['environment']['torch'] != b['environment']['torch'], 'Source stack pair is not distinct')
    fixed = dict(epochs=200, min_epochs=30, patience=30, batch_size=128, threads=2,
        normalization_method='min-max', deterministic=True, allow_tf32=False, grad_clip_norm=0,
        schedule='constant_all_lanes', validation='mean physical battery RMSE', best_epoch_tie='earliest')
    candidates = []
    for lr in (.0001, .0003, .001):
        for freeze, alpha, beta in [('all_u', 0., 0.), ('all_u', .1, .01), ('head_only', .1, .01)]:
            candidates.append(dict(candidate_id='candidate_%02d' % (len(candidates)+1),
                parameters=dict(lr=lr, alpha=alpha, beta=beta, freeze=freeze,
                    mono='mean_eval', weight_decay=0., initialization='source')))
    target_only = [dict(candidate_id='candidate_%02d' % (i*3+j+1),
        parameters=dict(lr=lr, alpha=0., beta=0., freeze='all_u', mono='mean_eval',
                        weight_decay=wd, initialization='random'))
        for i, lr in enumerate((.0001,.0003,.001)) for j, wd in enumerate((0.,1e-5,1e-4))]
    controls = {name:dict(candidate_id='original_loss_control', parameters=dict(lr=lr, alpha=.7, beta=50.,
        freeze='all_u', mono='sum_train', weight_decay=0., initialization='source')) for name, (_, _, lr) in DIRECTIONS.items()}
    return dict(format='pinn4soh-transfer-robustness-prereg-v1', status='frozen_before_new_target_training',
        historical_test_observed=True, label='exploratory_improvement_not_paper_reproduction',
        stage_a_registration_sha256=checked['registration_sha256'],
        target_splits=target, target_identity={p:physical['identities'][p] for r in target.values() for p in r['paths']},
        unused_target_train_count=80, target_train_rule='Same original first MIT target battery May1; verified pure UID. No score used.',
        validation_test_rule='Reuse frozen Stage A MIT20 physical validation and22 historical test; mixed May5/6 excluded',
        fixed=fixed, selection_seeds=[2026111101,2026111102,2026111103],
        future_confirmation_seeds=list(range(2026111111,2026111121)),
        selection_source_repeat_indices=[1,2,3], confirmation_source_repeat_indices=list(range(1,11)),
        source_records=sources, transfer_candidates=candidates, target_only_candidates=target_only,
        original_loss_controls=controls,
        source_freeze='All G parameters frozen and G eval in every lane; head_only also freezes encoder and sets it eval',
        objective='Mean across3 paired target seeds of minimum validation physical-battery macro RMSE; tie candidate order',
        selection_policy='Modern only; complete both directions and target-only before locking all selections. No test selection; no expanding grid to ensure win.',
        confirmation_policy='Ten new target seeds use fixed selected configuration in both runtimes; historical source repeats paired by index, never reselected on target scores.',
        budget=dict(modern_selection_training=81,modern_original_control_training=6,
            future_training_per_stack=50,future_source_only_evaluations_per_stack=20,
            target_only_shared_between_directions=True,total_formal_target_training_two_stacks=187),
        limits=['Fixed one target training battery and one validation split do not establish selection robustness across batteries.',
            'Twenty additional labeled validation batteries guide selection and epoch stopping: this is not a strict one-labeled-target-battery budget.',
            'Historical source weights are reused. Ten fresh target seeds are not ten fresh source pretraining runs.',
            'Original-loss control preserves loss/constant learning rate but uses new identity-separated target splits, macro selection and min30/patience30; not original Table4 reproduction.',
            'Target-only is equally tuned and uses identical supervised endpoints, target data and budget, but it naturally has no source initialization.',
            'All lanes retain full-lifetime normalization and capacity-dependent3sigma. Offline transductive, not causal deployment.',
            'No target-test evaluation is necessary for selection; test stage requires all frozen selections and ten confirmation training results.',
            'Torch1.7 compatibility must be verified on its actual GPU. CPU modern smoke is execution evidence only.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive-root',type=Path,required=True)
    p.add_argument('--registration',type=Path,required=True)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    o=p.parse_args()
    require(not o.output.exists(), 'Use new output file; no overwrite of preregistration')
    protocol=make_protocol(o.archive_root,o.registration,o.repo)
    protocol['generator_sha256']=sha(__file__)
    protocol['implementation_sha256']={name:sha(Path(__file__).with_name(name)) for name in
        ('run_transfer_robustness.py','prepare_transfer_robustness.py','search_stage_a.py','search_replacement_grid.py','run_transfer.py','run_extended.py')}
    write(o.output,protocol)
    print('TRANSFER_PROTOCOL_FROZEN '+sha(o.output)+' '+identity(protocol['budget']))


if __name__=='__main__':main()
