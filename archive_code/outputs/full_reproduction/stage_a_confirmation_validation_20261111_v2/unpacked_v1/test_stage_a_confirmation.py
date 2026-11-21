"""Persistent low-cost Stage A confirmation verification; real test never opened."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tarfile

from run_stage_a_confirmation import group_plan,prepare_group,run_phase
from collect_stage_a_confirmation import audit_group,summary,paired_comparisons,FUTURE_SEEDS
from collect_stage_a import archive_verified
from restore_backup_json import restore
from search_replacement_grid import read,write,sha,identity


def rejected(action):
    try:action()
    except ValueError:return
    raise AssertionError('Invalid confirmation fixture was accepted')


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    base=parser.parse_args().output.resolve();base.mkdir(exist_ok=False)
    # Frozen prior synthetic MLP selection; no original battery test is loaded.
    prior=root/'outputs/full_reproduction/stage_a_cloud_validation_20261111_v4/fixture_suite'
    prior_plan=read(prior/'plan.json');selection=read(prior/'selection.json')
    source=dict(stage_a_selection_path=str(prior/'selection.json'),stage_a_selection_sha256=sha(prior/'selection.json'),
        stage_a_plan_path=str(prior/'plan.json'),stage_a_plan_sha256=identity(prior_plan))
    plan=group_plan(prior_plan['specification'],selection['selected'],FUTURE_SEEDS[:2],root/'PINN4SOH','cpu',source,
        'SYNTHETIC_ONLY_NO_FORMAL_REGISTRATION',True)
    folder=base/'synthetic_mlp';prepare_group(folder,plan)
    run_phase(folder,plan,'training',False)
    assert not (folder/'fixed_test').exists()
    assert audit_group(folder,False,False)['training_runs']==2
    run_phase(folder,plan,'fixed_test',False)
    audit=audit_group(folder,True,False)
    print('PASS two-newseed fixed train/test and independent metrics audit',flush=True)
    preserved={str(p):(sha(p),p.stat().st_mtime_ns) for p in folder.rglob('*') if p.name in {'result.json','best_model.pth','frozen_selection.json'}}
    run_phase(folder,plan,'training',True);run_phase(folder,plan,'fixed_test',True)
    assert all((sha(p),Path(p).stat().st_mtime_ns)==v for p,v in preserved.items())
    test_worker=folder/'fixed_test'/('seed_%d'%FUTURE_SEEDS[0])/'worker.json'
    original=test_worker.read_bytes();bad=read(test_worker);bad['checkpoint']=str(folder/'training'/('seed_%d'%FUTURE_SEEDS[1])/'best_model.pth');write(test_worker,bad)
    try:rejected(lambda:audit_group(folder,True,False))
    finally:test_worker.write_bytes(original)
    before=copy.deepcopy(plan);before['specification']['seeds']=FUTURE_SEEDS[1:3]
    rejected(lambda:prepare_group(folder,before))
    rejected(lambda:audit_group(folder,True,True))
    print('PASS immutable resume; cross-seed checkpoint, seed edits, smoke-as-formal rejected',flush=True)
    values=summary(range(10));assert values['mean']==4.5 and values['n']==10 and abs(values['sample_std']-3.0276503540974917)<1e-14
    mock={}
    for dataset in ('mit','hust'):
        for model,value in [('pinn',.02),('mlp',.03),('cnn',.04)]:
            mock[dataset+'_'+model]={'test_per_seed':[dict(seed=s,battery_macro_RMSE=value) for s in FUTURE_SEEDS]}
    paired=paired_comparisons(mock)
    assert abs(paired['mit_pinn_minus_mlp']['summary']['mean']+.01)<1e-15
    archive_dir=base/'archive_fixture';archive_dir.mkdir()
    paths={'fixture/'+p.relative_to(folder).as_posix():p for p in folder.rglob('*') if p.is_file()}
    frozen,archive,records=archive_verified(paths,archive_dir)
    write(archive_dir/'JSON_TRANSFER_MANIFEST.json',dict(format='pinn4soh-backup-json-base64-v1',archive_type='tar.gz',
        verified_without_errors=True,original_archives=1,wrapper_files=len(records),records=records))
    restore(archive_dir,base/'restored',settle_seconds=0)
    assert sha(base/'restored'/archive.name)==sha(archive)
    with tarfile.open(base/'restored'/archive.name,'r:gz') as tar:
        assert {m.name for m in tar.getmembers()}==set(frozen)
        assert all(hashlib.sha256(tar.extractfile(n).read()).hexdigest()==r['sha256'] for n,r in frozen.items())
    print('PASS complete-seed paired statistics and portable strict archive restore',flush=True)
    real={}
    for dataset in ('mit','hust'):
        prior=root/'outputs/full_reproduction/stage_a_validation_20261111'/(dataset+'_real_smoke')
        old=read(prior/'plan.json');selected=read(prior/'selection.json')['selected']
        origin=dict(stage_a_selection_path=str(prior/'selection.json'),stage_a_selection_sha256=sha(prior/'selection.json'),
            stage_a_plan_path=str(prior/'plan.json'),stage_a_plan_sha256=identity(old))
        realplan=group_plan(old['specification'],selected,FUTURE_SEEDS[:1],root/'PINN4SOH','cpu',origin,'REAL_SMOKE_ONLY',True)
        destination=base/(dataset+'_real_smoke');prepare_group(destination,realplan);run_phase(destination,realplan,'training',False)
        real[dataset]=audit_group(destination,False,False)
        assert real[dataset]['fixed_test_evaluations']==0 and not(destination/'fixed_test').exists()
        print('PASS '+dataset+' real active-PINN two-epoch newseed train/valid only',flush=True)
    files=['run_stage_a_confirmation.py','collect_stage_a_confirmation.py','test_stage_a_confirmation.py']
    write(base/'ACCEPTANCE.json',dict(status='passed_functional_and_real_smoke',formal_confirmation_started=False,
        real_test_loaded=False,synthetic_audit=audit,real_smoke=real,archive_members=len(frozen),
        checked=['unchanged StageA worker','two fixed new seeds no re-selection','same-seed checkpoint','immutable resume',
            'cross-seed checkpoint rejected','seed edits rejected','smoke cannot pass formal scope','all ten-seed paired t95 statistics',
            'archive member SHA and strict restore'],
        artifact_sha256={'reproduction/'+n:sha(root/'reproduction'/n) for n in files}))
    print('ALL_STAGE_A_CONFIRMATION_TESTS_PASSED',flush=True)


if __name__=='__main__':main()
