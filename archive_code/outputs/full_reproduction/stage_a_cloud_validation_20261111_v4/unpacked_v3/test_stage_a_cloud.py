"""Persistent synthetic collector/strict archive-restore QA, never formal training."""
import copy
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

from collect_stage_a import audit_suite, archive_verified
from restore_backup_json import restore
from search_replacement_grid import read, write, sha


def rejected(action):
    try:
        action()
    except ValueError:
        return
    raise AssertionError('Invalid fixture accepted')


def main():
    import numpy as np
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=root/'outputs/full_reproduction/stage_a_cloud_validation_20261111')
    directory=parser.parse_args().output.resolve()
    directory.mkdir(exist_ok=False)
    paths=[]
    for battery in range(4):
        path=directory/('battery%d.csv'%battery)
        cycle=np.arange(41+battery*3)
        features=np.stack([np.sin((cycle+battery+1)*(i+1)/50)+cycle/100 for i in range(16)],axis=1)
        np.savetxt(path,np.column_stack([features,2*(1-cycle*.001)]),delimiter=',',comments='',
            header=','.join(['feature%d'%i for i in range(16)]+['capacity']))
        paths.append(str(path))
    spec=dict(label='STAGE_A_COLLECTOR_SYNTHETIC_ONLY',dataset='MIT',model='MLP',smoke=True,seeds=[2026111101],
        splits=dict(train=dict(paths=paths[:1],physical_ids=['train'],nominal_capacity=2),
            valid=dict(paths=paths[1:3],physical_ids=['valid1','valid2'],nominal_capacity=2),
            test=dict(paths=paths[3:],physical_ids=['test'],nominal_capacity=2)),
        fixed=dict(epochs=2,min_epochs=2,patience=1,warmup_epochs=1,batch_size=64,threads=1,
            normalization_method='min-max',diagnostic_batch_size=8,grad_clip_norm=0,warmup_ratio=.1,
            final_lr_ratio=.01,deterministic=True,allow_tf32=False),
        structure=dict(u_layers=3,u_hidden=60,F_layers_num=3,F_hidden_dim=60),
        candidates=[dict(lr_u=.001,lr_F=0,alpha=0,beta_mean=0,weight_decay=0)],
        protocol_evidence=['Synthetic functional QA only; no formal battery test is opened'])
    spec_path=directory/'fixture_spec.json';write(spec_path,spec)
    suite=directory/'fixture_suite'
    command=[sys.executable,str(root/'reproduction/search_stage_a.py'),'--spec',str(spec_path),
        '--repo',str(root/'PINN4SOH'),'--output',str(suite),'--device','cpu','--execute','--allow-cpu','--evaluate-test']
    p=subprocess.run(command,text=True,capture_output=True)
    (directory/'synthetic_worker.log').write_text(p.stdout+'\n'+p.stderr,encoding='utf-8')
    if p.returncode:
        raise RuntimeError(p.stderr)
    audit=audit_suite(suite,True)
    print('PASS independent fixture candidate/selection/test audit',flush=True)
    worker=suite/'selected_test/seed_2026111101/worker.json'
    original=worker.read_bytes();bad=read(worker);bad['seed']+=1;write(worker,bad)
    try:rejected(lambda:audit_suite(suite,True))
    finally:worker.write_bytes(original)
    checkpoint=suite/'candidate_01/seed_2026111101/best_model.pth'
    original=checkpoint.read_bytes();checkpoint.write_bytes(original+b'corruption')
    try:rejected(lambda:audit_suite(suite,True))
    finally:checkpoint.write_bytes(original)
    selection=suite/'selection.json'
    original=selection.read_bytes();bad=read(selection);bad['plan_sha256']='0'*64;write(selection,bad)
    try:rejected(lambda:audit_suite(suite,True))
    finally:selection.write_bytes(original)
    print('PASS seed/checkpoint/selection tampering rejected',flush=True)
    archive_output=directory/'archive_fixture';archive_output.mkdir()
    inputs={'fixture/'+p.relative_to(suite).as_posix():p for p in suite.rglob('*') if p.is_file()}
    frozen,archive,records=archive_verified(inputs,archive_output)
    write(archive_output/'JSON_TRANSFER_MANIFEST.json',dict(format='pinn4soh-backup-json-base64-v1',archive_type='tar.gz',
        verified_without_errors=True,original_archives=1,wrapper_files=len(records),records=records))
    restored=directory/'restored'
    result=restore(archive_output,restored,settle_seconds=0)
    assert sha(restored/archive.name)==sha(archive)
    with tarfile.open(restored/archive.name,'r:gz') as reader:
        assert {m.name for m in reader.getmembers()}==set(frozen)
        for name,row in frozen.items():
            assert hashlib.sha256(reader.extractfile(name).read()).hexdigest()==row['sha256'],name
    print('PASS streamed archive, every member SHA, JSON wrapper and strict restore',flush=True)
    real={}
    for name in ('mit','hust'):
        real[name]=audit_suite(root/'outputs/full_reproduction/stage_a_validation_20261111'/(name+'_real_smoke'),False)
        assert real[name]['test_evaluations']==0 and real[name]['selected_test_seed_set_and_same_seed_checkpoint_verified'] is None
    evidence=dict(status='passed_collector_synthetic_and_real_smoke',synthetic_audit=audit,real_smoke_audit=real,
        rejection_tests=['wrong_test_seed','changed_checkpoint','changed_selection_plan'],archive_members=len(frozen),
        archive_sha256=sha(archive),strict_restore=result,formal_training_started=False,real_test_loaded=False,
        artifact_sha256={str(p.relative_to(root)):sha(p) for p in (root/'reproduction/collect_stage_a.py',root/'reproduction/test_stage_a_cloud.py')})
    write(directory/'COLLECTOR_ACCEPTANCE.json',evidence)
    print('ALL_STAGE_A_CLOUD_COLLECTOR_TESTS_PASSED',flush=True)


if __name__=='__main__':main()
