"""Independent array, expected-worker, selection and paired-transfer audit."""
import argparse
import csv
import math
from pathlib import Path
import statistics
from functools import lru_cache

from search_replacement_grid import read,write,sha,identity
from collect_stage_a import archive_verified
from collect_stage_a_confirmation import physical_metrics,metric_record_matches

LANES=('HUST-to-MIT','XJTU-to-MIT','target-only','HUST-to-MIT-original','XJTU-to-MIT-original')
METRICS=('battery_macro_RMSE','battery_macro_MAE','battery_macro_MAPE_pct')
TRAIN_ARTIFACTS={'data_identity.json','valid_truth.npy','valid_prediction.npy','valid_metrics.json','best_model.pth','history.json'}
TEST_ARTIFACTS={'data_identity.json','test_truth.npy','test_prediction.npy','test_metrics.json'}


def require(value,message):
    if not value:raise ValueError(message)


def checksums(root,protocol,archive,stack,device,bundle):
    """Plan/source binding uses explicit caller paths, never worker self-report."""
    plan=read(root/'plan.json')
    require(plan['protocol']==protocol and plan['protocol_sha256']==identity_file(protocol,bundle/'PROTOCOL.json')
        and plan['runtime_stack']==stack and plan['repo']==str(Path(plan['repo']).resolve())
        and plan['archive']==str(archive) and plan['device']==device
        and plan['runner_sha256']==protocol['implementation_sha256']['run_transfer_robustness.py'], 'Transfer plan differs')
    repo=Path(plan['repo'])
    for name,checksum in protocol['implementation_sha256'].items():require(sha(bundle/name)==checksum,'Frozen implementation changed')
    source_files=sorted(repo.rglob('*.py'))+[bundle/n for n in ('run_transfer_robustness.py','search_stage_a.py',
        'search_replacement_grid.py','run_transfer.py','run_extended.py','prepare_transfer_robustness.py')]
    return plan,repo,{str(p.resolve()):sha(p) for p in source_files}


def identity_file(protocol,path):
    require(read(path)==protocol,'Protocol file content differs');return sha(path)


def expected_source(protocol,lane,stack,repeat,archive):
    if lane=='target-only':return None
    direction=lane[:-9] if lane.endswith('-original') else lane
    record=protocol['source_records'][direction][stack][str(repeat)]
    path=archive/record['relative_folder']/'model.pth'
    require(sha(path)==record['evidence_sha256']['model.pth'],'Source checkpoint changed')
    return dict(checkpoint=str(path),checkpoint_sha256=record['evidence_sha256']['model.pth'],
        historical_training_seed=record['training_seed'],source_repeat=repeat,source_record=record)


def candidate_list(protocol,lane):
    return protocol['target_only_candidates'] if lane=='target-only' else [protocol['original_loss_controls'][lane[:-9]]] if lane.endswith('-original') else protocol['transfer_candidates']


def audit_worker(folder,protocol,lane,candidate,seed,repeat,mode,stack,repo,archive,device,source_files,protocol_sha,lockpath=None,trained=None):
    import numpy as np
    worker,result=read(folder/'worker.json'),read(folder/'result.json')
    roles=('train','valid') if mode=='train' else ('test',)
    splits={role:dict(paths=[str((repo/p).resolve()) for p in protocol['target_splits'][role]['paths']],
        physical_ids=protocol['target_splits'][role]['physical_ids'],nominal_capacity=protocol['target_splits'][role]['nominal_capacity']) for role in roles}
    inputs={p:sha(p) for row in splits.values() for p in row['paths']}
    expected=dict(format='transfer-robustness-worker-v1',lane=lane,parameters=candidate['parameters'],candidate_id=candidate['candidate_id'],
        mode=mode,seed=seed,source_repeat=repeat,stack=stack,device=device,repo=str(repo),output=str(folder),protocol_sha256=protocol_sha,
        fixed=protocol['fixed'],source=expected_source(protocol,lane,stack,repeat,archive),splits=splits,
        input_sha256=inputs,source_sha256=source_files)
    if trained:expected.update(trained_checkpoint=str(trained),trained_checkpoint_sha256=sha(trained),selection_lock_sha256=sha(lockpath))
    require(worker==expected,'Expected worker lane/seed/source/split/config differs: '+str(folder))
    require(result['status']=='complete' and result['worker_identity']==identity(expected) and result['lane']==lane
        and result['seed']==seed and result['candidate_id']==candidate['candidate_id'] and result['source']==expected['source']
        and result['smoke'] is False and result['test_loaded']==(mode!='train') and result['G_frozen_unchanged'] is True
        and result['encoder_frozen']==(candidate['parameters']['freeze']=='head_only'), 'Completion/source/freeze identity differs')
    arts=TRAIN_ARTIFACTS if mode=='train' else TEST_ARTIFACTS
    require(set(result['artifact_sha256'])==arts,'Artifact set differs')
    for name,checksum in result['artifact_sha256'].items():require(sha(folder/name)==checksum,'Artifact SHA differs')
    require(result['environment']['device']=='cuda' and result['environment']['deterministic_requested'] is True,
        'Formal result must use requested deterministic GPU')
    if mode=='source_test':require(expected['source'] is not None and result['initial_solution_sha256']==result['final_solution_sha256'],
        'Source-only evaluation updated parameters or has no source')
    version=result['environment']['torch']
    require(version.startswith('1.7.1') if stack=='legacy' else not version.startswith('1.7.'),'Recorded runtime stack differs')
    if expected['source']:
        source_path=Path(expected['source']['checkpoint'])
        require(result['initial_solution_sha256']==source_state_hash(source_path),'Source initialization u differs from historical source')
        if mode=='train' or trained:
            saved=checkpoint_state(folder/'best_model.pth' if mode=='train' else trained)
            source=checkpoint_state(source_path)
            require(tensor_state_hash(saved['dynamical_F'])==tensor_state_hash(source['dynamical_F']),'Frozen G checkpoint changed')
            if candidate['parameters']['freeze']=='head_only':
                require(tensor_state_hash({k:v for k,v in saved['solution_u'].items() if k.startswith('encoder.')})==
                    tensor_state_hash({k:v for k,v in source['solution_u'].items() if k.startswith('encoder.')}),'Frozen encoder checkpoint changed')
    data=read(folder/'data_identity.json');require(set(data)==set(splits),'Loaded roles differ')
    for role,s in splits.items():
        batteries=data[role]['batteries']
        require([r['path'] for r in batteries]==s['paths'] and [r['physical_id'] for r in batteries]==s['physical_ids']
            and all(isinstance(r['n_pairs'],int) and r['n_pairs']>0 for r in batteries),'Physical boundaries differ')
    role='valid' if mode=='train' else 'test'
    metrics=physical_metrics(np.load(folder/(role+'_truth.npy'),allow_pickle=False),np.load(folder/(role+'_prediction.npy'),allow_pickle=False),data[role]['batteries'])
    require(metric_record_matches(metrics,read(folder/(role+'_metrics.json'))) and math.isclose(metrics['battery_macro_RMSE'],result['objective'],rel_tol=1e-13,abs_tol=1e-15),
        'Independent metrics differ')
    if mode=='train':
        h=read(folder/'history.json');fixed=protocol['fixed']
        require([r['epoch'] for r in h]==list(range(1,len(h)+1)) and len(h)==result['epochs_completed'] and fixed['min_epochs']<=len(h)<=fixed['epochs'], 'Epoch sequence/budget differs')
        best=min(h,key=lambda r:(r['validation_battery_macro_RMSE'],r['epoch']))
        require(best['epoch']==result['best_epoch'] and best['validation_battery_macro_RMSE']==result['objective'],'Best epoch differs')
        for r in h:
            require(r['lr']==candidate['parameters']['lr'] and r['G_training'] is False and r['gradient_finite'] is True
                and r['encoder_frozen']==(candidate['parameters']['freeze']=='head_only')
                and all(math.isfinite(v) for v in r['losses'].values())
                and all(math.isfinite(v) and v>=0 for v in r['diagnostic_weighted_gradient_l2'].values()),'Training diagnostics differ')
        require(len(h)==fixed['epochs'] or h[-1]['stale_epochs']>=fixed['patience'],'Early stop before rule')
    return dict(result_sha256=sha(folder/'result.json'),metrics=metrics,objective=result['objective'],
        tensor_identity=identity(data),artifact_checks=len(arts),initial_u=result['initial_solution_sha256'],environment=result['environment'])


@lru_cache(maxsize=256)
def checkpoint_state(path):
    """Independent hash of audited source u; permits legacy self-generated states."""
    import hashlib
    import inspect
    import torch
    kwargs=dict(map_location='cpu')
    if 'weights_only' in inspect.signature(torch.load).parameters:kwargs['weights_only']=True
    return torch.load(str(path),**kwargs)


def tensor_state_hash(state):
    import hashlib
    digest=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        array=tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode());digest.update(str(array.dtype).encode());digest.update(str(array.shape).encode());digest.update(array.tobytes())
    return digest.hexdigest()


def source_state_hash(path):return tensor_state_hash(checkpoint_state(Path(path))['solution_u'])


def audit_selection(root,protocol,archive,bundle,device):
    root,archive,bundle=[Path(p).resolve() for p in (root,archive,bundle)]
    plan,repo,source_files=checksums(root,protocol,archive,'modern',device,bundle)
    expected_results=set();selections={};allrows=[];cohort=None
    for lane in LANES:
        rows=[];ranking=[]
        for candidate in candidate_list(protocol,lane):
            scores=[]
            for repeat,seed in enumerate(protocol['selection_seeds'],1):
                folder=root/'selection'/lane/candidate['candidate_id']/('seed_%d'%seed)
                a=audit_worker(folder,protocol,lane,candidate,seed,repeat,'train','modern',repo,archive,device,source_files,plan['protocol_sha256'])
                if cohort is None:cohort=a['tensor_identity']
                require(a['tensor_identity']==cohort,'Target tensor identities differ across lanes/candidates/seeds')
                rows.append(dict(candidate_id=candidate['candidate_id'],seed=seed,objective=a['objective'],result_sha256=a['result_sha256']))
                scores.append(a['objective']);expected_results.add(folder/'result.json');allrows.append(dict(lane=lane,seed=seed,**a))
            ranking.append(dict(candidate,validation_macro_RMSE_mean=sum(scores)/len(scores),per_seed=scores))
        ranking.sort(key=lambda r:(r['validation_macro_RMSE_mean'],r['candidate_id']))
        expected=dict(selected=ranking[0],ranking=ranking,test_used_for_selection=False,rows=rows)
        path=root/'selection'/lane/'selection.json';require(read(path)==expected,'Independent selection differs')
        selections[lane]=dict(path='selection/'+lane+'/selection.json',sha256=sha(path),selected=ranking[0])
    actual=set((root/'selection').glob('*/*/seed_*/result.json'))
    require(actual==expected_results and len(actual)==87,'Formal selection scope must be87 complete workers')
    locked=read(root/'ALL_SELECTIONS_LOCKED.json')
    require(locked==dict(protocol_sha256=plan['protocol_sha256'],runner_sha256=plan['runner_sha256'],modern_only=True,test_used=False,selections=selections),
        'Global lock differs from independent ranking')
    return dict(status='passed',trainings=87,selections=selections,lock_sha256=sha(root/'ALL_SELECTIONS_LOCKED.json'),rows=allrows)


def audit_confirmation(root,selection_root,protocol,archive,bundle,stack,device,require_marker=True):
    root,selection_root,archive,bundle=[Path(p).resolve() for p in (root,selection_root,archive,bundle)]
    selected=audit_selection(selection_root,protocol,archive,bundle,device)
    plan,repo,source_files=checksums(root,protocol,archive,stack,device,bundle)
    expected_results=set();rows=[];cohort=None;result_map={}
    for lane in LANES:
        candidate=selected['selections'][lane]['selected']
        for repeat,seed in enumerate(protocol['future_confirmation_seeds'],1):
            folder=root/'confirmation'/lane/('seed_%d'%seed)
            a=audit_worker(folder,protocol,lane,candidate,seed,repeat,'train',stack,repo,archive,device,source_files,plan['protocol_sha256'])
            if cohort is None:cohort=a['tensor_identity']
            require(a['tensor_identity']==cohort,'Confirmation target tensor cohort differs')
            expected_results.add(folder/'result.json');result_map[(folder/'result.json').relative_to(root).as_posix()]=a['result_sha256']
            rows.append(dict(lane=lane,seed=seed,source_repeat=repeat,**a))
    require(set((root/'confirmation').glob('*/seed_*/result.json'))==expected_results and len(expected_results)==50,'Confirmation matrix must be50')
    if require_marker:
        require(read(root/'CONFIRMATION_TRAINING_COMPLETE.json')==dict(selection_lock_sha256=selected['lock_sha256'],protocol_sha256=plan['protocol_sha256'],stack=stack,training_count=50,results=result_map),
            'Confirmation global completion marker differs')
    return dict(status='passed',training_runs=50,stack=stack,selection_lock_sha256=selected['lock_sha256'],rows=rows)


def audit_test(root,selection_root,protocol,archive,bundle,stack,device):
    root,selection_root,archive,bundle=[Path(p).resolve() for p in (root,selection_root,archive,bundle)]
    confirmation=audit_confirmation(root,selection_root,protocol,archive,bundle,stack,device)
    selected=read(selection_root/'ALL_SELECTIONS_LOCKED.json')
    gate=read(root/'TEST_EVALUATION_GATE.json')
    require(gate['status']=='passed' and gate['selection_lock_sha256']==sha(selection_root/'ALL_SELECTIONS_LOCKED.json')
        and gate['confirmation_marker_sha256']==sha(root/'CONFIRMATION_TRAINING_COMPLETE.json')
        and gate['expected_test_evaluations']==70 and gate['confirmation_result_sha256']=={r['lane']+'/seed_%d'%r['seed']:r['result_sha256'] for r in confirmation['rows']},
        'External source-only/test gate does not bind the full confirmation cohort')
    plan,repo,source_files=checksums(root,protocol,archive,stack,device,bundle)
    rows=[];expected_results=set();truth_sha=None;test_tensor_sha=None
    for lane in LANES+('HUST-to-MIT-source-only','XJTU-to-MIT-source-only'):
        parent=lane[:-12] if lane.endswith('-source-only') else lane
        candidate=selected['selections'][parent]['selected']
        for repeat,seed in enumerate(protocol['future_confirmation_seeds'],1):
            folder=root/'test'/lane/('seed_%d'%seed);source_only=lane.endswith('-source-only')
            trained=None if source_only else root/'confirmation'/lane/('seed_%d'%seed)/'best_model.pth'
            a=audit_worker(folder,protocol,parent,candidate,seed,repeat,'source_test' if source_only else 'test',stack,repo,archive,device,
                source_files,plan['protocol_sha256'],selection_root/'ALL_SELECTIONS_LOCKED.json',trained)
            if source_only:
                source=expected_source(protocol,parent,stack,repeat,archive)
                require(a['initial_u']==source_state_hash(Path(source['checkpoint'])),'Source-only u differs from its historical checkpoint')
            ysha=sha(folder/'test_truth.npy')
            if truth_sha is None:truth_sha=ysha;test_tensor_sha=a['tensor_identity']
            require(ysha==truth_sha and a['tensor_identity']==test_tensor_sha,'Test truth/tensor cohort differs')
            rows.append(dict(lane=lane,seed=seed,source_repeat=repeat,**a));expected_results.add(folder/'result.json')
    require(set((root/'test').glob('*/seed_*/result.json'))==expected_results and len(expected_results)==70,'Test cohort must be70')
    return dict(status='passed',stack=stack,training_runs=50,test_evaluations=70,rows=rows,
        gate_sha256=sha(root/'TEST_EVALUATION_GATE.json'),confirmation=confirmation)


def write_gate(root,report,selection_root):
    root,selection_root=Path(root),Path(selection_root)
    evidence=dict(status='passed',selection_lock_sha256=sha(selection_root/'ALL_SELECTIONS_LOCKED.json'),
        confirmation_marker_sha256=sha(root/'CONFIRMATION_TRAINING_COMPLETE.json'),expected_test_evaluations=70,
        confirmation_result_sha256={r['lane']+'/seed_%d'%r['seed']:r['result_sha256'] for r in report['rows']},
        source_only_binding='External immutable gate: same locked selected config, same historical source index, no parameter update; worker has no fake trained-checkpoint field')
    path=root/'TEST_EVALUATION_GATE.json'
    require(not path.exists() or read(path)==evidence,'Test gate changed')
    if not path.exists():write(path,evidence)
    return evidence


def summarize(values):
    require(len(values)==10 and all(math.isfinite(x) for x in values),'Exactly ten finite paired seeds required')
    mean=statistics.mean(values);sd=statistics.stdev(values);se=sd/math.sqrt(10)
    return dict(n=10,mean=mean,sample_sd=sd,standard_error=se,t95=[mean-2.2621571628540993*se,mean+2.2621571628540993*se],
        uncertainty='Target training seed paired with fixed historical source repeat; fixed test batteries, not independent dataset sampling')


def comparisons(audits):
    rows=[]
    for stack,audit in audits.items():
        lookup={(r['lane'],r['seed']):r for r in audit['rows']}
        for direction in ('HUST-to-MIT','XJTU-to-MIT'):
            for comparator in (direction+'-source-only','target-only',direction+'-original'):
                for metric in METRICS:
                    delta=[lookup[(direction,s)]['metrics'][metric]-lookup[(comparator,s)]['metrics'][metric]
                        for s in range(2026111111,2026111121)]
                    rows.append(dict(stack=stack,direction=direction,comparison='improved-minus-'+comparator,metric=metric,
                        negative_mean_improved=statistics.mean(delta)<0,negative_seed_count=sum(v<0 for v in delta),per_seed=delta,**summarize(delta)))
    for direction in ('HUST-to-MIT','XJTU-to-MIT'):
        for metric in METRICS:
            lookup={s:{(r['lane'],r['seed']):r for r in a['rows']} for s,a in audits.items()}
            delta=[lookup['legacy'][(direction,s)]['metrics'][metric]-lookup['modern'][(direction,s)]['metrics'][metric] for s in range(2026111111,2026111121)]
            rows.append(dict(stack='paired-runtime',direction=direction,comparison='legacy-minus-modern',metric=metric,per_seed=delta,**summarize(delta)))
    return rows


def figures(audits,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'modern':'#2468a4','legacy':'#b34a35'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),constrained_layout=True)
    for ax,direction in zip(axes,('HUST-to-MIT','XJTU-to-MIT')):
        for stack,audit in audits.items():
            lookup={(r['lane'],r['seed']):r for r in audit['rows']}
            for lane,marker,label in [(direction,'o','improved'),(direction+'-source-only','x','source only'),('target-only','^','target only'),(direction+'-original','s','original-loss control')]:
                values=[lookup[(lane,s)]['metrics']['battery_macro_RMSE']*100 for s in range(2026111111,2026111121)]
                ax.plot(range(1,11),values,marker=marker,linewidth=.8,markersize=4,color=colors[stack],alpha=.8,
                    linestyle='-' if stack=='modern' else '--',label=stack+' '+label)
        ax.set(title=direction,xlabel='Prespecified target seed index',ylabel='Physical battery macro RMSE (percentage points)')
        ax.grid(alpha=.2);ax.set_xticks(range(1,11))
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.12),ncol=4,fontsize=7)
    for ext in ('png','pdf'):fig.savefig(output/('paired_seed_RMSE.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)


def collect(modern,legacy,protocol_path,archive,bundle,output,device='0'):
    modern,legacy,protocol_path,archive,bundle,output=[Path(x).resolve() for x in (modern,legacy,protocol_path,archive,bundle,output)]
    require(not output.exists(),'Fresh independent collection output required')
    protocol=read(protocol_path);audits={stack:audit_test(root,modern,protocol,archive,bundle,stack,device) for stack,root in [('modern',modern),('legacy',legacy)]}
    require(audits['modern']['rows'][0]['tensor_identity']==audits['legacy']['rows'][0]['tensor_identity'],'Cross-runtime target tensors differ')
    require(sha(modern/'test/HUST-to-MIT/seed_2026111111/test_truth.npy')==sha(legacy/'test/HUST-to-MIT/seed_2026111111/test_truth.npy'),
        'Cross-runtime test truth differs')
    effects=comparisons(audits);output.mkdir(parents=True)
    write(output/'TRANSFER_INDEPENDENT_AUDIT.json',dict(status='passed',selection=audit_selection(modern,protocol,archive,bundle,device),
        stacks=audits,paired_effects=effects,protocol_sha256=sha(protocol_path),formal_trainings=187,fixed_test_evaluations=140,
        source_only_external_gate_binding=True,historical_test_observed=True,limitations=protocol['limits']))
    with (output/'paired_effects.csv').open('w',encoding='utf-8-sig',newline='') as f:
        keys=['stack','direction','comparison','metric','n','mean','sample_sd','standard_error','negative_mean_improved','negative_seed_count','per_seed','t95','uncertainty']
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for row in effects:writer.writerow({k:row.get(k) for k in keys})
    figures(audits,output)
    paths={}
    for label,root in [('modern_results',modern),('legacy_results',legacy),('tools',bundle)]:
        for p in root.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':
                require(not p.is_symlink(),'Result/tool symlink rejected');paths[label+'/'+p.relative_to(root).as_posix()]=p
    for direction,stacks in protocol['source_records'].items():
        for stack,records in stacks.items():
            for repeat,record in records.items():
                for name,checksum in record['evidence_sha256'].items():
                    p=archive/record['relative_folder']/name;require(sha(p)==checksum,'Source changed during collection')
                    paths['source_evidence/'+direction+'/'+stack+'/repeat'+repeat+'/'+name]=p.resolve()
    repo=Path(read(modern/'plan.json')['repo'])
    for stacks in protocol['source_records'].values():
        for records in stacks.values():
            for record in records.values():
                for name,checksum in record['input_tensor_identity']['input_file_sha256'].items():
                    require(sha(repo/name)==checksum,'Historical source CSV changed before archive')
                    paths['repo_snapshot/'+name]=repo/name
    for p in sorted(repo.rglob('*.py')):paths['repo_snapshot/'+p.relative_to(repo).as_posix()]=p
    for role,row in protocol['target_splits'].items():
        for name in row['paths']:paths['repo_snapshot/'+name]=repo/name
    for p in output.iterdir():paths['audit/'+p.name]=p
    parent=modern.parent
    require(legacy.parent==parent,'Two runtimes must share pipeline root')
    for p in parent.iterdir():
        if p.is_file():paths['pipeline_evidence/'+p.name]=p
    for label in ('modern_smokes','legacy_smokes'):
        for p in (parent/label).rglob('*'):
            if p.is_file() and p.suffix!='.pyc' and '__pycache__' not in p.parts:paths['pipeline_evidence/'+label+'/'+p.relative_to(parent/label).as_posix()]=p
    relocation=archive/'VERIFIED_RELOCATION.json'
    if relocation.is_file():paths['pipeline_evidence/VERIFIED_RELOCATION.json']=relocation
    frozen,tar,records=archive_verified(paths,output)
    evidence=dict(status='complete',archive_bytes=tar.stat().st_size,archive_sha256=sha(tar),all_archived_member_hashes_verified=True,
        protocol_sha256=sha(protocol_path),formal_trainings=187,fixed_test_evaluations=140,
        inventory=[dict(path=n,**r) for n,r in sorted(frozen.items())])
    write(output/'TRANSFER_COLLECTION_VERIFIED.json',evidence)
    write(output/'JSON_TRANSFER_MANIFEST.json',dict(format='pinn4soh-backup-json-base64-v1',archive_type='tar.gz',verified_without_errors=True,
        original_archives=1,wrapper_files=len(records),records=records))
    return evidence


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('modern','legacy','protocol-path','archive','bundle','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',default='0')
    result=collect(**vars(parser.parse_args()));print(result['archive_sha256'])
