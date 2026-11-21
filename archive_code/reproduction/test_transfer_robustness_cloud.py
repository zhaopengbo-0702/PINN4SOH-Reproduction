"""Synthetic tests for independent phase gates, tamper rejection and portable bytes.

Fabricated fixture results exercise audit logic only, never scientific accuracy.
"""
import copy
import hashlib
from pathlib import Path
import tempfile
import os
from unittest.mock import patch

from search_replacement_grid import read,write,sha,identity
from collect_transfer_robustness import audit_selection,audit_confirmation,audit_test,write_gate,physical_metrics,figures,comparisons
from run_transfer_robustness import make_worker
from run_transfer_robustness_cloud import relocate_sources,wait_prior,SUITES
from collect_transfer_robustness import source_state_hash


def fixture_worker(folder,worker,truth,pred,epochs=30):
    import numpy as np
    import torch
    folder.mkdir(parents=True);write(folder/'worker.json',worker)
    roles=worker['splits'];role='valid' if worker['mode']=='train' else 'test';data={}
    for name,record in roles.items():
        data[name]=dict(batteries=[dict(path=p,physical_id=u,n_pairs=1) for p,u in zip(record['paths'],record['physical_ids'])],sha256=['fixture']*4)
    n=len(data[role]['batteries']);y=np.ones((n,1))*truth;predictions=np.ones((n,1))*pred
    write(folder/'data_identity.json',data);np.save(folder/(role+'_truth.npy'),y);np.save(folder/(role+'_prediction.npy'),predictions)
    metrics=physical_metrics(y,predictions,data[role]['batteries']);write(folder/(role+'_metrics.json'),metrics)
    names=['data_identity.json',role+'_truth.npy',role+'_prediction.npy',role+'_metrics.json']
    if worker['mode']=='train':
        state=torch.load(worker['source']['checkpoint'],weights_only=True) if worker['source'] else dict(solution_u={'fixture':torch.tensor([1.])},dynamical_F={'fixture':torch.tensor([2.])})
        torch.save(state,folder/'best_model.pth')
        history=[dict(epoch=i,validation_battery_macro_RMSE=metrics['battery_macro_RMSE'],stale_epochs=i,
            lr=worker['parameters']['lr'],G_training=False,gradient_finite=True,encoder_frozen=worker['parameters']['freeze']=='head_only',
            losses=dict(data=.1,pde=.1,mono=.1,total=.1),diagnostic_weighted_gradient_l2=dict(data=.1,pde=.1,mono=.1)) for i in range(1,epochs+1)]
        write(folder/'history.json',history);names+=['best_model.pth','history.json']
    result=dict(status='complete',worker_identity=identity(worker),lane=worker['lane'],seed=worker['seed'],candidate_id=worker['candidate_id'],source=worker['source'],smoke=False,
        test_loaded=role=='test',G_frozen_unchanged=True,encoder_frozen=worker['parameters']['freeze']=='head_only',objective=metrics['battery_macro_RMSE'],
        epochs_completed=epochs if worker['mode']=='train' else 0,best_epoch=1,initial_solution_sha256='synthetic_same_source',final_solution_sha256='synthetic_same_source',
        environment=dict(torch='1.7.1+cu110' if worker['stack']=='legacy' else '1.13.1+cu116',device='cuda',deterministic_requested=True),
        artifact_sha256={name:sha(folder/name) for name in names})
    if worker['source']:
        result['initial_solution_sha256']=result['final_solution_sha256']=source_state_hash(Path(worker['source']['checkpoint']))
    write(folder/'result.json',result);return result


def main():
    root=Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='transfer_cloud_fixture_') as temp:
        temp=Path(temp).resolve();repo=temp/'repo';archive=temp/'sources';modern=temp/'modern';legacy=temp/'legacy';bundle=temp/'bundle'
        repo.mkdir();archive.mkdir();bundle.mkdir()
        for name in ('run_transfer_robustness.py','search_stage_a.py','search_replacement_grid.py','run_transfer.py','run_extended.py','prepare_transfer_robustness.py'):
            (bundle/name).write_bytes((root/'reproduction'/name).read_bytes())
        paths=[]
        for i in range(3):p=repo/('battery%d.csv'%i);p.write_text('fixture only');paths.append(p.name)
        record=dict(relative_folder='source',dataset='HUST',batch='all',training_seed=123,best_epoch=1,environment={'torch':'fixture'},source_fingerprint='f',
            input_tensor_identity={},evidence_sha256={'model.pth':'not-set'},source_selection='same-index',historical_source_seed=True)
        import torch
        (archive/'source').mkdir();torch.save(dict(solution_u={'fixture':torch.tensor([1.])},dynamical_F={'fixture':torch.tensor([2.])}),archive/'source/model.pth');record['evidence_sha256']['model.pth']=sha(archive/'source/model.pth')
        parameter=dict(lr=.001,alpha=.1,beta=.01,freeze='all_u',mono='mean_eval',weight_decay=0.,initialization='source')
        candidates=[dict(candidate_id='candidate_%02d'%i,parameters=dict(parameter,lr=.0001*i)) for i in range(1,10)]
        sources={d:{s:{str(i):dict(copy.deepcopy(record),training_seed=i) for i in range(1,11)} for s in ('modern','legacy')} for d in ('HUST-to-MIT','XJTU-to-MIT')}
        protocol=dict(format='fixture',status='fixture',historical_test_observed=True,selection_seeds=[2026111101,2026111102,2026111103],
            future_confirmation_seeds=list(range(2026111111,2026111121)),source_records=sources,target_splits={r:dict(paths=[paths[i]],physical_ids=[r],nominal_capacity=1.1) for i,r in enumerate(('train','valid','test'))},
            fixed=dict(epochs=200,min_epochs=30,patience=30,batch_size=128,threads=2,normalization_method='min-max',deterministic=True,allow_tf32=False,grad_clip_norm=0),
            transfer_candidates=candidates,target_only_candidates=[dict(c,parameters=dict(c['parameters'],alpha=0,beta=0,initialization='random')) for c in candidates],
            original_loss_controls={d:dict(candidate_id='original_loss_control',parameters=parameter) for d in sources},
            implementation_sha256={'run_transfer_robustness.py':sha(bundle/'run_transfer_robustness.py')})
        write(bundle/'PROTOCOL.json',protocol);p_sha=sha(bundle/'PROTOCOL.json')
        for stack,dest in [('modern',modern),('legacy',legacy)]:
            dest.mkdir();write(dest/'plan.json',dict(protocol=protocol,protocol_sha256=p_sha,runtime_stack=stack,repo=str(repo),archive=str(archive),device='0',runner_sha256=protocol['implementation_sha256']['run_transfer_robustness.py']))
        lanes=['HUST-to-MIT','XJTU-to-MIT','target-only','HUST-to-MIT-original','XJTU-to-MIT-original'];selections={}
        # make_worker sources paths are computed from the executing code folder;
        # replace only source_sha tree to model the packaged script location.
        files={str(p.resolve()):sha(p) for p in [bundle/n for n in ('run_transfer_robustness.py','search_stage_a.py','search_replacement_grid.py','run_transfer.py','run_extended.py','prepare_transfer_robustness.py')]}
        for lane in lanes:
            cs=protocol['target_only_candidates'] if lane=='target-only' else [protocol['original_loss_controls'][lane[:-9]]] if lane.endswith('-original') else candidates
            rows=[];ranking=[]
            for index,c in enumerate(cs):
                scores=[]
                for repeat,seed in enumerate(protocol['selection_seeds'],1):
                    folder=modern/'selection'/lane/c['candidate_id']/('seed_%d'%seed)
                    w=make_worker(protocol,lane,c,seed,repeat,'train','modern',repo,archive,folder,p_sha,'0');w['source_sha256']=files
                    result=fixture_worker(folder,w,1,1+.01*(index+1));scores.append(result['objective'])
                    rows.append(dict(candidate_id=c['candidate_id'],seed=seed,objective=result['objective'],result_sha256=sha(folder/'result.json')))
                ranking.append(dict(c,validation_macro_RMSE_mean=sum(scores)/len(scores),per_seed=scores))
            ranking.sort(key=lambda r:(r['validation_macro_RMSE_mean'],r['candidate_id']))
            selection=dict(selected=ranking[0],ranking=ranking,test_used_for_selection=False,rows=rows);path=modern/'selection'/lane/'selection.json';write(path,selection)
            selections[lane]=dict(path='selection/'+lane+'/selection.json',sha256=sha(path),selected=ranking[0])
        locked=dict(protocol_sha256=p_sha,runner_sha256=protocol['implementation_sha256']['run_transfer_robustness.py'],modern_only=True,test_used=False,selections=selections)
        write(modern/'ALL_SELECTIONS_LOCKED.json',locked)
        report=audit_selection(modern,protocol,archive,bundle,'0');assert report['trainings']==87
        print('PASS independent87matrix, best epoch, frozen selection ranking',flush=True)
        audits={}
        for stack,dest in [('modern',modern),('legacy',legacy)]:
            result_map={}
            for lane in lanes:
                c=selections[lane]['selected']
                for repeat,seed in enumerate(protocol['future_confirmation_seeds'],1):
                    folder=dest/'confirmation'/lane/('seed_%d'%seed)
                    w=make_worker(protocol,lane,c,seed,repeat,'train',stack,repo,archive,folder,p_sha,'0');w['source_sha256']=files
                    fixture_worker(folder,w,1,1.01);result_map[(folder/'result.json').relative_to(dest).as_posix()]=sha(folder/'result.json')
            write(dest/'CONFIRMATION_TRAINING_COMPLETE.json',dict(selection_lock_sha256=sha(modern/'ALL_SELECTIONS_LOCKED.json'),protocol_sha256=p_sha,stack=stack,training_count=50,results=result_map))
            conf=audit_confirmation(dest,modern,protocol,archive,bundle,stack,'0');write_gate(dest,conf,modern)
            for lane in lanes+['HUST-to-MIT-source-only','XJTU-to-MIT-source-only']:
                parent=lane[:-12] if lane.endswith('-source-only') else lane;c=selections[parent]['selected']
                for repeat,seed in enumerate(protocol['future_confirmation_seeds'],1):
                    folder=dest/'test'/lane/('seed_%d'%seed);source_only=lane.endswith('-source-only')
                    w=make_worker(protocol,parent,c,seed,repeat,'source_test' if source_only else 'test',stack,repo,archive,folder,p_sha,'0');w['source_sha256']=files
                    if not source_only:
                        trained=dest/'confirmation'/lane/('seed_%d'%seed)/'best_model.pth'
                        w.update(trained_checkpoint=str(trained),trained_checkpoint_sha256=sha(trained),selection_lock_sha256=sha(modern/'ALL_SELECTIONS_LOCKED.json'))
                    fixture_worker(folder,w,1,1.02 if source_only else 1.01)
            audits[stack]=audit_test(dest,modern,protocol,archive,bundle,stack,'0')
            assert audits[stack]['test_evaluations']==70
            print('PASS '+stack+'50 pretest gate and70 exact same-seed checkpoint/source evaluations',flush=True)
        effects=comparisons(audits);assert len(effects)==42
        fig=temp/'figures';fig.mkdir();figures(audits,fig);assert len(list(fig.iterdir()))==2
        # Rehashing a tampered result cannot hide a changed prespecified seed.
        target=legacy/'confirmation/HUST-to-MIT/seed_2026111111/worker.json';original=target.read_bytes();changed=read(target);changed['source_repeat']=2;write(target,changed)
        try:audit_confirmation(legacy,modern,protocol,archive,bundle,'legacy','0');raise AssertionError('Wrong source repeat accepted')
        except ValueError:pass
        target.write_bytes(original)
        from collect_stage_a import archive_verified
        backup=temp/'backup';backup.mkdir();frozen,tar,chunks=archive_verified({'audit/paired.png':fig/'paired_seed_RMSE.png'},backup)
        assert len(chunks)==1 and frozen
        prior=temp/'prior.json';write(prior,dict(status='failed'))
        try:wait_prior(prior,temp/'confirmation',False);raise AssertionError('Failed predecessor accepted')
        except ValueError:pass
        absent=temp/'not_created_yet.json';sleep_calls=[]
        def create_finished_status(seconds):
            sleep_calls.append(seconds)
            write(absent,dict(status='complete',request=dict(output=str(temp/'confirmation'))))
        with patch('run_transfer_robustness_cloud.time.sleep',side_effect=create_finished_status):
            waited=wait_prior(absent,temp/'confirmation',True)
            assert waited['status']=='complete' and sleep_calls==[30]
        print('PASS missing predecessor status waits without probe then verifies completed root',flush=True)
        # Relocation never rewrites protocol paths; test source bytes through an
        # alias map. Windows may need a fake alias mkdir if symlink privilege is absent.
        workspace=temp/'workspace';workspace.mkdir();aliasroot=temp/'aliases';rproto=copy.deepcopy(protocol)
        for suite in SUITES:(workspace/suite/'source').mkdir(parents=True)
        for d,stacks in rproto['source_records'].items():
            for stack,records in stacks.items():
                suite=SUITES[0] if stack=='modern' else SUITES[1]
                for r in records.values():
                    r['relative_folder']=suite+'/source';(workspace/suite/'source/model.pth').write_bytes((archive/'source/model.pth').read_bytes())
        if os.name=='nt':
            # Create actual directory junctions with native mklink is unnecessary
            # for fixture logic: a temporary symlink attempt proves privilege.
            try:
                link=temp/'privilege_probe';link.symlink_to(workspace,target_is_directory=True);link.unlink()
                relocation=relocate_sources(rproto,workspace,aliasroot)
                assert relocation['protocol_unchanged']
                print('PASS verified Windows source alias relocation',flush=True)
            except OSError as e:
                print('SKIP native source-alias creation on Windows; Linux symlink operation awaits cloud preflight: '+str(e.winerror),flush=True)
        else:
            relocation=relocate_sources(rproto,workspace,aliasroot);assert relocation['protocol_unchanged']
        print('PASS wrong-repeat rejection, charts, verified portable tar and failed-prior no-probe gate',flush=True)
    print('ALL_TRANSFER_CLOUD_FIXTURE_TESTS_PASSED')


if __name__=='__main__':main()
