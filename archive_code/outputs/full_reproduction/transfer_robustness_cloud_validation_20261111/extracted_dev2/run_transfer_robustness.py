"""Single-worker transfer improvement; selection, locked confirmation, then test."""
import argparse
import copy
import hashlib
import importlib
import inspect
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from search_replacement_grid import read, write, sha, identity
from search_stage_a import battery_metrics, mono_eval_with_grad, require, grad_l2
from run_transfer import state_hash
from run_extended import task_lock
from prepare_transfer_robustness import long_archive


def load_checkpoint(path, device):
    import torch
    # Our own SHA-bound, previously audited source/result files only. Torch1.7
    # lacks weights_only; modern stacks use the restricted loader.
    kwargs=dict(map_location=device)
    if 'weights_only' in inspect.signature(torch.load).parameters:kwargs['weights_only']=True
    return torch.load(str(path),**kwargs)


def verify_protocol(protocol, repo, archive):
    require(protocol['format']=='pinn4soh-transfer-robustness-prereg-v1'
            and protocol['status']=='frozen_before_new_target_training' and protocol['historical_test_observed'], 'Protocol identity differs')
    require(protocol['selection_seeds']==[2026111101,2026111102,2026111103]
            and protocol['future_confirmation_seeds']==list(range(2026111111,2026111121)), 'Registered seed sets differ')
    for name,checksum in protocol.get('implementation_sha256',{}).items():
        require(sha(Path(__file__).with_name(name))==checksum,'Registered implementation changed: '+name)
    seen_paths,seen_uids=set(),{}
    for role, record in protocol['target_splits'].items():
        require(role in {'train','valid','test'} and len(record['paths'])==len(record['physical_ids']), 'Invalid target split')
        for path,uid in zip(record['paths'],record['physical_ids']):
            source=protocol['target_identity'][path]
            require(path not in seen_paths and uid not in seen_uids, 'Target physical identity/path overlap')
            require(source['physical_ids']==[uid] and sha(repo/path)==source['csv_sha256'], 'Target bytes/physical identity changed')
            seen_paths.add(path);seen_uids[uid]=role
    require(len(protocol['target_splits']['train']['paths'])==1 and len(protocol['target_splits']['valid']['paths'])==20
            and len(protocol['target_splits']['test']['paths'])==22, 'Registered target size differs')
    for direction,stacks in protocol['source_records'].items():
        require(direction in {'HUST-to-MIT','XJTU-to-MIT'}, 'Unknown direction')
        for stack,records in stacks.items():
            require(stack in {'modern','legacy'} and set(records)==set(map(str,range(1,11))), 'Source cohort differs')
            for source in records.values():
                for name,checksum in source['evidence_sha256'].items():
                    require(sha(archive/source['relative_folder']/name)==checksum, 'Source evidence changed: '+name)
                inventory=read(archive/source['relative_folder']/'source_inventory.json')
                for name,checksum in inventory['sha256'].items():
                    if name.endswith('.py'):require(sha(repo/name)==checksum,'Author source differs from historical pretraining')
    return True


def source_for(protocol, lane, stack, repeat, archive):
    if lane=='target-only':return None
    direction=lane.removesuffix('-original') if hasattr(str,'removesuffix') else (lane[:-9] if lane.endswith('-original') else lane)
    record=protocol['source_records'][direction][stack][str(repeat)]
    folder=archive/record['relative_folder']
    return dict(checkpoint=str(folder/'model.pth'), checkpoint_sha256=record['evidence_sha256']['model.pth'],
        historical_training_seed=record['training_seed'],source_repeat=repeat,source_record=record)


def make_worker(protocol, lane, candidate, seed, repeat, mode, stack, repo, archive, output, protocol_sha, device):
    roles=('train','valid') if mode=='train' else ('valid',) if mode=='source_valid' else ('test',)
    splits={r:copy.deepcopy(protocol['target_splits'][r]) for r in roles}
    for record in splits.values():record['paths']=[str((repo/p).resolve()) for p in record['paths']]
    files=sorted(repo.rglob('*.py'))+[Path(__file__).resolve(),Path(__file__).with_name('search_stage_a.py'),
        Path(__file__).with_name('search_replacement_grid.py'),Path(__file__).with_name('run_transfer.py'),Path(__file__).with_name('run_extended.py'),
        Path(__file__).with_name('prepare_transfer_robustness.py')]
    return dict(format='transfer-robustness-worker-v1',lane=lane,parameters=candidate['parameters'],candidate_id=candidate['candidate_id'],
        mode=mode,seed=seed,source_repeat=repeat,stack=stack,device=device,repo=str(repo),output=str(output),protocol_sha256=protocol_sha,
        fixed=protocol['fixed'],source=source_for(protocol,lane,stack,repeat,archive),splits=splits,
        input_sha256={p:sha(p) for r in splits.values() for p in r['paths']}, source_sha256={str(p):sha(p) for p in files})


def configure_modes(model, freeze):
    model.eval()  # G always remains eval; frozen encoder dropout stays eval.
    model.solution_u.train()
    if freeze=='head_only':model.solution_u.encoder.eval()


def compute_losses(model, batch, params):
    import torch
    x1,x2,y1,y2=batch
    if params['alpha']:
        u1,f1=model(x1);u2,f2=model(x2)
        pde=.5*(f1.square().mean()+f2.square().mean())
    else:
        u1,u2=model.solution_u(x1),model.solution_u(x2)
        pde=torch.zeros((),device=u1.device)
    data=.5*((u1-y1).square().mean()+(u2-y2).square().mean())
    if params['mono']=='sum_train':
        mono=torch.relu(u2-u1).sum()
    else:
        # Restore each submodule's mode, including a frozen encoder. The shared
        # context restores the root flag; individual modes are restored below.
        modes={m:m.training for m in model.solution_u.modules()}
        with mono_eval_with_grad(model.solution_u):
            v1,v2=model.solution_u(x1),model.solution_u(x2)
            mono=torch.relu(v2-v1).mean()
        for module,training in modes.items():module.training=training
    return dict(data=data,pde=pde,mono=mono,total=data+params['alpha']*pde+params['beta']*mono)


def worker_main(path):
    import numpy as np
    import torch
    from types import SimpleNamespace
    from torch.utils.data import TensorDataset,DataLoader
    w=read(path);output=Path(w['output']);fixed=w['fixed'];params=w['parameters']
    version=torch.__version__.split('+')[0].split('.')
    actual_stack='legacy' if (int(version[0]),int(version[1]))<(1,8) else 'modern'
    require(actual_stack==w['stack'],'Actual runtime does not match requested stack')
    require(Path(path).resolve().parent==output.resolve(), 'Worker path mismatch')
    for name,checksum in {**w['input_sha256'],**w['source_sha256']}.items():require(sha(name)==checksum,'Worker source/input changed')
    if w['source']:require(sha(w['source']['checkpoint'])==w['source']['checkpoint_sha256'],'Source checkpoint changed')
    require(w['device']=='cpu' or torch.cuda.is_available(),'GPU unavailable')
    use_cuda=w['device']!='cpu'
    device=torch.device('cuda' if use_cuda else 'cpu')
    torch.set_num_threads(fixed['threads'])
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=fixed['deterministic']
    for backend in (torch.backends.cudnn,torch.backends.cuda.matmul):
        if hasattr(backend,'allow_tf32'):backend.allow_tf32=fixed['allow_tf32']
    if hasattr(torch,'use_deterministic_algorithms'):torch.use_deterministic_algorithms(fixed['deterministic'])
    elif hasattr(torch,'set_deterministic'):torch.set_deterministic(fixed['deterministic'])
    else:require(not fixed['deterministic'],'No deterministic API in runtime')
    random.seed(w['seed']);np.random.seed(w['seed']);torch.manual_seed(w['seed'])
    if use_cuda:torch.cuda.manual_seed_all(w['seed'])
    sys.path.insert(0,w['repo'])
    module=importlib.import_module('Model.Model')
    from dataloader.dataloader import DF
    args=SimpleNamespace(epochs=fixed['epochs'],warmup_epochs=0,warmup_lr=params['lr'],lr=params['lr'],
        final_lr=params['lr'],lr_F=0.,F_layers_num=3,F_hidden_dim=60,alpha=params['alpha'],beta=params['beta'],
        save_folder=str(output),log_dir='author_args.log')
    model=module.PINN(args).to(device)
    if w['source']:
        checkpoint=load_checkpoint(w['source']['checkpoint'],device)
        model.solution_u.load_state_dict(checkpoint['solution_u'],strict=True)
        model.dynamical_F.load_state_dict(checkpoint['dynamical_F'],strict=True)
    for p in model.dynamical_F.parameters():p.requires_grad_(False)
    for p in model.solution_u.encoder.parameters():p.requires_grad_(params['freeze']!='head_only')
    active=[p for p in model.solution_u.parameters() if p.requires_grad]
    before_g=state_hash(model.dynamical_F.state_dict());before_encoder=state_hash(model.solution_u.encoder.state_dict())
    before_u=state_hash(model.solution_u.state_dict())
    torch.manual_seed((w['seed']+101)%(2**32))
    if use_cuda:torch.cuda.manual_seed_all((w['seed']+101)%(2**32))
    data=DF(SimpleNamespace(normalization_method=fixed['normalization_method']))
    loaders,identities={},{}
    for role,record in w['splits'].items():
        pieces,batteries=[],[]
        for name,uid in zip(record['paths'],record['physical_ids']):
            (x1,y1),(x2,y2)=data.load_one_battery(name,record['nominal_capacity'])
            values=[torch.tensor(v,dtype=torch.float32) for v in (x1,x2,y1[:,None],y2[:,None])]
            require(len(x1)>0 and values[0].shape[1]==17 and all(torch.isfinite(v).all().item() for v in values),'Invalid target tensor')
            pieces.append(values);batteries.append(dict(path=name,physical_id=uid,n_pairs=len(x1)))
        tensors=[torch.cat([v[i] for v in pieces]) for i in range(4)]
        loaders[role]=DataLoader(TensorDataset(*tensors),batch_size=fixed['batch_size'],shuffle=role=='train',num_workers=0,
            generator=torch.Generator().manual_seed((w['seed']+102)%(2**32)))
        identities[role]=dict(batteries=batteries,sha256=[hashlib.sha256(t.numpy().tobytes()).hexdigest() for t in tensors])
    write(output/'data_identity.json',identities)
    def predict(role):
        model.eval();truth,prediction=[],[]
        with torch.no_grad():
            for x1,_,y1,_ in loaders[role]:
                truth.append(y1.numpy());prediction.append(model.solution_u(x1.to(device)).cpu().numpy())
        return np.concatenate(truth),np.concatenate(prediction)
    started=time.perf_counter();history=[];best=float('inf');best_epoch=0;best_prediction=None
    if w['mode']=='train':
        require(set(loaders)=={'train','valid'},'Training must not load test')
        optimizer=torch.optim.Adam(active,lr=params['lr'],weight_decay=params['weight_decay'])
        diagnostic=tuple(t[:min(16,len(t))].clone().to(device) for t in loaders['train'].dataset.tensors)
        stale=0
        for epoch in range(1,fixed['epochs']+1):
            configure_modes(model,params['freeze']);meters={k:0. for k in ('data','pde','mono','total')};seen=0
            with torch.random.fork_rng(devices=[torch.cuda.current_device()] if use_cuda else []):
                torch.manual_seed((w['seed']+313)%(2**32))
                d=compute_losses(model,[t.clone() for t in diagnostic],params)
                gradients=dict(data=grad_l2(d['data'],active),pde=grad_l2(d['pde'],active,params['alpha']),
                    mono=grad_l2(d['mono'],active,params['beta']))
                require(all(math.isfinite(x) for x in gradients.values()),'Nonfinite fixed diagnostics')
            for batch in loaders['train']:
                values=compute_losses(model,[v.to(device) for v in batch],params)
                require(all(torch.isfinite(v).all().item() for v in values.values()),'Nonfinite loss')
                optimizer.zero_grad();values['total'].backward()
                require(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in active),'Nonfinite gradient')
                optimizer.step();n=len(batch[0]);seen+=n
                for k,v in values.items():meters[k]+=float(v.detach().cpu())*n
            truth,pred=predict('valid');metrics=battery_metrics(truth,pred,identities['valid']['batteries'])
            score=metrics['battery_macro_RMSE'];stale+=1
            if score<best:
                best,best_epoch,stale=score,epoch,0;best_prediction=pred.copy()
                torch.save(dict(solution_u=copy.deepcopy(model.solution_u.state_dict()),
                    dynamical_F=copy.deepcopy(model.dynamical_F.state_dict())),output/'best_model.pth')
            history.append(dict(epoch=epoch,validation_battery_macro_RMSE=score,stale_epochs=stale,
                losses={k:v/seen for k,v in meters.items()},lr=params['lr'],G_training=False,
                encoder_frozen=params['freeze']=='head_only',gradient_finite=True,
                diagnostic_weighted_gradient_l2=gradients,diagnostic_n=len(diagnostic[0]),
                diagnostic_unweighted_losses={k:float(d[k].detach().cpu()) for k in ('data','pde','mono')}))
            write(output/'history.json',history)
            if epoch>=fixed['min_epochs'] and stale>=fixed['patience']:break
        checkpoint=load_checkpoint(output/'best_model.pth',device)
        model.solution_u.load_state_dict(checkpoint['solution_u']);model.dynamical_F.load_state_dict(checkpoint['dynamical_F'])
        truth,pred=predict('valid');require(np.array_equal(pred,best_prediction),'Best checkpoint reload differs')
        role='valid'
    else:
        role='valid' if w['mode']=='source_valid' else 'test'
        require(set(loaders)=={role},'Evaluation role mismatch')
        if w.get('trained_checkpoint'):
            require(sha(w['trained_checkpoint'])==w['trained_checkpoint_sha256'],'Confirmation checkpoint changed')
            c=load_checkpoint(w['trained_checkpoint'],device)
            model.solution_u.load_state_dict(c['solution_u']);model.dynamical_F.load_state_dict(c['dynamical_F'])
        truth,pred=predict(role)
    require(before_g==state_hash(model.dynamical_F.state_dict()),'Frozen G changed')
    if params['freeze']=='head_only':require(before_encoder==state_hash(model.solution_u.encoder.state_dict()),'Frozen encoder changed')
    metrics=battery_metrics(truth,pred,identities[role]['batteries'])
    np.save(output/(role+'_truth.npy'),truth);np.save(output/(role+'_prediction.npy'),pred);write(output/(role+'_metrics.json'),metrics)
    names=['data_identity.json',role+'_truth.npy',role+'_prediction.npy',role+'_metrics.json']
    if w['mode']=='train':names+=['best_model.pth','history.json']
    import pandas
    write(output/'result.json',dict(status='complete',worker_identity=identity(w),lane=w['lane'],seed=w['seed'],
        candidate_id=w['candidate_id'],objective=metrics['battery_macro_RMSE'],best_epoch=best_epoch,
        epochs_completed=len(history),test_loaded=role=='test',G_frozen_unchanged=True,
        encoder_frozen=params['freeze']=='head_only',initial_solution_sha256=before_u,
        final_solution_sha256=state_hash(model.solution_u.state_dict()),source=w['source'],smoke=w.get('smoke',False),
        seconds=time.perf_counter()-started,environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,pandas=pandas.__version__,
        cuda=torch.version.cuda,device=str(device),gpu=torch.cuda.get_device_name(0) if use_cuda else None,
        deterministic_requested=fixed['deterministic'],cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG')),
        artifact_sha256={n:sha(output/n) for n in names}))


def check_completed(folder,worker):
    import numpy as np
    r=read(folder/'result.json');require(r['status']=='complete' and r['worker_identity']==identity(worker),'Completed identity changed')
    for name,checksum in r['artifact_sha256'].items():require(sha(folder/name)==checksum,'Completed artifact changed')
    role='valid' if worker['mode'] in {'train','source_valid'} else 'test'
    metrics=battery_metrics(np.load(folder/(role+'_truth.npy'),allow_pickle=False),np.load(folder/(role+'_prediction.npy'),allow_pickle=False),read(folder/'data_identity.json')[role]['batteries'])
    require(metrics['battery_macro_RMSE']==r['objective'],'Completed objective disagrees with arrays')
    return r


def launch(folder,worker,options):
    if (folder/'result.json').exists():
        require(options.resume,'Existing result needs explicit resume');return check_completed(folder,worker)
    if folder.exists() and any(folder.iterdir()):
        require(options.resume,'Incomplete folder needs explicit resume')
        folder.rename(folder.with_name(folder.name+'.incomplete.'+str(time.time_ns())))
    folder.mkdir(parents=True,exist_ok=True);write(folder/'worker.json',worker)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if options.device=='cpu' else options.device,
        OMP_NUM_THREADS=str(worker['fixed']['threads']),MKL_NUM_THREADS=str(worker['fixed']['threads']),CUBLAS_WORKSPACE_CONFIG=':4096:8',
        PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    with (folder/'console.log').open('w',encoding='utf-8') as stream:
        p=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker-json',str(folder/'worker.json')],env=env,stdout=stream,stderr=subprocess.STDOUT)
    require(p.returncode==0,'Transfer worker failed; inspect '+str(folder/'console.log'))
    return check_completed(folder,worker)


def candidates(protocol,lane):
    if lane=='target-only':return protocol['target_only_candidates']
    if lane.endswith('-original'):return [protocol['original_loss_controls'][lane[:-9]]]
    return protocol['transfer_candidates']


def freeze_selection(rows,candidate_list,seeds):
    require({(r['candidate_id'],r['seed']) for r in rows}=={(c['candidate_id'],s) for c in candidate_list for s in seeds}
            and len(rows)==len(candidate_list)*len(seeds),'Complete candidate/seed matrix required')
    ranked=[]
    for c in candidate_list:
        scores=[r['objective'] for r in rows if r['candidate_id']==c['candidate_id']]
        require(all(math.isfinite(x) for x in scores),'Invalid validation scores')
        ranked.append(dict(c,validation_macro_RMSE_mean=sum(scores)/len(scores),per_seed=scores))
    ranked.sort(key=lambda x:(x['validation_macro_RMSE_mean'],x['candidate_id']))
    return dict(selected=ranked[0],ranking=ranked,test_used_for_selection=False)


def run(o):
    protocol=read(o.protocol);repo=o.repo.resolve();archive=long_archive(o.archive_root);output=o.output.resolve()
    require(o.device=='cpu' or o.device.isdigit(),'Only one GPU index allowed')
    require(not o.execute or o.device!='cpu' or o.allow_cpu,'CPU execute requires allow-cpu')
    require(output!=repo and repo not in output.parents,'Output outside author repo required')
    verify_protocol(protocol,repo,archive)
    plan=dict(protocol_sha256=sha(o.protocol),runtime_stack=o.stack,repo=str(repo),archive=str(archive),device=o.device,
        runner_sha256=sha(__file__),protocol=protocol)
    if output.exists():require(read(output/'plan.json')==plan,'Existing plan differs')
    else:output.mkdir(parents=True);write(output/'plan.json',plan)
    if not o.execute:
        print('TRANSFER_DRY_RUN '+identity(plan));return
    require(o.phase!='selection' or o.stack=='modern','Legacy may confirm fixed modern selection, never retune')
    lanes=['HUST-to-MIT','XJTU-to-MIT','target-only','HUST-to-MIT-original','XJTU-to-MIT-original']
    lockpath=o.selection_root.resolve()/'ALL_SELECTIONS_LOCKED.json' if o.selection_root else output/'ALL_SELECTIONS_LOCKED.json'
    with task_lock(output/'.transfer-robustness.lock'):
        state=dict(phase=o.phase,stack=o.stack,status='running',completed=0,protocol_sha256=sha(o.protocol))
        write(output/'status.json',state)
        def job(lane,candidate,seed,repeat,mode,folder,trained=None):
            if o.pause_file and o.pause_file.exists():
                state['status']='paused_after_completed_worker';write(output/'status.json',state);return None
            worker=make_worker(protocol,lane,candidate,seed,repeat,mode,o.stack,repo,archive,folder,sha(o.protocol),o.device)
            if trained:
                worker.update(trained_checkpoint=str(trained),trained_checkpoint_sha256=sha(trained),selection_lock_sha256=sha(lockpath))
            result=launch(folder,worker,o);state['completed']+=1;write(output/'status.json',state);return result
        if o.phase=='selection':
            for lane in lanes:
                rows=[]
                for c in candidates(protocol,lane):
                    for repeat,seed in enumerate(protocol['selection_seeds'],1):
                        folder=output/'selection'/lane/c['candidate_id']/('seed_%d'%seed)
                        r=job(lane,c,seed,repeat,'train',folder)
                        if r is None:return
                        rows.append(dict(candidate_id=c['candidate_id'],seed=seed,objective=r['objective'],result_sha256=sha(folder/'result.json')))
                selection=freeze_selection(rows,candidates(protocol,lane),protocol['selection_seeds']);selection['rows']=rows
                path=output/'selection'/lane/'selection.json'
                if path.exists():require(read(path)==selection,'Frozen selection changed')
                else:write(path,selection)
            locked=dict(protocol_sha256=sha(o.protocol),runner_sha256=sha(__file__),modern_only=True,test_used=False,
                selections={lane:dict(path='selection/'+lane+'/selection.json',sha256=sha(output/'selection'/lane/'selection.json'),
                    selected=read(output/'selection'/lane/'selection.json')['selected']) for lane in lanes})
            if lockpath.exists():require(read(lockpath)==locked,'Global selection lock changed')
            else:write(lockpath,locked)
        else:
            require(o.selection_root is not None and lockpath.is_file(),'Confirmation/test requires global frozen modern selection')
            locked=read(lockpath);require(locked['protocol_sha256']==sha(o.protocol) and locked['test_used'] is False,'Selection protocol differs')
            require(locked['runner_sha256']==sha(__file__),'Confirmation runner differs from locked selection')
            for lane,row in locked['selections'].items():require(sha(o.selection_root/row['path'])==row['sha256'],'Selection SHA changed')
            if o.phase=='test':
                confirm=read(output/'CONFIRMATION_TRAINING_COMPLETE.json')
                require(confirm['selection_lock_sha256']==sha(lockpath) and confirm['training_count']==50
                        and len(confirm['results'])==50,'All confirmation training must complete before test')
                # Verify the whole confirmation matrix before loading the first test,
                # so a missing later lane cannot leak an earlier test evaluation.
                for lane in lanes:
                    for seed in protocol['future_confirmation_seeds']:
                        folder=output/'confirmation'/lane/('seed_%d'%seed)
                        relative=(folder/'result.json').relative_to(output).as_posix()
                        require(relative in confirm['results'] and sha(folder/'result.json')==confirm['results'][relative],
                                'Confirmation result changed before test')
                        check_completed(folder,read(folder/'worker.json'))
            for lane in lanes:
                c=locked['selections'][lane]['selected']
                for repeat,seed in enumerate(protocol['future_confirmation_seeds'],1):
                    training=output/'confirmation'/lane/('seed_%d'%seed)
                    if o.phase=='confirmation':
                        r=job(lane,c,seed,repeat,'train',training)
                    else:
                        train_worker=read(training/'worker.json');check_completed(training,train_worker)
                        r=job(lane,c,seed,repeat,'test',output/'test'/lane/('seed_%d'%seed),training/'best_model.pth')
                    if r is None:return
            if o.phase=='confirmation':
                complete=dict(selection_lock_sha256=sha(lockpath),protocol_sha256=sha(o.protocol),stack=o.stack,training_count=50,
                    results={p.relative_to(output).as_posix():sha(p) for p in (output/'confirmation').glob('*/seed_*/result.json')})
                write(output/'CONFIRMATION_TRAINING_COMPLETE.json',complete)
            else:
                for lane in ('HUST-to-MIT','XJTU-to-MIT'):
                    c=locked['selections'][lane]['selected']
                    for repeat,seed in enumerate(protocol['future_confirmation_seeds'],1):
                        if job(lane,c,seed,repeat,'source_test',output/'test'/(lane+'-source-only')/('seed_%d'%seed)) is None:return
        state.update(status='complete');write(output/'status.json',state)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path);p.add_argument('--repo',type=Path);p.add_argument('--archive-root',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--phase',choices=['selection','confirmation','test'],default='selection');p.add_argument('--stack',choices=['modern','legacy'],default='modern')
    p.add_argument('--selection-root',type=Path);p.add_argument('--device',default='0');p.add_argument('--execute',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--allow-cpu',action='store_true');p.add_argument('--pause-file',type=Path)
    p.add_argument('--worker-json',type=Path)
    o=p.parse_args()
    if o.worker_json:worker_main(o.worker_json)
    else:
        require(all(getattr(o,k) is not None for k in ('protocol','repo','archive_root','output')),'Explicit paths required');run(o)


if __name__=='__main__':main()
