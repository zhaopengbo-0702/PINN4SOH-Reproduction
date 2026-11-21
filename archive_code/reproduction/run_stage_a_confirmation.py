"""Fixed Stage A configuration, ten preregistered new seeds; never another search.

The immutable Stage A worker executes unchanged. All six original selections are
verified before writing this plan; all 60 new trainings precede any fixed test.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from search_replacement_grid import read, write, sha, identity
from search_stage_a import build_plan, launch_worker
from collect_stage_a import audit_suite, require
from run_extended import task_lock
from verify_stage_a_registration import verify

FORMAT='pinn4soh-stage-a-fixed-newseed-confirmation-v1'
GROUPS=[d+'_'+m for d in ('mit','hust') for m in ('pinn','mlp','cnn')]


def now():return datetime.now(timezone.utc).isoformat()


def verify_bundle(bundle):
    manifest=read(bundle/'TOOLS_SHA256.json')
    for name,row in manifest['files'].items():
        path=bundle/name
        require(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Confirmation tool changed: '+name)
    original=read(bundle/'STAGE_A_V3_TOOLS_SHA256.json')
    for name in ('search_stage_a.py','search_replacement_grid.py','run_extended.py','collect_stage_a.py','verify_stage_a_registration.py'):
        require(sha(bundle/name)==original['files'][name]['sha256'],'Frozen Stage A implementation changed')
    return sha(bundle/'TOOLS_SHA256.json')


def frozen_stage_a(stage_a_root,registration,repo):
    """No confirmation work is dispatched before all original suites verify."""
    checked=verify(registration,repo)
    document=read(registration/'REGISTRATION.json')
    seeds=document['future_confirmation_seeds']
    require(len(seeds)==10 and len(set(seeds))==10 and not set(seeds)&set(document['selection_seeds']),
            'Ten distinct preregistered future seeds required')
    records={}
    for group in GROUPS:
        suite=stage_a_root/group
        plan=read(suite/'plan.json')
        spec=read(registration/(group+'_stage_a9.json'))
        require(plan['specification']==spec and Path(plan['repo']).resolve()==repo and
                plan['device'].isdigit() and plan['candidates']==[dict(candidate_id='candidate_%02d'%(i+1),parameters=p)
                    for i,p in enumerate(spec['candidates'])],'Stage A group differs from registration/repo')
        audit=audit_suite(suite,True)
        require(not audit['smoke'] and audit['candidate_runs']==27 and audit['test_evaluations']==3,
                'Original Stage A group is not complete 27+3')
        selection=read(suite/'selection.json')
        records[group]=dict(stage_a_selection_path=str(suite/'selection.json'),stage_a_selection_sha256=sha(suite/'selection.json'),
            stage_a_plan_path=str(suite/'plan.json'),stage_a_plan_sha256=identity(plan),selected=selection['selected'],
            original_stage_a_audit=audit)
    return checked,seeds,records


def group_plan(spec,selected,seeds,repo,device,source,registration_sha256,smoke=False):
    """Validate shape with the original validator; override only its 9x3 budget."""
    fixed=copy.deepcopy(spec)
    fixed.update(label='STAGE_A_FIXED_CONFIG_NEW_SEED_CONFIRMATION',seeds=seeds,
                 candidates=[copy.deepcopy(selected['parameters'])],smoke=smoke)
    probe=copy.deepcopy(fixed);probe['smoke']=True
    plan=build_plan(probe,repo)
    plan.update(format=FORMAT,specification=fixed,device=device,training_runs=len(seeds),workers=1,
        selected=copy.deepcopy(selected),registration_sha256=registration_sha256,
        stage_a_selection_path=source['stage_a_selection_path'],stage_a_selection_sha256=source['stage_a_selection_sha256'],
        stage_a_plan_path=source['stage_a_plan_path'],stage_a_plan_sha256=source['stage_a_plan_sha256'],
        hyperparameter_selection='None; all six Stage A selections frozen before any new seed training',
        fixed_test_policy='Every prespecified seed evaluated once; no seed/config winner selection')
    # Candidate identity retains the selected original grid ID, including CNN/MLP.
    plan['candidates']=[dict(candidate_id=selected['candidate_id'],parameters=selected['parameters'])]
    plan['source_sha256'][str(Path(__file__).resolve())]=sha(Path(__file__))
    if not smoke:
        require(len(seeds)==10 and fixed['fixed']==spec['fixed'] and fixed['structure']==spec['structure'] and
            fixed['splits']==spec['splits'] and fixed['fixed']['min_epochs']>=30 and fixed['fixed']['patience']>=30,
            'Formal confirmation must keep complete original fixed protocol')
    return plan


def prepare_group(folder,plan):
    require(not folder.exists() or read(folder/'plan.json')==plan,'Existing confirmation plan/source/input differs')
    if not folder.exists():
        folder.mkdir(parents=True)
        write(folder/'plan.json',plan)
        (folder/'frozen_selection.json').write_bytes(Path(plan['stage_a_selection_path']).read_bytes())
    require(sha(folder/'frozen_selection.json')==plan['stage_a_selection_sha256'],'Frozen original selection differs')


def worker_record(folder,plan,seed,mode):
    spec=plan['specification'];selected=plan['selected']
    roles=('train','valid') if mode=='train' else ('test',)
    worker=dict(mode=mode,model=spec['model'],repo=plan['repo'],output=str(folder),fixed=spec['fixed'],
        parameters=selected['parameters'],candidate_id=selected['candidate_id'],seed=seed,device=plan['device'],
        source_sha256=plan['source_sha256'],splits={r:plan['resolved_splits'][r] for r in roles},
        input_sha256={p:plan['input_sha256'][p] for r in roles for p in plan['resolved_splits'][r]['paths']})
    if mode=='test':
        group=folder.parent.parent;checkpoint=group/'training'/('seed_%d'%seed)/'best_model.pth'
        worker.update(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),selection=str(group/'frozen_selection.json'),
            selection_sha256=sha(group/'frozen_selection.json'))
    return worker


def run_phase(folder,plan,phase,resume):
    state_path=folder/'status.json'
    state=read(state_path) if state_path.exists() else dict(format=FORMAT,status='running',plan_sha256=identity(plan),
        completed_training=0,total_training=plan['training_runs'],completed_test=0,total_test=plan['training_runs'],workers=1)
    require(state['plan_sha256']==identity(plan),'Confirmation state plan differs')
    options=argparse.Namespace(resume=resume,device=plan['device'])
    mode='train' if phase=='training' else 'test'
    if mode=='test':require(state['completed_training']==plan['training_runs'],'All group training must finish before test')
    state.update(status='running',phase=phase,updated_at=now());write(state_path,state)
    try:
        for index,seed in enumerate(plan['specification']['seeds'],1):
            worker_folder=folder/phase/('seed_%d'%seed)
            worker=worker_record(worker_folder,plan,seed,mode)
            if (worker_folder/'worker.json').exists() and (worker_folder/'result.json').exists():
                require(read(worker_folder/'worker.json')==worker,'Completed confirmation worker changed')
            launch_worker(worker_folder,worker,options)
            state['completed_training' if mode=='train' else 'completed_test']=index
            state['updated_at']=now();write(state_path,state)
        state.update(status='training_complete' if mode=='train' and state['completed_test']<plan['training_runs'] else 'complete',
                     updated_at=now())
    except BaseException as error:
        state.update(status='failed',error=repr(error),updated_at=now());raise
    finally:write(state_path,state)


def modern_probe(gpu):
    code="import inspect,json,torch,numpy,pandas,scipy,sklearn,sys; assert 'weights_only' in inspect.signature(torch.load).parameters; assert torch.cuda.is_available() and torch.cuda.device_count()==1; print(json.dumps(dict(python=sys.version,torch=torch.__version__,numpy=numpy.__version__,pandas=pandas.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda)))"
    p=subprocess.run([sys.executable,'-c',code],text=True,capture_output=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu))
    require(p.returncode==0,'Modern single GPU probe failed: '+p.stderr)
    return json.loads(p.stdout)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','stage-a-root','prior-pipeline','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--gpu',default='0');parser.add_argument('--execute',action='store_true')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--skip-gpu-probe',action='store_true')
    parser.add_argument('--wait-prior',action='store_true',help='Queue without any training until Stage A pipeline is complete')
    options=parser.parse_args()
    require(options.gpu.isdigit() and not(options.execute and options.skip_gpu_probe),'One real GPU required for execution')
    bundle=Path(__file__).resolve().parent;registration=bundle/'registration'
    repo,stage_a_root,output=[p.resolve() for p in (options.repo,options.stage_a_root,options.output)]
    require(re.fullmatch(r'[A-Za-z0-9_-]+',output.name),'Output name must satisfy strict archive grammar')
    require(output!=repo and repo not in output.parents and output!=stage_a_root and stage_a_root not in output.parents and
        output!=bundle and bundle not in output.parents,'Confirmation output must be separate from all frozen inputs')
    manifest_sha=verify_bundle(bundle)
    if options.wait_prior:
        require(options.execute,'Waiting queue is only available with explicit execute')
        while True:
            prior=read(options.prior_pipeline)
            if prior['status']=='complete':break
            require(prior['status'] in ('running','waiting'),'Prior Stage A failed/stopped; inspect it before resuming')
            print('WAITING_FOR_STAGE_A '+now(),flush=True);time.sleep(30)
    prior=read(options.prior_pipeline)
    require(prior['status']=='complete','Stage A pipeline must complete before confirmation preflight')
    require(Path(prior['request']['output']).resolve()==stage_a_root,'Prior pipeline output is not requested Stage A root')
    require(verify_bundle(bundle)==manifest_sha,'Confirmation tool manifest changed while waiting')
    registration_check,seeds,selections=frozen_stage_a(stage_a_root,registration,repo)
    environment={} if options.skip_gpu_probe else modern_probe(options.gpu)
    plans={group:group_plan(read(registration/(group+'_stage_a9.json')),selections[group]['selected'],seeds,repo,options.gpu,
        selections[group],registration_check['registration_sha256']) for group in GROUPS}
    request=dict(format=FORMAT,bundle=str(bundle),bundle_manifest_sha256=manifest_sha,repo=str(repo),stage_a_root=str(stage_a_root),output=str(output),
        prior_pipeline=str(options.prior_pipeline.resolve()),registration_sha256=registration_check['registration_sha256'],gpu=options.gpu,
        frozen_selection_sha256={g:selections[g]['stage_a_selection_sha256'] for g in GROUPS},training_runs=60,fixed_test_evaluations=60)
    if not options.execute:
        print(json.dumps(dict(status='preflight_passed',request=request,environment=environment,registration=registration_check),indent=2));return
    state_path=output.with_name(output.name+'_pipeline_status.json')
    with task_lock(output.with_name('.'+output.name+'.pipeline.lock')):
        previous=read(state_path) if state_path.exists() else None
        require(not previous or (options.resume and previous['request']==request),'Existing confirmation pipeline requires unchanged request and explicit resume')
        if previous and previous['status']=='complete':print('CONFIRMATION_ALREADY_COMPLETE');return
        require(previous or not output.exists(),'Fresh confirmation output required')
        output.mkdir(parents=True,exist_ok=True)
        state=dict(format=FORMAT,status='running',request=request,pid=os.getpid(),started_at=now(),phase='freeze',steps=[],prior_attempt=previous)
        write(state_path,state)
        try:
            write(output/'preflight.json',dict(environment=environment,registration=registration_check,prior_pipeline_sha256=sha(options.prior_pipeline)))
            # Commit every fixed selection and complete future-seed plan before
            # dispatching the first training worker.
            for group,plan in plans.items():prepare_group(output/group,plan)
            for phase in ('training','fixed_test'):
                if phase=='fixed_test':
                    require(all(read(output/g/'status.json')['completed_training']==10 for g in GROUPS),
                            'All 60 trainings must complete before any fixed test')
                state['phase']=phase;write(state_path,state)
                for group,plan in plans.items():
                    state['active_group']=group;write(state_path,state)
                    print('START '+group+' '+phase,flush=True)
                    run_phase(output/group,plan,phase,options.resume)
                    state['steps'].append(dict(group=group,phase=phase,status='complete',finished_at=now()));write(state_path,state)
                    print('COMPLETE '+group+' '+phase,flush=True)
            from collect_stage_a_confirmation import collect
            collection=output.with_name(output.name+'_backup')
            collect(output,bundle,registration,stage_a_root,collection)
            state.update(status='complete',phase='complete',finished_at=now(),collection=str(collection))
            print('STAGE_A_CONFIRMATION_PIPELINE_COMPLETE',flush=True)
        except BaseException as error:
            state.update(status='failed',error=repr(error),failed_at=now());raise
        finally:state['updated_at']=now();write(state_path,state)


if __name__=='__main__':main()
