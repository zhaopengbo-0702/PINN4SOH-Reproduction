"""Modern single-GPU StageA queue: smoke, all selections, then selected tests, archive."""
import argparse
from datetime import datetime,timezone
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from search_replacement_grid import read,write,sha
from run_extended import task_lock
from verify_stage_a_registration import verify


def now():return datetime.now(timezone.utc).isoformat()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prior-pipeline',type=Path,required=True)
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--skip-gpu-probe',action='store_true')
    options=parser.parse_args()
    if not options.gpu.isdigit() or (options.execute and options.skip_gpu_probe):raise ValueError('One GPU and real GPU preflight required')
    bundle=Path(__file__).resolve().parent;repo=options.repo.resolve();output=options.output.resolve();registration=bundle/'registration'
    if not re.fullmatch(r'[A-Za-z0-9_-]+',output.name):raise ValueError('Output name must satisfy strict archive restore grammar')
    if output==repo or repo in output.parents or output==bundle or bundle in output.parents:raise ValueError('Results must be outside immutable sources')
    if read(options.prior_pipeline)['status']!='complete':raise ValueError('Prior XJTU pipeline must complete before StageA')
    manifest=read(bundle/'TOOLS_SHA256.json')
    for name,row in manifest['files'].items():
        path=bundle/name
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('Tool bundle changed: '+name)
    inputs=read(bundle/'EXPECTED_AUTHOR_INPUTS.json')
    if {p.relative_to(repo).as_posix() for p in repo.rglob('*.py')}!=set(inputs['author_python_sha256']):raise ValueError('Author source set differs')
    for name,checksum in inputs['author_python_sha256'].items():
        if sha(repo/name)!=checksum:raise ValueError('Author source changed')
    registration_check=verify(registration,repo)
    environment={}
    if not options.skip_gpu_probe:
        code="import inspect,json,torch,numpy,pandas,scipy,sklearn,sys; assert 'weights_only' in inspect.signature(torch.load).parameters; assert torch.cuda.is_available() and torch.cuda.device_count()==1; print(json.dumps(dict(python=sys.version,torch=torch.__version__,numpy=numpy.__version__,pandas=pandas.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda)))"
        probe=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=options.gpu))
        if probe.returncode:raise RuntimeError('Modern GPU dependency probe failed: '+probe.stderr)
        environment=json.loads(probe.stdout)
    request=dict(bundle=str(bundle),bundle_manifest_sha256=sha(bundle/'TOOLS_SHA256.json'),repo=str(repo),output=str(output),gpu=options.gpu,
        prior_pipeline=str(options.prior_pipeline.resolve()),registration_sha256=registration_check['registration_sha256'])
    if not options.execute:
        print(json.dumps(dict(status='preflight_passed',request=request,registration=registration_check,environment=environment),indent=2));return
    state_path=output.with_name(output.name+'_pipeline_status.json')
    with task_lock(output.with_name('.'+output.name+'.pipeline.lock')):
        previous=read(state_path) if state_path.exists() else None
        if previous and (not options.resume or previous['request']!=request):raise ValueError('Existing pipeline requires unchanged request and explicit resume')
        if previous and previous['status']=='complete':print('STAGE_A_ALREADY_COMPLETE');return
        if not previous and output.exists():raise ValueError('Fresh pipeline output required')
        output.mkdir(parents=True,exist_ok=True)
        state=dict(format='pinn4soh-stagea-pipeline-v1',status='running',request=request,pid=os.getpid(),started_at=now(),steps=[],prior_attempt=previous)
        write(state_path,state)
        def step(name,command):
            record=dict(name=name,status='running',started_at=now(),command=command);state['steps'].append(record);write(state_path,state)
            print('START '+name,flush=True)
            code=subprocess.call(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=options.gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',CUBLAS_WORKSPACE_CONFIG=':4096:8'))
            record.update(status='complete' if code==0 else 'failed',returncode=code,finished_at=now());write(state_path,state)
            if code:raise RuntimeError(name+' failed: '+str(code))
            print('COMPLETE '+name,flush=True)
        try:
            write(output/'preflight.json',dict(registration=registration_check,environment=environment))
            # Genuine dataset GPU smoke, one PINN candidate with active PDE+mono,
            # short train/valid only; no smoke test evaluation.
            for dataset in ('mit','hust'):
                spec=read(registration/(dataset+'_pinn_stage_a9.json'))
                spec.update(label='STAGE_A_GPU_TWO_EPOCH_SMOKE_ONLY',smoke=True,seeds=spec['seeds'][:1])
                spec['fixed'].update(epochs=2,min_epochs=2,patience=1,warmup_epochs=1,diagnostic_batch_size=8)
                for role,count in [('train',1),('valid',2),('test',1)]:
                    for key in ('paths','physical_ids'):spec['splits'][role][key]=spec['splits'][role][key][:count]
                spec['candidates']=[dict(lr_u=.001,lr_F=.001,alpha=.1,beta_mean=.01,weight_decay=0)]
                spec_path=output/(dataset+'_gpu_smoke_spec.json')
                if spec_path.exists() and read(spec_path)!=spec:raise ValueError('Smoke spec changed')
                write(spec_path,spec)
                step(dataset+'_gpu_smoke',[sys.executable,str(bundle/'search_stage_a.py'),'--spec',str(spec_path),'--repo',str(repo),'--output',str(output/(dataset+'_gpu_smoke')),'--device',options.gpu,'--execute','--resume'])
            # All six validation searches finish before any formal selected test.
            for phase in ('selection','selected_test'):
                for dataset in ('mit','hust'):
                    for model in ('pinn','mlp','cnn'):
                        group=dataset+'_'+model
                        command=[sys.executable,str(bundle/'search_stage_a.py'),'--spec',str(registration/(group+'_stage_a9.json')),'--repo',str(repo),'--output',str(output/group),'--device',options.gpu,'--execute','--resume']
                        if phase=='selected_test':command+=['--evaluate-test']
                        step(group+'_'+phase,command)
            collection=output.with_name(output.name+'_backup')
            if collection.exists():raise ValueError('Collection exists; preserve and inspect before separate resumed collection')
            step('audit_and_collection',[sys.executable,str(bundle/'collect_stage_a.py'),'--root',str(output),'--bundle',str(bundle),'--registration',str(registration),'--output',str(collection)])
            state.update(status='complete',finished_at=now(),collection=str(collection))
            print('STAGE_A_PIPELINE_COMPLETE',flush=True)
        except BaseException as error:
            state.update(status='failed',error=repr(error),failed_at=now());raise
        finally:state['updated_at']=now();write(state_path,state)


if __name__=='__main__':main()
