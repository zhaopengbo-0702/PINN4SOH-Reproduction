"""Queue one transfer pipeline after Stage A confirmation, on its existing GPU."""
import argparse
import copy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from search_replacement_grid import read,write,sha,identity
from run_extended import task_lock
from install_legacy_runtime import isolated_environment
from collect_transfer_robustness import audit_selection,audit_confirmation,audit_test,write_gate,collect,require


SUITES=('pinn4soh_full_paper_stage1','pinn4soh_legacy_formal_control_20261110','pinn4soh_legacy_selected_extension_20261110')


def now():return datetime.now(timezone.utc).isoformat()


def bundle_check(bundle):
    manifest=read(bundle/'TOOLS_SHA256.json')
    require(manifest.get('deployable') is True,'Development-only bundle cannot run cloud queue')
    for name,row in manifest['files'].items():
        path=bundle/name
        require(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Transfer bundle changed: '+name)
    protocol=read(bundle/'PROTOCOL.json')
    for name,checksum in protocol['implementation_sha256'].items():require(sha(bundle/name)==checksum,'Protocol implementation differs')
    return sha(bundle/'TOOLS_SHA256.json')


def relocate_sources(protocol,workspace,destination):
    """A verified alias tree changes no frozen protocol path or source bytes."""
    destination.mkdir(parents=True,exist_ok=True);mapping={};evidence={}
    for suite in SUITES:
        source=(workspace/suite).resolve();alias=destination/suite
        require(source.is_dir(),'Historical cloud suite missing: '+str(source))
        if alias.is_symlink():require(alias.resolve()==source,'Source alias points elsewhere')
        elif alias.exists():require(alias.resolve()==source,'Source alias collision')
        else:alias.symlink_to(source,target_is_directory=True)
        mapping[suite]=str(source)
    for direction,stacks in protocol['source_records'].items():
        for stack,records in stacks.items():
            for repeat,record in records.items():
                folder=destination/record['relative_folder']
                for name,checksum in record['evidence_sha256'].items():require(sha(folder/name)==checksum,'Relocated evidence differs: '+str(folder/name))
                evidence[direction+'/'+stack+'/'+repeat]=dict(relative_folder=record['relative_folder'],resolved_folder=str(folder.resolve()),evidence_sha256=record['evidence_sha256'])
        for repeat in map(str,range(1,11)):
            a,b=[stacks[s][repeat] for s in ('modern','legacy')]
            require(a['training_seed']==b['training_seed'] and a['source_fingerprint']==b['source_fingerprint']
                and a['input_tensor_identity']==b['input_tensor_identity'],'Source runtime pair differs')
    result=dict(format='transfer-source-relocation-v1',mapping=mapping,records=evidence,protocol_unchanged=True)
    target=destination/'VERIFIED_RELOCATION.json'
    require(not target.exists() or read(target)==result,'Relocation map changed')
    if not target.exists():write(target,result)
    return result


def runtime_environment(stack,legacy_prefix,gpu):
    env=isolated_environment(legacy_prefix) if stack=='legacy' else dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',
        PYTHONUNBUFFERED='1',PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    return env


def gpu_probe(python,environment,stack,repo):
    # Imports, exact stack and single allocation checks happen after predecessor.
    code="import json,torch,numpy,pandas,scipy,sklearn,sys; assert torch.cuda.is_available() and torch.cuda.device_count()==1; print(json.dumps(dict(python=sys.version,torch=torch.__version__,numpy=numpy.__version__,pandas=pandas.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0))))"
    p=subprocess.run([str(python),'-c',code],env=environment,text=True,capture_output=True)
    require(p.returncode==0,'Transfer GPU probe failed: '+p.stderr)
    probe=json.loads(p.stdout)
    require(probe['torch']=='1.7.1+cu110' if stack=='legacy' else probe['torch'].startswith('1.13.1'),'Expected runtime differs')
    if stack=='legacy':require(sys.platform=='linux','Legacy runtime only on Linux')
    return probe


def source_input_check(protocol,repo,bundle):
    expected=read(bundle/'EXPECTED_AUTHOR_INPUTS.json')['author_python_sha256']
    require({p.relative_to(repo).as_posix() for p in repo.rglob('*.py')}==set(expected),'Author source set differs')
    for name,checksum in expected.items():require(sha(repo/name)==checksum,'Author source changed')
    for record in protocol['target_identity'].values():require(len(record['physical_ids'])==1,'Mixed target UID')
    for name,record in protocol['target_identity'].items():require(sha(repo/name)==record['csv_sha256'],'Target CSV changed')
    for stacks in protocol['source_records'].values():
        for records in stacks.values():
            for record in records.values():
                for name,checksum in record['input_tensor_identity']['input_file_sha256'].items():
                    require(sha(repo/name)==checksum,'Historical pretraining CSV changed')


def smoke(python,environment,protocol,repo,archive,folder,stack,device,bundle):
    from run_transfer_robustness import make_worker,check_completed
    from types import SimpleNamespace
    # Run each source and the encoder-freeze branch; no test role is passed.
    for lane,candidate_index in [('HUST-to-MIT',1),('XJTU-to-MIT',2)]:
        out=folder/(lane+'_smoke');candidate=protocol['transfer_candidates'][candidate_index]
        worker=make_worker(protocol,lane,candidate,2026111101,1,'train',stack,repo,archive,out,sha(bundle/'PROTOCOL.json'),device)
        worker['fixed']=copy.deepcopy(worker['fixed']);worker['fixed'].update(epochs=2,min_epochs=2,patience=1)
        for key in ('paths','physical_ids'):worker['splits']['valid'][key]=worker['splits']['valid'][key][:2]
        keep={p for row in worker['splits'].values() for p in row['paths']};worker['input_sha256']={p:v for p,v in worker['input_sha256'].items() if p in keep}
        worker['smoke']=True
        if (out/'result.json').exists():check_completed(out,worker);continue
        require(not out.exists(),'Incomplete GPU smoke requires separate inspected output')
        out.mkdir(parents=True);write(out/'worker.json',worker)
        with (out/'console.log').open('w',encoding='utf-8') as stream:
            p=subprocess.run([str(python),str(bundle/'run_transfer_robustness.py'),'--worker-json',str(out/'worker.json')],env=environment,stdout=stream,stderr=subprocess.STDOUT)
        require(p.returncode==0,'Transfer real GPU smoke failed: '+str(out/'console.log'))
        r=check_completed(out,worker);require(r['test_loaded'] is False and r['smoke'] is True,'Smoke unexpectedly loaded test')


def wait_prior(path,expected_root,wait):
    while True:
        if not path.exists():
            require(wait,'Stage A confirmation status does not exist; no GPU probe allowed')
            print('WAITING_FOR_STAGE_A_CONFIRMATION_STATUS '+now(),flush=True)
            time.sleep(30)
            continue
        prior=read(path)
        if prior['status']=='complete':
            require(Path(prior['request']['output']).resolve()==expected_root,'Predecessor root mismatch')
            return prior
        require(wait and prior['status'] in ('running','waiting'),'Stage A confirmation not complete or failed; no GPU probe allowed')
        print('WAITING_FOR_STAGE_A_CONFIRMATION '+now(),flush=True);time.sleep(30)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','workspace','output','prior-pipeline','stage-a-confirmation-root','legacy-prefix'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu',default='0');p.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--wait-prior',action='store_true')
    o=p.parse_args();bundle=Path(__file__).resolve().parent;repo=o.repo.resolve();workspace=o.workspace.resolve();output=o.output.resolve()
    require(o.gpu=='0','Use the single allocated logical GPU0')
    require(re.fullmatch(r'[A-Za-z0-9_-]+',output.name),'Output archive grammar differs')
    for protected in (bundle,repo,o.stage_a_confirmation_root.resolve()):require(output!=protected and protected not in output.parents,'Fresh separate transfer output required')
    manifest_sha=bundle_check(bundle);protocol=read(bundle/'PROTOCOL.json')
    prior=wait_prior(o.prior_pipeline,o.stage_a_confirmation_root.resolve(),o.wait_prior and o.execute)
    require(bundle_check(bundle)==manifest_sha,'Bundle changed while waiting')
    previous_collection=Path(prior['collection'])/'STAGE_A_CONFIRMATION_COLLECTION_VERIFIED.json'
    completed=read(previous_collection)
    require(completed['status']=='complete' and completed['training_runs']==60 and completed['fixed_test_evaluations']==60
        and completed['all_archived_member_hashes_verified'] is True,'Predecessor confirmation lacks full60+60 archive')
    source_input_check(protocol,repo,bundle)
    request=dict(bundle_manifest_sha256=manifest_sha,protocol_sha256=sha(bundle/'PROTOCOL.json'),repo=str(repo),workspace=str(workspace),
        output=str(output),legacy_prefix=str(o.legacy_prefix.resolve()),gpu=o.gpu,prior_pipeline=str(o.prior_pipeline.resolve()),
        prior_confirmation_collection_sha256=sha(previous_collection))
    if not o.execute:
        print(json.dumps(dict(status='read_only_preflight_passed_no_GPU_probe',request=request),indent=2));return
    state_path=output.with_name(output.name+'_pipeline_status.json')
    with task_lock(output.with_name('.'+output.name+'.pipeline.lock')):
        previous=read(state_path) if state_path.exists() else None
        require(not previous or o.resume and previous['request']==request,'Resume request differs')
        if previous and previous['status']=='complete':print('TRANSFER_PIPELINE_ALREADY_COMPLETE');return
        require(previous or not output.exists(),'Fresh output required')
        output.mkdir(parents=True,exist_ok=True)
        state=dict(status='running',request=request,pid=os.getpid(),phase='source_relocation',started_at=now(),steps=[],prior_attempt=previous)
        write(state_path,state)
        archive=output/'verified_source_archive';modern=output/'modern';legacy=output/'legacy'
        def save():state['updated_at']=now();write(state_path,state)
        def step(name,command,env):
            state['phase']=name;save();print('START '+name,flush=True)
            code=subprocess.call(command,env=env)
            state['steps'].append(dict(name=name,returncode=code,finished_at=now()));save()
            require(code==0,'Transfer step failed: '+name)
        try:
            relocation=relocate_sources(protocol,workspace,archive)
            common=['--protocol',str(bundle/'PROTOCOL.json'),'--repo',str(repo),'--archive-root',str(archive),'--device',o.gpu,'--execute']
            for stack,root in [('modern',modern),('legacy',legacy)]:
                python=Path(sys.executable) if stack=='modern' else o.legacy_prefix.resolve()/'bin/python'
                environment=runtime_environment(stack,o.legacy_prefix.resolve(),o.gpu)
                state['phase']=stack+'_GPU_probe';save()
                probe=gpu_probe(python,environment,stack,repo);probe_path=output/(stack+'_GPU_probe.json')
                if probe_path.exists():require(read(probe_path)==probe,'Runtime probe changed across resume')
                else:write(probe_path,probe)
                state['phase']=stack+'_real_smoke';save();smoke(python,environment,protocol,repo,archive,output/(stack+'_smokes'),stack,o.gpu,bundle)
                command=[str(python),str(bundle/'run_transfer_robustness.py'),*common,'--output',str(root),'--stack',stack]
                if o.resume:command+=['--resume']
                if stack=='modern':
                    step('modern_selection87',command+['--phase','selection'],environment)
                    audit=audit_selection(modern,protocol,archive,bundle,o.gpu);write(output/'MODERN_SELECTION_GATE.json',audit)
                else:
                    audit_selection(modern,protocol,archive,bundle,o.gpu)
                step(stack+'_confirmation50',command+['--phase','confirmation','--selection-root',str(modern)]+(['--resume'] if stack=='modern' and not o.resume else []),environment)
                report=audit_confirmation(root,modern,protocol,archive,bundle,stack,o.gpu)
                write(output/(stack+'_CONFIRMATION_GATE.json'),report);write_gate(root,report,modern)
                step(stack+'_fixed_test70',command+['--phase','test','--selection-root',str(modern)]+(['--resume'] if not o.resume else []),environment)
                audited=audit_test(root,modern,protocol,archive,bundle,stack,o.gpu);write(output/(stack+'_TEST_AUDIT.json'),audited)
            state['phase']='independent_collection';save()
            backup=output.with_name(output.name+'_backup')
            collect(modern,legacy,bundle/'PROTOCOL.json',archive,bundle,backup,o.gpu)
            state.update(status='complete',phase='complete',collection=str(backup),finished_at=now());print('TRANSFER_ROBUSTNESS_PIPELINE_COMPLETE',flush=True)
        except BaseException as error:state.update(status='failed',error=repr(error),failed_at=now());raise
        finally:save()


if __name__=='__main__':main()
