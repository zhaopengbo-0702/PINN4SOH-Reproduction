"""Independent Stage A array/identity/selection audit and verified archive collector."""
import argparse
import base64
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import tarfile

from search_replacement_grid import read, write, sha, identity


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_tree(v) for v in value)
    return not isinstance(value, (int, float)) or math.isfinite(value)


def audit_data_identity(data, splits):
    require(set(data) == set(splits), "Loaded roles differ from worker")
    for role, split in splits.items():
        batteries = data[role]['batteries']
        require([r['path'] for r in batteries] == split['paths'] and
                [r['physical_id'] for r in batteries] == split['physical_ids'] and
                all(isinstance(r['n_pairs'], int) and r['n_pairs'] > 0 for r in batteries),
                "Physical boundaries differ from registered split")


def objective(truth, prediction, batteries):
    import numpy as np
    truth = np.asarray(truth).reshape(-1).astype(np.float64)
    prediction = np.asarray(prediction).reshape(-1).astype(np.float64)
    require(truth.shape == prediction.shape and truth.size and np.isfinite(truth).all() and np.isfinite(prediction).all(), "Bad arrays")
    grouped, start = {}, 0
    for row in batteries:
        end = start + row["n_pairs"]
        grouped.setdefault(row["physical_id"], []).extend((prediction[start:end] - truth[start:end]).tolist())
        start = end
    require(start == len(truth), "Physical boundaries differ")
    return sum(math.sqrt(sum(v * v for v in values) / len(values)) for values in grouped.values()) / len(grouped)


def audit_suite(suite, require_test=True):
    import numpy as np
    suite = Path(suite).resolve()
    plan, state, selection = [read(suite / name) for name in ("plan.json", "status.json", "selection.json")]
    require(state["status"] == "complete" and state["plan_sha256"] == identity(plan), "Incomplete suite")
    require(selection['plan_sha256'] == identity(plan), "Selection plan differs")
    require(state["completed_runs"] == state["total_runs"] == plan["candidate_runs"], "Count incomplete")
    spec, rows, checks, expected_dirs = plan["specification"], [], 0, set()
    for name, checksum in {**plan["source_sha256"], **plan["input_sha256"]}.items():
        require(sha(name) == checksum, "Source/input changed since plan")
    tensor_cohort = {}
    for candidate in plan["candidates"]:
        for seed in spec["seeds"]:
            folder = suite / candidate["candidate_id"] / ("seed_%d" % seed)
            expected_dirs.add(folder.relative_to(suite).as_posix())
            worker, result = read(folder / "worker.json"), read(folder / "result.json")
            expected_splits = {r: plan["resolved_splits"][r] for r in ("train", "valid")}
            require(worker["mode"] == "train" and worker['repo'] == plan['repo'] and Path(worker['output']).resolve() == folder and
                worker["seed"] == seed and worker["candidate_id"] == candidate["candidate_id"] and
                worker["parameters"] == candidate["parameters"] and worker["fixed"] == spec["fixed"] and worker["model"] == spec["model"] and
                worker["splits"] == expected_splits and worker["source_sha256"] == plan["source_sha256"] and worker["device"] == plan["device"] and
                worker["input_sha256"] == {p: plan["input_sha256"][p] for r in ("train", "valid") for p in expected_splits[r]["paths"]}, "Candidate worker differs")
            require(result["status"] == "complete" and result["worker_identity"] == identity(worker) and result["test_loaded"] is False and
                    result["checkpoint_reload_bitwise_prediction_equal"] is True, "Candidate completion/reload invalid")
            require(set(result["artifact_sha256"]) == {"best_model.pth","valid_truth.npy","valid_prediction.npy","validation_metrics.json","data_identity.json","history.json","gradient_diagnostics.json","diagnostic_batch.json"}, "Missing artifact set")
            for name, checksum in result["artifact_sha256"].items():
                require(sha(folder / name) == checksum, "Completed candidate artifact changed")
                checks += 1
            data = read(folder / "data_identity.json")
            audit_data_identity(data, expected_splits)
            tensor = identity(data)
            require(seed not in tensor_cohort or tensor_cohort[seed] == tensor, "Candidate tensors differ within seed")
            tensor_cohort[seed] = tensor
            score = objective(np.load(folder/'valid_truth.npy',allow_pickle=False),np.load(folder/'valid_prediction.npy',allow_pickle=False),data['valid']['batteries'])
            history = read(folder/'history.json')
            require([r['epoch'] for r in history] == list(range(1,len(history)+1)), "Bad history sequence")
            require(history and len(history) <= spec['fixed']['epochs'] and finite_tree(history) and
                    all(r['gradients_finite'] and r['mono_branch_mode'] == 'eval_with_grad' and
                        r['data_branch_mode'] == 'train' and 0 < r['batch_min'] <= r['batch_max'] <= spec['fixed']['batch_size']
                        for r in history), "Invalid/nonfinite training history")
            best = min(history,key=lambda r:(r['validation']['battery_macro_RMSE'],r['epoch']))
            require(math.isclose(score,result['objective'],rel_tol=1e-14,abs_tol=1e-15) and result['objective']==best['validation']['battery_macro_RMSE'] and
                result['best_epoch']==best['epoch'] and result['epochs_completed']==len(history), "Independent macro/best history mismatch")
            require(spec['smoke'] or len(history)>=spec['fixed']['min_epochs'], "Stopped before minimum epochs")
            diagnostics=read(folder/'gradient_diagnostics.json')
            require([r['epoch'] for r in diagnostics] == [r['epoch'] for r in history] and finite_tree(diagnostics) and
                    all(r['mono_requires_grad'] and r['mono_dropout']=='eval_with_grad' and r['data_dropout']=='train' and
                        r['pde_computed'] == (spec['model']=='PINN' and candidate['parameters']['alpha']>0) for r in diagnostics),
                    "Mono gradient diagnostics missing")
            rows.append(dict(candidate_id=candidate['candidate_id'],seed=seed,objective=result['objective'],result_sha256=sha(folder/'result.json')))
    actual={p.parent.relative_to(suite).as_posix() for p in suite.glob('candidate_*/seed_*/result.json')}
    require(actual==expected_dirs,"Candidate matrix differs")
    ranking=[]
    for candidate in plan['candidates']:
        scores=[r['objective'] for r in rows if r['candidate_id']==candidate['candidate_id']]
        require(len(scores)==len(spec['seeds']),"Missing seed")
        ranking.append(dict(**candidate,mean_validation_battery_macro_RMSE=sum(scores)/len(scores),per_seed_objectives=scores))
    ranking.sort(key=lambda r:(r['mean_validation_battery_macro_RMSE'],r['candidate_id']))
    require(selection['ranking']==ranking and selection['selected']==ranking[0] and selection['candidate_results']==rows and selection['test_metrics_used'] is False,
            "Independent selection differs")
    tests=list(suite.glob('selected_test/seed_*/result.json'))
    require(not require_test or {p.parent.name for p in tests}=={'seed_%d'%s for s in spec['seeds']},"Selected test seed set differs")
    selected=ranking[0]
    test_rows=[]
    for path in tests:
        worker,result=read(path.with_name('worker.json')),read(path)
        seed=worker['seed']
        require(worker['mode']=='test' and worker['repo']==plan['repo'] and Path(worker['output']).resolve()==path.parent and
            path.parent.name=='seed_%d'%seed and seed in spec['seeds'] and worker['candidate_id']==selected['candidate_id'] and worker['parameters']==selected['parameters'] and
            worker['splits']=={'test':plan['resolved_splits']['test']} and worker['fixed']==spec['fixed'] and worker['model']==spec['model'] and
            worker['source_sha256']==plan['source_sha256'] and worker['device']==plan['device'] and worker['input_sha256']=={p:plan['input_sha256'][p] for p in plan['resolved_splits']['test']['paths']},"Test worker differs")
        checkpoint=suite/selected['candidate_id']/('seed_%d'%seed)/'best_model.pth'
        require(Path(worker['checkpoint']).resolve()==checkpoint and sha(checkpoint)==worker['checkpoint_sha256'] and sha(suite/'selection.json')==worker['selection_sha256']==result['selection_sha256'],"Test same-seed checkpoint/selection mismatch")
        require(result['worker_identity']==identity(worker) and result['status']=='complete' and result['test_used_for_selection'] is False,"Test identity invalid")
        require(set(result['artifact_sha256'])=={'test_truth.npy','test_prediction.npy','test_metrics.json','data_identity.json'},"Test artifacts missing")
        for name,checksum in result['artifact_sha256'].items():
            require(sha(path.parent/name)==checksum,"Test artifact changed")
            checks+=1
        data=read(path.with_name('data_identity.json'))
        audit_data_identity(data, worker['splits'])
        score=objective(np.load(path.with_name('test_truth.npy'),allow_pickle=False),np.load(path.with_name('test_prediction.npy'),allow_pickle=False),data['test']['batteries'])
        require(math.isclose(score,result['test_battery_macro_RMSE'],rel_tol=1e-14,abs_tol=1e-15),"Test macro differs")
        test_rows.append(dict(seed=seed,battery_macro_RMSE=result['test_battery_macro_RMSE']))
    return dict(status='passed',candidate_runs=len(rows),candidates=len(ranking),seeds=spec['seeds'],selected=selected,
        test_evaluations=len(test_rows),test_per_seed=test_rows,artifact_sha256_checks=checks,selection_recomputed_independently=True,
        selected_test_seed_set_and_same_seed_checkpoint_verified=True if require_test else (True if len(tests)==len(spec['seeds']) else None),
        model=spec['model'],dataset=spec['dataset'],smoke=spec['smoke'])


def collect(root,bundle,registration,output):
    root,bundle,registration,output=[Path(p).resolve() for p in (root,bundle,registration,output)]
    require(not output.exists(),"Fresh collection output required")
    require(re.fullmatch(r'[A-Za-z0-9_-]+',output.name), "Backup folder name must satisfy strict restore grammar")
    tools=read(bundle/'TOOLS_SHA256.json')
    for name,row in tools['files'].items():
        path=bundle/name
        require(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'], 'Tool bundle changed before collection')
    from verify_stage_a_registration import verify
    registered_repo=Path(read(root/'mit_pinn/plan.json')['repo']).resolve()
    verified_registration=verify(registration,registered_repo)
    audits={}
    for dataset in ('mit','hust'):
        for model in ('pinn','mlp','cnn'):
            group=dataset+'_'+model
            spec=read(registration/(group+'_stage_a9.json'))
            require(read(root/group/'plan.json')['repo']==str(registered_repo) and
                    read(root/group/'plan.json')['specification']==spec and spec['smoke'] is False and
                    spec['model']==model.upper() and spec['dataset']==dataset.upper() and
                    len(spec['seeds'])==3 and len(spec['candidates'])==9, "Formal group differs from registration")
            audits[group]=audit_suite(root/group,True)
    require(sum(r['candidate_runs'] for r in audits.values())==162 and sum(r['test_evaluations'] for r in audits.values())==18,"Formal collection scope differs")
    output.mkdir(parents=True)
    write(output/'STAGE_A_INDEPENDENT_AUDIT.json',dict(status='passed',groups=audits,selection_runs=162,selected_test_evaluations=18,
        registration=verified_registration,tools_manifest_sha256=sha(bundle/'TOOLS_SHA256.json')))
    paths={}
    for label,target in [('results',root),('tools',bundle),('registration',registration),('audit',output/'STAGE_A_INDEPENDENT_AUDIT.json')]:
        for path in target.rglob('*') if target.is_dir() else [target]:
            require(not path.is_symlink(),'Symlink rejected')
            if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':
                name=label+'/'+(path.relative_to(target).as_posix() if target.is_dir() else path.name)
                paths[name]=path
    # The portable archive retains original input bytes, so an independent local
    # audit can rebuild tensors without relying on the running server filesystem.
    for group in audits:
        plan=read(root/group/'plan.json');repo=Path(plan['repo']).resolve()
        for name,checksum in {**plan['source_sha256'],**plan['input_sha256']}.items():
            path=Path(name).resolve()
            if repo in path.parents:
                require(sha(path)==checksum and not path.is_symlink(), 'Portable author/input snapshot changed')
                paths['repo_snapshot/'+path.relative_to(repo).as_posix()]=path
    frozen,archive,records=archive_verified(paths,output)
    size,checksum=archive.stat().st_size,sha(archive)
    evidence=dict(format='pinn4soh-stagea-collection-v1',status='complete',archive_bytes=size,archive_sha256=checksum,all_archived_member_hashes_verified=True,
        selection_runs=162,selected_test_evaluations=18,groups=audits,inventory=[dict(path=name,**r) for name,r in sorted(frozen.items())],
        runtime_note='Modern runtime; LEGACY_STAGEA name prefix solely preserves existing strict restore grammar',
        limits=['New improvement experiment, historical test already observed','No 10-seed confirmation or old-runtime replication performed by this pipeline'])
    write(output/'STAGE_A_COLLECTION_VERIFIED.json',evidence)
    write(output/'JSON_TRANSFER_MANIFEST.json',dict(format='pinn4soh-backup-json-base64-v1',archive_type='tar.gz',verified_without_errors=True,original_archives=1,wrapper_files=len(records),records=records))
    return evidence


def archive_verified(paths,output):
    """Stream and independently reread exact file bytes; also used by fixture QA."""
    output=Path(output).resolve()
    frozen={name:dict(bytes=p.stat().st_size,sha256=sha(p)) for name,p in paths.items()}
    archive=output/('PINN4SOH_LEGACY_STAGEA_'+output.name+'.tar.gz')
    with archive.open('wb') as compressed:
        with gzip.GzipFile(fileobj=compressed,mode='wb',compresslevel=1,mtime=0) as gz:
            with tarfile.open(fileobj=gz,mode='w|') as tar:
                for name,path in sorted(paths.items()):
                    before=path.stat(); h=hashlib.sha256()
                    with path.open('rb') as stream:
                        class Reader:
                            def read(self,size):
                                block=stream.read(size);h.update(block);return block
                        tar.addfile(tar.gettarinfo(str(path),arcname=name),Reader())
                    after=path.stat()
                    require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns) and h.hexdigest()==frozen[name]['sha256'],'Source changed during archive')
    seen=set()
    with tarfile.open(archive,'r|gz') as tar:
        for member in tar:
            require(member.isfile() and member.name in frozen and member.name not in seen,'Bad archive member')
            h=hashlib.sha256();size=0;stream=tar.extractfile(member)
            for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block);size+=len(block)
            require(size==frozen[member.name]['bytes'] and h.hexdigest()==frozen[member.name]['sha256'],'Archive member hash failed')
            seen.add(member.name)
    require(seen==set(frozen),'Archive set differs')
    size,checksum=archive.stat().st_size,sha(archive);chunk_size=48*1024*1024;count=max(1,(size+chunk_size-1)//chunk_size);records=[]
    with archive.open('rb') as stream:
        for i in range(count):
            data=stream.read(chunk_size)
            meta=dict(format='pinn4soh-backup-json-base64-v1',archive_type='tar.gz',original_zip_name=archive.name,original_zip_bytes=size,original_zip_sha256=checksum,
                chunk_index=i,chunk_count=count,chunk_offset=i*chunk_size,chunk_bytes=len(data),chunk_sha256=hashlib.sha256(data).hexdigest(),encoding='base64')
            path=output/('results.stagea.'+output.name+'.chunk%03d.json'%(i+1))
            path.write_text(json.dumps(dict(meta,data_base64=base64.b64encode(data).decode('ascii')),separators=(',',':'))+'\n',encoding='ascii')
            require(base64.b64decode(read(path)['data_base64'],validate=True)==data,'Wrapper roundtrip failed')
            records.append(dict(meta,wrapper_name=path.name,wrapper_bytes=path.stat().st_size,wrapper_sha256=sha(path),roundtrip_verified=True))
    return frozen,archive,records


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','bundle','registration','output'):parser.add_argument('--'+name,type=Path,required=True)
    result=collect(**vars(parser.parse_args()))
    print(json.dumps({k:v for k,v in result.items() if k!='inventory'},indent=2))
