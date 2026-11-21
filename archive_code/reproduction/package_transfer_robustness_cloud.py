"""Package unchanged transfer protocol/runner plus independently reviewed cloud tools."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package(root,output,allow_unaccepted=False):
    root,output=Path(root).resolve(),Path(output).resolve();source=root/'reproduction'
    protocol=root/'outputs/full_reproduction/transfer_robustness_preregistered_20261111/PROTOCOL.json'
    accepted=root/'outputs/full_reproduction/transfer_robustness_preregistered_20261111/ACCEPTANCE.json'
    require=lambda v,m:None if v else (_ for _ in ()).throw(ValueError(m))
    require(not output.exists() and output.parent.is_dir(),'Fresh output archive required')
    prior=json.loads(accepted.read_text(encoding='utf-8'))
    require(prior['status']=='passed_functional_and_real_cpu_smoke_not_formal' and prior['protocol_sha256']==sha(protocol),'Base acceptance differs')
    for name,checksum in prior['artifact_sha256'].items():require(sha(root/name)==checksum,'Accepted source changed: '+name)
    names=['run_transfer_robustness.py','prepare_transfer_robustness.py','run_transfer_robustness_cloud.py','collect_transfer_robustness.py',
        'package_transfer_robustness_cloud.py','search_stage_a.py','search_replacement_grid.py','run_extended.py','run_transfer.py',
        'run_reproduction.py','verify_stage_a_registration.py','install_legacy_runtime.py','probe_legacy_runtime.py','collect_stage_a.py',
        'collect_stage_a_confirmation.py','restore_backup_json.py','TRANSFER_ROBUSTNESS_RUN.md','TRANSFER_ROBUSTNESS_CLOUD.md',
        'test_transfer_robustness.py','test_transfer_robustness_cloud.py']
    files={name:(source/name).read_bytes() for name in names}
    files['PROTOCOL.json']=protocol.read_bytes();files['LOCAL_RUNNER_ACCEPTANCE.json']=accepted.read_bytes()
    qa=root/'outputs/full_reproduction/transfer_robustness_cloud_validation_20261111/ACCEPTANCE.json'
    if not allow_unaccepted:
        q=json.loads(qa.read_text(encoding='utf-8'));require(q['status']=='passed_local_wrapper_collector_package','Cloud tools QA must pass before deployable package')
        for name,checksum in q['artifact_sha256'].items():require(sha(root/name)==checksum,'Cloud QA changed since acceptance')
        files['LOCAL_CLOUD_ACCEPTANCE.json']=qa.read_bytes()
    legacy_files=['run_transfer_robustness.py','prepare_transfer_robustness.py','search_stage_a.py','search_replacement_grid.py',
        'run_extended.py','run_transfer.py','run_reproduction.py','verify_stage_a_registration.py']
    for name,data in files.items():
        if name.endswith('.py'):ast.parse(data,filename=name,feature_version=(3,7) if name in legacy_files else (3,10))
    expected=dict(author_python_sha256={p.relative_to(root/'PINN4SOH').as_posix():sha(p) for p in sorted((root/'PINN4SOH').rglob('*.py'))},
        protocol_sha256=sha(protocol),prior_runner_acceptance_sha256=sha(accepted),formal_target_trainings=187,
        fixed_test_evaluations=140,same_platform_job=True,single_GPU=True,single_worker=True,
        source_archive_aliases=['pinn4soh_full_paper_stage1','pinn4soh_legacy_formal_control_20261110','pinn4soh_legacy_selected_extension_20261110'],
        predecessor='Stage A future-seed confirmation complete60training+60test+archive, then probeGPU',
        native_legacy_checkpoint_loader_sha256=sha(source/'run_transfer_robustness.py'),
        modern_python='/usr/bin/python3',legacy_python='/workspace/legacy37/bin/python',actual_GPU_compatibility_performed=False)
    files['EXPECTED_AUTHOR_INPUTS.json']=json.dumps(expected,indent=2).encode()
    hashes={n:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in files.items()}
    files['TOOLS_SHA256.json']=json.dumps(dict(format='pinn4soh-transfer-robustness-cloud-v1',files=hashes,deployable=not allow_unaccepted),indent=2).encode()
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED) as z:
        for n,b in files.items():z.writestr(n,b)
    with zipfile.ZipFile(output) as z:require(not z.testzip() and set(z.namelist())==set(files) and all(z.read(n)==b for n,b in files.items()),'Package roundtrip differs')
    record=dict(filename=output.name,bytes=output.stat().st_size,sha256=sha(output),files=hashes,deployable=not allow_unaccepted,
        frozen_protocol_unchanged=True,frozen_runner_unchanged=True,py37_worker_import_tree_AST=True,cloud_accessed=False,
        formal_target_training_started=False,actual_GPU_compatibility_performed=False)
    output.with_suffix('.manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8');return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-unaccepted',action='store_true',help='Fixture-only development package, never deploy')
    print(json.dumps({k:v for k,v in package(**vars(p.parse_args())).items() if k!='files'},indent=2))
