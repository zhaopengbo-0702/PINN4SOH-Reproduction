"""Package StageA runner and immutable pre-registration; no training or cloud access."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def package(root,output):
    root,output=Path(root).resolve(),Path(output).resolve();source=root/'reproduction';registration=root/'outputs/full_reproduction/stage_a9_preregistered_20261111'
    if output.exists() or not output.parent.is_dir():raise ValueError('Fresh package required')
    names=['search_stage_a.py','search_replacement_grid.py','run_extended.py','verify_stage_a_registration.py','collect_stage_a.py','run_stage_a_cloud.py',
        'package_stage_a_cloud.py','STAGE_A_RUN.md','STAGE_A_CLOUD.md','restore_backup_json.py','test_stage_a_cloud.py']
    files={name:(source/name).read_bytes() for name in names}
    for path in registration.iterdir():
        if path.is_file():files['registration/'+path.name]=path.read_bytes()
    accepted=json.loads((root/'outputs/full_reproduction/stage_a_validation_20261111/ACCEPTANCE.json').read_text())
    if sha(source/'search_stage_a.py')!=accepted['artifact_sha256']['reproduction/search_stage_a.py']:raise ValueError('Accepted StageA runner changed')
    cloud_qa=root/'outputs/full_reproduction/stage_a_cloud_validation_20261111_v3/COLLECTOR_ACCEPTANCE.json'
    checked=json.loads(cloud_qa.read_text())
    if checked['status']!='passed_collector_synthetic_and_real_smoke' or any(sha(root/name)!=checksum for name,checksum in checked['artifact_sha256'].items()):
        raise ValueError('Collector QA missing or changed since acceptance')
    files['LOCAL_RUNNER_ACCEPTANCE.json']=(root/'outputs/full_reproduction/stage_a_validation_20261111/ACCEPTANCE.json').read_bytes()
    files['LOCAL_COLLECTOR_ACCEPTANCE.json']=cloud_qa.read_bytes()
    for name,data in files.items():
        if name.endswith('.py'):ast.parse(data,filename=name,feature_version=(3,10))
    expected=dict(author_python_sha256={p.relative_to(root/'PINN4SOH').as_posix():sha(p) for p in sorted((root/'PINN4SOH').rglob('*.py'))},
        accepted_stage_a_validation_sha256=sha(root/'outputs/full_reproduction/stage_a_validation_20261111/ACCEPTANCE.json'),
        accepted_collector_validation_sha256=sha(cloud_qa),
        target='modern Python3.10 torch1.13 CUDA; old runtime not validated',selection_runs=162,selected_test_evaluations=18,
        future_confirmation_training_runs=60,future_confirmation_performed=False,old_runtime_confirmation_performed=False)
    files['EXPECTED_AUTHOR_INPUTS.json']=json.dumps(expected,indent=2).encode()
    hashes={name:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in files.items()}
    files['TOOLS_SHA256.json']=json.dumps(dict(format='pinn4soh-stagea-cloud-package-v1',files=hashes),indent=2).encode()
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED) as archive:
        for name,data in files.items():archive.writestr(name,data)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() or set(archive.namelist())!=set(files):raise ValueError('ZIP check failed')
        for name,data in files.items():
            if archive.read(name)!=data:raise ValueError('ZIP member bytes changed')
    record=dict(format='pinn4soh-stagea-cloud-package-evidence-v1',filename=output.name,bytes=output.stat().st_size,sha256=sha(output),files=hashes,
        accepted_runner_unchanged=True,python310_ast_passed=True,formal_selection_training_started=False,cloud_accessed=False)
    output.with_suffix('.manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--output',type=Path,required=True)
    result=package(**vars(parser.parse_args()));print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))
