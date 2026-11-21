"""Package only fixed ten-newseed confirmation; frozen Stage A sources unchanged."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package(root,output):
    root,output=Path(root).resolve(),Path(output).resolve();source=root/'reproduction'
    require=lambda v,t:None if v else (_ for _ in ()).throw(ValueError(t))
    require(not output.exists() and output.parent.is_dir(),'Fresh package required')
    original=root/'outputs/full_reproduction/stage-a-cloud-20261111-v3.zip'
    require(sha(original)=='740c64e14a8475dbc874a97d03678c3571f69ba0bdd1a27a5aa80b5002043ca6','Stage A v3 archive differs')
    with zipfile.ZipFile(original) as archive:
        manifest=json.loads(archive.read('TOOLS_SHA256.json'))
        frozen=['search_stage_a.py','search_replacement_grid.py','run_extended.py','collect_stage_a.py','verify_stage_a_registration.py','restore_backup_json.py']
        files={name:archive.read(name) for name in frozen}
        for name in archive.namelist():
            if name.startswith('registration/'):files[name]=archive.read(name)
    for name in frozen:
        require(hashlib.sha256(files[name]).hexdigest()==manifest['files'][name]['sha256'] and
            files[name]==(source/name).read_bytes(),'Frozen Stage A source differs: '+name)
    files['STAGE_A_V3_TOOLS_SHA256.json']=json.dumps(manifest,indent=2).encode()
    for name in ['run_stage_a_confirmation.py','collect_stage_a_confirmation.py','package_stage_a_confirmation.py',
                 'test_stage_a_confirmation.py','STAGE_A_CONFIRMATION.md']:
        files[name]=(source/name).read_bytes()
    qa=root/'outputs/full_reproduction/stage_a_confirmation_validation_20261111_v2/ACCEPTANCE.json'
    evidence=json.loads(qa.read_text())
    require(evidence['status']=='passed_functional_and_real_smoke' and not evidence['real_test_loaded'] and
        all(sha(root/name)==checksum for name,checksum in evidence['artifact_sha256'].items()),'Confirmation acceptance differs')
    files['LOCAL_CONFIRMATION_ACCEPTANCE.json']=qa.read_bytes()
    author={p.relative_to(root/'PINN4SOH').as_posix():sha(p) for p in sorted((root/'PINN4SOH').rglob('*.py'))}
    files['EXPECTED_AUTHOR_INPUTS.json']=json.dumps(dict(author_python_sha256=author,training_runs=60,fixed_test_evaluations=60,
        confirmation_seeds=list(range(2026111111,2026111121)),new_grid=False,old_runtime_performed=False,
        stage_a_v3_archive_sha256=sha(original),confirmation_acceptance_sha256=sha(qa)),indent=2).encode()
    for name,data in files.items():
        if name.endswith('.py'):ast.parse(data,filename=name,feature_version=(3,10))
    hashes={name:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in files.items()}
    files['TOOLS_SHA256.json']=json.dumps(dict(format='pinn4soh-stagea-fixed-confirmation-cloud-v1',files=hashes),indent=2).encode()
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED) as archive:
        for name,data in files.items():archive.writestr(name,data)
    with zipfile.ZipFile(output) as archive:
        require(not archive.testzip() and set(archive.namelist())==set(files) and all(archive.read(n)==b for n,b in files.items()),'ZIP roundtrip differs')
    record=dict(format='pinn4soh-stagea-fixed-confirmation-package-evidence-v1',filename=output.name,bytes=output.stat().st_size,
        sha256=sha(output),files=hashes,stage_a_worker_unchanged=True,python310_ast_passed=True,formal_confirmation_started=False,
        modern_runtime_only=True,training_runs=60,fixed_test_evaluations=60,old_runtime_training_performed=False)
    output.with_suffix('.manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,required=True)
    print(json.dumps({k:v for k,v in package(**vars(parser.parse_args())).items() if k!='files'},indent=2))
