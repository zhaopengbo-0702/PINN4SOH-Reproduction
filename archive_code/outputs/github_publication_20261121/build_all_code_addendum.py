"""Archive every code/notebook and related configuration, including withdrawn extras."""
from pathlib import Path
import os
import json
import hashlib
import zipfile
import importlib.util
from datetime import datetime, timezone

ROOT = Path('\\\\?\\' + os.getcwd())
PUB = ROOT / 'outputs/github_publication_20261121'
OUT = PUB / 'full_archive'
CODE_EXTS = {'.py', '.ps1', '.sh', '.bat', '.cmd', '.js', '.ts', '.ipynb', '.R', '.r'}
CONFIG_EXTS = {'.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.patch', '.md', '.txt'}
SKIP_DIRS = {'.git', '__pycache__', '.pytest_cache', 'node_modules', 'wheels', '.venv', 'venv', 'site-packages'}
spec = importlib.util.spec_from_file_location('cas_build', str(PUB / 'build_full_cas_archive.py'))
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def run():
    archive = OUT / 'pinn4soh-all-code-and-publication-tools.zip'
    assert not archive.exists(), 'Existing code addendum; do not overwrite'
    index = json.loads((OUT / 'index.json').read_text(encoding='utf-8'))
    archived_code = {f['path']: f['sha256'] for f in index['files'] if Path(f['path']).suffix in CODE_EXTS}
    code_paths = []
    nearby = set()
    for parent, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            path = Path(parent) / name
            rel = path.relative_to(ROOT).as_posix()
            if path.suffix in CODE_EXTS:
                code_paths.append((rel, path))
                nearby.add(path.parent)
    selected = {rel: path for rel, path in code_paths}
    for folder in nearby:
        for path in folder.iterdir():
            if path.is_file() and path.suffix.lower() in CONFIG_EXTS and path.stat().st_size <= 40 * 1024**2:
                rel = path.relative_to(ROOT).as_posix()
                # Intermediate upload command files only contain source text/base64 tooling;
                # full training outputs are covered by CAS or intentionally withdrawn.
                selected[rel] = path
        reg = folder / 'registration'
        if reg.is_dir():
            for path in reg.rglob('*'):
                if path.is_file() and path.suffix.lower() in CONFIG_EXTS:
                    selected[path.relative_to(ROOT).as_posix()] = path
    rows = []
    matches = []
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for rel, path in sorted(selected.items()):
            data = path.read_bytes()
            for label, pattern in builder.SECRETS:
                if pattern.search(data):
                    matches.append({'path': rel, 'rule': label})
            z.writestr(rel, data)
            rows.append({'path': rel, 'bytes': len(data), 'sha256': sha(data),
                         'is_code_or_notebook': path.suffix in CODE_EXTS,
                         'already_in_main_CAS_identical': archived_code.get(rel) == sha(data) if path.suffix in CODE_EXTS else None})
        report = {'format': 'pinn4soh-all-code-coverage-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
                  'index_sha256': sha((OUT / 'index.json').read_bytes()), 'all_code_count': len(code_paths),
                  'all_code_paths': [rel for rel, p in sorted(code_paths)],
                  'includes_withdrawn_extra_code_but_not_training_outputs': True,
                  'publication_tools_included': True, 'source_files_modified': False,
                  'files': rows, 'sensitive_scan_matches': matches, 'sensitive_scan_passed': not matches,
                  'license_note': '第三方代码/数据/Notebook的权利与许可保持归原权利人，本归档不施加覆盖它们的统一许可。'}
        report_data = (json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
        z.writestr('ALL_CODE_COVERAGE.json', report_data)
    assert not matches, 'Credentials suspected; do not publish code addendum'
    expected = {r['path']: (r['bytes'], r['sha256']) for r in rows}
    with zipfile.ZipFile(archive, 'r') as z:
        assert len(z.namelist()) == len(expected) + 1
        for name in expected:
            data = z.read(name)
            assert (len(data), sha(data)) == expected[name], name
        assert z.read('ALL_CODE_COVERAGE.json') == report_data
    (OUT / 'ALL_CODE_COVERAGE.json').write_bytes(report_data)
    receipt = {'format': 'all-code-addendum-receipt-v1', 'archive': archive.name, 'bytes': archive.stat().st_size,
               'sha256': builder.sha_file(archive), 'code_files': len(code_paths), 'all_files': len(rows),
               'all_members_readback_sha256_passed': True, 'sensitive_scan_passed': True,
               'coverage_json_sha256': sha(report_data), 'publication_snapshot_at_utc': report['created_utc']}
    (OUT / 'ALL_CODE_ADDENDUM_VERIFIED.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    run()
