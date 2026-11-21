"""Build reviewed experiment evidence assets without editing source evidence."""
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tarfile
from datetime import datetime, timezone

BASE = Path('\\\\?\\' + os.getcwd())
FULL = BASE / 'outputs/full_reproduction'
OUT = BASE / 'outputs/github_publication_20261121/release_assets'
OUT.mkdir(parents=True, exist_ok=True)

PACKAGES = [
    ('pinn4soh-main-2120', '主实验2120项：常规440、小样本1120、补充target-only80、迁移480。完成训练不等于所有论文结论一致。', [
        ('workspace/pinn4soh_full_paper_stage1', FULL / 'cloud_emergency_recovered/workspace/pinn4soh_full_paper_stage1'),
        ('workspace/pinn4soh_full_paper_stage2', FULL / 'cloud_emergency_recovered/workspace/pinn4soh_full_paper_stage2'),
        ('workspace/full_report_tools', FULL / 'cloud_emergency_recovered/workspace/full_report_tools'),
        ('workspace/full_supplement_tools_v1', FULL / 'cloud_emergency_recovered/workspace/full_supplement_tools_v1'),
    ]),
    ('pinn4soh-current-defaults110-and-mit40', '作者当前修正版默认110项、MIT物理身份敏感性40项、现代栈正式CPU计时。默认110与主2120使用不同配置，不能混为同一结论。', [
        ('workspace/pinn4soh_main_20261109', FULL / 'cloud_emergency_recovered/workspace/pinn4soh_main_20261109'),
        ('workspace/pinn4soh_mit_physical_purity_sensitivity_20261110', FULL / 'cloud_emergency_recovered/workspace/pinn4soh_mit_physical_purity_sensitivity_20261110'),
        ('workspace/efficiency_formal_20261110', FULL / 'cloud_emergency_recovered/workspace/efficiency_formal_20261110'),
    ]),
    ('pinn4soh-legacy-controls-and-seed110', '旧栈正式40项对照、身份审计与补充；扩展99项新训练加11项已完成repeat1复用构成110配对，不是110项新训练。现代参考原件随对应恢复目录保留。', [
        ('extension', FULL / 'legacy_extension_recovered_20261111/restored/extracted'),
        ('control40/pinn4soh_legacy_formal_control_20261110', FULL / 'legacy_overlay_recovered_20261110/extracted/pinn4soh_legacy_formal_control_20261110'),
        ('control40/legacy_independent_overlay_20261110', FULL / 'legacy_overlay_recovered_20261110/extracted/legacy_independent_overlay_20261110'),
    ]),
    ('pinn4soh-replacement-search192', 'XJTU/2C透明替代搜索192项候选训练及3项选定测试。使用新split和3个seed，选中β=0；不能称恢复作者历史搜索或证明单调项贡献。', [
        ('replacement_search', FULL / 'replacement_search_recovered_20261111/restored/extracted'),
    ]),
]

SECRET_PATTERNS = [
    ('github_token', re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,})\b')),
    ('private_key', re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')),
    ('api_key', re.compile(rb'\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}\b')),
    ('credential_assignment', re.compile(rb'(?i)(?:["\x27]?(?:password|passwd|access_token|refresh_token|authorization|cookie)["\x27]?\s*[:=]\s*["\x27])[^"\x27\r\n]{8,}')),
]
TEXT_EXT = {'.json', '.jsonl', '.log', '.txt', '.csv', '.md', '.py', '.ps1', '.yaml', '.yml', '.ini', '.cfg'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def jbytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def run():
    catalog = {'format': 'pinn4soh-public-release-assets-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
               'date': '2026-11-21', 'cloud_only_stage_a59_included': False,
               'scope_note': '仅本地已持有的训练证据；未包含第三方原始数据、依赖安装包、云登录截图、私有凭据或未下载的59项追加实验。',
               'packages': []}
    for name, summary, trees in PACKAGES:
        archive = OUT / (name + '.tar.gz')
        assert not archive.exists(), 'Do not overwrite an existing release asset: ' + name
        entries = []
        omitted = []
        sensitive = []
        for prefix, source in trees:
            assert source.is_dir(), str(source)
            for parent, dirs, files in os.walk(source):
                dirs[:] = sorted(d for d in dirs if d not in {'__pycache__', '.git', 'node_modules'})
                for filename in sorted(files):
                    path = Path(parent) / filename
                    relative = path.relative_to(source).as_posix()
                    if filename.endswith(('.lock', '.pyc', '.tmp')) or filename in {'.parallel.lock', '.full_suite.lock'}:
                        omitted.append(prefix + '/' + relative)
                        continue
                    arcname = prefix + '/' + relative
                    assert not path.is_symlink()
                    if path.suffix.lower() in TEXT_EXT:
                        data = path.read_bytes()
                        for label, pattern in SECRET_PATTERNS:
                            if pattern.search(data):
                                sensitive.append({'path': arcname, 'pattern': label})
                    entries.append((arcname, path, path.stat().st_size, digest(path)))
        assert not sensitive, json.dumps(sensitive, ensure_ascii=True)
        entries.sort(key=lambda e: e[0])
        assert len({e[0] for e in entries}) == len(entries)
        manifest = {'format': 'pinn4soh-release-member-sha256-v1', 'package': name, 'summary': summary,
                    'scope': catalog['scope_note'], 'source_files_unmodified': True,
                    'original_absolute_paths_inside_metadata_preserved': True,
                    'omitted_transient_runtime_markers': omitted,
                    'files': [{'path': n, 'bytes': s, 'sha256': h} for n, p, s, h in entries]}
        manifest_data = jbytes(manifest)
        readme = ('# 训练证据下载包\n\n' + summary + '\n\n' + catalog['scope_note'] +
                  '\n\n目录中的模型、预测/标签数组、日志、配置和manifest均保持原始字节。'
                  '原元数据中的/workspace路径是历史训练路径，不表示本次解压路径。'
                  '运行标记锁、缓存及临时文件未打包；科学数据不因此改变。'
                  '\n\n`RELEASE_MEMBER_SHA256.json`列出每份原文件的大小与SHA256。'
                  '解压后先核此清单，再使用仓库中的对应审计脚本。'
                  '本包不自带第三方原始数据或完整运行环境；数据来源与代码见GitHub中文使用指南。\n').encode('utf-8')
        with tarfile.open(archive, 'w:gz', compresslevel=6, format=tarfile.PAX_FORMAT) as tar:
            for arcname, path, size, checksum in entries:
                tar.add(str(path), arcname=arcname, recursive=False)
            for arcname, data in [('RELEASE_MEMBER_SHA256.json', manifest_data), ('README_下载包说明.md', readme)]:
                info = tarfile.TarInfo(arcname); info.size = len(data); info.mtime = 0
                tar.addfile(info, io.BytesIO(data))
        expected = {n: (s, h) for n, p, s, h in entries}
        expected.update({'RELEASE_MEMBER_SHA256.json': (len(manifest_data), hashlib.sha256(manifest_data).hexdigest()),
                         'README_下载包说明.md': (len(readme), hashlib.sha256(readme).hexdigest())})
        observed = set()
        with tarfile.open(archive, 'r:gz') as tar:
            for member in tar:
                assert member.isfile() and member.name in expected and member.name not in observed
                data = tar.extractfile(member).read()
                assert (len(data), hashlib.sha256(data).hexdigest()) == expected[member.name], member.name
                observed.add(member.name)
        assert observed == set(expected)
        size = archive.stat().st_size
        assert size < 2 * 1024**3
        item = {'filename': archive.name, 'summary': summary, 'bytes': size, 'sha256': digest(archive),
                'source_file_count': len(entries), 'archive_file_members': len(observed),
                'source_uncompressed_bytes': sum(e[2] for e in entries), 'member_roundtrip_sha256_verified': True,
                'sensitive_text_scan_passed': True, 'omitted_transient_files': len(omitted),
                'source_trees': [{'archive_prefix': p, 'workspace_relative_source': str(s.relative_to(BASE)).replace('\\', '/')} for p, s in trees]}
        (OUT / (name + '.members.sha256.json')).write_bytes(manifest_data)
        (OUT / (name + '.verification.json')).write_bytes(jbytes(item))
        catalog['packages'].append(item)
        (OUT / 'RELEASE_ASSETS_CATALOG.json').write_bytes(jbytes(catalog))
        print(json.dumps(item, ensure_ascii=True), flush=True)
    lines = [p['sha256'] + '  ' + p['filename'] for p in catalog['packages']]
    (OUT / 'SHA256SUMS.txt').write_text('\n'.join(lines) + '\n', encoding='ascii')
    print('ALL_RELEASE_ASSETS_VERIFIED', flush=True)


if __name__ == '__main__':
    run()
