"""Scan all eligible text bytes using overlap, emitting only paths and rule names."""
from pathlib import Path
import os
import json
import hashlib
import importlib.util

ROOT = Path('\\\\?\\' + os.getcwd())
PUB = ROOT / 'outputs/github_publication_20261121'
spec = importlib.util.spec_from_file_location('cas_builder', str(PUB / 'build_full_cas_archive.py'))
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
patterns = builder.SECRETS + [
    ('bearer_token', __import__('re').compile(rb'(?i)\bBearer\s+[A-Za-z0-9_.-]{20,}')),
    ('session_assignment', __import__('re').compile(rb'(?i)["\x27]?(?:sessionid|session_id|sessionToken|JSESSIONID)["\x27]?\s*[:=]\s*["\x27][A-Za-z0-9_./+=-]{16,}["\x27]')),
]
hits = []
scanned_files = 0
scanned_bytes = 0
for parent, dirs, files in os.walk(ROOT):
    dirs[:] = [d for d in dirs if not builder.excluded_reason((Path(parent) / d).relative_to(ROOT).as_posix(), True)]
    for name in files:
        path = Path(parent) / name
        rel = path.relative_to(ROOT).as_posix()
        if builder.excluded_reason(rel) or path.suffix.lower() not in builder.TEXT_EXT:
            continue
        scanned_files += 1
        overlap = b''
        file_rules = set()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(4 * 1024**2), b''):
                scanned_bytes += len(block)
                candidate = overlap + block
                for label, pattern in patterns:
                    if pattern.search(candidate):
                        file_rules.add(label)
                overlap = candidate[-2048:]
        if file_rules:
            hits.append({'path': rel, 'rules': sorted(file_rules)})
report = {'format': 'full-cas-source-text-secret-scan-v1', 'all_eligible_text_sizes_scanned': True,
          'scanned_files': scanned_files, 'scanned_bytes': scanned_bytes, 'matches': hits,
          'passed_automatic_rules': not hits,
          'note': '仅输出路径和规则，无凭据值。命中不自动判断为秘密，需人工查看上下文；二进制科学数据不按文本扫描。'}
(PUB / 'full_archive/FULL_SOURCE_TEXT_SECRET_SCAN.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report, ensure_ascii=True), flush=True)
