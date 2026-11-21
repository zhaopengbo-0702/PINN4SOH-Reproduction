"""Convert a verified full-CAS index into a compact cloud deduplication lookup."""
import argparse, hashlib, json
from pathlib import Path

p = argparse.ArgumentParser(__doc__)
p.add_argument('--index', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
x = p.parse_args()
raw = x.index.read_bytes(); source = json.loads(raw)
assert source['format'] == 'pinn4soh-local-cas-v1' and source['files']
known = {}
for row in source['files']:
    h, size = row['sha256'], row['bytes']
    assert len(h) == 64 and size >= 0
    assert h not in known or known[h] == size
    known[h] = size
obj = dict(format='pinn4soh-known-local-sha256-v1',
           basis=dict(index=str(x.index), index_sha256=hashlib.sha256(raw).hexdigest(),
                      status=source.get('status'), source_file_count=len(source['files']),
                      requirement='Only upload after the local full-CAS index and archive independently verify.'),
           sha256_to_bytes=known)
x.output.parent.mkdir(parents=True, exist_ok=True)
assert not x.output.exists()
x.output.write_text(json.dumps(obj, ensure_ascii=True, separators=(',', ':')) + '\n', encoding='utf-8')
print(json.dumps(dict(files=len(source['files']), unique=len(known), bytes=x.output.stat().st_size,
                     sha256=hashlib.sha256(x.output.read_bytes()).hexdigest())))
