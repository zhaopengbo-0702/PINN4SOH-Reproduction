"""Read-only cloud inventory and verified, deduplicated delta backup; never delete."""
import argparse, hashlib, json, os, stat, zipfile
from pathlib import Path
from datetime import datetime, timezone


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def write(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    a = argparse.ArgumentParser(__doc__)
    a.add_argument('--root', type=Path, required=True)
    a.add_argument('--known-local', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--max-mib', type=int, default=600)
    a.add_argument('--runtime-policy', choices=['archive', 'index-only'], default='archive')
    x = a.parse_args()
    root, out = x.root.resolve(strict=True), x.output.resolve()
    assert root.is_dir() and out != root and not out.exists()
    assert 70 <= x.max_mib <= 1900
    known = json.loads(x.known_local.read_text(encoding='utf-8'))
    assert known['format'] == 'pinn4soh-known-local-sha256-v1'
    local = known['sha256_to_bytes']
    assert all(len(k) == 64 and isinstance(v, int) and v >= 0 for k, v in local.items())
    out.mkdir(parents=True)
    idx = dict(format='pinn4soh-cloud-delta-v1', created_utc=datetime.now(timezone.utc).isoformat(),
               root=str(root), source_files_modified=False, deletion_performed=False,
               known_local_index_sha256=digest(x.known_local), known_local_basis=known.get('basis'),
               runtime_policy=x.runtime_policy, chunk_bytes=64 * 1024**2,
               max_part_bytes=x.max_mib * 1024**2, files=[], blobs={}, parts=[], errors=[], top_level={})
    removed = ('stage_a', 'stage-a', 'stagea', 'confirmation', 'transfer_robustness', 'transfer-robustness', 'deadline')
    runtimes = {'legacy37', 'site-packages', 'offline', 'legacy_offline', 'legacy_runtime'}
    paths = []
    # Never follow symbolic links. Every non-output entry is indexed, including discarded/cache entries.
    for parent, dirs, files in os.walk(root, followlinks=False):
        dirs.sort(); files.sort()
        for name in list(dirs):
            p = Path(parent) / name
            if p == out:
                dirs.remove(name); continue
            if p.is_symlink():
                paths.append(p); dirs.remove(name)
        paths.extend(Path(parent) / name for name in files)
    paths.sort(key=lambda p: p.relative_to(root).as_posix())
    z = None
    payload = 0

    def close():
        nonlocal z, payload
        if z:
            name = Path(z.filename); z.close()
            assert name.stat().st_size <= idx['max_part_bytes']
            with zipfile.ZipFile(name) as check:
                for info in check.infolist():
                    h = hashlib.sha256()
                    with check.open(info) as f:
                        for b in iter(lambda: f.read(4 * 1024**2), b''): h.update(b)
                    assert h.hexdigest() == info.filename.split('/')[-1]
            idx['parts'].append(dict(name=name.name, bytes=name.stat().st_size, sha256=digest(name), members_verified=True))
            write(out / 'PACK_PROGRESS.json', dict(files=len(idx['files']), parts=idx['parts']))
            z = None; payload = 0

    def blob(b):
        nonlocal z, payload
        h = hashlib.sha256(b).hexdigest()
        if h not in idx['blobs']:
            if z is None or payload + len(b) + 1024 > idx['max_part_bytes'] - 1024**2:
                close()
                path = out / ('cloud-delta-part-%03d.zip' % (len(idx['parts']) + 1))
                z = zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_STORED, allowZip64=True)
            name = 'blobs/' + h
            z.writestr(name, b)
            payload += len(b) + 1024
            idx['blobs'][h] = dict(bytes=len(b), archive=Path(z.filename).name, member=name)
        return dict(bytes=len(b), sha256=h)

    for n, p in enumerate(paths, 1):
        rel = p.relative_to(root).as_posix()
        row = dict(path=rel)
        try:
            before = p.lstat()
            top = rel.split('/')[0]
            tally = idx['top_level'].setdefault(top, dict(files=0, bytes=0, symlinks=0))
            tally['files'] += 1
            lower = rel.lower()
            reason = ('user_discarded_optional_experiments' if any(k in lower for k in removed)
                      else 'python_cache' if '__pycache__' in p.parts or p.suffix in {'.pyc', '.pyo'}
                      else 'runtime_index_only_explicit_policy' if top in runtimes and x.runtime_policy == 'index-only'
                      else None)
            if stat.S_ISLNK(before.st_mode):
                row.update(type='symlink', target=os.readlink(p), status=reason or 'symlink_metadata_preserved')
                tally['symlinks'] += 1
            elif stat.S_ISREG(before.st_mode):
                size, h = before.st_size, digest(p)
                after = p.stat()
                assert (size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'changed while hashing'
                row.update(type='file', bytes=size, sha256=h)
                tally['bytes'] += size
                if reason:
                    row['status'] = reason
                elif local.get(h) == size:
                    row['status'] = 'matched_verified_local_sha256'
                else:
                    row.update(status='cloud_delta_archived', chunks=[])
                    full = hashlib.sha256()
                    with p.open('rb') as f:
                        for b in iter(lambda: f.read(idx['chunk_bytes']), b''):
                            full.update(b); row['chunks'].append(blob(b))
                    after = p.stat()
                    assert full.hexdigest() == h and (size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'changed while packing'
            else:
                row.update(type='special', status='not_archived_special_file', mode=before.st_mode)
                idx['errors'].append(dict(path=rel, error='unsupported special file'))
        except Exception as e:
            row['status'] = 'error'; row['error'] = str(e)
            idx['errors'].append(dict(path=rel, error=str(e)))
        idx['files'].append(row)
        if n % 2000 == 0: print('PROGRESS', n, len(paths), len(idx['parts']), flush=True)
    close()
    # Independently reassemble every delta file from ZIP blobs; full-file SHA must match source inventory.
    opened = {}
    try:
        for row in idx['files']:
            if row['status'] != 'cloud_delta_archived': continue
            h = hashlib.sha256(); size = 0
            for chunk in row['chunks']:
                b = idx['blobs'][chunk['sha256']]
                if b['archive'] not in opened: opened[b['archive']] = zipfile.ZipFile(out / b['archive'])
                with opened[b['archive']].open(b['member']) as f:
                    for data in iter(lambda: f.read(4 * 1024**2), b''): h.update(data); size += len(data)
            assert h.hexdigest() == row['sha256'] and size == row['bytes'], row['path']
    finally:
        for f in opened.values(): f.close()
    counts = {}
    for row in idx['files']: counts[row['status']] = counts.get(row['status'], 0) + 1
    idx.update(status='passed_read_only_inventory_delta_backup' if not idx['errors'] else 'incomplete_with_errors',
               counts=counts, delta_files_reassembled_verified=True,
               known_local_files_not_embedded=True, output_subtree_excluded=str(out),
               limitations=['Matched local bytes depend on uploaded verified local index.',
                            'Discarded experiments/cache are indexed but not archived.',
                            'Symlink targets are preserved as metadata; links are not followed.',
                            'No source deletion or modification performed.'])
    write(out / 'CLOUD_INVENTORY.json', idx)
    (out / 'SHA256SUMS.txt').write_text(''.join(p['sha256'] + '  ' + p['name'] + '\n' for p in idx['parts'])
        + digest(out / 'CLOUD_INVENTORY.json') + '  CLOUD_INVENTORY.json\n', encoding='ascii')
    print(json.dumps(dict(status=idx['status'], files=len(idx['files']), counts=counts, parts=len(idx['parts']),
                         index_sha256=digest(out / 'CLOUD_INVENTORY.json')), ensure_ascii=True), flush=True)
    if idx['errors']: raise SystemExit(2)


if __name__ == '__main__': main()
