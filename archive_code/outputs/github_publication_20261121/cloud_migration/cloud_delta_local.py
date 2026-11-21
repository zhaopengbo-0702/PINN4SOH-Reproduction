"""Plan missing cloud content, then verify or safely restore the combined archive."""
import argparse, hashlib, json, os, re, stat, tempfile, zipfile
from pathlib import Path, PurePosixPath

HASH = re.compile(r'^[0-9a-f]{64}$')
REMOVED = {'user_discarded_optional_experiments', 'python_cache'}
PART = re.compile(r'^(?:full-cas|cloud-delta|cloud-needed)-part-\d+\.zip$')


def require(v, text):
    if not v: raise ValueError(text)


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''): h.update(b)
    return h.hexdigest()


def pairs(rows):
    result = {}
    for k, v in rows:
        require(k not in result, 'Duplicate JSON key: ' + k); result[k] = v
    return result


def load(p):
    with Path(p).open(encoding='utf-8-sig') as f: return json.load(f, object_pairs_hook=pairs)


def relative(v):
    require(isinstance(v, str) and v and '\\' not in v and '\0' not in v, 'Unsafe relative path')
    p = PurePosixPath(v)
    require(not p.is_absolute() and str(p) == v and all(s not in ('', '.', '..') for s in p.parts), 'Noncanonical path: ' + v)
    for s in p.parts:
        require(not any(c in s for c in ':<>"|?*') and not s.endswith((' ', '.')) and
                not re.fullmatch(r'(?i)(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?', s), 'Unsafe path component: ' + s)
    return p


def no_links(p):
    while True:
        if p.exists() or p.is_symlink():
            s = p.lstat()
            require(not stat.S_ISLNK(s.st_mode) and not (getattr(s, 'st_file_attributes', 0) & 0x400), 'Symlink/reparse ancestor: ' + str(p))
        if p.parent == p: return
        p = p.parent


def long(p):
    p = Path(os.path.abspath(str(p)))
    return Path('\\\\?\\' + str(p)) if os.name == 'nt' and not str(p).startswith('\\\\?\\') else p


def write(p, v):
    p = Path(p); no_links(p)
    require(not p.exists(), 'Refusing overwrite: ' + str(p))
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f: json.dump(v, f, ensure_ascii=False, indent=2); f.write('\n')


def validated(cloud_path, local_path):
    c, l = load(cloud_path), load(local_path)
    require(c['format'] == 'pinn4soh-cloud-delta-v1' and c['status'] == 'passed_read_only_inventory_delta_backup'
            and not c['errors'] and c['delta_files_reassembled_verified'], 'Cloud inventory incomplete')
    require(l['format'] == 'pinn4soh-local-cas-v1', 'Local CAS format differs')
    files, seen, case = {}, set(), set()
    for row in l['files']:
        relative(row['path']); require(HASH.fullmatch(row['sha256']) and type(row['bytes']) is int, 'Invalid local row')
        files.setdefault(row['sha256'], row)
    for row in c['files']:
        path = row['path']; relative(path)
        require(path not in seen, 'Duplicate cloud path: ' + path)
        seen.add(path); case.add(path.casefold())
        require(row['type'] in ('file', 'symlink') and row['status'] not in ('error', 'not_archived_special_file'), 'Unsupported cloud row')
        if row['type'] == 'file':
            require(HASH.fullmatch(row['sha256']) and type(row['bytes']) is int and row['bytes'] >= 0, 'Invalid cloud file')
            if row['status'] == 'cloud_delta_archived':
                require(sum(x['bytes'] for x in row['chunks']) == row['bytes'], 'Cloud chunks size differs')
                for chunk in row['chunks']:
                    require(HASH.fullmatch(chunk['sha256']) and chunk['bytes'] <= 64 * 1024**2
                            and c['blobs'][chunk['sha256']]['bytes'] == chunk['bytes'], 'Invalid cloud chunk')
        else:
            require(isinstance(row['target'], str) and '\0' not in row['target'], 'Invalid symlink target')
    return c, l, files


def plan(cloud_path, local_path, output):
    c, l, files = validated(cloud_path, local_path)
    requests, missing_matched, counts = {}, [], {}
    for row in c['files']:
        if row['type'] != 'file' or row['status'] in REMOVED: continue
        h = row['sha256']
        if h in files:
            require(files[h]['bytes'] == row['bytes'], 'Same SHA different size'); key = 'whole_file_in_local_CAS'
        elif row['status'] == 'cloud_delta_archived':
            key = 'resolved_by_local_chunks'; offset = 0
            for chunk in row['chunks']:
                bh = chunk['sha256']
                if bh in l['blobs']:
                    require(l['blobs'][bh]['bytes'] == chunk['bytes'], 'Same chunk SHA different size')
                else:
                    key = 'needs_new_cloud_chunks'
                    requests.setdefault(bh, dict(sha256=bh, bytes=chunk['bytes'], path=row['path'], offset=offset,
                                                source_file_sha256=h, source_file_bytes=row['bytes']))
                offset += chunk['bytes']
        else:
            key = 'missing_previously_matched_file'; missing_matched.append(row)
        counts[key] = counts.get(key, 0) + 1
    require(not missing_matched, 'Previously matched cloud files missing from local CAS: ' + str(missing_matched[:3]))
    obj = dict(format='pinn4soh-cloud-needed-chunks-v1', cloud_index_sha256=sha(Path(cloud_path)),
               local_index_sha256=sha(Path(local_path)), counts=counts,
               requests=sorted(requests.values(), key=lambda r: (r['path'], r['offset'])),
               needed_unique_bytes=sum(r['bytes'] for r in requests.values()),
               old_cloud_delta_bytes=sum(p['bytes'] for p in c['parts']),
               excluded_counts={s:sum(r['status'] == s for r in c['files']) for s in REMOVED},
               symlink_metadata_count=sum(r['type'] == 'symlink' for r in c['files']),
               mode_and_empty_directory_metadata_available=False,
               scope='File bytes and symlink targets; Linux executable modes and empty directories were not recorded.')
    write(output, obj)
    print(json.dumps({k:v for k,v in obj.items() if k != 'requests'}, ensure_ascii=True))


class Store:
    def __init__(self, index, root, part_key):
        self.idx, self.root, self.parts, self.opened = index, long(root), {p['name']:p for p in index[part_key]}, {}
    def stream(self, h):
        b = self.idx['blobs'][h]; name = b['archive']
        require(PART.fullmatch(name) and b['member'] == 'blobs/' + h, 'Unsafe archive/member name')
        if name not in self.opened:
            p = self.root / name; no_links(p); row = self.parts[name]
            require(p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], 'Archive size/SHA failed: ' + name)
            z = zipfile.ZipFile(p); names = set()
            for info in z.infolist():
                require(info.filename not in names and not info.is_dir() and not info.flag_bits & 1
                        and not stat.S_ISLNK(info.external_attr >> 16), 'Invalid ZIP member')
                bh = info.filename.removeprefix('blobs/')
                require(HASH.fullmatch(bh) and bh in self.idx['blobs'] and self.idx['blobs'][bh]['archive'] == name
                        and info.file_size == self.idx['blobs'][bh]['bytes'], 'Unexpected ZIP member')
                names.add(info.filename)
            expected = {v['member'] for v in self.idx['blobs'].values() if v['archive'] == name}
            require(names == expected, 'ZIP members differ'); self.opened[name] = z
        full, size = hashlib.sha256(), 0
        with self.opened[name].open(b['member']) as f:
            for data in iter(lambda:f.read(4 * 1024**2), b''):
                full.update(data);size += len(data);yield data
        require(full.hexdigest() == h and size == b['bytes'], 'Blob size/SHA failed')
    def close(self):
        for z in self.opened.values():z.close()


def recover(x):
    c,l,files = validated(x.cloud_index,x.local_index)
    local = Store(l,x.local_parts,'archive_parts')
    extra = None
    if x.needed_index:
        n = load(x.needed_index)
        require(n['format']=='pinn4soh-cloud-needed-archive-v1' and n['status']=='verified'
                and n['cloud_index_sha256']==sha(Path(x.cloud_index)) and n['local_index_sha256']==sha(Path(x.local_index)), 'Needed archive binding differs')
        extra=Store(n,x.needed_parts,'parts')
    elif x.cloud_parts:
        extra=Store(c,x.cloud_parts,'parts')
    dest=long(x.output) if x.output else None
    if dest:
        if os.name == 'nt':
            lowered = [r['path'].casefold() for r in c['files'] if r['status'] not in REMOVED]
            require(len(lowered) == len(set(lowered)), 'Cloud contains Linux case-distinct paths; restore on case-sensitive Linux, or verify only on Windows')
        no_links(dest);require(not dest.exists(), 'Restore target must be a fresh directory');dest.mkdir(parents=True)
    receipt=dict(format='pinn4soh-combined-cloud-coverage-v1',mode='restore' if dest else 'verify-only',
                 cloud_index_sha256=sha(Path(x.cloud_index)),local_index_sha256=sha(Path(x.local_index)),
                 files_verified=0,bytes_verified=0,symlinks_metadata=[],excluded=[],source_files_modified=False,
                 symlinks_created=False,executable_mode_or_empty_directory_restore_claimed=False)
    try:
        for row in c['files']:
            if row['status'] in REMOVED:
                receipt['excluded'].append(dict(path=row['path'],status=row['status']));continue
            if row['type']=='symlink':
                receipt['symlinks_metadata'].append(dict(path=row['path'],target=row['target']));continue
            require(row['type']=='file','Unsupported row')
            h=row['sha256']; ref=files.get(h)
            chunks=ref['chunks'] if ref else row.get('chunks')
            require(chunks is not None,'Missing file bytes: '+row['path'])
            target=dest.joinpath(*relative(row['path']).parts) if dest else None
            stream=None;tmp=None
            if target:
                no_links(target);target.parent.mkdir(parents=True,exist_ok=True);no_links(target.parent)
                fd,name=tempfile.mkstemp(prefix='.restore-',dir=str(target.parent));tmp=Path(name);stream=os.fdopen(fd,'wb')
            try:
                full,size=hashlib.sha256(),0
                for chunk in chunks:
                    bh=chunk['sha256'];store=local if bh in l['blobs'] else extra
                    require(store and bh in store.idx['blobs'],'Missing blob: '+bh)
                    for data in store.stream(bh):
                        full.update(data);size+=len(data)
                        if stream:stream.write(data)
                require(size==row['bytes'] and full.hexdigest()==h,'Cloud file reconstruction differs: '+row['path'])
                if stream:
                    stream.flush();os.fsync(stream.fileno());stream.close();stream=None
                    no_links(target);require(not target.exists(),'Destination appeared');os.link(tmp,target);tmp.unlink();tmp=None
                receipt['files_verified']+=1;receipt['bytes_verified']+=size
            finally:
                if stream:stream.close()
                if tmp and tmp.exists():tmp.unlink()
        receipt.update(status='passed_all_retained_cloud_file_bytes',archives_read=len(local.opened)+(len(extra.opened) if extra else 0),
                       retained_symlink_targets_recorded=len(receipt['symlinks_metadata']),
                       limitations=['POSIX modes and empty directories were not recorded by source inventory.',
                                    'Symlinks are metadata only, never created or followed.',
                                    'User-discarded experiments and Python cache remain excluded.'])
        write(x.receipt,receipt)
        print(json.dumps({k:v for k,v in receipt.items() if k not in ('excluded','symlinks_metadata')},ensure_ascii=True))
    finally:
        local.close()
        if extra:extra.close()


def main():
    p=argparse.ArgumentParser(__doc__);s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('plan');a.add_argument('--cloud-index',required=True);a.add_argument('--local-index',required=True);a.add_argument('--output',required=True)
    a=s.add_parser('verify');a.add_argument('--cloud-index',required=True);a.add_argument('--local-index',required=True)
    a.add_argument('--local-parts',required=True);a.add_argument('--cloud-parts');a.add_argument('--needed-index');a.add_argument('--needed-parts');a.add_argument('--output');a.add_argument('--receipt',required=True)
    x=p.parse_args()
    if x.command=='plan':plan(x.cloud_index,x.local_index,x.output)
    else:recover(x)


if __name__=='__main__':main()
