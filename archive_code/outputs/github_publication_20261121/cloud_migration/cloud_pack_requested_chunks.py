"""Read only requested byte ranges; emit verified small delta, never delete."""
import argparse,hashlib,json,os,zipfile,stat,re
from pathlib import Path,PurePosixPath


def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()


def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--request',type=Path);p.add_argument('--inventory',type=Path);p.add_argument('--known-hashes',type=Path);p.add_argument('--local-index-sha');p.add_argument('--output',type=Path,required=True);p.add_argument('--max-mib',type=int,default=600);a=p.parse_args()
 if a.inventory:
  assert a.known_hashes and a.local_index_sha and not a.request
  c=json.loads(a.inventory.read_text());known=set(a.known_hashes.read_text().split());requests={}
  assert c['status']=='passed_read_only_inventory_delta_backup' and all(len(h)==64 for h in known)
  for row in c['files']:
   if row['type']!='file' or row['status'] in {'user_discarded_optional_experiments','python_cache'} or row['sha256'] in known:continue
   assert row['status']=='cloud_delta_archived'
   offset=0
   for b in row['chunks']:
    if b['sha256'] not in known:requests.setdefault(b['sha256'],dict(sha256=b['sha256'],bytes=b['bytes'],path=row['path'],offset=offset,source_file_sha256=row['sha256'],source_file_bytes=row['bytes']))
    offset+=b['bytes']
  q=dict(format='pinn4soh-cloud-needed-chunks-v1',cloud_index_sha256=sha(a.inventory),local_index_sha256=a.local_index_sha,requests=sorted(requests.values(),key=lambda r:(r['path'],r['offset'])))
  request_sha=hashlib.sha256(json.dumps(q,sort_keys=True).encode()).hexdigest()
 else:
  assert a.request;q=json.loads(a.request.read_text());request_sha=sha(a.request)
 r=a.root.resolve(strict=True);o=a.output.resolve();assert q['format']=='pinn4soh-cloud-needed-chunks-v1';assert not o.exists() and o!=r;assert 70<=a.max_mib<=1900;o.mkdir(parents=True)
 idx=dict(format='pinn4soh-cloud-needed-archive-v1',cloud_index_sha256=q['cloud_index_sha256'],local_index_sha256=q['local_index_sha256'],request_sha256=request_sha,blobs={},parts=[],source_files_modified=False,deletion_performed=False)
 patterns=[('token',re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|sk-(?:proj-)?[A-Za-z0-9_-]{24,})\b')),('private_key',re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')),('credential_assignment',re.compile(rb'(?i)(?:["\x27]?(?:password|passwd|access_token|refresh_token|authorization|cookie)["\x27]?\s*[:=]\s*["\x27])[^"\x27\r\n]{8,}'))]
 suffixes={'.py','.ps1','.md','.json','.jsonl','.txt','.log','.csv','.html','.js','.yaml','.yml','.ini','.cfg','.url','.pem','.key'}
 scan=dict(files=0,matched=[],bytes=0)
 z=None;payload=0;checked={}
 def close():
  nonlocal z,payload
  if not z:return
  f=Path(z.filename);z.close();assert f.stat().st_size<=a.max_mib*1024**2
  with zipfile.ZipFile(f) as t:
   for m in t.infolist():assert hashlib.sha256(t.read(m)).hexdigest()==m.filename.split('/')[-1]
  idx['parts'].append(dict(name=f.name,bytes=f.stat().st_size,sha256=sha(f),members_verified=True));z=None;payload=0
 for row in q['requests']:
  v=row['path'];rel=PurePosixPath(v);assert not rel.is_absolute() and str(rel)==v and '\\' not in v and '..' not in rel.parts
  f=r.joinpath(*rel.parts);assert f.is_file();assert all(not t.is_symlink() for t in [f,*f.parents] if t!=r.parent)
  st=f.stat();assert st.st_size==row['source_file_bytes']
  if v not in checked:
   assert sha(f)==row['source_file_sha256'];checked[v]=(st.st_size,st.st_mtime_ns)
   if f.suffix.lower() in suffixes and st.st_size<=64*1024**2:
    data=f.read_bytes();scan['files']+=1;scan['bytes']+=len(data)
    for label,pattern in patterns:
     if pattern.search(data):scan['matched'].append(dict(path=v,pattern=label))
  assert checked[v]==(st.st_size,st.st_mtime_ns)
  with f.open('rb') as s:s.seek(row['offset']);b=s.read(row['bytes'])
  h=hashlib.sha256(b).hexdigest();assert h==row['sha256'] and len(b)==row['bytes'] and len(b)<=64*1024**2
  assert h not in idx['blobs']
  if z is None or payload+len(b)+1024>a.max_mib*1024**2-1024**2:
   close();z=zipfile.ZipFile(o/('cloud-needed-part-%03d.zip'%(len(idx['parts'])+1)),'x',compression=zipfile.ZIP_STORED,allowZip64=True)
  z.writestr('blobs/'+h,b);payload+=len(b)+1024;idx['blobs'][h]=dict(bytes=len(b),archive=Path(z.filename).name,member='blobs/'+h)
 close()
 for v,st in checked.items():
  f=r.joinpath(*PurePosixPath(v).parts);s=f.stat();assert st==(s.st_size,s.st_mtime_ns)
 # Fresh read-only lstat inventory preserves POSIX modes, empty directories and link targets.
 # Do not traverse links; metadata of the output subtree itself is excluded.
 removed=('stage_a','stage-a','stagea','confirmation','transfer_robustness','transfer-robustness','deadline')
 meta=dict(format='pinn4soh-cloud-filesystem-metadata-v1',root=str(r),entries=[],symlinks_followed=False,source_files_modified=False,generated_exclusions=[])
 def generated(v):
  return v.split('/')[0].startswith(('CLOUD_MIGRATION_BACKUP_','CLOUD_NEEDED_','KNOWN_LOCAL','NEEDED_CHUNKS','cloud_inventory_backup_20261121','cloud_pack_requested_chunks_20261121'))
 for parent,dirs,names in os.walk(r,followlinks=False):
  dirs.sort();names.sort()
  for name in list(dirs):
   f=Path(parent)/name
   if f==o or generated(f.relative_to(r).as_posix()):meta['generated_exclusions'].append(f.relative_to(r).as_posix());dirs.remove(name);continue
   if f.is_symlink():dirs.remove(name);names.append(name)
  for name in sorted(dirs+names):
   f=Path(parent)/name;v=f.relative_to(r).as_posix();s=f.lstat()
   if generated(v):meta['generated_exclusions'].append(v);continue
   if any(k in v.lower() for k in removed) or '__pycache__' in f.parts or f.suffix in {'.pyc','.pyo'}:continue
   row=dict(path=v,mode=stat.S_IMODE(s.st_mode),mtime_ns=s.st_mtime_ns)
   if stat.S_ISLNK(s.st_mode):row.update(type='symlink',target=os.readlink(f))
   elif stat.S_ISDIR(s.st_mode):row.update(type='directory')
   elif stat.S_ISREG(s.st_mode):row.update(type='file',bytes=s.st_size)
   else:row.update(type='special')
   meta['entries'].append(row)
 metadata=o/'FILESYSTEM_METADATA.json';metadata.write_text(json.dumps(meta,indent=2)+'\n')
 # Keep abandoned experiment code/config separately, without its models, predictions or training results.
 code=dict(format='pinn4soh-discarded-experiment-code-only-v1',members=[],results_included=False)
 with zipfile.ZipFile(o/'discarded-code-only.zip','x',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as cz:
  for parent,dirs,names in os.walk(r,followlinks=False):
   dirs[:]=[d for d in dirs if not (Path(parent)/d).is_symlink() and not generated((Path(parent)/d).relative_to(r).as_posix()) and d!='__pycache__']
   for name in names:
    f=Path(parent)/name;v=f.relative_to(r).as_posix()
    if f.is_symlink() or not any(k in v.lower() for k in removed):continue
    iscode=f.suffix.lower() in {'.py','.sh','.ps1','.ipynb','.yaml','.yml','.toml','.ini','.cfg'}
    isconfig=f.suffix.lower()=='.json' and any(k in name.lower() for k in ('config','spec','registration','protocol','plan','worker','arguments','settings','manifest'))
    if not (iscode or isconfig) or f.stat().st_size>16*1024**2:continue
    data=f.read_bytes();scan['files']+=1;scan['bytes']+=len(data)
    for label,pattern in patterns:
     if pattern.search(data):scan['matched'].append(dict(path=v,pattern=label))
    cz.writestr(v,data);code['members'].append(dict(path=v,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
 with zipfile.ZipFile(o/'discarded-code-only.zip') as cz:
  for row in code['members']:assert hashlib.sha256(cz.read(row['path'])).hexdigest()==row['sha256']
 code['archive']=dict(name='discarded-code-only.zip',bytes=(o/'discarded-code-only.zip').stat().st_size,sha256=sha(o/'discarded-code-only.zip'));(o/'DISCARDED_CODE_ONLY.json').write_text(json.dumps(code,indent=2)+'\n')
 scan['passed']=not scan['matched'];idx.update(status='verified' if scan['passed'] else 'held_sensitive_text_match',requested_chunks=len(q['requests']),unique_bytes=sum(b['bytes'] for b in idx['blobs'].values()),sensitive_text_scan=scan,filesystem_metadata=dict(name=metadata.name,bytes=metadata.stat().st_size,sha256=sha(metadata)),discarded_code_only_manifest=dict(name='DISCARDED_CODE_ONLY.json',sha256=sha(o/'DISCARDED_CODE_ONLY.json')),discarded_code_only_archive=code['archive'])
 (o/'NEEDED_ARCHIVE.json').write_text(json.dumps(idx,indent=2)+'\n');print(json.dumps(dict(status=idx['status'],parts=len(idx['parts']),bytes=idx['unique_bytes'],chunks=len(idx['blobs']),text_scan_passed=scan['passed'],metadata_entries=len(meta['entries']),index_sha256=sha(o/'NEEDED_ARCHIVE.json'))))


if __name__=='__main__':main()
