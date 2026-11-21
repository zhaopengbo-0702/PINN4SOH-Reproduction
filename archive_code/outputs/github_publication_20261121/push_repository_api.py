"""Publish Git blobs through authenticated API when smart HTTPS is unreliable."""
from pathlib import Path
import subprocess,json,base64,hashlib,time,concurrent.futures
from github_api import api,auth,REPO

ROOT=Path(__file__).resolve().parent/'repository'
CACHE=ROOT.parent/'GITHUB_BLOB_CACHE.json'
headers=auth()
cache=json.loads(CACHE.read_text(encoding='utf-8')) if CACHE.exists() else {}
paths=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode('utf-8').split('\0');paths=[x for x in paths if x]

def put(name):
    raw=(ROOT/name).read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if digest in cache:return name,cache[digest],digest
    for attempt in range(4):
        try:
            result=api('POST','/repos/'+REPO+'/git/blobs',{'content':base64.b64encode(raw).decode(),'encoding':'base64'},headers=headers)
            expected=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
            assert result['sha']==expected
            return name,result['sha'],digest
        except Exception:
            if attempt==3:raise
            time.sleep(2+attempt)

tree=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
    for i,(name,oid,digest) in enumerate(pool.map(put,paths),1):
        tree.append({'path':name,'mode':'100644','type':'blob','sha':oid});cache[digest]=oid
        if i%25==0:
            CACHE.write_text(json.dumps(cache),encoding='utf-8');print(json.dumps({'blobs_verified':i,'total':len(paths)}),flush=True)
CACHE.write_text(json.dumps(cache),encoding='utf-8')
remote_tree=api('POST','/repos/'+REPO+'/git/trees',{'tree':tree},headers=headers)
parent=None
try:parent=api('GET','/repos/'+REPO+'/git/ref/heads/main',headers=headers)['object']['sha']
except RuntimeError as e:
    if ': 404 ' not in str(e) and ': 409 ' not in str(e):raise
commit=api('POST','/repos/'+REPO+'/git/commits',{'message':'发布中文复现成果、图表和经过核验的证据索引','tree':remote_tree['sha'],'parents':[parent] if parent else []},headers=headers)
if parent:api('PATCH','/repos/'+REPO+'/git/refs/heads/main',{'sha':commit['sha'],'force':False},headers=headers)
else:api('POST','/repos/'+REPO+'/git/refs',{'ref':'refs/heads/main','sha':commit['sha']},headers=headers)
verify=api('GET','/repos/'+REPO+'/git/trees/'+remote_tree['sha']+'?recursive=1',headers=headers)
assert not verify['truncated']
actual={x['path']:x['sha'] for x in verify['tree'] if x['type']=='blob'}
assert actual=={x['path']:x['sha'] for x in tree}
receipt={'status':'published_remote_tree_verified','url':'https://github.com/'+REPO,'commit':commit['sha'],'tree':remote_tree['sha'],'files':len(tree)}
(ROOT.parent/'GITHUB_REPOSITORY_PUBLISHED.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt),flush=True)
