"""Upload stable release assets; credentials stay in memory."""
from pathlib import Path
import argparse, hashlib, json, urllib.parse, urllib.request, urllib.error, time, os
from github_api import api, auth, REPO

BASE=Path(__file__).resolve().parent

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def release(tag,title):
    try:return api('GET','/repos/'+REPO+'/releases/tags/'+tag)
    except RuntimeError as e:
        if ': 404 ' not in str(e):raise
    body=('中文成果与模型证据。主实验2120项全部完成，但不代表原论文全部数值与结论一致。\n\n'
          '请先阅读仓库中文首页、实验结果、未复现与限制。训练证据四包已逐成员SHA回读，全部结果保留有利与不利记录。\n\n'
          '完整档案另见 archive-20261121；附件按SHA索引核验，历史第三方材料权利归原来源。')
    return api('POST','/repos/'+REPO+'/releases',{'tag_name':tag,'target_commitish':'main','name':title,'body':body,'draft':False,'prerelease':False})

def upload(rel,p):
    assets=api('GET','/repos/'+REPO+'/releases/'+str(rel['id'])+'/assets?per_page=100')
    digest=sha(p)
    for a in assets:
        if a['name']==p.name:
            assert a['size']==p.stat().st_size
            if a.get('digest'):assert a['digest']=='sha256:'+digest
            return dict(name=p.name,bytes=a['size'],sha256=digest,url=a['browser_download_url'],already_present=True)
    url=rel['upload_url'].split('{')[0]+'?'+urllib.parse.urlencode({'name':p.name})
    h=auth();h.update({'Content-Type':'application/octet-stream','Content-Length':str(p.stat().st_size)})
    with p.open('rb') as stream:
        req=urllib.request.Request(url,data=stream,headers=h,method='POST')
        try:
            with urllib.request.urlopen(req,timeout=1800) as f:a=json.load(f)
        except urllib.error.HTTPError as e:raise RuntimeError('Upload failed '+str(e.code)+' '+str(json.load(e).get('message',''))) from None
    assert a['state']=='uploaded' and a['size']==p.stat().st_size
    if a.get('digest'):assert a['digest']=='sha256:'+digest
    return dict(name=p.name,bytes=a['size'],sha256=digest,url=a['browser_download_url'],github_digest=a.get('digest'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--title',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--files',nargs='*');args=p.parse_args()
    rel=release(args.tag,args.title);rows=[]; receipt=BASE/('UPLOAD_'+args.tag+'.json')
    if receipt.exists():rows=json.loads(receipt.read_text(encoding='utf-8')).get('assets',[])
    files=[args.directory/n for n in args.files] if args.files else [f for f in args.directory.iterdir() if f.is_file()]
    for f in sorted(files):
        if f.name.endswith(('.tmp','.part')):continue
        print(json.dumps({'uploading':f.name,'bytes':f.stat().st_size}),flush=True)
        r=upload(rel,f);rows=[x for x in rows if x['name']!=f.name]+[r]
        receipt.write_text(json.dumps({'release':rel['html_url'],'tag':args.tag,'assets':rows},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'uploaded':f.name,'bytes':r['bytes'],'sha256':r['sha256']}),flush=True)
    print(json.dumps({'release':rel['html_url'],'verified_assets':len(rows)}),flush=True)

if __name__=='__main__':main()
