"""GitHub helper. Credentials stay in memory and never appear in output."""
import os, json, subprocess, urllib.request, urllib.error, time

OWNER = 'zhaopengbo-0702'
NAME = 'PINN4SOH-Reproduction'
REPO = OWNER + '/' + NAME

def auth():
    r = subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',
        text=True,capture_output=True,timeout=30,
        env=dict(os.environ,GCM_INTERACTIVE='Never',GIT_TERMINAL_PROMPT='0'))
    c=dict(x.split('=',1) for x in r.stdout.splitlines() if '=' in x)
    token=c.get('password')
    if not token: raise RuntimeError('No stored GitHub authorization')
    return {'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json',
            'X-GitHub-Api-Version':'2022-11-28','User-Agent':'pinn4soh-reproduction-publisher'}

def api(method, endpoint, payload=None, headers=None):
    h=auth() if headers is None else headers
    data=None if payload is None else json.dumps(payload,ensure_ascii=False).encode('utf-8')
    req=urllib.request.Request('https://api.github.com'+endpoint,data=data,
        headers=dict(h,**{'Content-Type':'application/json'}),method=method)
    try:
        with urllib.request.urlopen(req,timeout=60) as f:
            b=f.read(); return json.loads(b) if b else {}
    except urllib.error.HTTPError as e:
        raise RuntimeError('GitHub '+method+' '+endpoint+': '+str(e.code)+' '+str(json.load(e).get('message','request failed'))) from None

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['identity','create']);a=p.parse_args()
    h=auth();user=api('GET','/user',headers=h); assert user['login']==OWNER
    if a.action=='identity': print(json.dumps({'login':user['login']}))
    elif a.action=='create':
        d=api('POST','/user/repos',{'name':NAME,'description':'PINN4SOH论文复现：中文成果看板、2120项主实验、真实结果、未复现清单、模型日志与完整档案','private':False,'auto_init':False},headers=h)
        assert d['full_name']==REPO and not d['private']
        print(json.dumps({'url':d['html_url'],'private':d['private'],'full_name':d['full_name']}))
