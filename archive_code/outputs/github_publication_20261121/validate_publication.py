from pathlib import Path
import re,json,hashlib,ast,urllib.parse
ROOT=Path(__file__).resolve().parent/'repository'
errors=[];checked=0
for p in [ROOT/'README.md',* (ROOT/'docs').glob('*.md')]:
    s=p.read_text(encoding='utf-8');assert '\ufffd' not in s
    for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',s):
        if link.startswith(('http:','https:','#','../../releases/')):continue
        link=urllib.parse.unquote(link.split('#')[0])
        if link and not (p.parent/link).exists():errors.append({'document':p.name,'missing':link})
        checked+=1
for p in (ROOT/'reproduction').glob('*.py'):
    ast.parse(p.read_text(encoding='utf-8-sig'),filename=p.name)
protected={'suite_report.json':'ebdc9f8a7ddeef608ee93a4b4aa6215b8ad06ac4b26e75d744233419f76d9738',
'independent_review_complete/claim_decisions_21.csv':'01e16e72ddcf8c28f55d7c57773cc8c42a412e8cb92c26744de2fae05ee63428',
'independent_review_complete/figure_table_evidence_map_25.csv':'106ada4af838f80b533c2e7a99bc2cb6874142786110ae6276cede6f550dcf96'}
for name,digest in protected.items():assert hashlib.sha256((ROOT/'outputs/full_reproduction/final_results_2120'/name).read_bytes()).hexdigest()==digest
secret=re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|sk-proj-[A-Za-z0-9_-]{25,})\b|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')
for p in ROOT.rglob('*'):
    if not p.is_file() or '.git' in p.parts:continue
    if p.suffix in {'.py','.ps1','.txt','.json','.md','.csv','.html','.yml','.yaml'} and secret.search(p.read_bytes()):errors.append({'secret_pattern':str(p.relative_to(ROOT))})
    if p.stat().st_size>95*1024*1024:errors.append({'too_large':str(p.relative_to(ROOT))})
result={'status':'passed' if not errors else 'failed','checked_primary_links':checked,'protected_hashes':3,'errors':errors}
(ROOT.parent/'PUBLICATION_VALIDATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=True));assert not errors
