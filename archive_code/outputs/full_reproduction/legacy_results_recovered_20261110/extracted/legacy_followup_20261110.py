import json,time,subprocess,os,sys,datetime
from pathlib import Path
r=Path('/workspace'); state=r/'legacy_followup_state_20261110.json'; suite=r/'pinn4soh_legacy_formal_control_20261110/manifest.json'
def save(stage,**kw):
 d={'stage':stage,'utc':datetime.datetime.utcnow().isoformat()+'Z',**kw}; state.write_text(json.dumps(d,indent=2)); print(json.dumps(d),flush=True)
save('waiting_for_40_control')
while True:
 d=json.loads(suite.read_text())
 statuses=[t['status'] for t in d['tasks'].values()]
 if d.get('status')=='complete' and len(statuses)==40 and all(s=='complete' for s in statuses): break
 if d.get('status')=='failed' or any(s=='failed' for s in statuses): save('control_failed',suite_status=d.get('status')); sys.exit(1)
 time.sleep(15)
env=dict(os.environ,PYTHONPATH='/workspace/site-packages'); tools=r/'legacy_single_gpu_v3/tools'; comp=r/'pinn4soh_legacy_control_comparison_20261110'
save('comparing_40')
cmd=[sys.executable,str(tools/'compare_legacy_control.py'),'--plan',str(r/'legacy_single_gpu_v3/control/legacy_control_plan.json'),'--legacy-suite',str(r/'pinn4soh_legacy_formal_control_20261110'),'--current-suite',str(r/'pinn4soh_full_paper_stage1'),'--current-suite',str(r/'pinn4soh_full_paper_stage2'),'--output',str(comp)]
p=subprocess.run(cmd,env=env); save('comparison_finished',returncode=p.returncode)
if p.returncode: sys.exit(p.returncode)
p=subprocess.run([sys.executable,str(r/'collect_legacy_results.py'),'--output-prefix',str(r/'legacy_results_20261110')],env=env); save('collection_finished',returncode=p.returncode); sys.exit(p.returncode)