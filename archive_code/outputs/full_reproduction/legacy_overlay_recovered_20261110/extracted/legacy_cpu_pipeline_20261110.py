import sys,subprocess
from pathlib import Path
r=Path('/workspace'); sys.path.insert(0,str(r/'legacy_single_gpu_v3/tools')); from install_legacy_runtime import isolated_environment
env=isolated_environment(r/'legacy37')
for tag,extra in [('smoke',['--smoke']),('formal',['--threads','1'])]:
 cmd=[str(r/'legacy37/bin/python'),'-B','-I',str(r/'benchmark_efficiency_legacy.py'),'--repo',str(r/'PINN4SOH-main'),'--output',str(r/'legacy_independent_overlay_20261110'/('legacy_cpu_'+tag)),*extra]; subprocess.run(cmd,env=env,check=True)
print('LEGACY_CPU_COMPLETE',flush=True)