import json,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent
from analyze_oracle_v57 import analyze
oracle=analyze(r)
assert all(all(g>10 for g in v['round_gains']) for v in oracle['comparisons']['oracle7'].values()), 'Insufficient oracle headroom: do not launch'
subprocess.run([sys.executable,str(r/'check_dynamic_v58.py')],check=True)
assert json.loads((r/'dynamic_v58_checks.json').read_text())['passed']
subprocess.run([sys.executable,str(r/'benchmark_dynamic_v58.py'),'--role','ar','--round','-1'],check=True)
for rd,order in enumerate(json.loads((r/'dynamic_v58_plan.json').read_text())['orders']):
 for role in order:subprocess.run([sys.executable,str(r/'benchmark_dynamic_v58.py'),'--role',role,'--round',str(rd)],check=True)
print('ALL_V58_COMPLETE',flush=True)
