import json,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent
from analyze_diagnostic_v59 import analyze
d=analyze(r)['roles']
assert all(d['recency4096']['domains'][domain]['mean_accepted_prefix']<d['identity']['domains'][domain]['mean_accepted_prefix'] for domain in d['identity']['domains'])
assert json.loads((r/'dynamic_v58_checks.json').read_text())['passed']
subprocess.run([sys.executable,str(r/'benchmark_coverage_v60.py'),'--role','ar','--round','-1'],check=True)
for rd,order in enumerate(json.loads((r/'coverage_v60_plan.json').read_text())['orders']):
 for role in order:subprocess.run([sys.executable,str(r/'benchmark_coverage_v60.py'),'--role',role,'--round',str(rd)],check=True)
print('ALL_V60_COMPLETE',flush=True)
