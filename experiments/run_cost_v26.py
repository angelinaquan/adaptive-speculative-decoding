import json,subprocess,sys
from pathlib import Path
from analyze_cost_v25 import analyze
r=Path(__file__).resolve().parent
summary=analyze(r);(r/'cost_v25_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if not summary['launch_gate']:
 (r/'cost_v26_not_launched.json').write_text(json.dumps({'reason':'Measured cost policy did not beat all required validation comparators','selection':summary},indent=2)+'\n')
 print('HELDOUT_NOT_LAUNCHED: validation gate failed',flush=True);sys.exit(0)
plan=json.loads((r/'cost_v26_plan.json').read_text())
for rd,order in enumerate(plan['orders']):
 for role in order:subprocess.run([sys.executable,str(r/'benchmark_cost_v26.py'),'--role',role,'--round',str(rd)],check=True)
print('ALL_V26_COMPLETE',flush=True)
