import json,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent
from analyze_context_v55 import analyze
summary=analyze(r)
if any(all(g>10 for g in values['native']['round_gains']) for role,values in summary['comparisons'].items() if role!='native'):
 print('V55 has a >10% candidate; skip oracle and prioritize confirmation.',flush=True)
 sys.exit(0)
subprocess.run([sys.executable,str(r/'benchmark_oracle_v56.py'),'--role','ar','--round','-1'],check=True)
for rd,order in enumerate(json.loads((r/'oracle_v56_plan.json').read_text())['orders']):
 for role in order:subprocess.run([sys.executable,str(r/'benchmark_oracle_v56.py'),'--role',role,'--round',str(rd)],check=True)
print('ALL_V56_COMPLETE',flush=True)
