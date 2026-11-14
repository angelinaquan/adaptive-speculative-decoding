import json,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(r/'benchmark_cost_v38.py'),'--role','ar','--round','-1'],check=True)
for rd,order in enumerate(json.loads((r/'cost_v38_plan.json').read_text())['orders']):
 for role in order:
  subprocess.run([sys.executable,str(r/'benchmark_cost_v38.py'),'--role',role,'--round',str(rd)],check=True)
print('ALL_V38_COMPLETE',flush=True)
