import json,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(r/'benchmark_context_v55.py'),'--role','ar','--round','-1'],check=True)
for rd,order in enumerate(json.loads((r/'context_v55_plan.json').read_text())['orders']):
 for role in order:subprocess.run([sys.executable,str(r/'benchmark_context_v55.py'),'--role',role,'--round',str(rd)],check=True)
print('ALL_V55_COMPLETE',flush=True)
