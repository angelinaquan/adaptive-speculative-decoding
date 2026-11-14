import json,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent
for method in json.loads((r/'cost_v29_plan.json').read_text())['methods']:
 subprocess.run([sys.executable,str(r/'benchmark_cost_v29.py'),'--method',method],check=True)
print('ALL_V29_COMPLETE',flush=True)
