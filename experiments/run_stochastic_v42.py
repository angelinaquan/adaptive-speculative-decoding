import json,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent
for method in json.loads((r/'stochastic_v42_plan.json').read_text())['methods']:
 subprocess.run([sys.executable,str(r/'benchmark_stochastic_v42.py'),'--method',method],check=True)
print('ALL_V42_COMPLETE',flush=True)
