import json,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(r/'benchmark_vocab_v53.py'),'--role','ar','--round','-1'],check=True)
for rd,order in enumerate(json.loads((r/'vocab_v53_plan.json').read_text())['orders']):
 for role in order:subprocess.run([sys.executable,str(r/'benchmark_vocab_v53.py'),'--role',role,'--round',str(rd)],check=True)
print('ALL_V53_COMPLETE',flush=True)
