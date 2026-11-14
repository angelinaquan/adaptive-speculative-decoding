import json,sys,subprocess
from pathlib import Path
from analyze_dynamic_v58 import analyze
r=Path(__file__).resolve().parent
s=analyze(r)
if any(all(g>10 for g in s['comparisons'][role]['native7']['round_gains']) for role in ['recency1024','recency4096']):
 print('Deployable candidate>10%; skip diagnostic and prioritize confirmation.',flush=True);sys.exit(0)
for role in ['identity','recency4096']:
 subprocess.run([sys.executable,str(r/'probe_dynamic_v59.py'),'--role',role],check=True)
