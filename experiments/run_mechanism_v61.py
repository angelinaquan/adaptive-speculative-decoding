import json,subprocess,sys
from pathlib import Path
from analyze_coverage_v60 import analyze
r=Path(__file__).resolve().parent
s=analyze(r)
assert s==json.loads((r/'coverage_v60_summary.json').read_text())
assert not any(all(g>10 for g in c['native7']['round_gains']) for name,c in s['comparisons'].items() if name!='native7'), 'Expand successful candidate instead'
for role in ['identity','oracle1024']:
 subprocess.run([sys.executable,str(r/'probe_mechanism_v61.py'),'--role',role],check=True)
subprocess.run([sys.executable,str(r/'publish_mechanism_v61.py')],check=True)
