"""Run only after all validation candidates finish and pass their gates."""
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
scratch = Path(os.environ['SERVING_SCRATCH'])
summary_path = scratch/'serving_v15_summary.json'
subprocess.run([sys.executable, str(root/'analyze_serving_v15.py'),
                str(scratch/'probe-results-v15'), '--output', str(summary_path)], check=True)
summary = json.loads(summary_path.read_text())
if not summary['batches']['8']['adaptive_beats_both_descriptively']:
    print('NO_HELDOUT_LAUNCH: adaptive did not beat both validation comparators at the predeclared primary batch size', flush=True)
    sys.exit(0)
orders = [['ar','fixed','adaptive'], ['fixed','adaptive','ar'], ['adaptive','ar','fixed']]
for batch in [8,1,32]:
    for round_index, order in enumerate(orders):
        for role in order:
            subprocess.run([sys.executable, str(root/'benchmark_serving_v16.py'),
                '--role',role,'--batch',str(batch),'--round',str(round_index)], check=True)
print('ALL_HELDOUT_SESSIONS_COMPLETE', flush=True)
