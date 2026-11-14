import json,os
from pathlib import Path
import wandb
r=Path(__file__).resolve().parent;s=json.loads((r/'history_v74_summary.json').read_text())
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='history-v74-replay',name='past-feedback-only-budget-allocation',dir=os.environ['SERVING_SCRATCH'],config=json.loads((r/'history_v74_plan.json').read_text()))
run.summary.update({k:s[k] for k in ['passed','history_gain_percent','oracle_gain_percent','fixed3_reward','history_reward','hindsight_oracle_reward']})
a=wandb.Artifact('history-v74-replay','offline-diagnostic')
for name in ['history_v74_plan.json','history_v74_summary.json','history_v74_trace.json','replay_history_v74.py','allocation_headroom_v73.py','labels_v73_checks.json','labels_v73_summary.json','publish_history_v74.py']:a.add_file(str(r/name))
run.log_artifact(a);(r/'history_v74_run.json').write_text(json.dumps({'url':run.url},indent=2)+'\n');run.finish()
