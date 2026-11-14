import json,os
from pathlib import Path
import wandb
r=Path(__file__).resolve().parent;s=json.loads((r/'context_v75_summary.json').read_text())
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='context-v75-prediction',name='nested-prompt-crossvalidation',dir=os.environ['SERVING_SCRATCH'],config=json.loads((r/'context_v75_plan.json').read_text()))
run.summary['passed']=s['passed'];run.summary['models']=s['models'];run.summary['fixed3_reward']=s['fixed3_reward'];run.summary['oracle_reward']=s['oracle_reward']
a=wandb.Artifact('context-v75-prediction','offline-diagnostic')
for name in ['context_probe_v75.py','context_v75_plan.json','context_v75_summary.json','context_v75_records.json','context_v75_features.npz','context_v75_predictions.npz','allocation_headroom_v73.py','labels_v73_checks.json','publish_context_v75.py']:a.add_file(str(r/name))
run.log_artifact(a);(r/'context_v75_run.json').write_text(json.dumps({'url':run.url})+'\n');run.finish()
