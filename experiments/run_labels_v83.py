import os,json,subprocess
from pathlib import Path
import wandb
r=Path(__file__).resolve().parent;plan=json.loads((r/'labels_v83_plan.json').read_text());run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='labels-v83-scale-gate',name='480prompt-train-validation-independent-widths',dir=os.environ['SERVING_SCRATCH'],config=plan)
status='failed'
try:
 for name in ['check_labels_v83.py','analyze_labels_v83.py']:
  subprocess.run(['/tmp/aquan-forespark-v63/ForeSpark/.venv/bin/python',str(r/name)],check=True)
 summary=json.loads((r/'labels_v83_summary.json').read_text());assert summary['passed'];run.summary['label_checks']=summary;status='completed'
finally:
 run.summary['status']=status;a=wandb.Artifact('labels-v83-scale-gate','training-labels')
 for name in ['check_labels_v83.py','analyze_labels_v83.py','run_labels_v83.py','run_labels_v83.sh','labels_v83_plan.json','labels_v83_manifest.json','labels_v83_checks.json','labels_v83_summary.json','labels_v83_features.pt','labels_v83_features.npz']:
  if (r/name).exists():a.add_file(str(r/name))
 run.log_artifact(a);(r/'labels_v83_run.json').write_text(json.dumps({'url':run.url,'status':status},indent=2)+'\n');run.finish()
