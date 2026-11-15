import json,os,subprocess
from pathlib import Path
import wandb
r=Path(__file__).resolve().parent;plan=json.loads((r/'selector_v88_plan.json').read_text());run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='selector-v88-headroom',name='frozen-selector-scaling-and-candidate-ceiling',dir=os.environ['SERVING_SCRATCH'],config=plan);status='failed'
try:
 for name in ['check_selector_v88.py','analyze_selector_v88.py']:subprocess.run(['/tmp/aquan-forespark-v63/ForeSpark/.venv/bin/python',str(r/name)],check=True)
 summary=json.loads((r/'selector_v88_summary.json').read_text());assert summary['passed'];run.summary['results']=summary['results'];status='completed'
finally:
 run.summary['status']=status;a=wandb.Artifact('selector-v88-headroom','diagnostic')
 for pattern in ['selector_v88*.json','check_selector_v88.py','analyze_selector_v88.py','run_selector_v88.py','run_selector_v88.sh']:
  for p in r.glob(pattern):a.add_file(str(p))
 run.log_artifact(a);(r/'selector_v88_run.json').write_text(json.dumps({'url':run.url,'status':status},indent=2)+'\n');run.finish()
