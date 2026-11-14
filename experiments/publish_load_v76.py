import json,os
from pathlib import Path
import wandb
r=Path(__file__).resolve().parent
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='load-v76-confirmation',name='native-width-load-confirmation',dir=os.environ['SERVING_SCRATCH'],config=json.loads((r/'load_v76_plan.json').read_text()))
statuses={}
for p in r.glob('load_v76_b*_w*_r*.json'):
 x=json.loads(p.read_text());statuses[p.stem]=x['status']
run.summary['statuses']=statuses
if (r/'load_v76_summary.json').exists():run.summary['audit']=json.loads((r/'load_v76_summary.json').read_text())
a=wandb.Artifact('load-v76-confirmation','serving-confirmation')
for pattern in ['*v76*.py','*v76*.json','run_load_v76.sh']:
 for p in r.glob(pattern):a.add_file(str(p))
run.log_artifact(a);(r/'load_v76_run.json').write_text(json.dumps({'url':run.url,'statuses':statuses},indent=2)+'\n');run.finish()
