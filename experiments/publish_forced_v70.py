import json,os
from pathlib import Path
import wandb
r=Path(__file__).resolve().parent
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='forced-v70-serving-gate',name='forced-width-parity-and-cost',dir=os.environ['SERVING_SCRATCH'],config=json.loads((r/'forced_v70_plan.json').read_text()))
statuses={}
for role in ['ar','native8','forced']:
 path=r/f'forced_v70_{role}.json'
 if not path.exists():statuses[role]='not_run';continue
 result=json.loads(path.read_text());statuses[role]=result['status']
 # Failed or incomplete runs never publish throughput as a performance result.
 if result['status']=='completed':
  rows=[x for x in result['records'] if x['phase']=='timed']
  for width in sorted({x['width'] for x in rows}):
   selected=[x for x in rows if x['width']==width]
   run.summary[f'{role}/width{width}/tokens_per_second']=sum(x['tokens'] for x in selected)/sum(x['seconds'] for x in selected)
run.summary['statuses']=statuses
summary=r/'forced_v70_summary.json'
if summary.exists():run.summary['audit']=json.loads(summary.read_text())
artifact=wandb.Artifact('forced-v70-serving-gate','exploratory-serving')
for pattern in ['*v70*.py','*v70*.json','run_forced_v70.sh']:
 for path in r.glob(pattern):artifact.add_file(str(path))
run.log_artifact(artifact);(r/'forced_v70_run.json').write_text(json.dumps({'url':run.url,'statuses':statuses},indent=2)+'\n');run.finish()
