import json,os
from pathlib import Path
import wandb
r=Path(__file__).resolve().parent
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='fixed-v72-native-controls',name='balanced-native-width-sweep',dir=os.environ['SERVING_SCRATCH'],config=json.loads((r/'fixed_v72_plan.json').read_text()))
statuses={}
for path in sorted(r.glob('fixed_v72_*_round*.json')):
 result=json.loads(path.read_text());statuses[path.stem]=result['status']
 if result['status']=='completed':
  rows=[x for x in result['records'] if x['phase']=='timed'];run.summary[path.stem+'/tokens_per_second']=sum(x['tokens'] for x in rows)/sum(x['seconds'] for x in rows)
run.summary['statuses']=statuses
summary=r/'fixed_v72_summary.json'
if summary.exists():run.summary['audit']=json.loads(summary.read_text())
artifact=wandb.Artifact('fixed-v72-native-controls','exploratory-serving')
for pattern in ['*v72*.py','*v72*.json','run_fixed_v72.sh']:
 for path in r.glob(pattern):artifact.add_file(str(path))
artifact.add_file(str(r/'vocab_v54_manifest.json'))
run.log_artifact(artifact);(r/'fixed_v72_run.json').write_text(json.dumps({'url':run.url,'statuses':statuses},indent=2)+'\n');run.finish()
