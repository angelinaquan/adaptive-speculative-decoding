import json,os
from pathlib import Path
import wandb
r=Path(__file__).resolve().parent
result=json.loads((r/'backbone_v65_checks.json').read_text())
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='backbone-v65-compatibility',name='released-dflash2-to-frozen-heads',dir=os.environ['SERVING_SCRATCH'],config={'scope':result['scope'],'model':result['model'],'revision':result['revision']})
run.summary.update({'status':result['status'],'passed':result.get('passed',False)})
a=wandb.Artifact('backbone-v65-compatibility','correctness')
for name in ['backbone_v65_checks.json','check_backbone_v65.py','run_backbone_v65.sh','publish_backbone_v65.py']:a.add_file(str(r/name))
run.log_artifact(a);(r/'backbone_v65_run.json').write_text(json.dumps({'run_url':run.url})+'\n');run.finish()
