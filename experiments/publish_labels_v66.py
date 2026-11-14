import json,os
from pathlib import Path
import wandb
r=Path(__file__).resolve().parent
result=json.loads((r/'labels_v66_checks.json').read_text())
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='labels-v66-real-context',name='independent-width-labels',dir=os.environ['SERVING_SCRATCH'],config={'scope':result['scope'],'model':result['model'],'revision':result['revision']})
run.summary.update({'status':result['status'],'passed':result.get('passed',False)})
a=wandb.Artifact('labels-v66-real-context','correctness')
for name in ['labels_v66_checks.json','check_labels_v66.py','run_labels_v66.sh','publish_labels_v66.py','labels_v66_plan.json','oracle_v57_manifest.json']+(['labels_v66_features.pt'] if (r/'labels_v66_features.pt').exists() else []):a.add_file(str(r/name))
run.log_artifact(a);(r/'labels_v66_run.json').write_text(json.dumps({'run_url':run.url})+'\n');run.finish()
