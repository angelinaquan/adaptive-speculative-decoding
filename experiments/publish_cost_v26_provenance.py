import json,os
from pathlib import Path
import wandb
r=Path(__file__).resolve().parent
p=json.loads((r/'cost_v26_provenance_audit.json').read_text());assert p['passed']
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='cost-v26-heldout',name='provenance-and-precision-qualification',dir=os.environ['SERVING_SCRATCH'],config={'precision_qualification':p['precision_qualification']})
run.summary['provenance_passed']=True
art=wandb.Artifact('cost-v26-provenance-precision','audit')
for name in ['cost_v26_provenance_audit.json','kernel_precision_v28.json','audit_kernel_precision_v28.py']:
 art.add_file(str(r/name))
run.log_artifact(art)
(r/'cost_v26_provenance_run.json').write_text(json.dumps({'run_url':run.url})+'\n')
run.finish()
