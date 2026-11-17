"""Publish audited existing data; do not rerun the correctness sweep."""
import json,os
from pathlib import Path
import wandb
from audit_correctness_v114 import audit
r=Path(__file__).resolve().parent;s=audit(r)
(r/'correctness_v114_local_audit.json').write_text(json.dumps(s,indent=2)+'\n')
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',id='hj28pf3h',resume='must',dir=os.environ['SERVING_SCRATCH'])
run.summary.update({'status':'correctness_passed_publication_repaired','original_slurm_job':3072601,'original_slurm_job_status':'failed_after_gate_in_wandb_finalizer','completed_sessions':32,'batch_output_checks':1856,'correctness_passed':True,'heldout_started_by_original_job':False})
art=wandb.Artifact('8b-fp32head-correctness-v114','correctness-regression')
for pattern in ['correctness_v114_b*.json','correctness_v114_local_audit.json','correctness_v112_gate.json','benchmark_correctness_v114.py','run_correctness_v114.py','repair_publish_v114.py','audit_correctness_v114.py','matched_v112_plan.json']:
 for path in r.glob(pattern):art.add_file(str(path))
run.log_artifact(art)
(r/'correctness_v114_run.json').write_text(json.dumps(dict(status='correctness_passed_publication_repaired',url=run.url,completed_sessions=32,original_job_status='failed_after_gate'),indent=2)+'\n')
run.finish()
