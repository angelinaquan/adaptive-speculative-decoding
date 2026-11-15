import json,os,subprocess
from pathlib import Path
import wandb
r=Path(__file__).resolve().parent;plan=json.loads((r/'score_v90_plan.json').read_text());run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='rerank-v90-frozen-scores',name='heldout-score-collection',dir=os.environ['SERVING_SCRATCH'],config=plan);status='failed'
try:
 subprocess.run(['/tmp/aquan-forespark-v63/ForeSpark/.venv/bin/python',str(r/'check_score_v90.py')],check=True)
 raw=json.loads((r/'score_v90_checks.json').read_text());assert raw['passed'];run.summary['prompts']=len(raw['records']);run.summary['correctness_passed']=True;status='completed'
finally:
 run.summary['status']=status;a=wandb.Artifact('score-v90-heldout','candidate-grids')
 for pattern in ['score_v90*.json','rerank_v90_plan.json','check_score_v90.py','run_score_v90.py','run_score_v90.sh','prefix_path_v89.py']:
  for p in r.glob(pattern):a.add_file(str(p))
 run.log_artifact(a);(r/'score_v90_run.json').write_text(json.dumps({'url':run.url,'status':status},indent=2)+'\n');run.finish()
