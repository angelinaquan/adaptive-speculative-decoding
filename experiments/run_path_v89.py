import json,os,subprocess
from pathlib import Path
import wandb
r=Path(__file__).resolve().parent;plan=json.loads((r/'path_v89_plan.json').read_text());run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='path-v89-prefix-objective',name='prefix-reward-vs-greedy-global-map',dir=os.environ['SERVING_SCRATCH'],config=plan);status='failed'
try:
 for name in ['prefix_path_v89.py','check_path_v89.py','analyze_path_v89.py']:subprocess.run(['/tmp/aquan-forespark-v63/ForeSpark/.venv/bin/python',str(r/name)],check=True)
 summary=json.loads((r/'path_v89_summary.json').read_text());assert summary['passed'];run.summary['results']=summary['results'];status='completed'
finally:
 run.summary['status']=status;a=wandb.Artifact('path-v89-prefix-objective','diagnostic')
 for pattern in ['path_v89*.json','check_path_v89.py','analyze_path_v89.py','run_path_v89.py','run_path_v89.sh','prefix_path_v89.py']:
  for p in r.glob(pattern):a.add_file(str(p))
 run.log_artifact(a);(r/'path_v89_run.json').write_text(json.dumps({'url':run.url,'status':status},indent=2)+'\n');run.finish()
