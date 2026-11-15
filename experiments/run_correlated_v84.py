import os,json,subprocess
from pathlib import Path
import wandb
r=Path(__file__).resolve().parent;plan=json.loads((r/'correlated_v84_plan.json').read_text())
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='correlated-v84-scale-gate',name='width-correlated-feedback-training-validation',dir=os.environ['SERVING_SCRATCH'],config=plan);status='failed'
try:
 for script in ['test_correlated_v84.py','correlated_v84.py']:
  subprocess.run(['/tmp/aquan-forespark-v63/ForeSpark/.venv/bin/python',str(r/script)],check=True)
 s=json.loads((r/'correlated_v84_summary.json').read_text());assert s['passed'];run.summary['primary_reward_gate_passed']=s['primary_gate_passed'];run.summary['throughput_claim']=False
 rows=[]
 for row in s['learning_curve']:
  for name,result in row['models'].items():rows.append([row['training_per_domain'],name,result['reward'],result['gain_vs_fixed3_percent']])
 run.log({'learning_curve':wandb.Table(columns=['training_prompts_per_domain','model','reward','gain_vs_fixed3_percent'],data=rows)});run.summary['primary']=s['learning_curve'][-1];status='completed'
finally:
 run.summary['status']=status;a=wandb.Artifact('correlated-v84-scale-gate','validation-replay')
 for name in ['correlated_v84.py','test_correlated_v84.py','run_correlated_v84.py','run_correlated_v84.sh','correlated_v84_plan.json','correlated_v84_summary.json','correlated_v84_records.json','labels_v83_summary.json','labels_v83_features.npz','labels_v83_manifest.json','allocation_headroom_v73.py']:
  if (r/name).exists():a.add_file(str(r/name))
 run.log_artifact(a);(r/'correlated_v84_run.json').write_text(json.dumps({'url':run.url,'status':status},indent=2)+'\n');run.finish()
