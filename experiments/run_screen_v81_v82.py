"""Reproduce and log two failed/passing diagnostic gates without calling reward throughput."""
import os,json,subprocess,hashlib
from pathlib import Path
import wandb
r=Path(__file__).resolve().parent
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='context-allocation-v81-v82',name='nonlinear-selection-and-causal-residual-screen',dir=os.environ['SERVING_SCRATCH'],config={'scope':'Exploratory teacher-anchor reward replay, not serving throughput','input_prompts':47,'states':277,'widths':list(range(1,9)),'budget_per_state':3,'outer_folds':4,'inner_folds':3})
status='failed'
try:
 for stem in ['decision_v81','residual_v82']:
  subprocess.run(['/tmp/aquan-forespark-v63/ForeSpark/.venv/bin/python',str(r/(stem+'.py'))],check=True,stdout=subprocess.DEVNULL)
  s=json.loads((r/(stem+'_summary.json')).read_text());expected=json.loads((r/(stem+'_local_expected.json')).read_text());assert s['passed']
  assert s['models'].keys()==expected['models'].keys()
  for name,m in s['models'].items():
   assert m['reward']==expected['models'][name]['reward'],(stem,name,'Remote reward differs')
   for metric,value in m.items():run.summary[stem+'/'+name+'/'+metric]=value
  assert s['fixed3_reward']==expected['fixed3_reward']
  run.summary[stem+'/checks_passed']=True
  run.summary[stem+'/limitations']=s['limitations']
  run.log({stem+'/results':wandb.Table(columns=['model','reward','gain_vs_fixed3_percent'],data=[[name,m['reward'],m['gain_vs_fixed3_percent']] for name,m in s['models'].items()])})
 status='completed'
finally:
 run.summary['status']=status
 artifact=wandb.Artifact('context-allocation-v81-v82','diagnostic-replay')
 for pattern in ['decision_v81*','residual_v82*','context_v75_features.npz','context_v75_summary.json','allocation_headroom_v73.py','run_screen_v81_v82.py']:
  for p in r.glob(pattern):artifact.add_file(str(p))
 run.log_artifact(artifact)
 (r/'screen_v81_v82_run.json').write_text(json.dumps({'url':run.url,'status':status},indent=2)+'\n')
 run.finish()
