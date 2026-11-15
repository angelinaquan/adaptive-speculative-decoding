import json,os,subprocess
from pathlib import Path
import wandb
r=Path(__file__).resolve().parent;plan=json.loads((r/'model_v102_plan.json').read_text());paths=json.loads((r/'model_v102_paths.json').read_text());assert paths['passed'];run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='second-model-v102',name='qwen3-8b-native-parity-and-cost-smoke',dir=os.environ['SERVING_SCRATCH'],config=plan);status='failed';sessions=[]
try:
 for role,width,rnd in [('ar',0,-1),('native',3,0),('native',7,0),('native',7,1),('native',3,1)]:
  subprocess.run(['/tmp/aquan-forespark-v63/ForeSpark/.venv/bin/python',str(r/'benchmark_model_v102.py'),'--role',role,'--width',str(width),'--round',str(rnd)],check=True)
  s=json.loads((r/f'model_v102_{role}{width}_round{rnd}.json').read_text());assert s['status']=='completed';sessions.append(s);run.log({'completed_sessions':len(sessions)})
 totals={}
 for width in [3,7]:
  rows=[row for s in sessions if s['role']=='native' and s['width']==width for row in s['records'] if row['phase']=='timed'];totals[str(width)]=sum(x['tokens'] for x in rows)/sum(x['seconds'] for x in rows)
 out={'passed':True,'throughputs':totals,'width3_vs_width7_percent':100*(totals['3']/totals['7']-1),'scope':plan['scope'],'limitations':plan['limitations']};(r/'model_v102_summary.json').write_text(json.dumps(out,indent=2)+'\n');run.summary.update(out);status='completed'
finally:
 run.summary['status']=status;a=wandb.Artifact('qwen3-8b-v102-smoke','baseline-correctness')
 for pattern in ['model_v102*.json','prepare_model_v102.py','benchmark_model_v102.py','run_model_v102.py','run_model_v102.sh','vocab_v54_manifest.json']:
  for p in r.glob(pattern):a.add_file(str(p))
 run.log_artifact(a);(r/'model_v102_run.json').write_text(json.dumps({'url':run.url,'status':status},indent=2)+'\n');run.finish()
