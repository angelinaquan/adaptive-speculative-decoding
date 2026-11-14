"""W&B orchestration; one allocation, fail closed on any correctness error."""
import os,json,subprocess,time,hashlib,traceback
from pathlib import Path
import wandb
r=Path(__file__).resolve().parent;p='/tmp/aquan-forespark-v63/ForeSpark/.venv/bin/python';start=time.time()
plan=json.loads((r/'profile_v77_plan.json').read_text())
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='profile-v77-budget-transfer',name='native-width-profiling-and-transfer',dir=os.environ['SERVING_SCRATCH'],config=plan)
(r/'profile_v77_run.json').write_text(json.dumps({'url':run.url,'status':'running'},indent=2)+'\n')
status='failed';sessions=[]
try:
 jobs=[(64,0,-1),(8,0,-1)]+[(b,w,order) for order in range(2) for b in ([8,64] if order==0 else [64,8]) for w in (range(1,9) if order==0 else range(8,0,-1))]
 for step,(b,w,order) in enumerate(jobs):
  subprocess.run([p,str(r/'benchmark_profile_v77.py'),'--batch',str(b),'--width',str(w),'--round',str(order)],check=True)
  x=json.loads((r/f'profile_v77_b{b}_w{w}_r{order}.json').read_text());assert x['status']=='completed';sessions.append({'batch':b,'width':w,'round':order})
  run.log({'completed_sessions':len(sessions),'batch':b,'width':w,'round':order,'elapsed_seconds':time.time()-start,'parity_passed':1},step=step)
 subprocess.run([p,str(r/'analyze_profile_v77.py')],check=True)
 summary=json.loads((r/'profile_v77_summary.json').read_text());assert summary['passed'];run.summary['analysis']=summary
 for name in ['groups','full_profile','pooled']:
  rows=summary[name];columns=sorted(rows[0]);run.log({name:wandb.Table(columns=columns,data=[[json.dumps(row.get(c)) if isinstance(row.get(c),(dict,list)) else row.get(c) for c in columns] for row in rows])})
 status='completed'
except BaseException:
 run.summary['error']=traceback.format_exc();raise
finally:
 run.summary['status']=status;run.summary['completed_sessions']=sessions;run.summary['elapsed_seconds']=time.time()-start
 a=wandb.Artifact('profile-v77-budget-transfer','profiling-experiment')
 for pattern in ['*profile_v77*.py','*profile_v77*.json','run_profile_v77.sh']:
  for f in r.glob(pattern):a.add_file(str(f))
 run.log_artifact(a);(r/'profile_v77_run.json').write_text(json.dumps({'url':run.url,'status':status,'sessions':sessions,'elapsed_seconds':time.time()-start},indent=2)+'\n');run.finish()
