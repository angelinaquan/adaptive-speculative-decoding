"""W&B orchestration; one allocation, fail closed on any correctness error."""
import os,json,subprocess,time,hashlib,traceback
from pathlib import Path
import wandb
r=Path(__file__).resolve().parent;p='/tmp/aquan-forespark-v63/ForeSpark/.venv/bin/python';start=time.time()
plan=json.loads((r/'transfer_v78_plan.json').read_text())
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='transfer-v78-frozen-confirmation',name='frozen-width-new-benchmarks-longer-generation',dir=os.environ['SERVING_SCRATCH'],config=plan)
(r/'transfer_v78_run.json').write_text(json.dumps({'url':run.url,'status':'running'},indent=2)+'\n')
status='failed';sessions=[]
try:
 jobs=[(64,0,-1),(8,0,-1)]+[(b,w,order) for order in range(2) for b in ([8,64] if order==0 else [64,8]) for w in (range(1,9) if order==0 else range(8,0,-1))]
 for step,(b,w,order) in enumerate(jobs):
  subprocess.run([p,str(r/'benchmark_transfer_v78.py'),'--batch',str(b),'--width',str(w),'--round',str(order)],check=True)
  x=json.loads((r/f'transfer_v78_b{b}_w{w}_r{order}.json').read_text());assert x['status']=='completed';sessions.append({'batch':b,'width':w,'round':order})
  run.log({'completed_sessions':len(sessions),'batch':b,'width':w,'round':order,'elapsed_seconds':time.time()-start,'parity_passed':1},step=step)
 subprocess.run([p,str(r/'analyze_transfer_v78.py')],check=True)
 summary=json.loads((r/'transfer_v78_summary.json').read_text());assert summary['passed'];run.summary['analysis']=summary
 for name in ['comparisons']:
  rows=summary[name];columns=sorted(rows[0]);run.log({name:wandb.Table(columns=columns,data=[[json.dumps(row.get(c)) if isinstance(row.get(c),(dict,list)) else row.get(c) for c in columns] for row in rows])})
 status='completed'
except BaseException:
 run.summary['error']=traceback.format_exc();raise
finally:
 run.summary['status']=status;run.summary['completed_sessions']=sessions;run.summary['elapsed_seconds']=time.time()-start
 a=wandb.Artifact('transfer-v78-frozen-confirmation','profiling-experiment')
 for pattern in ['*transfer_v78*.py','*transfer_v78*.json','run_transfer_v78.sh']:
  for f in r.glob(pattern):a.add_file(str(f))
 run.log_artifact(a);(r/'transfer_v78_run.json').write_text(json.dumps({'url':run.url,'status':status,'sessions':sessions,'elapsed_seconds':time.time()-start},indent=2)+'\n');run.finish()
