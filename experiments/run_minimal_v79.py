import json,time,subprocess,os,hashlib
from pathlib import Path
import wandb
r=Path(__file__).resolve().parent;plan=json.loads((r/'minimal_v79_plan.json').read_text());p='/tmp/aquan-forespark-v63/ForeSpark/.venv/bin/python';run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='minimal-v79-profiler',name='three-disjoint-panels-one-warmup-one-pass',config=plan,dir=os.environ['SERVING_SCRATCH']);status='failed';records=[]
(r/'minimal_v79_run.json').write_text(json.dumps({'url':run.url,'status':'running'})+'\n')
try:
 for panel,order in enumerate(plan['width_orders']):
  for w in order:
   t=time.perf_counter();subprocess.run([p,str(r/'benchmark_minimal_v79.py'),'--panel',str(panel),'--width',str(w)],check=True);wall=time.perf_counter()-t;x=json.loads((r/f'minimal_v79_p{panel}_w{w}.json').read_text());assert x['status']=='completed' and len(x['records'])==6 and all(z['parity'] for z in x['records'])
   assert x['source_sha256']==hashlib.sha256((r/'benchmark_minimal_v79.py').read_bytes()).hexdigest() and x['plan_sha256']==hashlib.sha256((r/'minimal_v79_plan.json').read_bytes()).hexdigest()
   records.append({'panel':panel,'width':w,'subprocess_wall_seconds':wall,'timed_seconds':sum(z['seconds'] for z in x['records'] if z['repeat']==0),'generation_seconds_including_warmup':sum(z['seconds'] for z in x['records'])});run.log({'completed_sessions':len(records),'last_subprocess_seconds':wall})
 selected=[min([z for z in records if z['panel']==panel],key=lambda z:(z['timed_seconds'],z['width']))['width'] for panel in range(3)]
 # Only now load test outcomes; they cannot influence profiling selection.
 test=json.loads((r/'transfer_v78_summary.json').read_text());assert test['passed'];test=next(z for z in test['comparisons'] if z['batch']==64 and z['domain']=='pooled');rates=test['throughputs'];rows=[]
 for panel,k in enumerate(selected):
  rows_p=[z for z in records if z['panel']==panel];cost=sum(z['subprocess_wall_seconds'] for z in rows_p);gen=sum(z['generation_seconds_including_warmup'] for z in rows_p);saving=1/rates[7]-1/rates[k-1]
  rows.append({'panel':panel,'selected_width':k,'test_loss_to_best_percent':100*(1-rates[k-1]/max(rates)),'total_measured_profiling_seconds':cost,'generation_component_seconds':gen,'break_even_tokens_vs8':cost/saving if saving>0 else None,'test_throughput_lookup':rates[k-1]})
 result={'passed':True,'primary_pass':all(z['test_loss_to_best_percent']<=5 for z in rows),'panels':rows,'records':records,'limitations':plan['limitations']};(r/'minimal_v79_summary.json').write_text(json.dumps(result,indent=2)+'\n');run.summary['analysis']=result;status='completed'
finally:
 run.summary['status']=status;a=wandb.Artifact('minimal-v79-profiler','profiling-experiment')
 for pattern in ['*minimal_v79*.py','minimal_v79*.json','run_minimal_v79.sh']:
  for f in r.glob(pattern):a.add_file(str(f))
 run.log_artifact(a);(r/'minimal_v79_run.json').write_text(json.dumps({'url':run.url,'status':status},indent=2)+'\n');run.finish()
