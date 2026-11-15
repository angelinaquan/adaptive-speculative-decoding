import json,os,subprocess,hashlib
from pathlib import Path
import wandb
r=Path(__file__).resolve().parent;plan=json.loads((r/'cost_v99_plan.json').read_text());run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='postdraft-cost-v99',name='draft8-verify3-vs-native3-cost-gate',dir=os.environ['SERVING_SCRATCH'],config=plan);status='failed';sessions=[]
try:
 for role,width,round in [('ar',0,-1),('native',3,0),('forced',3,0),('forced',8,0),('forced',8,1),('forced',3,1),('native',3,1)]:
  subprocess.run(['/tmp/aquan-forespark-v63/ForeSpark/.venv/bin/python',str(r/'benchmark_cost_v99.py'),'--role',role,'--width',str(width),'--round',str(round)],check=True)
  p=r/f'cost_v99_{role}{width}_round{round}.json';s=json.loads(p.read_text());assert s['status']=='completed';sessions.append(s);run.log({'completed_sessions':len(sessions)})
 totals={}
 for role,width in [('native',3),('forced',3),('forced',8)]:
  rows=[x for s in sessions if s['role']==role and s['width']==width for x in s['records'] if x['phase']=='timed'];totals[f'{role}{width}']=sum(x['tokens'] for x in rows)/sum(x['seconds'] for x in rows)
 for width in [3,8]:
  audit=next(s for s in sessions if s['role']=='forced' and s['width']==width and s['round']==0);assert any(x['exact_gpu_audit_batches']>0 for x in audit['audits'])
 result={'passed':True,'throughputs':totals,'forced8_vs_native3_percent':100*(totals['forced8']/totals['native3']-1),'forced8_vs_forced3_percent':100*(totals['forced8']/totals['forced3']-1),'forced3_vs_native3_percent':100*(totals['forced3']/totals['native3']-1),'scope':plan['scope'],'interpretation':plan['interpretation']};(r/'cost_v99_summary.json').write_text(json.dumps(result,indent=2)+'\n');run.summary.update(result);status='completed'
finally:
 run.summary['status']=status;a=wandb.Artifact('postdraft-cost-v99','serving-cost-gate')
 for pattern in ['cost_v99*.json','benchmark_cost_v99.py','run_cost_v99.py','run_cost_v99.sh','fixed_verify_v99.py','forced_policy_v72.py','vocab_v54_manifest.json']:
  for p in r.glob(pattern):a.add_file(str(p))
 run.log_artifact(a);(r/'cost_v99_run.json').write_text(json.dumps({'url':run.url,'status':status},indent=2)+'\n');run.finish()
