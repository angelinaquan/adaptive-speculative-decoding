import json,os,subprocess
from pathlib import Path
import wandb
r=Path(__file__).resolve().parent;plan=json.loads((r/'profile_v85_plan.json').read_text());run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='profile-v85-draft-components',name='native3-vs-forced3-component-headroom',dir=os.environ['SERVING_SCRATCH'],config=plan);status='failed';sessions=[]
try:
 for role,width,round in [('ar',0,-1),('native',3,0),('forced',3,0),('forced',3,1),('native',3,1)]:
  subprocess.run(['/tmp/aquan-forespark-v63/ForeSpark/.venv/bin/python',str(r/'benchmark_profile_v85.py'),'--role',role,'--width',str(width),'--round',str(round)],check=True)
  p=r/f'profile_v85_{role}{width}_round{round}.json';s=json.loads(p.read_text());assert s['status']=='completed';sessions.append(s);run.log({'completed_sessions':len(sessions)})
 totals={}
 for role in ['native','forced']:
  rows=[x for s in sessions if s['role']==role for x in s['records'] if x['phase']=='timed'];totals[role]=sum(x['tokens'] for x in rows)/sum(x['seconds'] for x in rows)
 prof=next(s for s in sessions if s['role']=='forced' and s['round']==0);rows=[x for x in prof['records'] if x['phase']=='profile'];wall_ms=1000*sum(x['seconds'] for x in rows);stages={}
 for name in ['pre','context','backbone','tail']:
  es=[x for x in prof['stage_profile']['records'] if x['stage']==name];assert es;stages[name]={'calls':len(es),'total_ms':sum(x['ms'] for x in es)}
 measured_tps=sum(x['tokens'] for x in rows)/(wall_ms/1000)
 result={'passed':True,'throughputs_uninstrumented':totals,'forced_vs_native_percent':100*(totals['forced']/totals['native']-1),'profiled_forced_tps':measured_tps,'profile_vs_uninstrumented_forced_percent':100*(measured_tps/totals['forced']-1),'profile_wall_ms':wall_ms,'stages':stages,'tail_ms_fraction_of_profile_wall':stages['tail']['total_ms']/wall_ms,'scope':plan['scope'],'interpretation':plan['interpretation']};(r/'profile_v85_summary.json').write_text(json.dumps(result,indent=2)+'\n');run.summary.update(result);status='completed'
finally:
 run.summary['status']=status;a=wandb.Artifact('profile-v85-draft-components','profiling')
 for pattern in ['profile_v85*.json','benchmark_profile_v85.py','run_profile_v85.py','run_profile_v85.sh','stage_profile_v85.py','forced_policy_v72.py','vocab_v54_manifest.json']:
  for p in r.glob(pattern):a.add_file(str(p))
 run.log_artifact(a);(r/'profile_v85_run.json').write_text(json.dumps({'url':run.url,'status':status},indent=2)+'\n');run.finish()
