"""Independent saved-output and arithmetic audit for v85/v87 profiling."""
import json,hashlib,math,sys
from pathlib import Path
r=Path(__file__).resolve().parent;version=sys.argv[1];assert version in ['85','87'];stem='profile_v'+version;batch=64 if version=='85' else 8;width=3 if version=='85' else 8
plan=json.loads((r/(stem+'_plan.json')).read_text());source=r/('benchmark_'+stem+'.py');summary=json.loads((r/(stem+'_summary.json')).read_text());ref=json.loads((r/(stem+'_ar0_round-1.json')).read_text());assert ref['status']=='completed' and len(ref['outputs'])==len(plan['indices']);checked=0;sessions=[]
for role,w,round in [('ar',0,-1),('native',width,0),('forced',width,0),('forced',width,1),('native',width,1)]:
 s=json.loads((r/f'{stem}_{role}{w}_round{round}.json').read_text());sessions.append(s)
 assert s['status']=='completed' and s['source_sha256']==hashlib.sha256(source.read_bytes()).hexdigest() and s['plan_sha256']==hashlib.sha256((r/(stem+'_plan.json')).read_bytes()).hexdigest()
 assert s['config']['max_num_seqs']==batch and s['config']['dtype']=='bfloat16'
 for x in s['records']:
  tokens=ref['outputs'][x['start']:x['start']+batch];assert x['parity'] and x['tokens']==sum(map(len,tokens)) and x['seconds']>0
  assert x['output_sha256']==hashlib.sha256(json.dumps(tokens).encode()).hexdigest();checked+=1
 if role=='forced':assert all(a['target_unchanged'] and a['logging_disabled'] and a['width']==width for a in s['audits'])
for role in ['native','forced']:
 rows=[x for s in sessions if s['role']==role for x in s['records'] if x['phase']=='timed'];actual=sum(x['tokens'] for x in rows)/sum(x['seconds'] for x in rows)
 assert math.isclose(actual,summary['throughputs_uninstrumented'][role],rel_tol=1e-12)
prof=next(s for s in sessions if s['role']=='forced' and s['round']==0);rows=[x for x in prof['records'] if x['phase']=='profile'];wall=1000*sum(x['seconds'] for x in rows)
assert math.isclose(wall,summary['profile_wall_ms'],rel_tol=1e-12)
for name in ['pre','context','backbone','tail']:
 es=[e for e in prof['stage_profile']['records'] if e['stage']==name];assert es and all(math.isfinite(e['ms']) and e['ms']>=0 for e in es)
 assert len(es)==summary['stages'][name]['calls'] and math.isclose(sum(e['ms'] for e in es),summary['stages'][name]['total_ms'],rel_tol=1e-12)
fraction=summary['stages']['tail']['total_ms']/wall;assert math.isclose(fraction,summary['tail_ms_fraction_of_profile_wall'],rel_tol=1e-12)
audit={'passed':True,'sessions':len(sessions),'output_batch_checks':checked,'source_plan_hashes_match':True,'tail_fraction':fraction,'idealized_zero_tail_gain_percent':100*(1/(1-fraction)-1),'caveat':'Hypothetical deletion of the entire tail in the instrumented forced runtime, not achievable head-only savings or a native speed bound.'}
(r/(stem+'_local_audit.json')).write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit))
