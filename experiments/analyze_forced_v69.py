"""Audit completed forced-width screen; never treat its oracle as a deployable method."""
import json,hashlib,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);a=p.parse_args();r=a.root
plan=json.loads((r/'forced_v69_plan.json').read_text());results={role:json.loads((r/f'forced_v69_{role}.json').read_text()) for role in ['ar','native8','forced']}
assert all(x['status']=='completed' and x['all_widths_parity_passed'] for x in results.values())
assert len(results['ar']['outputs'])==24
summary={'scope':'24reusedprompts,3batches,2reversewidthorders. Descriptiveonly; nofreshgeneralizationorindependentconfidenceinterval.','throughputs':{},'gates_passed':True}
source_hash=hashlib.sha256((r/'benchmark_forced_v69.py').read_bytes()).hexdigest();plan_hash=hashlib.sha256((r/'forced_v69_plan.json').read_bytes()).hexdigest()
for role,res in results.items():
 assert res['source_sha256']==source_hash and res['plan_sha256']==plan_hash
 assert res['config']['dtype']=='bfloat16' and res['config']['max_num_seqs']==8
 widths=list(range(1,9)) if role=='forced' else [8 if role=='native8' else 0]
 assert len(res['records'])==len(widths)*6*3 and all(x['parity'] for x in res['records'])
 for width in widths:
  rows=[x for x in res['records'] if x['width']==width and x['phase']=='timed']
  assert len(rows)==12
  assert {(x['order'],x['repeat'],x['start']) for x in rows}=={(o,t,b) for o in [0,1] for t in [0,1] for b in [0,8,16]}
  for row in rows:assert row['tokens']==sum(map(len,results['ar']['outputs'][row['start']:row['start']+8]))
  summary['throughputs'][f'{role}_{width}']=sum(x['tokens'] for x in rows)/sum(x['seconds'] for x in rows)
 if role=='forced':
  assert len(res['audits'])==48
  for check in res['audits']:
   assert check['target_unchanged'] and check['logging_disabled'] and check['forced_steps']>0
   assert check['backbone_graphs']>0 and check['pre_graphs']>0 and check['tail_graphs']>0
   assert check['policy_source_sha256']==hashlib.sha256((r/'forced_policy_v69.py').read_bytes()).hexdigest()
forced={k:v for k,v in summary['throughputs'].items() if k.startswith('forced_')}
best=max(forced,key=forced.get);summary['best_forced_width']=best
summary['best_forced_vs_forced8_percent']=100*(forced[best]/forced['forced_8']-1)
summary['best_forced_vs_native8_percent_DIAGNOSTIC_ONLY']=100*(forced[best]/summary['throughputs']['native8_8']-1)
(r/'forced_v69_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
