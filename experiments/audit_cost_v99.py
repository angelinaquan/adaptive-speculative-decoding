import json,hashlib,math
from pathlib import Path
r=Path(__file__).resolve().parent;plan=json.loads((r/'cost_v99_plan.json').read_text());ref=json.loads((r/'cost_v99_ar0_round-1.json').read_text());expected=ref['outputs'];assert len(expected)==192;source_hash=hashlib.sha256((r/'benchmark_cost_v99.py').read_bytes()).hexdigest();plan_hash=hashlib.sha256((r/'cost_v99_plan.json').read_bytes()).hexdigest();sessions=[];checks=0;gpu_batches=0
for role,width,round in [('ar',0,-1),('native',3,0),('forced',3,0),('forced',8,0),('forced',8,1),('forced',3,1),('native',3,1)]:
 s=json.loads((r/f'cost_v99_{role}{width}_round{round}.json').read_text());assert s['status']=='completed' and s['source_sha256']==source_hash and s['plan_sha256']==plan_hash
 for row in s['records']:
  ids=expected[row['start']:row['start']+64];assert row['parity'] and row['tokens']==sum(map(len,ids));assert row['output_sha256']==hashlib.sha256(json.dumps(ids).encode()).hexdigest();assert math.isfinite(row['seconds']) and row['seconds']>0;checks+=1
 if role=='forced':
  assert s['config']['speculative_config']['num_speculative_tokens']==8
  for a in s['audits']:
   assert a['width']==width and a['verification_cap']==3 and a['target_unchanged'] and a['logging_disabled'] and a['quota_calls']>0
   assert a['allocation_source_sha256']==hashlib.sha256((r/'fixed_verify_v99.py').read_bytes()).hexdigest();gpu_batches+=a['exact_gpu_audit_batches']
  if round==0:assert any(a['exact_gpu_audit_batches']>0 for a in s['audits'])
 elif role=='native':assert s['config']['speculative_config']['num_speculative_tokens']==3
 sessions.append(s)
totals={};orders=[]
for rnd in [None,0,1]:
 result={}
 for role,width in [('native',3),('forced',3),('forced',8)]:
  rows=[row for s in sessions if s['role']==role and s['width']==width and (rnd is None or s['round']==rnd) for row in s['records'] if row['phase']=='timed'];result[f'{role}{width}']=sum(x['tokens'] for x in rows)/sum(x['seconds'] for x in rows)
 if rnd is None:totals=result
 else:orders.append({'round':rnd,'throughputs':result,'forced8_vs_native3_percent':100*(result['forced8']/result['native3']-1),'forced8_vs_forced3_percent':100*(result['forced8']/result['forced3']-1)})
summary=json.loads((r/'cost_v99_summary.json').read_text());assert totals==summary['throughputs'];assert gpu_batches>0
out={'passed':True,'sessions':len(sessions),'batch_output_hash_checks':checks,'untimed_gpu_allocation_audit_batches':gpu_batches,'local_throughputs_exactly_match_remote':True,'orders':orders,'scope':'Raw artifact audit; no statistical significance or generalization claim from two orders/three reused domain batches'};(r/'cost_v99_local_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
