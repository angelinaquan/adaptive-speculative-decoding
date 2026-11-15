import json,hashlib,math
from pathlib import Path
r=Path(__file__).resolve().parent;plan=json.loads((r/'minimal_v79_plan.json').read_text());summary=json.loads((r/'minimal_v79_summary.json').read_text());ref=json.loads((r/'profile_v77_b64_w0_r-1.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert sha(r/'profile_v77_manifest.json')==plan['source_manifest_sha256'];assert summary['passed'] and ref['status']=='completed'
assert len(set(sum(plan['panels'],[])))==576
checks=0;panels=[]
for panel in range(3):
 measured=[];expected=[ref['outputs'][i] for i in plan['panels'][panel]]
 for w in range(1,9):
  x=json.loads((r/f'minimal_v79_p{panel}_w{w}.json').read_text());assert x['status']=='completed' and x['panel']==panel and x['width']==w
  assert x['source_sha256']==sha(r/'benchmark_minimal_v79.py') and x['plan_sha256']==sha(r/'minimal_v79_plan.json')
  assert x['config']['max_num_seqs']==64 and x['config']['dtype']=='bfloat16' and x['config']['speculative_config']['num_speculative_tokens']==w and not x['config']['enable_prefix_caching']
  assert len(x['records'])==6 and {(v['repeat'],v['start']) for v in x['records']}=={(rep,j) for rep in [-1,0] for j in [0,64,128]}
  for v in x['records']:
   ids=expected[v['start']:v['start']+64];assert v['parity'] and v['output_sha256']==hashlib.sha256(json.dumps(ids).encode()).hexdigest() and v['tokens']==sum(map(len,ids));checks+=1
  rec=next(z for z in summary['records'] if z['panel']==panel and z['width']==w);assert rec['timed_seconds']==sum(v['seconds'] for v in x['records'] if v['repeat']==0);assert rec['generation_seconds_including_warmup']==sum(v['seconds'] for v in x['records']);assert rec['subprocess_wall_seconds']>=x['wall_seconds_before_shutdown'];measured.append(rec)
 ordered=sorted(measured,key=lambda z:(z['timed_seconds'],z['width']));k=ordered[0]['width'];assert k==summary['panels'][panel]['selected_width'];wall=sum(z['subprocess_wall_seconds'] for z in measured);gen=sum(z['generation_seconds_including_warmup'] for z in measured);assert wall==summary['panels'][panel]['total_measured_profiling_seconds'] or math.isclose(wall,summary['panels'][panel]['total_measured_profiling_seconds'],abs_tol=1e-10)
 rates=next(z for z in json.loads((r/'transfer_v78_summary.json').read_text())['comparisons'] if z['batch']==64 and z['domain']=='pooled')['throughputs'];assert math.isclose(wall/(1/rates[7]-1/rates[k-1]),summary['panels'][panel]['break_even_tokens_vs8'],rel_tol=1e-12)
 panels.append({'panel':panel,'selected_width':k,'second_width':ordered[1]['width'],'profile_gap_percent':100*(ordered[1]['timed_seconds']/ordered[0]['timed_seconds']-1),'non_generation_share_percent':100*(1-gen/wall)})
out={'passed':True,'sessions':24,'exact_batch_output_checks':checks,'panels':panels,'wall_time_evidence':'Controller-recorded subprocess elapsed; arithmetic verified, not independently remeasured.','test_evidence':'v78 throughput lookup, not another test-generation run.'};(r/'minimal_v79_local_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
