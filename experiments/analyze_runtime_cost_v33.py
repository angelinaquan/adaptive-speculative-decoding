"""Audit actual-event records against CPU budget counts before cost summaries."""
import json,hashlib,math
from pathlib import Path
import numpy as np

def analyze(root):
 x=json.loads((root/'runtime_cost_v33.json').read_text());assert x['status']=='completed'
 assert x['source_sha256']==hashlib.sha256((root/'probe_runtime_cost_v33.py').read_bytes()).hexdigest()
 assert x['extension_sha256']==hashlib.sha256((root/'diagnostic_runtime_cost_v33.py').read_bytes()).hexdigest()
 assert len(x['traces'])==12 and {(t['mode'],t['start']) for t in x['traces']}=={(m,s) for m in ['original','measured'] for s in range(0,384,64)}
 groups={};alignment=[];unmatched=[]
 for t in x['traces']:
  b=t['budgets'];samples=t['samples']
  # All twelve calls have exactly one initial step before draft-budget selection.
  assert len(samples)==len(b)+1
  unmatched.append(dict(mode=t['mode'],start=t['start'],sample=samples[0]))
  assert not samples[0]['full_graph'],'Unmatched first step must not be silently included as decode'
  samples=samples[1:]
  for u,v in zip(b,samples):
   assert u['requests']==v['requests'] and u['non_draft']+u['budget']==v['tokens']
   assert all(math.isfinite(v[k]) and v[k]>=0 for k in ['forward_ms','draft_ms'])
   if u['non_draft']!=u['requests'] or not v['full_graph']:continue
   bucket=next(n for n in [1,2,4,8,16,32,64,128,256,512] if n>=v['tokens'])
   key=(t['mode'],t['domain'],bucket)
   groups.setdefault(key,[]).append(dict(profile_forward_ms=u['profile_forward_ms'],actual_forward_ms=v['forward_ms'],profile_draft_ms=u['profile_draft_ms'],actual_draft_ms=v['draft_ms'],trimmed=u['budget']<u['available']))
  alignment.append(dict(mode=t['mode'],start=t['start'],matched_steps=len(b),unmatched_initial_steps=1))
 summary=[]
 for (mode,domain,bucket),rows in sorted(groups.items()):
  summary.append(dict(mode=mode,domain=domain,bucket=bucket,n=len(rows),trimmed_steps=sum(z['trimmed'] for z in rows),**{k:float(np.median([z[k] for z in rows])) for k in ['profile_forward_ms','actual_forward_ms','profile_draft_ms','actual_draft_ms']}))
 return dict(passed=True,scope='Instrumented GPU-event diagnostic, not throughput evidence. Matched full-graph decode steps only; unmatched initial steps retained separately.',alignment=alignment,unmatched_initial_steps=unmatched,group_medians=summary,run_url=x['run_url'],limitations=['Sampling and host scheduling overhead are not fully represented by target+draft events.','Profile and runtime request shapes/contexts differ; aggregate residuals are descriptive, not causal latency estimates.','Observed methods induce different step distributions. Do not infer speed gains by comparing their median GPU-event times.'])
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();s=analyze(a.root);(a.root/'runtime_cost_v33_summary.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps({'passed':s['passed'],'matched_steps':sum(z['matched_steps'] for z in s['alignment']),'run_url':s['run_url']},indent=2))
