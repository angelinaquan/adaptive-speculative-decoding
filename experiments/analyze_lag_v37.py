"""Audit lag-allocation pilot; descriptive, no held-out inference."""
import json,math,hashlib
from pathlib import Path

def analyze(root):
 x=json.loads((root/'lag_v37_results.json').read_text());assert x['status']=='completed';assert x['allocation_checks']['passed'] and x['allocation_checks']['cases']==33
 for key,name in [('source_sha256','pilot_lag_v37.py'),('module_sha256','lag_allocation_v37.py'),('manifest_sha256','cost_v26_manifest.json')]:assert x[key]==hashlib.sha256((root/name).read_bytes()).hexdigest()
 methods=['identity','lag1','lag4'];rows=x['records'];assert len(rows)==54 and {(z['round'],z['method'],z['start']) for z in rows}=={(rd,m,s) for rd in range(3) for m in methods for s in range(0,384,64)}
 ref=json.loads((root/'cost_v29_ar.json').read_text());assert ref['status']=='completed'
 for row in rows:
  assert math.isfinite(row['seconds']) and row['seconds']>0
  assert row['tokens']==sum(map(len,ref['outputs'][row['start']:row['start']+64]))
  assert row['domain']==['alpaca','gsm8k','mbpp'][row['start']//128]
 for m,value in [('identity',0),('lag1',1),('lag4',4)]:assert x['costs'][m]['strength']==value
 def rate(m,rd=None,domain=None):
  selected=[z for z in rows if z['method']==m and (rd is None or z['round']==rd) and (domain is None or z['domain']==domain)]
  return sum(z['tokens'] for z in selected)/sum(z['seconds'] for z in selected)
 comparisons={m:dict(gain_percent=100*(rate(m)/rate('identity')-1),round_gains_percent=[100*(rate(m,rd)/rate('identity',rd)-1) for rd in range(3)],domain_gain_percent={d:100*(rate(m,domain=d)/rate('identity',domain=d)-1) for d in ['alpaca','gsm8k','mbpp']}) for m in ['lag1','lag4']}
 return dict(passed=True,scope=x['plan']['scope'],throughputs={m:rate(m) for m in methods},comparisons=comparisons,run_url=x['run_url'],limitations=['Reused validation prompts, same-engine exploratory comparison; no fresh gain claim.','Identity includes custom allocation overhead. Positive results require native baseline, tuned fixed widths, and independent balanced engines.','Finite-batch completion objective; does not demonstrate continuous-serving throughput.'])
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();s=analyze(a.root);(a.root/'lag_v37_summary.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s,indent=2))
