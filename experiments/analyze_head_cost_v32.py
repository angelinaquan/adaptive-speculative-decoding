"""Descriptive balanced-stream pilot; no independent-batch inference."""
import json,math,hashlib
from pathlib import Path

def analyze(root):
 x=json.loads((root/'head_cost_v32_results.json').read_text());assert x['status']=='completed'
 ref=json.loads((root/'cost_v29_ar.json').read_text());assert ref['status']=='completed';outputs=ref['outputs'];assert len(outputs)==384
 for field,name in [('source_sha256','pilot_head_cost_v32.py'),('module_sha256','head_cost_v32.py'),('manifest_sha256','cost_v26_manifest.json'),('source_fit_file_sha256','head_calibration_v18_calibration.json')]:assert x[field]==hashlib.sha256((root/name).read_bytes()).hexdigest()
 methods=['identity','frozen','online100'];rows=x['records'];keys={(z['round'],z['method'],z['start']) for z in rows};assert len(rows)==len(keys)==54 and keys=={(rd,m,start) for rd in range(3) for m in methods for start in range(0,384,64)}
 for row in rows:
  assert math.isfinite(row['seconds']) and row['seconds']>0 and row['tokens']==sum(map(len,outputs[row['start']:row['start']+64]))
  assert row['domain']==['alpaca','gsm8k','mbpp'][row['start']//128]
 states={(z['round'],z['method']):z for z in x['states']};assert len(x['states'])==len(states)==9
 for (rd,m),state in states.items():
  initial=state['initial'];final=state['final'];assert initial['refits']==0 and initial['learning']==(m=='online100')
  if m=='online100':assert final['refits']>0
  else:assert final['refits']==0 and initial['slope']==final['slope'] and initial['intercepts']==final['intercepts']
 def rate(m,rd=None,d=None):
  selected=[z for z in rows if z['method']==m and (rd is None or z['round']==rd) and (d is None or z['domain']==d)]
  return sum(z['tokens'] for z in selected)/sum(z['seconds'] for z in selected)
 comparisons={}
 for base,method in [('identity','frozen'),('identity','online100'),('frozen','online100')]:
  comparisons[method+'_vs_'+base]=dict(gain_percent=100*(rate(method)/rate(base)-1),round_gains_percent=[100*(rate(method,rd)/rate(base,rd)-1) for rd in range(3)],domain_gain_percent={d:100*(rate(method,d=d)/rate(base,d=d)-1) for d in ['alpaca','gsm8k','mbpp']})
 return dict(scope=x['plan']['scope'],throughputs={m:rate(m) for m in methods},comparisons=comparisons,run_url=x['run_url'],checks_passed=True,limitations=['Descriptive reused-prompt single-engine pilot. Online state couples batches; no independent-prompt confidence interval.','Identity includes shared extension overhead. A benefit must be confirmed against native zero-overhead cost policy and strongest fixed baseline.'])
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();print(json.dumps(analyze(a.root),indent=2))
