"""Audit exploratory batch128 sweep; do not infer significance from this sweep."""
import json,hashlib,math
from pathlib import Path

def analyze(root):
 plan=json.loads((root/'cost_v48_plan.json').read_text())
 data={m:json.loads((root/f'cost_v48_{m}.json').read_text()) for m in plan['methods']}
 ref=data['ar']['outputs'];assert len(ref)==384
 rates={};domain_times={};hashes={}
 for method,x in data.items():
  assert x['status']=='completed' and x['outputs']==ref
  m=x['metadata'];assert m['method']==method and m['batch']==128 and m['engine']['max_num_seqs']==128
  for key,name in [('manifest_sha256','cost_v26_manifest.json'),('source_sha256','benchmark_cost_v48.py'),('cost_source_sha256','cost_only_v26.py')]:
   digest=hashlib.sha256((root/name).read_bytes()).hexdigest();assert m[key]==digest;hashes[key]=digest
  rows=x['records'];keys={(z['start'],z['repeat']) for z in rows};assert len(rows)==len(keys)==6 and keys=={(start,rep) for start in range(0,384,128) for rep in range(2)}
  domain_times[method]={d:0. for d in ['alpaca','gsm8k','mbpp']}
  for row in rows:
   assert row['tokens']==sum(map(len,ref[row['start']:row['start']+128]))
   assert math.isfinite(row['seconds']) and row['seconds']>0
   assert row['domain']==['alpaca','gsm8k','mbpp'][row['start']//128]
   domain_times[method][row['domain']]+=row['seconds']
  rates[method]=sum(z['tokens'] for z in rows)/sum(z['seconds'] for z in rows)
 best=max((f'fixed{i}' for i in range(1,8)),key=rates.get)
 gains={m:100*(rates['measured']/rates[m]-1) for m in ['ar','original',best]}
 domain_best={d:min((f'fixed{i}' for i in range(1,8)),key=lambda m:domain_times[m][d]) for d in ['alpaca','gsm8k','mbpp']}
 domain_baseline_time=sum(domain_times[domain_best[d]][d] for d in domain_best)
 gains['domain_fixed']=100*(domain_baseline_time/sum(domain_times['measured'].values())-1)
 return dict(selected_domain_fixed=domain_best,positive_against_strong_baselines=all(gains[m]>0 for m in ['original',best,'domain_fixed']),scope=plan['scope'],all_correctness_checks_passed=True,throughputs=rates,selected_fixed=best,measured_gain_percent=gains,meets_exploratory_5_percent_gate=all(v>=5 for v in gains.values()),domain_gain_percent={d:{m:100*(domain_times[m][d]/domain_times['measured'][d]-1) for m in ['original',best]} for d in ['alpaca','gsm8k','mbpp']},hashes=hashes,runs={m:x['run_url'] for m,x in data.items()},limitation='Unbalanced engine order and reused prompts. Selection only; expanded balanced cross-load benchmark required. Existing v38 suite is reused, not new unseen data.')
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();print(json.dumps(analyze(a.root),indent=2))
