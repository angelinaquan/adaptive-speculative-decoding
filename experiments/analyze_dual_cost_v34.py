import json,hashlib,math
from pathlib import Path
import numpy as np

def analyze(root):
 x=json.loads((root/'dual_cost_v34_results.json').read_text());assert x['status']=='completed'
 for k,n in [('source_sha256','pilot_dual_cost_v34.py'),('module_sha256','dual_cost_v34.py'),('manifest_sha256','cost_v26_manifest.json')]:assert x[k]==hashlib.sha256((root/n).read_bytes()).hexdigest()
 ref=json.loads((root/'cost_v29_ar.json').read_text())['outputs'];rows=x['records'];methods=['original','target_only','both']
 assert len(rows)==54 and {(z['round'],z['method'],z['start']) for z in rows}=={(rd,m,s) for rd in range(3) for m in methods for s in range(0,384,64)}
 for z in rows:assert math.isfinite(z['seconds']) and z['seconds']>0 and z['tokens']==sum(map(len,ref[z['start']:z['start']+64]))
 raw=x['costs']['both']['raw_draft_samples'];sizes=np.array(sorted({n for n,v in raw}));values=np.array([np.median([v for n,v in raw if n==size]) for size in sizes]);assert np.array_equal(values[np.searchsorted(sizes,np.arange(65))],x['costs']['both']['cost_tables'][0]);assert x['costs']['both']['cost_tables'][1]==x['costs']['target_only']['cost_tables'][1]
 def rate(m,rd=None):
  selected=[z for z in rows if z['method']==m and (rd is None or z['round']==rd)];return sum(z['tokens'] for z in selected)/sum(z['seconds'] for z in selected)
 comparisons={base:dict(gain_percent=100*(rate('both')/rate(base)-1),round_gains_percent=[100*(rate('both',rd)/rate(base,rd)-1) for rd in range(3)]) for base in ['original','target_only']}
 return dict(passed=True,scope=x['plan']['scope'],throughputs={m:rate(m) for m in methods},both_vs=comparisons,run_url=x['run_url'],interpretation='No consistent improvement over target-only: mean is driven by one slow target-only block; retain all observations. No fresh or significant gain claim.')
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();x=analyze(a.root);(a.root/'dual_cost_v34_summary.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x,indent=2))
