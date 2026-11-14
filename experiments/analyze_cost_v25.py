"""Audit the complete static-width sweep before any held-out launch."""
import argparse,hashlib,json,math
from pathlib import Path
METHODS=['ar','original','measured']+[f'fixed{i}' for i in range(1,8)]
def analyze(root):
 data={m:json.loads((root/f'cost_v25_{m}.json').read_text()) for m in METHODS}
 assert all(x['status']=='completed' and x['method']==m for m,x in data.items())
 assert len({x['manifest_sha256'] for x in data.values()})==1
 metrics={};tokens={}
 for method,x in data.items():
  rows=x['records'];lookup={(z['repeat'],z['start']):z for z in rows}
  assert len(rows)==len(lookup)==15 and set(lookup)=={(rd,start) for rd in range(3) for start in range(0,144,32)}
  for key,row in lookup.items():
   assert math.isfinite(row['seconds']) and row['seconds']>0 and row['tokens']>0
   if key in tokens:assert tokens[key]==row['tokens']
   tokens[key]=row['tokens']
  throughput=sum(z['tokens'] for z in rows)/sum(z['seconds'] for z in rows)
  assert math.isclose(throughput,x['throughput'],rel_tol=1e-10)
  metrics[method]=dict(tokens_per_second=throughput,run_url=x['run_url'],source_sha256=x['source_sha256'])
 best=max([m for m in METHODS if m.startswith('fixed')],key=lambda m:metrics[m]['tokens_per_second'])
 measured=metrics['measured']['tokens_per_second']
 return dict(scope='Validation selection; unbalanced separate-engine order, no significance claim',methods=metrics,selected_fixed=best,measured_gains_percent={m:100*(measured/metrics[m]['tokens_per_second']-1) for m in ['ar','original',best]},launch_gate=all(measured>metrics[m]['tokens_per_second'] for m in ['ar','original',best]),manifest_sha256=data['ar']['manifest_sha256'],teardown_note='fixed4 used original runner; later methods add explicit bounded shutdown after artifact upload, outside timing. Original source preserved as benchmark_cost_v25_pre_shutdown.py.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=analyze(a.root);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
