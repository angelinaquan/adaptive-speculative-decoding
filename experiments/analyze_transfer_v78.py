import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,hashlib
from pathlib import Path
import numpy as np
r=Path(__file__).resolve().parent;plan=json.loads((r/'transfer_v78_plan.json').read_text());tasks=json.loads((r/'transfer_v78_manifest.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(r/'transfer_v78_manifest.json')==plan['manifest_sha256'];ref={b:json.loads((r/f'transfer_v78_b{b}_w0_r-1.json').read_text()) for b in [8,64]};assert all(x['status']=='completed' for x in ref.values()) and ref[8]['outputs']==ref[64]['outputs'];assert len(ref[8]['outputs'])==384
T=np.zeros((2,8,2,6));N=np.zeros_like(T);checks=0
for bi,b in enumerate([8,64]):
 for wi,w in enumerate(range(1,9)):
  for o in range(2):
   x=json.loads((r/f'transfer_v78_b{b}_w{w}_r{o}.json').read_text());assert x['status']=='completed' and (x['batch'],x['width'],x['round'])==(b,w,o)
   assert x['source_sha256']==sha(r/'benchmark_transfer_v78.py') and x['plan_sha256']==sha(r/'transfer_v78_plan.json') and x['manifest_sha256']==plan['manifest_sha256']
   assert x['config']['max_num_seqs']==b and x['config']['speculative_config']['num_speculative_tokens']==w
   assert len(x['records'])==4*384//b and {(v['repeat'],v['start']) for v in x['records']}=={(rep,j) for rep in [-2,-1,0,1] for j in range(0,384,b)}
   for v in x['records']:
    ids=ref[b]['outputs'][v['start']:v['start']+b];assert v['parity'] and v['output_sha256']==hashlib.sha256(json.dumps(ids).encode()).hexdigest() and v['tokens']==sum(map(len,ids));checks+=1
    assert v['domain']==tasks[v['start']]['domain']
    if v['repeat']>=0:T[bi,wi,o,v['start']//64]+=v['seconds']/2;N[bi,wi,o,v['start']//64]+=v['tokens']/2
assert np.all(N==N[:,0:1]) and np.array_equal(N[0],N[1])
rng=np.random.default_rng(1865778);summary={'passed':True,'checks':checks,'scope':plan['question'],'comparisons':[],'limitations':plan['limitations']}
for bi,b in enumerate([8,64]):
 k=plan['frozen_selected_widths'][str(b)]-1
 for name,ix in [('pooled',list(range(6))),('alpaca',[0,1]),('math500',[2,3]),('lbpp',[4,5])]:
  times=T[bi][:,:,ix].mean(axis=1).sum(axis=1);tokens=N[bi,0][:,ix].mean(axis=0).sum();best=int(np.argmin(times));gain=100*(times[7]/times[k]-1);boots=[]
  for _ in range(5000):
   rr=rng.integers(0,2,2);jj=np.concatenate([rng.choice([2*d,2*d+1],2) for d in range(3)]) if name=='pooled' else rng.choice(ix,len(ix));boots.append(100*(T[bi,7][np.ix_(rr,jj)].sum()/T[bi,k][np.ix_(rr,jj)].sum()-1))
  summary['comparisons'].append({'batch':b,'domain':name,'frozen_width':k+1,'hindsight_best_width':best+1,'throughputs':(tokens/times).tolist(),'loss_to_best_percent':float(100*(1-times.min()/times[k])),'gain_vs8_percent':float(gain),'paired_bootstrap95_vs8':np.percentile(boots,[2.5,97.5]).tolist(),'gain_vs8_by_round':[float(100*(T[bi,7,o,ix].sum()/T[bi,k,o,ix].sum()-1)) for o in range(2)]})
summary['primary_pass']=all(z['loss_to_best_percent']<=5 for z in summary['comparisons'] if z['domain']=='pooled')
(r/'transfer_v78_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
