import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,hashlib
from pathlib import Path
import numpy as np
r=Path(__file__).resolve().parent;plan=json.loads((r/'load_v76_plan.json').read_text());manifest=json.loads((r/'load_v76_manifest.json').read_text())
refs={b:json.loads((r/f'load_v76_b{b}_w0_r-1.json').read_text()) for b in [8,64]}
assert all(x['status']=='completed' for x in refs.values()) and refs[8]['outputs']==refs[64]['outputs'];assert len(refs[8]['outputs'])==768
source_sha=hashlib.sha256((r/'benchmark_load_v76.py').read_bytes()).hexdigest();plan_sha=hashlib.sha256((r/'load_v76_plan.json').read_bytes()).hexdigest()
assert hashlib.sha256((r/'load_v76_manifest.json').read_bytes()).hexdigest()==plan['manifest_sha256']
times={};tokens={};summary={'passed':True,'scope':plan['scope'],'throughputs':{},'comparisons':{},'round_gains_percent':{},'domains':{}}
for b in [8,64]:
 for w in [3,8]:
  ts=np.zeros((4,12));ns=np.zeros((4,12))
  for order in range(4):
   x=json.loads((r/f'load_v76_b{b}_w{w}_r{order}.json').read_text());assert x['status']=='completed' and x['batch']==b and x['width']==w and x['round']==order
   assert x['source_sha256']==source_sha and x['plan_sha256']==plan_sha and x['manifest_sha256']==plan['manifest_sha256']
   assert x['config']['max_num_seqs']==b and x['config']['dtype']=='bfloat16' and x['config']['speculative_config']['num_speculative_tokens']==w
   assert len(x['records'])==4*768//b
   assert {(v['repeat'],v['start']) for v in x['records']}=={(rep,start) for rep in [-2,-1,0,1] for start in range(0,768,b)}
   for row in x['records']:
    ids=refs[b]['outputs'][row['start']:row['start']+b]
    assert row['parity'] and row['tokens']==sum(map(len,ids)) and row['output_sha256']==hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    assert row['domain']==manifest[row['start']]['domain']
    if row['repeat']>=0:ts[order,row['start']//64]+=row['seconds']/2;ns[order,row['start']//64]+=row['tokens']/2
  times[b,w]=ts;tokens[b,w]=ns;summary['throughputs'][f'b{b}w{w}']=float(ns.sum()/ts.sum())
for b in [8,64]:assert np.array_equal(tokens[b,3],tokens[b,8])
assert np.array_equal(tokens[8,3],tokens[64,3])
# Gain directions are fixed in the plan: long draft atlowload, short athighload.
def gains(sums):return [100*(sums[8,3]/sums[8,8]-1),100*(sums[64,8]/sums[64,3]-1),np.log(sums[8,3]/sums[8,8])-np.log(sums[64,3]/sums[64,8])]
point=gains({k:v.sum() for k,v in times.items()});rng=np.random.default_rng(18657);draws=[]
for _ in range(10000):
 orders=rng.integers(0,4,4);blocks=np.concatenate([d*4+rng.integers(0,4,4) for d in range(3)])
 draws.append(gains({k:v[np.ix_(orders,blocks)].sum() for k,v in times.items()}))
draws=np.array(draws)
for i,name in enumerate(['batch8_width8_vs3_percent','batch64_width3_vs8_percent','log_time_interaction']):summary['comparisons'][name]={'estimate':float(point[i]),'paired_bootstrap_95':np.percentile(draws[:,i],[2.5,97.5]).tolist()}
for b,method,baseline in [(8,8,3),(64,3,8)]:
 summary['round_gains_percent'][str(b)]=(100*(times[b,baseline].sum(axis=1)/times[b,method].sum(axis=1)-1)).tolist()
 for d,name in enumerate(['alpaca','gsm8k','mbpp']):summary['domains'][f'b{b}_{name}']=float(100*(times[b,baseline][:,d*4:(d+1)*4].sum()/times[b,method][:,d*4:(d+1)*4].sum()-1))
high=summary['comparisons']['batch64_width3_vs8_percent'];summary['primary_pass']=high['estimate']>10 and high['paired_bootstrap_95'][0]>0;summary['limitations']=plan['limitations'];summary['bootstrap_scope']=plan['uncertainty']
(r/'load_v76_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
