"""Paired held-out throughput analysis, conditional and engine-resampled intervals."""
import argparse,json
from pathlib import Path
import numpy as np
ROLES=['ar','fixed','original','measured']
def compare(base,proposed):
 # Arrays: 12 prompt batches (four per domain), four engine rounds, two repeats.
 assert base.shape==proposed.shape==(12,4,2)
 assert np.isfinite(base).all() and np.isfinite(proposed).all() and (base>0).all() and (proposed>0).all()
 rng=np.random.default_rng(18657)
 groups=np.concatenate([rng.integers(offset,offset+4,size=(10000,4)) for offset in [0,4,8]],axis=1)
 bt=base.sum(axis=2);pt=proposed.sum(axis=2)
 conditional=100*(bt[groups].sum(axis=(1,2))/pt[groups].sum(axis=(1,2))-1)
 rounds=rng.integers(0,4,(10000,4))
 b=bt[groups[:,:,None],rounds[:,None,:]];p=pt[groups[:,:,None],rounds[:,None,:]]
 hierarchical=100*(b.sum(axis=(1,2))/p.sum(axis=(1,2))-1)
 return dict(gain_percent=float(100*(base.sum()/proposed.sum()-1)),round_gains_percent=(100*(base.sum(axis=(0,2))/proposed.sum(axis=(0,2))-1)).tolist(),stratified_batch_bootstrap_95=np.quantile(conditional,[.025,.975]).tolist(),batch_and_round_bootstrap_95=np.quantile(hierarchical,[.025,.975]).tolist())
def analyze(root):
 plan=json.loads((root/'cost_v26_plan.json').read_text());data={(role,rd):json.loads((root/f'cost_v26_{role}-round{rd}.json').read_text()) for role in ROLES for rd in range(4)}
 reference=data['ar',0]['outputs'];assert len(reference)==384
 for key,x in data.items():
  assert x['status']=='completed' and x['outputs']==reference
  assert (x['metadata']['role'],x['metadata']['round'])==key
  assert x['metadata']['manifest_sha256']==plan['manifest_sha256']
 for field in ['selection_sha256','source_sha256','cost_source_sha256']:assert len({x['metadata'][field] for x in data.values()})==1
 times={role:np.zeros((12,4,2)) for role in ROLES};rates={}
 for (role,rd),x in data.items():
  rows=x['records'];lookup={(z['start'],z['repeat']):z for z in rows}
  assert len(rows)==len(lookup)==24 and set(lookup)=={(s,rep) for s in range(0,384,32) for rep in range(2)}
  for (start,rep),row in lookup.items():
   assert row['tokens']==sum(len(seq) for seq in reference[start:start+32])
   assert row['domain']==['alpaca','gsm8k','mbpp'][start//128]
   times[role][start//32,rd,rep]=row['seconds']
 for role in ROLES:rates[role]=sum(len(x) for x in reference)*8/float(times[role].sum())
 comparisons={role:compare(times[role],times['measured']) for role in ['ar','fixed','original']}
 domains={}
 for index,domain in enumerate(['alpaca','gsm8k','mbpp']):
  domains[domain]={role:float(100*(times[role][index*4:index*4+4].sum()/times['measured'][index*4:index*4+4].sum()-1)) for role in ['ar','fixed','original']}
 passed=all(min(v['round_gains_percent'])>0 and v['stratified_batch_bootstrap_95'][0]>0 and v['batch_and_round_bootstrap_95'][0]>0 for v in comparisons.values())
 return dict(scope='Held-out batch32 greedy throughput on one H200 with one model pair and batch-invariant FP32 kernels',passed_predeclared_speed_criteria=passed,selected_fixed=data['fixed',0]['metadata']['method'],throughputs=rates,comparisons=comparisons,domain_gain_percent=domains,manifest_sha256=plan['manifest_sha256'],runs={f'{role}-round{rd}':x['run_url'] for (role,rd),x in data.items()},limitations=['384fresh prompts, only four prompt batches per domain and four engine rounds; intervals do not establish generality across hardware, models or concurrency.','Conditional interval resamples prompt batches within domain; second interval also resamples the four observed paired engine rounds.','Code-domain evaluation uses MBPP whereas validation used HumanEval.','Cost-model correction only; no online calibration benefit or novel scheduler claim.'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=analyze(a.root);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
