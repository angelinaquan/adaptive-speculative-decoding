"""Fresh v38 audit with fixed global and prespecified domain-specific baselines."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
ROLES=['fixed3','fixed7','original','measured']
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
 plan=json.loads((root/'cost_v38_plan.json').read_text())
 expected_hashes={k:hashlib.sha256((root/n).read_bytes()).hexdigest() for k,n in [('manifest_sha256','cost_v38_manifest.json'),('plan_sha256','cost_v38_plan.json'),('source_sha256','benchmark_cost_v38.py'),('cost_source_sha256','cost_only_v26.py')]}
 assert plan['manifest_sha256']==expected_hashes['manifest_sha256']
 ref=json.loads((root/'cost_v38_ar-reference.json').read_text());assert ref['status']=='completed';outputs=ref['outputs'];assert len(outputs)==768
 data={(role,rd):json.loads((root/f'cost_v38_{role}-round{rd}.json').read_text()) for role in ROLES for rd in range(4)}
 for key,x in [('reference',ref)]+list(data.items()):
  assert x['status']=='completed' and x['outputs']==outputs
  m=x['metadata'];assert all(m[k]==v for k,v in expected_hashes.items())
  assert m['batch_invariant'] is True and m['tf32_override']=='0'
  assert m['engine']['dtype']=='float32' and m['engine']['max_num_seqs']==64
  assert m['versions']=={'vllm':'0.33.0','torch':'2.14.0','wandb':'0.25.1'}
  if key!='reference':assert (m['role'],m['round'])==key and m['order']==plan['orders'][key[1]]
  rows=x['records'];assert len(rows)==24 and {(z['start'],z['repeat']) for z in rows}=={(s,i) for s in range(0,768,64) for i in range(2)}
  for z in rows:
   assert z['tokens']==sum(map(len,outputs[z['start']:z['start']+64]))
   assert z['domain']==['alpaca','gsm8k','mbpp'][z['start']//256]
   assert np.isfinite(z['seconds']) and z['seconds']>0
 times={m:np.zeros((12,4,2)) for m in ROLES}
 for (role,rd),x in data.items():
  for z in x['records']:times[role][z['start']//64,rd,z['repeat']]=z['seconds']
 assert plan['global_fixed']=='fixed7' and plan['domain_fixed']=={'alpaca':'fixed3','gsm8k':'fixed7','mbpp':'fixed7'}
 times['domain_fixed']=np.concatenate([times[plan['domain_fixed'][d]][i*4:(i+1)*4] for i,d in enumerate(['alpaca','gsm8k','mbpp'])],axis=0)
 rates={m:sum(map(len,outputs))*8/float(t.sum()) for m,t in times.items()}
 comparisons={m:compare(times[m],times['measured']) for m in ['fixed7','domain_fixed','original']}
 statistical=all(min(v['round_gains_percent'])>0 and v['stratified_batch_bootstrap_95'][0]>0 and v['batch_and_round_bootstrap_95'][0]>0 for v in comparisons.values())
 practical=all(v['gain_percent']>=5 for v in comparisons.values())
 domains={d:{m:100*(float(times[m][i*4:(i+1)*4].sum())/float(times['measured'][i*4:(i+1)*4].sum())-1) for m in ['fixed3','fixed7','domain_fixed','original']} for i,d in enumerate(['alpaca','gsm8k','mbpp'])}
 return dict(scope=plan['scope'],all_correctness_and_provenance_checks_passed=True,passed_statistical_criteria=statistical,passed_5_percent_practical_target=practical,passed_primary_success=statistical and practical,throughputs=rates,comparisons=comparisons,domain_gain_percent=domains,hashes=expected_hashes,runs={f'{m}-round{rd}':x['run_url'] for (m,rd),x in data.items()},reference_run=ref['run_url'],limitations=['One model pair, H200, batch64 and FP32 tensor/batch-invariant backend only; globally strict IEEE arithmetic not established.','Only four prompt batches per domain and four engine rounds; repeats are not independent workload samples.','Code source expands from sanitized MBPP257-test subset to official full500-test split, with task-ID and normalized-text overlap exclusions.','Domain-specific fixed baseline is assembled from prespecified choices, not selected on this fresh set.','A positive chat result does not replace mixed-workload primary success; no online-calibration benefit claim.'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();s=analyze(a.root);a.output.write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s,indent=2))
