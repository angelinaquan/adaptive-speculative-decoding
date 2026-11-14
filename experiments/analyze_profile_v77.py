"""Audit native sweep and evaluate selectors without test feedback."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,hashlib
from pathlib import Path
import numpy as np
r=Path(__file__).resolve().parent
plan=json.loads((r/'profile_v77_plan.json').read_text());manifest=json.loads((r/'profile_v77_manifest.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(r/'profile_v77_manifest.json')==plan['manifest_sha256']
ref={b:json.loads((r/f'profile_v77_b{b}_w0_r-1.json').read_text()) for b in [8,64]}
assert all(x['status']=='completed' for x in ref.values()) and ref[8]['outputs']==ref[64]['outputs']
# Each observation is an entire contiguous64promptblock, averaged across timed passes.
T=np.zeros((2,8,2,21));N=np.zeros_like(T);W=np.zeros_like(T);init=np.zeros((2,8,2));warm=np.zeros_like(init)
for bi,b in enumerate([8,64]):
 for wi,w in enumerate(range(1,9)):
  for order in range(2):
   x=json.loads((r/f'profile_v77_b{b}_w{w}_r{order}.json').read_text());assert x['status']=='completed'
   assert (x['batch'],x['width'],x['round'])==(b,w,order)
   assert x['source_sha256']==sha(r/'benchmark_profile_v77.py') and x['plan_sha256']==sha(r/'profile_v77_plan.json') and x['manifest_sha256']==plan['manifest_sha256']
   c=x['config'];assert c['max_num_seqs']==b and c['speculative_config']['num_speculative_tokens']==w and c['dtype']=='bfloat16' and not c['enable_prefix_caching']
   assert len(x['records'])==4*1344//b and {(v['repeat'],v['start']) for v in x['records']}=={(rep,start) for rep in [-2,-1,0,1] for start in range(0,1344,b)}
   for v in x['records']:
    ids=ref[b]['outputs'][v['start']:v['start']+b];assert v['parity'] and v['tokens']==sum(map(len,ids)) and v['output_sha256']==hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    assert v['split']==manifest[v['start']]['split'] and v['domain']==manifest[v['start']]['domain']
    if v['repeat']>=0:T[bi,wi,order,v['start']//64]+=v['seconds']/2;N[bi,wi,order,v['start']//64]+=v['tokens']/2
    else:
     warm[bi,wi,order]+=v['seconds'];W[bi,wi,order,v['start']//64]+=v['seconds']
   init[bi,wi,order]=x['engine_init_seconds']
assert np.all(N==N[:,0:1,:,:]) and np.array_equal(N[0],N[1])
# Domain layout: fourprofileblocks, onevalidationblock, twotestblocks.
profile=np.array([[d*7+i for i in range(4)] for d in range(3)]);val=np.array([d*7+4 for d in range(3)]);test=np.array([[d*7+5,d*7+6] for d in range(3)])
t=T.mean(axis=2);n=N.mean(axis=2);domains=['alpaca','gsm8k','mbpp']
choose=lambda a:int(np.argmin(a))
def selectors(a):
 return {'global':choose(a[:,:,val].sum(axis=(0,2))),'batch':[choose(a[b][:,val].sum(axis=1)) for b in range(2)],'domain':[[choose(a[b,:,val[d]]) for d in range(3)] for b in range(2)]}
baselines=selectors(t)
poison=t.copy();poison[:,:,test.flatten()]=np.random.default_rng(1).uniform(1,1000,(2,8,6));assert selectors(poison)==baselines
records=[];rng=np.random.default_rng(plan['seed'])
for bi,b in enumerate([8,64]):
 for source in range(3):
  for draw in range(plan['subsets']):
   perm=rng.permutation(4)
   for blocks in [1,2,4]:
    selected_blocks=profile[source,perm[:blocks]];cost=t[bi][:,selected_blocks].sum(axis=1);chosen=choose(cost)
    assert chosen==choose(poison[bi][:,selected_blocks].sum(axis=1))
    for target in range(3):
     ix=test[target];test_times=t[bi][:,ix].sum(axis=1);tokens=float(n[bi,0,ix].sum());oracle=choose(test_times)
     row={'batch':b,'source':domains[source],'target':domains[target],'budget':blocks*64,'draw':draw,'width':chosen+1,'test_tps':tokens/test_times[chosen],'loss_to_test_best_percent':100*(1-test_times[oracle]/test_times[chosen]),'profile_generation_seconds':float((2*T[bi]+W[bi])[:,:,selected_blocks].sum())}
     for name,k in [('fixed8',7),('validation_global',baselines['global']),('validation_batch',baselines['batch'][bi]),('validation_domain',baselines['domain'][bi][target])]:
      row['gain_vs_'+name+'_percent']=100*(test_times[k]/test_times[chosen]-1)
     saved=(test_times[7]-test_times[chosen])/tokens
     row['amortization_tokens_vs8']=float((2*T[bi]+W[bi])[:,:,selected_blocks].sum()/saved) if saved>0 else None
     records.append(row)
summary={'passed':True,'scope':plan['question'],'baselines_zero_based':baselines,'selection_test_poison_passed':True,'groups':[],'full_profile':[],'pooled':[],'engine_init_seconds_by_batch_width_round':init.tolist(),'warmup_generation_seconds_by_batch_width_round':warm.tolist(),'limitations':plan['limitations']}
for b in [8,64]:
 for source in domains:
  for target in domains:
   for budget in [64,128,256]:
    rows=[z for z in records if (z['batch'],z['source'],z['target'],z['budget'])==(b,source,target,budget)]
    losses=np.array([z['loss_to_test_best_percent'] for z in rows]);h={str(k):sum(z['width']==k for z in rows) for k in range(1,9)}
    summary['groups'].append({'batch':b,'source':source,'target':target,'budget':budget,'width_counts':h,'mean_loss_percent':float(losses.mean()),'within1_percent':float(100*np.mean(losses<=1)),'within5_percent':float(100*np.mean(losses<=5)),'mean_gain_vs8_percent':float(np.mean([z['gain_vs_fixed8_percent'] for z in rows])),'mean_gain_vs_validation_batch_percent':float(np.mean([z['gain_vs_validation_batch_percent'] for z in rows])),'mean_profile_generation_seconds':float(np.mean([z['profile_generation_seconds'] for z in rows]))})
# Frozen full-profile selectors, round diagnostics, and limited test-block bootstrap.
for bi,b in enumerate([8,64]):
 for source in range(3):
  k=choose(t[bi][:,profile[source]].sum(axis=1))
  for target in range(3):
   ix=test[target];kb=baselines['batch'][bi];roundg=[100*(T[bi,kb,o,ix].sum()/T[bi,k,o,ix].sum()-1) for o in range(2)];boots=[]
   for _ in range(2000):
    rr=rng.integers(0,2,2);jj=rng.choice(ix,2);a=T[bi,kb][np.ix_(rr,jj)].sum();c=T[bi,k][np.ix_(rr,jj)].sum();boots.append(100*(a/c-1))
   summary['full_profile'].append({'batch':b,'source':domains[source],'target':domains[target],'width':k+1,'gain_vs_validation_batch_by_round':roundg,'paired_bootstrap95':np.percentile(boots,[2.5,97.5]).tolist()})
 for budget in [192,768]:
  ix=profile[:,0] if budget==192 else profile.flatten();k=choose(t[bi][:,ix].sum(axis=1));tt=t[bi][:,test.flatten()].sum(axis=1)
  summary['pooled'].append({'batch':b,'profile_total_prompts':budget,'width':k+1,'test_gain_vs8_percent':float(100*(tt[7]/tt[k]-1)),'test_gain_vs_validation_batch_percent':float(100*(tt[baselines['batch'][bi]]/tt[k]-1)),'test_loss_to_best_percent':float(100*(1-tt.min()/tt[k])),'profile_generation_seconds':float((2*T[bi]+W[bi])[:,:,ix].sum())})
(r/'profile_v77_records.json').write_text(json.dumps(records)+'\n');(r/'profile_v77_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'passed':True,'baselines':baselines,'pooled':summary['pooled']},indent=2))
