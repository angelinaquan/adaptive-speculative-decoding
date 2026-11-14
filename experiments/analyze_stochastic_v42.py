"""Exploratory throughput and prespecified finite-power sampling diagnostics."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix


def prefix_test(a,b,draws=5000,seed=424242):
    assert len(a)==len(b);n=len(a);rows=[];cols=[];vocabulary={}
    for i,seq in enumerate(a+b):
        padded=(seq+[-1]*4)[:4]
        for length in range(1,5):
            key=(length,tuple(padded[:length]))
            if key not in vocabulary:vocabulary[key]=len(vocabulary)
            rows.append(i);cols.append(vocabulary[key])
    features=csr_matrix((np.ones(len(rows),dtype=np.float32),(rows,cols)),shape=(2*n,len(vocabulary)))
    signs=np.r_[np.ones(n),-np.ones(n)]
    observed=float(np.square(features.T@signs).sum()/n**2)
    rng=np.random.default_rng(seed);exceed=0
    for start in range(0,draws,128):
        k=min(128,draws-start);selected=np.argsort(rng.random((k,2*n)),axis=1)[:,:n]
        weights=-np.ones((k,2*n),dtype=np.float32);weights[np.arange(k)[:,None],selected]=1
        distances=np.square(features.T@weights.T).sum(axis=0)/n**2
        exceed+=int(np.count_nonzero(distances>=observed-1e-12))
    return dict(statistic=observed,permutation_p=(exceed+1)/(draws+1),permutations=draws,draws_per_method=n)


def analyze(root):
    plan=json.loads((root/'stochastic_v42_plan.json').read_text());methods=plan['methods']
    data={m:json.loads((root/f'stochastic_v42_{m}.json').read_text()) for m in methods}
    reference=json.loads((root/'stochastic_v42_independent_ar.json').read_text());assert reference['status']=='completed' and reference['greedy_gate_passed']
    assert reference['source_sha256']==hashlib.sha256((root/'reference_sampling_v42.py').read_bytes()).hexdigest()
    assert reference['manifest_sha256']==hashlib.sha256((root/'cost_v26_manifest.json').read_bytes()).hexdigest()
    assert reference['amendment']==json.loads((root/'stochastic_v42_audit_amendment.json').read_text())
    assert reference['amendment']['new_ar_seed_base']==1700000000
    assert [z['prompt_index'] for z in reference['sampling_audit']]==[0,128,256]
    assert all(len(z['outputs'])==512 for z in reference['sampling_audit'])
    for m,x in data.items():
        assert x['status']=='completed' and x['greedy_gate_passed']
        metadata=x['metadata'];assert metadata['sampling_plan']==plan
        for key,file in [('source_sha256','benchmark_stochastic_v42.py'),('cost_source_sha256','cost_only_v26.py'),('manifest_sha256','cost_v26_manifest.json')]:assert metadata[key]==hashlib.sha256((root/file).read_bytes()).hexdigest()
        assert len(x['outputs'])==384 and len(x['records'])==12
        assert {(z['start'],z['repeat']) for z in x['records']}=={(s,i) for s in range(0,384,64) for i in range(2)}
        for z in x['records']:
            assert z['tokens']==sum(map(len,x['outputs'][z['start']:z['start']+64]))
            assert z['domain']==['alpaca','gsm8k','mbpp'][z['start']//128]
            assert np.isfinite(z['seconds']) and z['seconds']>0
        assert [z['prompt_index'] for z in x['sampling_audit']]==[0,128,256]
        for z in x['sampling_audit']:
            assert len(z['outputs'])==512
            assert all(len(seq)<=8 and all(isinstance(t,int) and t>=0 for t in seq) for seq in z['outputs'])
    def totals(method,domain=None):
        rows=[z for z in data[method]['records'] if domain is None or z['domain']==domain]
        return sum(z['tokens'] for z in rows),sum(z['seconds'] for z in rows)
    def rate(method,domain=None):
        tokens,seconds=totals(method,domain);return tokens/seconds
    rates={m:rate(m) for m in methods};fixed=[f'fixed{i}' for i in range(1,8)];best=max(fixed,key=lambda m:rates[m]);domains=['alpaca','gsm8k','mbpp']
    domain_best={d:max(fixed,key=lambda m:rate(m,d)) for d in domains}
    pairs=[totals(domain_best[d],d) for d in domains];domain_rate=sum(z[0] for z in pairs)/sum(z[1] for z in pairs)
    comparisons={best:100*(rates['measured']/rates[best]-1),'domain_fixed':100*(rates['measured']/domain_rate-1),'original':100*(rates['measured']/rates['original']-1)}
    tests=[]
    for m in methods[1:]:
        for i,d in enumerate(domains):
            z=prefix_test(reference['sampling_audit'][i]['outputs'],data[m]['sampling_audit'][i]['outputs']);z.update(method=m,domain=d);tests.append(z)
    threshold=.01/27;detected=[z for z in tests if z['permutation_p']<=threshold]
    return dict(scope=plan['scope'],rates=rates,best_fixed=best,domain_best_fixed=domain_best,domain_fixed_rate=domain_rate,gain_percent=comparisons,domain_rates={d:{m:rate(m,d) for m in methods} for d in domains},sampling_tests=tests,bonferroni_threshold=threshold,detected_distribution_differences=detected,no_distribution_difference_detected=not detected,passed_exploratory_promotion_gate=not detected and min(comparisons.values())>=5,runs={m:x['run_url'] for m,x in data.items()},independent_reference_run=reference['run_url'],audit_amendment=reference['amendment'],limitations=['Unbalanced engine order; reused prompts; selection on exploratory results. No fresh speed claim.','512 samples on one prompt/domain is a finite-power diagnostic; no detected difference is NOT distribution equivalence.','Generated lengths differ between policies; rates use actual output tokens.','Greedy parity, within-policy seeded repeatability, and finite-state kernel/allocator tests cover complementary properties, not a general proof.'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();s=analyze(a.root);a.output.write_text(json.dumps(s,indent=2)+'\n');print(json.dumps({k:v for k,v in s.items() if k!='sampling_tests'},indent=2))
