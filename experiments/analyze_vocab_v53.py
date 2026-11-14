import json,hashlib
from pathlib import Path
import numpy as np
def analyze(root):
    plan=json.loads((root/'vocab_v53_plan.json').read_text())
    hashes={key:hashlib.sha256((root/name).read_bytes()).hexdigest() for key,name in [('manifest_sha256','vocab_v53_manifest.json'),('plan_sha256','vocab_v53_plan.json'),('source_sha256','benchmark_vocab_v53.py')]}
    ref=json.loads((root/'vocab_v53_ar-reference.json').read_text())
    assert ref['status']=='completed' and len(ref['outputs'])==192
    data={(role,rd):json.loads((root/f'vocab_v53_{role}-round{rd}.json').read_text()) for role in plan['roles'] for rd in range(2)}
    times={role:np.zeros((3,2,2)) for role in plan['roles']}
    for key,x in [('ar',ref)]+list(data.items()):
        assert x['status']=='completed' and x['outputs']==ref['outputs']
        m=x['metadata'];assert all(m[k]==v for k,v in hashes.items())
        assert m['engine']['dtype']=='float32' and m['engine']['max_num_seqs']==64 and m['batch_invariant']
        rows=x['records'];assert len(rows)==6 and {(z['start'],z['repeat']) for z in rows}=={(s,i) for s in [0,64,128] for i in [0,1]}
        for z in rows:
            assert z['tokens']==sum(map(len,ref['outputs'][z['start']:z['start']+64]))
            assert z['domain']==['alpaca','gsm8k','mbpp'][z['start']//64]
            assert np.isfinite(z['seconds']) and z['seconds']>0
            if key!='ar':times[key[0]][z['start']//64,key[1],z['repeat']]=z['seconds']
        if key!='ar':assert (m['role'],m['round'])==key and m['order']==plan['orders'][key[1]]
    module_hash=hashlib.sha256((root/'active_vocab_v53.py').read_bytes()).hexdigest()
    assert hashlib.sha256((root/'prefix_v52_fit_summary.json').read_bytes()).hexdigest()==plan['source_fit_sha256']
    for (role,rd),x in data.items():
        assert x['metadata']['active_module_sha256']==module_hash
        spec=x['metadata']['engine']['speculative_config']
        assert spec['num_speculative_tokens']==7 and spec['draft_sample_method']=='greedy' and spec['enable_adaptive_verification'] is False
        if role!='native':
            v=x['active_vocab'];expected_hash=hashlib.sha256((root/plan['vocab_files'][role]).read_bytes()).hexdigest()
            assert v['sha256']==expected_hash==plan['vocab_hashes'][role]==x['metadata']['vocab_sha256']
            assert v['size']==(151936 if role=='identity' else 4096)
            assert v['target_parameter_signature_unchanged'] is True
            assert v['projection_check']['rtol']==v['projection_check']['atol']==1e-4
    rates={role:4*sum(map(len,ref['outputs']))/t.sum() for role,t in times.items()}
    fixed='native'
    comparisons={role:{base:dict(gain_percent=float(100*(times[base].sum()/t.sum()-1)),round_gains_percent=(100*(times[base].sum(axis=(0,2))/t.sum(axis=(0,2))-1)).tolist()) for base in ['native','identity','frequency4096']} for role,t in times.items()}
    return dict(scope=plan['scope'],all_output_and_provenance_checks_passed=True,throughputs=rates,best_tested_fixed=fixed,comparisons=comparisons,domain_gain_vs_native={role:{d:float(100*(times['native'][i].sum()/t[i].sum()-1)) for i,d in enumerate(['alpaca','gsm8k','mbpp'])} for role,t in times.items()},runs={f'{role}-{rd}':x['run_url'] for (role,rd),x in data.items()},hashes=hashes,limitations=['Two reversed engine orders and one prompt batch per domain; no significance or fresh-test claim.','Static prefix-weighted vocabulary pilot; online adaptation is not implemented.','Fixed width7 only; all widths, vocabulary budgets, and context-aware vocabulary controls must be retuned before confirmation.'])
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();print(json.dumps(analyze(a.root),indent=2))
