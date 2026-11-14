import json,hashlib
from pathlib import Path
import numpy as np
def analyze(root):
    plan=json.loads((root/'draft_v50_plan.json').read_text())
    hashes={key:hashlib.sha256((root/name).read_bytes()).hexdigest() for key,name in [('manifest_sha256','draft_v50_manifest.json'),('plan_sha256','draft_v50_plan.json'),('source_sha256','benchmark_draft_v50.py')]}
    ref=json.loads((root/'draft_v50_ar-reference.json').read_text())
    assert ref['status']=='completed' and len(ref['outputs'])==192
    data={(role,rd):json.loads((root/f'draft_v50_{role}-round{rd}.json').read_text()) for role in plan['roles'] for rd in range(2)}
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
    rates={role:4*sum(map(len,ref['outputs']))/t.sum() for role,t in times.items()}
    fixed=max((m for m in rates if m.startswith('fixed')),key=rates.get)
    comparisons={role:{base:dict(gain_percent=float(100*(times[base].sum()/t.sum()-1)),round_gains_percent=(100*(times[base].sum(axis=(0,2))/t.sum(axis=(0,2))-1)).tolist()) for base in ['adaptive_prob',fixed]} for role,t in times.items()}
    return dict(scope=plan['scope'],all_output_and_provenance_checks_passed=True,throughputs=rates,best_tested_fixed=fixed,comparisons=comparisons,domain_gain_vs_native={role:{d:float(100*(times['adaptive_prob'][i].sum()/t[i].sum()-1)) for i,d in enumerate(['alpaca','gsm8k','mbpp'])} for role,t in times.items()},runs={f'{role}-{rd}':x['run_url'] for (role,rd),x in data.items()},hashes=hashes,limitations=['Two reversed engine orders and one prompt batch per domain; no significance or fresh-test claim.','Existing greedy and top-k options are baseline optimizations, not a proposed adaptive method.','Fixed widths and top-k combinations not fully retuned; no superiority claim over best achievable static configuration.'])
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();print(json.dumps(analyze(a.root),indent=2))
