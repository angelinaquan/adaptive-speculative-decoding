import json,hashlib
from pathlib import Path
import numpy as np

def analyze(root):
    read=lambda n:json.loads((root/n).read_text())
    sha=lambda n:hashlib.sha256((root/n).read_bytes()).hexdigest()
    plan=read('oracle_v56_plan.json');ref=read('oracle_v56_ar-reference.json')
    oracle=read('oracle_v56_vocabularies.json');assert sha('oracle_v56_vocabularies.json')==plan['oracle_sha256']
    assert ref['outputs']==read('vocab_v54_ar-reference.json')['outputs']
    assert ref['status']=='completed' and len(ref['outputs'])==192
    hashes={k:sha(n) for k,n in [('manifest_sha256','vocab_v54_manifest.json'),('plan_sha256','oracle_v56_plan.json'),('source_sha256','benchmark_oracle_v56.py'),('active_module_sha256','active_vocab_v55.py'),('selector_sha256','select_context_v55.py'),('priors_sha256','context_v55_priors.json')]}
    data={(role,rd):read(f'oracle_v56_{role}-round{rd}.json') for role in plan['roles'] for rd in range(2)}
    times={role:np.zeros((3,2,2)) for role in plan['roles']}
    for key,x in [('ar',ref)]+list(data.items()):
        assert x['status']=='completed' and x['outputs']==ref['outputs']
        assert all(x['metadata'][k]==v for k,v in hashes.items())
        assert len(x['records'])==6
        assert {(z['start'],z['repeat']) for z in x['records']}=={(s,r) for s in [0,64,128] for r in [0,1]}
        for z in x['records']:
            assert z['tokens']==sum(map(len,ref['outputs'][z['start']:z['start']+64]))
            assert np.isfinite(z['seconds']) and z['seconds']>0
            if key!='ar':times[key[0]][z['start']//64,key[1],z['repeat']]=z['seconds']
        if key=='ar':continue
        role,rd=key;m=x['metadata'];assert (m['role'],m['round'])==key and m['order']==plan['orders'][rd]
        assert m['engine']['dtype']=='float32' and m['engine']['max_num_seqs']==64 and m['batch_invariant']
        spec=m['engine']['speculative_config'];assert spec['num_speculative_tokens']==7 and spec['draft_sample_method']=='greedy' and spec['enable_adaptive_verification'] is False
        if role=='native':assert not x['vocabulary_updates'];continue
        assert x['active_vocab']['target_parameter_signature_unchanged'] is True
        assert x['active_vocab']['projection_check']['rtol']==x['active_vocab']['projection_check']['atol']==1e-4
        updates=x['vocabulary_updates'];assert len(updates)==12
        assert {(z['start'],z['repeat']) for z in updates}=={(s,r) for s in [0,64,128] for r in [-2,-1,0,1]}
        for z in updates:
            ids=z['token_ids'];assert ids==oracle['vocabularies'][str(z['start'])]
            assert all(v in set(ids) for seq in ref['outputs'][z['start']:z['start']+64] for v in seq)
            assert ids==sorted(set(ids)) and len(ids)==(4096 if role.endswith('4096') else 8192)
            assert z['ids_sha256']==hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            assert z['target_unchanged'] and z['buffers_stable'] and z['projection_checked']==(z['repeat']<0)
            assert ids==next(q['token_ids'] for q in updates if q['start']==z['start'] and q['repeat']==-2)
    return dict(NONDEPLOYABLE_ORACLE_DIAGNOSTIC=True,all_checks_passed=True,scope=plan['scope'],throughputs={k:4*sum(map(len,ref['outputs']))/t.sum() for k,t in times.items()},comparisons={k:{b:dict(gain_percent=float(100*(times[b].sum()/t.sum()-1)),round_gains=(100*(times[b].sum(axis=(0,2))/t.sum(axis=(0,2))-1)).tolist()) for b in ['native']} for k,t in times.items()},runs={f'{r}-{rd}':x['run_url'] for (r,rd),x in data.items()},hashes=hashes,limitations=plan['limits'])
if __name__=='__main__':
 import sys
 print(json.dumps(analyze(Path(sys.argv[1])),indent=2))
