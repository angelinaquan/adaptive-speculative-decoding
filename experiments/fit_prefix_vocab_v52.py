"""Fit prefix-weighted and frequency vocabularies using only the frozen source set."""
import json,hashlib
from pathlib import Path
import numpy as np
from prefix_vocab_v51 import prefix_weights,lost_prefix_tokens

def fit(root,vocab_size=151936,budgets=(1024,2048,4096)):
    plan=json.loads((root/'prefix_v52_plan.json').read_text())
    ref=json.loads((root/'prefix_v52_ar-reference.json').read_text())
    data=json.loads((root/'prefix_v52_fixed7_greedy-round0.json').read_text())
    assert data['status']==ref['status']=='completed' and data['outputs']==ref['outputs']
    assert len(ref['outputs'])==256 and len(data['prefix_traces'])==4
    assert [t['start'] for t in data['prefix_traces']]==[0,64,128,192]
    for x in [ref,data]:
        for key,name in [('manifest_sha256','prefix_v52_manifest.json'),('plan_sha256','prefix_v52_plan.json'),('source_sha256','benchmark_prefix_v52.py')]:
            assert x['metadata'][key]==hashlib.sha256((root/name).read_bytes()).hexdigest()
    prefixes=[z['accepted_prefix'] for t in data['prefix_traces'] for z in t['records']]
    assert prefixes
    assert all(len(p)<=7 and all(0<=i<vocab_size for i in p) for p in prefixes)
    w=prefix_weights(prefixes,vocab_size)
    frequency=np.bincount([i for p in prefixes for i in p],minlength=vocab_size)
    source_hash=hashlib.sha256((root/'prefix_v52_fixed7_greedy-round0.json').read_bytes()).hexdigest()
    summary={'source_only':True,'states':len(prefixes),'accepted_tokens':sum(map(len,prefixes)),
             'unique_accepted_tokens':int(np.count_nonzero(frequency)),'source_sha256':source_hash,
             'scope':'Source fit only, not held-out calibration quality or timing evidence. Zero-frequency ties filled by token id, explicitly reported.','fits':[]}
    for method,weights in [('prefix',w),('frequency',frequency)]:
        order=np.lexsort((np.arange(vocab_size),-weights))
        for budget in budgets:
            ids=sorted(order[:budget].tolist());allowed=set(ids)
            result=dict(token_ids=ids,source='prefix_v52 source-only full-width greedy collection',source_sha256=source_hash,method=method,budget=budget,
                zero_weight_fill=int(np.sum(weights[ids]==0)),
                source_mean_positive_prefix_loss=float(np.mean([lost_prefix_tokens(p,allowed) for p in prefixes])),
                selection='Rank by source weights descending, break ties by token id. No validation/test outcomes used.',
                limitation='Static fit; source loss is optimistic. Prefix loss describes a frozen block and does not predict end-to-end speed.')
            (root/f'vocab_v52_{method}_{budget}.json').write_text(json.dumps(result,indent=2)+'\n')
            summary['fits'].append({k:v for k,v in result.items() if k!='token_ids'})
    (root/'prefix_v52_fit_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--vocab-size',type=int,required=True);a=p.parse_args()
    print(json.dumps(fit(a.root,a.vocab_size),indent=2))
