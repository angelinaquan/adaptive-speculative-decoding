"""Descriptive validation analysis; repeat timings are not independent prompts."""
import argparse,json
from pathlib import Path
import numpy as np


def analyze(data):
    assert data['status']=='completed'
    rows=data['records'];assert len(rows)==162
    lookup={(x['method'],x['round'],tuple(x['indices'])):x for x in rows}
    assert len(lookup)==len(rows)
    expected={(method,rd,tuple(range(start,start+8)))
              for method in ['frozen','online100','online25']
              for rd in range(3) for start in range(0,144,8)}
    assert set(lookup)==expected, 'Incomplete or unexpected comparison groups'
    for row in rows:
        start=row['indices'][0]
        assert row['domain']==('alpaca' if start<64 else 'gsm8k' if start<128 else 'humaneval')
        assert np.isfinite(row['seconds']) and row['seconds']>0
        assert isinstance(row['tokens'],int) and row['tokens']>0
    result={'scope':'Validation only; not evidence of held-out generalization','run_url':data['run_url'],'domains':{},'source_fit':data['source_fit']}
    for domain in ['all','alpaca','gsm8k','humaneval']:
        subset=[x for x in rows if domain=='all' or x['domain']==domain]
        groups=sorted({tuple(x['indices']) for x in subset})
        metrics={}
        for method in ['frozen','online100','online25']:
            selected=[x for x in subset if x['method']==method]
            for x in selected:
                baseline=lookup['frozen',x['round'],tuple(x['indices'])]
                assert x['tokens']==baseline['tokens'] and x['seconds']>0
            metrics[method]={'tokens_per_second':sum(x['tokens'] for x in selected)/sum(x['seconds'] for x in selected)}
            if method=='frozen':continue
            base=np.array([[lookup['frozen',rd,g]['seconds'] for rd in range(3)] for g in groups])
            timing=np.array([[lookup[method,rd,g]['seconds'] for rd in range(3)] for g in groups])
            metrics[method].update(
                gain_percent=100*(base.sum()/timing.sum()-1),
                round_gains_percent=(100*(base.sum(axis=0)/timing.sum(axis=0)-1)).tolist(),
                prompt_groups=len(groups))
        result['domains'][domain]=metrics
    result['limitations']=['Validation reused for exploration, no confirmatory significance claim.','No confidence interval: online state couples successive prompt groups; three repeats of the same stream are not independent workload samples.','Code domain has only two prompt groups.','Conditional calibration scores from different policies need not cover the same observations.','Upstream IRLS reused with pretrained-head features; not a new optimizer.','Requires comparison against STS and upstream qmax calibration before a method-level claim.']
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=analyze(json.loads(a.input.read_text()));a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
