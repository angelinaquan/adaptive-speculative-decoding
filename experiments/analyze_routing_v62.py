"""Optimistic whole-batch fixed-width routing on existing timings; NOT a new method."""
import hashlib,json,math
from pathlib import Path

def analyze(root):
    result={}
    for version,batch in [(25,32),(29,64),(48,128)]:
        data={}
        hashes={}
        for method in ['ar']+[f'fixed{i}' for i in range(1,8)]:
            p=root/f'cost_v{version}_{method}.json'
            x=json.loads(p.read_text()); assert x['status']=='completed'
            hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
            records={(z['start'],z['repeat']):z for z in x['records']}
            assert len(records)==len(x['records']) and records
            assert all(math.isfinite(z['seconds']) and z['seconds']>0 and z['tokens']>0 for z in records.values())
            data[method]=(x,records)
        reference=data['ar'][1]
        for method,(x,records) in data.items():
            assert records.keys()==reference.keys()
            assert all(z['tokens']==reference[k]['tokens'] for k,z in records.items())
            if version!=25:
                assert x['outputs']==data['ar'][0]['outputs']
                assert x['metadata']['manifest_sha256']==data['ar'][0]['metadata']['manifest_sha256']
            else:
                assert x['manifest_sha256']==data['ar'][0]['manifest_sha256']
        starts=sorted({k[0] for k in reference})
        repeats=sorted({k[1] for k in reference})
        assert set(reference)=={(s,t) for s in starts for t in repeats}
        means={m:{s:sum(rr[s,t]['seconds'] for t in repeats)/len(repeats) for s in starts} for m,(_,rr) in data.items()}
        best=min(means,key=lambda m:sum(means[m].values()))
        choices={s:min(means,key=lambda m:means[m][s]) for s in starts}
        fixed=sum(means[best].values()); oracle=sum(means[m][s] for s,m in choices.items())
        assert oracle<=fixed+1e-10
        result[f'v{version}']=dict(batch=batch,prompt_groups=len(starts),repeats=len(repeats),best_fixed=best,oracle_choices=choices,optimistic_gain_percent=100*(fixed/oracle-1),hashes=hashes)
    return dict(scope=__doc__,studies=result,limitations=['Same timings select and score the oracle: optimistic descriptive evidence, not generalization or statistical significance.','Separate engine order was not balanced in these screens; timing variation is retained.','Whole-batch, whole-generation routing only. Does not bound per-request or per-step scheduling.','v25 archived raw timing files omit output arrays; rely on its previously completed correctness audit rather than claim a new output audit here.','No new GPU timing, predictor, calibration result, or deployable speedup.'])
if __name__=='__main__':
 import sys
 print(json.dumps(analyze(Path(sys.argv[1])),indent=2))
