"""Audit v39 budget replay and report all exploratory timing comparisons."""
import argparse,json
from pathlib import Path
import numpy as np


def analyze(root):
    x=json.loads((root/'fresh_v39_results.json').read_text());assert x['status']=='completed'
    costs=x['target_costs']['cost_tables'];diagnostic=[]
    for row in x['diagnostic']:
        cap=np.asarray(row['capacities']);anchors=sum(row['anchors']);n=row['requests']
        limit=min(int(cap.sum()),max(0,row['max_logits']-n*row['bonus_tokens']))
        denominator=costs[0][n]+np.asarray(costs[1][anchors:anchors+limit+1])
        objectives={}
        for name in ['stale','fresh']:
            p=np.asarray(row[name+'_confidences']);assert p.shape==(n,7)
            scores=np.sort(np.cumprod(p,axis=1)[np.arange(7)[None,:]<cap[:,None]])[::-1][:limit]
            objectives[name]=(row['sampling_requests']+np.r_[0.,np.cumsum(scores)])/denominator
        old=row['old_budget'];fresh=row['fresh_budget']
        assert old==int(np.argmax(objectives['stale'])) and fresh==int(np.argmax(objectives['fresh']))
        assert row['old_tokens']==anchors+old and row['fresh_tokens']==anchors+fresh
        diagnostic.append(dict(changed=old!=fresh,old=old,fresh=fresh,
            expected_current_objective_gain_percent=100*(objectives['fresh'][fresh]/objectives['fresh'][old]-1) if objectives['fresh'][old]>0 else None))
    rows=x['records'];assert len(rows)==48
    assert {(z['method'],z['round'],z['start']) for z in rows}=={(m,rd,s) for m in ['native','fresh'] for rd in range(4) for s in range(0,384,64)}
    arrays={m:np.array([[next(z['seconds'] for z in rows if z['method']==m and z['round']==rd and z['start']==start) for rd in range(4)] for start in range(0,384,64)]) for m in ['native','fresh']}
    for arr in arrays.values():assert np.isfinite(arr).all() and (arr>0).all()
    gains=100*(arrays['native'].sum(axis=0)/arrays['fresh'].sum(axis=0)-1)
    gain=float(100*(arrays['native'].sum()/arrays['fresh'].sum()-1))
    domains={d:float(100*(arrays['native'][i*2:(i+1)*2].sum()/arrays['fresh'][i*2:(i+1)*2].sum()-1)) for i,d in enumerate(['alpaca','gsm8k','mbpp'])}
    return dict(scope=x['plan']['scope'],gain_percent=gain,round_gains_percent=gains.tolist(),domain_gain_percent=domains,
        passed_exploratory_gate=bool(gain>=5 and min(gains)>0),budget_replay_passed=True,
        diagnostic_steps=len(diagnostic),changed_budgets=sum(z['changed'] for z in diagnostic),diagnostic=diagnostic,run_url=x['run_url'],
        limitations=['Current confidence is a forecast, not ground-truth acceptance. Diagnostic objective gains are not real throughput or an oracle bound.','Diagnostic synchronization excluded from timings; fresh policy synchronization included.','Reused prompts, same-engine exploratory timing, no fresh generalization or significance claim.'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();s=analyze(a.root);a.output.write_text(json.dumps(s,indent=2)+'\n');print(json.dumps({k:v for k,v in s.items() if k!='diagnostic'},indent=2))
