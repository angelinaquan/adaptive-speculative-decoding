"""One-step decision proxies on recorded states; NOT a speedup estimate."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np


def step(row,tables):
    cap=np.asarray(row['capacities']);accepted=np.asarray(row['accepted']);p=np.asarray(row['confidence'])
    assert np.array_equal(cap,row['admitted']) and p.shape==(len(cap),7)
    assert np.all((accepted>=0)&(accepted<=cap)) and np.all((p>=0)&(p<=1))
    positions=np.broadcast_to(np.arange(7),p.shape);valid=positions<cap[:,None]
    scores=np.cumprod(p.astype(float),axis=1)[valid];labels=(positions<accepted[:,None])[valid].astype(float)
    # Score ties favor earlier positions, enforcing contiguous request prefixes.
    order=np.lexsort((positions[valid],-scores));scores=scores[order];labels=labels[order]
    anchors=sum(row['anchors']);n=len(cap);baseline=row['sampling_requests']
    cost=tables[0][n]+np.asarray(tables[1][anchors:anchors+len(scores)+1])
    expected=baseline+np.r_[0.,np.cumsum(scores)];realized=baseline+np.r_[0.,np.cumsum(labels)]
    oracle=baseline+np.minimum(np.arange(len(scores)+1),accepted.sum())
    head=int(np.argmax(expected/cost));oracle_budget=int(np.argmax(oracle/cost));full=len(scores)
    return dict(head_budget=head,oracle_budget=oracle_budget,full_budget=full,
        head_yield=float(realized[head]),head_cost=float(cost[head]),
        full_yield=float(realized[full]),full_cost=float(cost[full]),
        oracle_yield=float(oracle[oracle_budget]),oracle_cost=float(cost[oracle_budget]),
        head_one_step_rate=float(realized[head]/cost[head]),oracle_one_step_rate=float(oracle[oracle_budget]/cost[oracle_budget]))


def analyze(root):
    x=json.loads((root/'decision_v40_results.json').read_text());assert x['status']=='completed'
    ref=json.loads((root/'cost_v29_ar.json').read_text());assert len(x['parity_checks'])==6
    for row in x['parity_checks']:assert row['output_sha256']==hashlib.sha256(json.dumps(ref['outputs'][row['start']:row['start']+64]).encode()).hexdigest()
    assert x['source_sha256']==hashlib.sha256((root/'probe_decision_v40.py').read_bytes()).hexdigest()
    assert x['module_sha256']==hashlib.sha256((root/'decision_trace_v40.py').read_bytes()).hexdigest()
    rows=[];excluded=0
    for group in x['records']:
        for row in group['trace']:
            # Pure decode only; avoid conflating initial/chunked prefill with
            # verification cost and require complete seven-position feedback.
            if not all(v==1 for v in row['anchors']) or not all(v==7 for v in row['capacities']):excluded+=1;continue
            z=step(row,x['target_costs']['cost_tables']);z.update(domain=group['domain'],start=group['start']);rows.append(z)
    reports={}
    for domain in ['all','alpaca','gsm8k','mbpp']:
        subset=[z for z in rows if domain=='all' or z['domain']==domain];assert subset
        rates={m:sum(z[m+'_yield'] for z in subset)/sum(z[m+'_cost'] for z in subset) for m in ['full','head','oracle']}
        reports[domain]=dict(steps=len(subset),profile_based_tokens_per_ms=rates,head_vs_full_proxy_percent=100*(rates['head']/rates['full']-1),oracle_vs_head_proxy_percent=100*(rates['oracle']/rates['head']-1),head_shortened_steps=sum(z['head_budget']<z['full_budget'] for z in subset),oracle_shortened_steps=sum(z['oracle_budget']<z['full_budget'] for z in subset))
    return dict(scope='Post-hoc decision headroom on full-verification recorded states; no causal oracle and NO measured speedup.',reports=reports,excluded_non_full_decode_steps=excluded,checks={'source_hashes':True,'output_hashes':True,'capacity_feedback_alignment':True},run_url=x['run_url'],limitations=['Different policies change future states; ratios of recorded accepted-token sums to profiled cost sums are not closed-loop throughput.','Oracle uses future acceptance labels and is not implementable.','Sampler acceptance counts precede output truncation/EOS handling; proxy counts are not guaranteed delivered output tokens.','Reused validation prompts, one model/backend, no held-out generalization claim.'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();s=analyze(a.root);a.output.write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s,indent=2))
