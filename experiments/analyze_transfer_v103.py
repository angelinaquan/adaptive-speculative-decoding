"""Audit the completed b64 subset; never treat a failed sweep as a passed experiment."""
import json,hashlib
from pathlib import Path
r=Path(__file__).resolve().parent
ref=json.loads((r/'model_profile_v103_b8_ar0_round-1.json').read_text())
assert ref['status']=='completed'
checks=0; orders=[]
for order in [0,1]:
    times={}
    for width in range(1,8):
        s=json.loads((r/f'model_profile_v103_b64_native{width}_round{order}.json').read_text())
        assert s['status']=='completed' and len(s['records'])==12
        for key in ['source_sha256','plan_sha256','versions']:
            assert s[key]==ref[key]
        for rec in s['records']:
            ids=ref['outputs'][rec['start']:rec['start']+64]
            assert rec['parity'] and rec['output_sha256']==hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            assert rec['tokens']==sum(map(len,ids));checks+=1
        times[width]=sum(x['seconds'] for x in s['records'] if x['phase']=='timed')/2
    orders.append(dict(order=order,seconds_per_192=times,full_sweep_choice=min(times,key=times.get),transferred_choice=3,exploratory_3_or_max_choice=min([3,7],key=times.get)))
failed=json.loads((r/'model_profile_v103_b8_native2_round1.json').read_text())
assert failed['status']=='failed'
f=failed['parity_failure']; mismatch=[]
for i,(actual,expected) in enumerate(zip(f['actual'],f['expected'])):
    if actual!=expected:
        mismatch.append(dict(profile_index=f['start']+i,first_token_difference=next((j for j,(a,b) in enumerate(zip(actual,expected)) if a!=b),min(len(actual),len(expected)))))
result=dict(status='partial_diagnostic_only',job=3070220,job_status='FAILED',b64_batch_hash_checks=checks,orders=orders,failure=dict(batch=8,width=2,order=1,phase=f['phase'],mismatches=mismatch),limitations=['Global sweep correctness gate failed; no validated end-to-end transfer claim.','Profiling split only; no held-out 8B inference or total-cost comparison.','8B maximum width7, so {3,7} is an explicitly exploratory reinterpretation of the earlier {3,8} shortlist.','Two timing orders, not population significance.'],conclusion='Width3 is the observed b64 profiling winner in both orders; broader profiling finds no better width in this subset. Correctness failure blocks final claims.')
(r/'model_profile_v103_transfer_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
