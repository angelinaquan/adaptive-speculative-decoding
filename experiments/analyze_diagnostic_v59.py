import json,hashlib,math
from pathlib import Path

def analyze(root):
    result={}
    for role in ['identity','recency4096']:
        x=json.loads((root/f'diagnostic_v59_{role}.json').read_text())
        assert x['status']=='completed' and x['role']==role
        for key,name in [('source_sha256','probe_dynamic_v59.py'),('module_sha256','diagnose_dynamic_v59.py')]:
            assert x[key]==hashlib.sha256((root/name).read_bytes()).hexdigest()
        assert len(x['traces'])==6 and [t['start'] for t in x['traces']]==list(range(0,48,8))
        domains={}
        for t in x['traces']:
            a=t['audit'];assert a['enabled']==(role=='recency4096') and a['projection_checked'] and a['target_unchanged'] and a['buffers_stable']
            assert a['updates']==(len(t['rows']) if role=='recency4096' else 0)
            for row in t['rows']:
                assert all(math.isfinite(row[k]) and row[k]>=0 for k in ['verifier_ms','update_span_ms'])
                assert all(0<=v<=7 for v in row['full_width_accepted'])
            z=domains.setdefault(t['domain'],dict(steps=0,verifier_ms=0.,update_span_ms=0.,accepted=0,full_width_states=0))
            z['steps']+=len(t['rows'])
            for row in t['rows']:
                z['verifier_ms']+=row['verifier_ms'];z['update_span_ms']+=row['update_span_ms']
                z['accepted']+=sum(row['full_width_accepted']);z['full_width_states']+=len(row['full_width_accepted'])
        for z in domains.values():
            z['mean_accepted_prefix']=z['accepted']/z['full_width_states']
            z['mean_update_span_ms']=z['update_span_ms']/z['steps']
        result[role]=dict(domains=domains,run_url=x['run_url'])
    return dict(all_checks_passed=True,scope='Instrumented diagnostic, not throughput evidence. GPU update spans include launch gaps. Acceptance is before finalEOS/cap clipping.',roles=result)
if __name__=='__main__':
 import sys
 print(json.dumps(analyze(Path(sys.argv[1])),indent=2))
