"""Exploratory multi-width shared-engine screening, with full startup accounting."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import wandb

r = Path(__file__).resolve().parent
plan = json.loads((r/'screen_profile_v105_plan.json').read_text())
assert not (r/'screen_profile_v105_run.json').exists()
run = wandb.init(entity='aquan-world-model', project='18657-speculative-pilot',
                 group='shared-engine-screen-v105', name='4b-native-ranking-screen',
                 dir=os.environ['SERVING_SCRATCH'], config=plan)
status, wall = 'failed', {}
try:
    for role in ['ar', 'forced']:
        before = time.perf_counter()
        subprocess.run(['/tmp/aquan-forespark-v63/ForeSpark/.venv/bin/python',
                        str(r/'benchmark_screen_profile_v105.py'),'--role',role], check=True)
        wall[role] = time.perf_counter()-before
        run.log({f'{role}/subprocess_wall_seconds':wall[role]})
    ar=json.loads((r/'screen_profile_v105_ar.json').read_text())
    screen=json.loads((r/'screen_profile_v105_forced.json').read_text())
    assert ar['status']==screen['status']=='completed'
    assert len(screen['records'])==8*6*3 and len(screen['audits'])==8*6
    assert all(x['target_unchanged'] and x['logging_disabled'] for x in screen['audits'])
    for row in screen['records']:
        ids=ar['outputs'][row['start']:row['start']+64]
        assert row['parity'] and row['tokens']==sum(map(len,ids))
        assert row['output_sha256']==hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    shortlists={}
    for order in [0,1,'pooled']:
        times={w:sum(x['seconds'] for x in screen['records'] if x['phase']=='timed'
                     and x['width']==w and (order=='pooled' or x['order']==order)) for w in range(1,9)}
        assert all(t>0 for t in times.values())
        shortlists[str(order)]=sorted(times,key=lambda w:(times[w],w))[:2]
    raw=(r/'fixed_v72_summary.json').read_bytes()
    historical=json.loads(raw);assert historical['checks_passed']
    rates={w:historical['throughputs'][f'native{w}'] for w in range(1,9)}
    best=max(rates,key=lambda w:(rates[w],-w))
    summary=dict(correctness_passed=True,shortlists=shortlists,historical_native_best=best,
                 exploratory_gate_passed=all(best in x for x in shortlists.values()),
                 historical_reference_sha256=hashlib.sha256(raw).hexdigest(),
                 subprocess_wall_seconds=wall,shared_engine_initialization_seconds=screen['initialization_seconds'],
                 scope=plan['scope'],limitations=plan['limitations'],
                 measured_profiling_improvement=False,measured_inference_improvement=False)
    (r/'screen_profile_v105_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    run.summary.update(summary);status='completed'
finally:
    run.summary['status']=status
    artifact=wandb.Artifact('shared-engine-v105-screen','exploratory-profiling')
    for pattern in ['screen_profile_v105*.json','benchmark_screen_profile_v105.py',
                    'run_screen_profile_v105.py','run_screen_profile_v105.sh','forced_policy_v72.py',
                    'fixed_v72_summary.json','vocab_v54_manifest.json','export_v68_checks.json']:
        for path in r.glob(pattern):artifact.add_file(str(path))
    run.log_artifact(artifact)
    (r/'screen_profile_v105_run.json').write_text(json.dumps(dict(url=run.url,status=status),indent=2)+'\n')
    run.finish()
