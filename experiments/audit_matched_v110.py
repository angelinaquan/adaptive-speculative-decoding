"""Reconstruct selections, exact-output hashes, endpoints and primary decision."""
import hashlib
import json
from pathlib import Path
import math

r=Path(__file__).resolve().parent
plan=json.loads((r/'matched_v110_plan.json').read_text())
summary=json.loads((r/'matched_v110_summary.json').read_text())
ref=json.loads((r/'matched_v110_reference.json').read_text())
assert ref['status']=='completed' and len(ref['outputs'])==384
plan_hash=hashlib.sha256((r/'matched_v110_plan.json').read_bytes()).hexdigest()
source_hash=hashlib.sha256((r/'benchmark_matched_v110.py').read_bytes()).hexdigest()
checks=0;stages=0;audits=0
for path in r.glob('matched_v110_*.json'):
    obj=json.loads(path.read_text())
    if 'phase' not in obj:continue
    assert obj['status']=='completed' and obj['plan_sha256']==plan_hash and obj['source_sha256']==source_hash
    assert obj['versions']==ref['versions']
    assert obj['config']['model']==ref['config']['model']
    assert obj['config']['dtype']=='bfloat16' and obj['config']['max_num_seqs']==64
    assert obj['config']['enable_prefix_caching'] is False
    expected=ref['outputs'] if obj['phase']=='reference' else ref['outputs'][:192] if obj['phase']=='profile' else ref['outputs'][192:]
    for row in obj['records']:
        ids=expected[row['start']:row['start']+64]
        assert row['parity'] and row['tokens']==sum(map(len,ids))
        assert row['output_sha256']==hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        assert math.isfinite(row['seconds']) and row['seconds']>0
        checks+=1
    if obj['role']=='screen':
        assert len(obj['audits'])==16
        assert all(z['target_unchanged'] and z['logging_disabled'] for z in obj['audits'])
        audits+=len(obj['audits'])
    if obj['phase']=='deploy':
        assert len(obj['records'])==48 and all(not x['warmup'] for x in obj['records'])
        for h,endpoint in obj['milestones'].items():
            rows=[z for z in obj['records'] if z['repeat']<int(h)]
            assert endpoint['output_tokens']==sum(z['tokens'] for z in rows)
            assert math.isclose(endpoint['cumulative_inference_seconds'],sum(z['seconds'] for z in rows),abs_tol=1e-8)
            assert endpoint['prompts']==192*int(h)
    stages+=1

def timed_cost(obj,width):return sum(z['seconds'] for z in obj['records'] if z['width']==width and not z['warmup'])
for record in summary['pipelines']:
    method,order=record['method'],record['order'];tag=f'{method}_o{order}'
    if method=='zero_transfer':selected=3
    else:
        if method=='screen_confirm':
            screen=json.loads((r/f'matched_v110_{tag}_screen.json').read_text())
            candidates=sorted(range(1,9),key=lambda w:(timed_cost(screen,w),w))[:2]
        elif method=='native_pair':candidates=[3,8]
        else:assert method=='full_native';candidates=list(range(1,9))
        costs={w:timed_cost(json.loads((r/f'matched_v110_{tag}_profile{w}.json').read_text()),w) for w in candidates}
        selected=min(costs,key=lambda w:(costs[w],w))
    assert selected==record['selected_width']
    persisted=json.loads((r/f'matched_v110_{tag}_selection.json').read_text())
    assert persisted['selected_width']==selected and persisted['selection_used_test_outcomes'] is False
    deployment=json.loads((r/f'matched_v110_{tag}_deploy.json').read_text())
    assert deployment['milestones']==record['milestones']
    for endpoint in record['milestones'].values():
        assert endpoint['total_pipeline_seconds']>=record['profiling_wall_seconds']+deployment['initialization_seconds']+endpoint['cumulative_inference_seconds']

for row in summary['comparisons']:
    pipelines={z['method']:z for z in summary['pipelines'] if z['order']==row['order']}
    h=str(row['horizon_passes'])
    ours=pipelines['screen_confirm']['milestones'][h]['total_pipeline_seconds']
    baseline=pipelines[row['baseline']]['milestones'][h]['total_pipeline_seconds']
    assert math.isclose(row['total_time_reduction_percent'],100*(1-ours/baseline),abs_tol=1e-9)
primary=[x for x in summary['comparisons'] if x['horizon_passes']==16 and x['baseline'] in ['zero_transfer','native_pair']]
assert len(primary)==4
result=dict(passed=True,engine_stages=stages,batch_output_hash_checks=checks,
    reference_construction_batches_included=6,forced_target_parameter_audits=audits,
    selections_reconstructed_without_test_metrics=True,all_selected_widths=[x['selected_width'] for x in summary['pipelines']],
    primary_method_pass=all(x['total_time_reduction_percent']>=10 for x in primary),
    conclusion='Reject shared-engine screening for this setting: every method chose3; added screening cost loses to both cheap controls.',
    scope='Existing caches, one4B/H200/b64 workload, two method orders; no population significance claim.')
(r/'matched_v110_local_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
