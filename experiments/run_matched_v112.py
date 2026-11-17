"""Matched actual wall-clock comparison; all selections use profiling data only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import wandb

r=Path(__file__).resolve().parent
plan=json.loads((r/'matched_v112_plan.json').read_text())
assert json.loads((r/'correctness_v112_gate.json').read_text())['passed'], 'Correctness gate must pass before W&B or GPU stages'
assert not list(r.glob('matched_v112_*_deploy.json'))
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',
    group='matched-total-cost-v112',name='8b-heldout-transfer-total-cost',
    dir=os.environ['SERVING_SCRATCH'],config=plan)
(r/'matched_v112_run.json').write_text(json.dumps(dict(url=run.url,status='running'),indent=2)+'\n')
status='failed';pipelines=[]

def stage(role,phase,width,order,tag,start_ns=0):
    before=time.perf_counter()
    subprocess.run(['/tmp/aquan-forespark-v63/ForeSpark/.venv/bin/python',
        str(r/'benchmark_matched_v112.py'),'--role',role,'--phase',phase,'--width',str(width),
        '--order',str(order),'--batch',str(batch),'--tag',tag,'--pipeline-start-ns',str(start_ns)],check=True)
    wall=time.perf_counter()-before
    result=json.loads((r/f'matched_v112_{tag}.json').read_text())
    assert result['status']=='completed' and all(x['parity'] for x in result['records'])
    return result,wall

def costs(result):
    widths=sorted({x['width'] for x in result['records']})
    return {w:sum(x['seconds'] for x in result['records'] if x['width']==w and not x['warmup']) for w in widths}

try:
    reference_costs={}
    for batch in plan['batches']:
        _,reference_wall=stage('ar','reference',0,0,f'b{batch}_reference')
        reference_costs[batch]=reference_wall
        for order,methods in enumerate(plan['method_orders']):
            for method in methods:
                start_ns=time.monotonic_ns();tag=f'b{batch}_{method}_o{order}';native_costs={};stages=[]
                if method=='zero_transfer':selected=3 if batch==64 else 7
                else:
                    if method=='native_pair':candidates=[3,7]
                    elif method=='full_native':candidates=list(range(1,8))
                    else:raise AssertionError(method)
                    ordered=sorted(candidates,reverse=bool(order))
                    for width in ordered:
                        result,wall=stage('native','profile',width,order,tag+f'_profile{width}')
                        native_costs[width]=costs(result)[width]
                        stages.append(dict(stage='native_profile',width=width,seconds=wall))
                    selected=min(native_costs,key=lambda w:(native_costs[w],w))
                profile_seconds=(time.monotonic_ns()-start_ns)/1e9
                # Freeze selection to disk before any held-out inference for this pipeline.
                selection=dict(batch=batch,method=method,order=order,selected_width=selected,
                    profiling_wall_seconds=profile_seconds,native_profile_costs=native_costs,stages=stages,
                    selection_used_test_outcomes=False)
                (r/f'matched_v112_{tag}_selection.json').write_text(json.dumps(selection,indent=2)+'\n')
                deployment,wall=stage('native','deploy',selected,order,tag+'_deploy',start_ns)
                record=dict(**selection,milestones=deployment['milestones'],deployment_subprocess_seconds=wall)
                pipelines.append(record)
                (r/'matched_v112_progress.json').write_text(json.dumps(dict(status='running',pipelines=pipelines),indent=2)+'\n')
                run.log({'completed_pipelines':len(pipelines),'order':order,'batch':batch,
                         f'{method}/profiling_wall_seconds':profile_seconds,
                         **{f'{method}/total_seconds_h{h}':z['total_pipeline_seconds'] for h,z in record['milestones'].items()}})
    comparisons=[]
    for batch in plan['batches']:
        for order in [0,1]:
            rows={x['method']:x for x in pipelines if x['order']==order and x['batch']==batch}
            for horizon in plan['horizons_test_passes']:
                baseline=rows['full_native']['milestones'][str(horizon)]['total_pipeline_seconds']
                for method in ['zero_transfer','native_pair']:
                    seconds=rows[method]['milestones'][str(horizon)]['total_pipeline_seconds']
                    comparisons.append(dict(batch=batch,order=order,horizon_passes=horizon,method=method,
                        baseline='full_native',total_time_reduction_percent=100*(1-seconds/baseline)))
    summary=dict(correctness_passed=True,pipelines=pipelines,comparisons=comparisons,
        common_reference_seconds=reference_costs,scope=plan['scope'],limitations=plan['limitations'])
    (r/'matched_v112_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    run.summary.update(summary);status='completed'
finally:
    run.summary['status']=status
    artifact=wandb.Artifact('matched-v112-total-cost','matched-method-comparison')
    for pattern in ['matched_v112_*.json','benchmark_matched_v112.py','run_matched_v112.py',
                    'run_matched_v112.sh','correctness_v112_gate.json','model_split_v103_manifest.json','model_v102_paths.json']:
        for path in r.glob(pattern):artifact.add_file(str(path))
    run.log_artifact(artifact)
    (r/'matched_v112_run.json').write_text(json.dumps(dict(url=run.url,status=status),indent=2)+'\n')
    run.finish()
