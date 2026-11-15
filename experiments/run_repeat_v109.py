"""Uninstrumented cache-controlled reproduction; preserve every failed comparison."""
import json
import os
from pathlib import Path
import subprocess
import wandb

r=Path(__file__).resolve().parent
assert not list(r.glob('model_repeat_v109_*.json'))
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',
               group='8b-cache-v109',name='8b-uninstrumented-cache-control',dir=os.environ['SERVING_SCRATCH'],
               config={'scope':'Original192prompts, BF16, b64, AR/native3; no logprobs or worker observer.',
                       'states':['existing v102 caches','new caches first load','same new caches reload'],
                       'warmups':2,'timed_passes':2,'timing_claim':False})
status='failed';records=[]
try:
    for state in ['existing','fresh_first','fresh_reload']:
        for role,width,rnd in [('ar',0,-1),('native',3,0)]:
            process=subprocess.run(['/tmp/aquan-forespark-v63/ForeSpark/.venv/bin/python',
                str(r/'benchmark_repeat_v109.py'),'--state',state,'--role',role,'--width',str(width),'--round',str(rnd)])
            path=r/f'model_repeat_v109_{state}_{role}{width}.json'
            result=json.loads(path.read_text()) if path.exists() else {}
            record=dict(state=state,role=role,returncode=process.returncode,status=result.get('status','missing'),
                        exact_output_passed=result.get('status')=='completed',
                        error=result.get('error'),failure_start=result.get('parity_failure',{}).get('start'))
            records.append(record);run.log({'completed_sessions':len(records)})
            # Parity failures are the diagnostic's outcome, not a reason to
            # lose the remaining controlled comparisons. Other errors abort.
            if process.returncode and 'parity_failure' not in result:
                raise RuntimeError(record)
    ar_outputs={state:json.loads((r/f'model_repeat_v109_{state}_ar0.json').read_text())['outputs']
                for state in ['existing','fresh_first','fresh_reload']}
    old=json.loads((r/'model_v102_ar0_round-1.json').read_text())['outputs']
    summary=dict(scope='Cache-control diagnostic, no performance claim',records=records,
        ar_vs_original_changed_indices={s:[i for i,(a,b) in enumerate(zip(v,old)) if a!=b] for s,v in ar_outputs.items()},
        ar_existing_equals_fresh=ar_outputs['existing']==ar_outputs['fresh_first'],
        ar_first_equals_reload=ar_outputs['fresh_first']==ar_outputs['fresh_reload'])
    (r/'model_repeat_v109_summary.json').write_text(json.dumps(summary,indent=2)+'\n');run.summary.update(summary)
    status='completed'
finally:
    run.summary['status']=status
    artifact=wandb.Artifact('8b-v109-cache-control','correctness-diagnostic')
    for pattern in ['model_repeat_v109_*.json','benchmark_repeat_v109.py','run_repeat_v109.py',
                    'run_repeat_v109.sh','model_v102_plan.json','vocab_v54_manifest.json']:
        for path in r.glob(pattern):artifact.add_file(str(path))
    run.log_artifact(artifact)
    (r/'model_repeat_v109_run.json').write_text(json.dumps(dict(url=run.url,status=status),indent=2)+'\n')
    run.finish()
