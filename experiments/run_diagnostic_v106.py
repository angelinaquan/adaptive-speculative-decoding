import json
import os
from pathlib import Path
import subprocess
import wandb

r=Path(__file__).resolve().parent
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='8b-parity-v106',
               name='8b-first-divergence-logprob-diagnostic',dir=os.environ['SERVING_SCRATCH'],
               config={'scope':'Reproduce first128 v102 prompts with top20 logprobs; no timings or relaxed correctness.',
                       'roles':['ar','native3'],'dtype':'bfloat16','cache_policy':'Reuse respective v102 role/width caches'})
status='failed'
try:
    for role in ['ar','native']:
        subprocess.run(['/tmp/aquan-forespark-v63/ForeSpark/.venv/bin/python',str(r/'diagnose_model_v106.py'),'--role',role],check=True)
        result=json.loads((r/f'model_diagnostic_v106_{role}.json').read_text())
        run.summary[f'{role}/historical_outputs_reproduced']=result['historical_outputs_reproduced']
    status='completed'
finally:
    run.summary['status']=status
    artifact=wandb.Artifact('8b-v106-logprob-diagnostic','correctness-diagnostic')
    for pattern in ['model_diagnostic_v106_*.json','diagnose_model_v106.py','run_diagnostic_v106.py',
                    'model_v102_ar0_round-1.json','model_v102_native3_round0.json','model_v102_plan.json']:
        for path in r.glob(pattern):artifact.add_file(str(path))
    run.log_artifact(artifact)
    (r/'model_diagnostic_v106_run.json').write_text(json.dumps(dict(url=run.url,status=status),indent=2)+'\n')
    run.finish()
