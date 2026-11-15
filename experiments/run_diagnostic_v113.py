import json,os,subprocess
from pathlib import Path
import wandb
r=Path(__file__).resolve().parent
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='8b-correctness-v113',name='8b-width2-fp32-head-mitigation-test',dir=os.environ['SERVING_SCRATCH'],config={'scope':'Diagnostic only. Four no-logprob repeats then one instrumented repeat; original first8 profile prompts, AR/native2, graph with built-in FP32 output head. Compare AR/native without requiring legacy BF16 reference agreement; this is a changed numerical configuration, not a sampler patch.'})
status='failed'
try:
 for mode in ['graph']:
  for role in ['ar','native']:
   subprocess.run(['/tmp/aquan-forespark-v63/ForeSpark/.venv/bin/python',str(r/'diagnose_v113.py'),'--mode',mode,'--role',role],check=True)
   s=json.loads((r/f'diagnostic_v113_{mode}_{role}.json').read_text())
   run.log({f'{mode}/{role}/historical_mismatch_count':sum(len(z['mismatch_indices']) for z in s['records'][:4])})
 status='completed'
finally:
 run.summary['status']=status
 art=wandb.Artifact('8b-correctness-v113','diagnostic')
 for pattern in ['diagnostic_v113_*.json','diagnose_v113.py','run_diagnostic_v113.py','model_profile_v103_b8_native2_round1.json','model_profile_v103_b8_ar0_round-1.json']:
  for path in r.glob(pattern):art.add_file(str(path))
 run.log_artifact(art);(r/'diagnostic_v113_run.json').write_text(json.dumps(dict(url=run.url,status=status),indent=2)+'\n');run.finish()
