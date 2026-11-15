"""Strict fresh-reference gate before any heldout stage; no success by omission."""
import json,os,hashlib,subprocess
from pathlib import Path
import wandb
r=Path(__file__).resolve().parent
plan=json.loads((r/'matched_v112_plan.json').read_text())
assert not list(r.glob('correctness_v114_b*.json'))
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='8b-fp32head-correctness-v114',name='8b-fp32head-full-regression-before-heldout',dir=os.environ['SERVING_SCRATCH'],config={'scope':'Configuration mitigation, not demonstrated sampler fix. Full exact equality against fresh FP32-head AR; no heldout outcomes.','batches':[8,64],'widths':list(range(1,8)),'engine_orders':2,'passes':{'warmup':2,'timed':2},'prompts':200,'precision':'BF16 backbone, FP32 output head'})
status='failed';completed=[]
try:
 schedule=[(b,'ar',0,o) for o in [0,1] for b in [8,64]]
 schedule += [(b,'native',w,o) for o in [0,1] for b in [8,64] for w in (list(range(1,8)) if o==0 else list(range(7,0,-1)))]
 for b,role,w,o in schedule:
  subprocess.run(['/tmp/aquan-forespark-v63/ForeSpark/.venv/bin/python',str(r/'benchmark_correctness_v114.py'),'--batch',str(b),'--role',role,'--width',str(w),'--round',str(o)],check=True)
  path=r/f'correctness_v114_b{b}_{role}{w}_round{o}.json';s=json.loads(path.read_text());assert s['status']=='completed'
  assert len(s['records'])==4*((200+b-1)//b)
  completed.append(path.name);run.log({'completed_sessions':len(completed)})
 ref=json.loads((r/'correctness_v114_b8_ar0_round0.json').read_text())['outputs'];checks=0
 for name in completed:
  s=json.loads((r/name).read_text());b=s['batch']
  for z in s['records']:
   ids=ref[z['start']:z['start']+b]
   assert z['parity'] and z['output_sha256']==hashlib.sha256(json.dumps(ids).encode()).hexdigest()
   assert z['tokens']==sum(map(len,ids));checks+=1
 gate=dict(passed=True,sessions=len(completed),batch_output_checks=checks,manifest_sha256=plan['manifest_sha256'],plan_sha256=hashlib.sha256((r/'matched_v112_plan.json').read_bytes()).hexdigest(),benchmark_sha256=hashlib.sha256((r/'benchmark_matched_v112.py').read_bytes()).hexdigest(),precision={'backbone':'bfloat16','head':'float32'},cache_root='/tmp/aquan-forespark-v63/cache_v114',wandb_url=run.url,scope='Empirically validated precision mitigation. Original BF16 mismatch not reproduced or root-caused; no sampler-bug fix claim.')
 (r/'correctness_v112_gate.json').write_text(json.dumps(gate,indent=2)+'\n');run.summary.update(gate);status='completed'
finally:
 run.summary.update(status=status,completed_sessions=len(completed))
 art=wandb.Artifact('8b-fp32head-correctness-v114','correctness-regression')
 for pattern in ['correctness_v114_*.json','correctness_v112_gate.json','benchmark_correctness_v114.py','run_correctness_v114.py','matched_v112_plan.json','model_split_v103_manifest.json','vocab_v54_manifest.json','model_v102_plan.json']:
  for path in r.glob(pattern):art.add_file(str(path))
 run.log_artifact(art);(r/'correctness_v114_run.json').write_text(json.dumps(dict(status=status,url=run.url,completed_sessions=len(completed)),indent=2)+'\n');run.finish()
