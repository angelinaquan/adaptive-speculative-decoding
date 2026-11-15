import json, hashlib, os
from pathlib import Path
import wandb
r=Path(__file__).resolve().parent
s=json.loads((r/'rerank_v90_summary.json').read_text())
f=json.loads((r/'rerank_v90_fit.json').read_text())
assert s['passed'] and f['passed']
assert s['models_sha256']==hashlib.sha256((r/'rerank_v90_models.npz').read_bytes()).hexdigest()
assert s['heldout_sha256']==hashlib.sha256((r/'score_v90_checks.json').read_bytes()).hexdigest()
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='rerank-v90-frozen-scores',name='heldout-residual-reranker-evaluation',dir=os.environ['SERVING_SCRATCH'],config=json.loads((r/'rerank_v90_plan.json').read_text()))
rows=[]
for result in s['results']:
 for method,v in result['models'].items():
  rows.append([result['width'],method,v['reward'],v['gain_vs_native_percent']])
  run.summary[f"w{result['width']}/{method}/reward"]=v['reward']
  run.summary[f"w{result['width']}/{method}/gain_percent"]=v['gain_vs_native_percent']
 run.summary[f"w{result['width']}/gate_passed"]=result['gate_passed']
run.log({'heldout_results':wandb.Table(columns=['width','method','reward','gain_percent'],data=rows)})
a=wandb.Artifact('rerank-v90-evaluation','evaluation')
for name in ['rerank_v90_summary.json','rerank_v90_records.json','rerank_v90_fit.json','rerank_v90_models.npz','rerank_v90_plan.json','rerank_v90.py','fit_rerank_v90.py','evaluate_rerank_v90.py','publish_rerank_v90.py']:
 a.add_file(str(r/name))
run.log_artifact(a)
run.summary['candidate_status']='retired: both heldout gates failed'
url=run.url
run.finish()
(r/'rerank_v90_run.json').write_text(json.dumps({'url':url,'status':'completed'},indent=2)+'\n')
