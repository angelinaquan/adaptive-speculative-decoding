import os,json
from pathlib import Path
import wandb
from analyze_routing_v62 import analyze
r=Path(__file__).resolve().parent
s=analyze(r)
assert json.loads(json.dumps(s))==json.loads((r/'routing_v62_summary.json').read_text())
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='routing-v62-offline-headroom',name='whole-batch-oracle-NOT-METHOD',dir=os.environ['SERVING_SCRATCH'],config={'scope':s['scope'],'limitations':s['limitations']})
for study,info in s['studies'].items():run.summary[f'{study}/optimistic_gain_percent']=info['optimistic_gain_percent']
a=wandb.Artifact('routing-v62-offline-headroom','analysis')
for name in ['analyze_routing_v62.py','publish_routing_v62.py','routing_v62_summary.json']:a.add_file(str(r/name))
run.log_artifact(a)
(r/'routing_v62_audit_run.json').write_text(json.dumps({'run_url':run.url})+'\n')
run.finish()
