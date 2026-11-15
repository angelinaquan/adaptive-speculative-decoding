import json,os
from pathlib import Path
import wandb
r=Path(__file__).resolve().parent;files=['context_rerank_v96.py','test_context_rerank_v96.py','context_rerank_v96_plan.json','context_rerank_v96_summary.json','context_rerank_v96_checks.json'];assert all((r/name).is_file() for name in files)
s=json.loads((r/'context_rerank_v96_summary.json').read_text());assert json.loads((r/'context_rerank_v96_checks.json').read_text())['passed']
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='context-adapter-v96',name='context-conditioned-score-screen',dir=os.environ['SERVING_SCRATCH'],config=json.loads((r/'context_rerank_v96_plan.json').read_text()))
for row in s['results']:
 for method,v in row['models'].items():
  for key in ['reward','gain_vs_native_percent']:run.summary[f'w{row["width"]}/{method}/{key}']=v[key]
a=wandb.Artifact('context-v96-screen','exploratory-evaluation')
for name in files:a.add_file(str(r/name))
run.log_artifact(a);url=run.url;run.finish();(r/'context_v96_run.json').write_text(json.dumps({'url':url,'status':'completed'},indent=2)+'\n')
