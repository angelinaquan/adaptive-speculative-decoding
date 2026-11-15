import os,json
from pathlib import Path
import wandb
r=Path(__file__).resolve().parent
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='tree-feasibility-v93',name='shared-tree-budget-screen',dir=os.environ['SERVING_SCRATCH'],config={'scope':'Exploratory reused training-pool replay only; no throughput or novelty claim','node_budgets':[3,8]})
a=wandb.Artifact('tree-feasibility-v93','exploratory-evaluation')
for version,prefix in [(93,'batch_tree')]:
 summary=json.loads((r/f'{prefix}_v{version}_summary.json').read_text());checks=json.loads((r/f'{prefix}_v{version}_checks.json').read_text());assert checks['passed']
 for row in summary['results']:
  for method,val in row['rewards'].items():run.summary[f'v{version}/w{row["width"]}/{method}/reward']=val
 for name in [f'{prefix}_v{version}.py',f'test_{prefix}_v{version}.py',f'{prefix}_v{version}_summary.json',f'{prefix}_v{version}_checks.json']:a.add_file(str(r/name))
run.log_artifact(a);url=run.url;run.finish();(r/'batch_tree_v93_run.json').write_text(json.dumps({'url':url,'status':'completed'},indent=2)+'\n')
