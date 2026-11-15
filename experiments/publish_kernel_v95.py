import os,json
from pathlib import Path
import wandb
r=Path(__file__).resolve().parent
files=['kernel_v95.py','test_kernel_v95.py','kernel_v95_summary.json','kernel_v95_checks.json']
assert all((r/name).is_file() for name in files)
s=json.loads((r/'kernel_v95_summary.json').read_text());assert json.loads((r/'kernel_v95_checks.json').read_text())['passed']
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='kernel-feature-v95',name='linear-rbf-matched-budget-screen',dir=os.environ['SERVING_SCRATCH'],config={'scope':s['scope'],'limitations':s['limitations']})
for row in s['results']:
 for key in ['reward','gain_vs_fixed3_percent','selected_lambda']:run.summary[row['model']+'/'+key]=row[key]
a=wandb.Artifact('kernel-v95-screen','exploratory-evaluation')
for name in files:a.add_file(str(r/name))
run.log_artifact(a);url=run.url;run.finish();(r/'kernel_v95_run.json').write_text(json.dumps({'url':url,'status':'completed'},indent=2)+'\n')
