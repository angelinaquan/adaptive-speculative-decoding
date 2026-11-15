import os,json
from pathlib import Path
import wandb
r=Path(__file__).resolve().parent;names=['shortverify_v101.py','test_shortverify_v101.py','shortverify_v101_summary.json','shortverify_v101_min1_summary.json','shortverify_v101_checks.json'];assert all((r/n).is_file() for n in names);assert json.loads((r/'shortverify_v101_checks.json').read_text())['passed'];s=json.loads((r/'shortverify_v101_summary.json').read_text())
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='shortverify-v101',name='draft3-budget2-zero-arm-screen',dir=os.environ['SERVING_SCRATCH'],config={'scope':s['scope'],'limitations':s['limitations']})
for name,row in s['results'].items():
 for key in ['reward','gain_vs_native3_percent']:run.summary[name+'/'+key]=row[key]
a=wandb.Artifact('shortverify-v101-screen','exploratory-evaluation')
for n in names:a.add_file(str(r/n))
run.log_artifact(a);url=run.url;run.finish();(r/'shortverify_v101_run.json').write_text(json.dumps({'url':url,'status':'completed'},indent=2)+'\n')
