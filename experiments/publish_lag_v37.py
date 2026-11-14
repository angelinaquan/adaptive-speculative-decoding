import os,json
from pathlib import Path
import wandb
from analyze_lag_v37 import analyze
r=Path(__file__).resolve().parent;s=analyze(r);out=r/'lag_v37_summary.json';out.write_text(json.dumps(s,indent=2)+'\n')
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='lag-v37-exploration',name='complete-pilot-audit',dir=os.environ['SERVING_SCRATCH'],config={'scope':s['scope'],'limitations':s['limitations']})
for m,v in s['comparisons'].items():run.summary[m+'/gain_percent']=v['gain_percent']
art=wandb.Artifact('lag-v37-audit','exploratory-audit')
for p in [out,r/'lag_v37_results.json',r/'analyze_lag_v37.py']:art.add_file(str(p))
run.log_artifact(art);(r/'lag_v37_audit_run.json').write_text(json.dumps({'run_url':run.url})+'\n');run.finish()
