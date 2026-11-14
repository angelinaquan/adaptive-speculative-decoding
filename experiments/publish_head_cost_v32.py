import os,json
from pathlib import Path
import wandb
from analyze_head_cost_v32 import analyze
r=Path(__file__).resolve().parent;s=analyze(r);out=r/'head_cost_v32_summary.json';out.write_text(json.dumps(s,indent=2)+'\n')
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='head-cost-v32-batch64',name='descriptive-calibration-audit',dir=os.environ['SERVING_SCRATCH'],config={'scope':s['scope'],'limitations':s['limitations']})
for k,v in s['comparisons'].items():run.summary[k+'/gain_percent']=v['gain_percent']
art=wandb.Artifact('head-cost-v32-audit','exploratory-audit')
for p in [out,r/'head_cost_v32_results.json',r/'analyze_head_cost_v32.py']:art.add_file(str(p))
run.log_artifact(art);(r/'head_cost_v32_audit_run.json').write_text(json.dumps({'run_url':run.url})+'\n');run.finish()
