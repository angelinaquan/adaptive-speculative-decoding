import os,json
from pathlib import Path
import wandb
from analyze_draft_v50 import analyze
r=Path(__file__).resolve().parent;s=analyze(r);out=r/'draft_v50_summary.json';out.write_text(json.dumps(s,indent=2)+'\n')
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='draft-v50-headroom',name='complete-baseline-screen-audit',dir=os.environ['SERVING_SCRATCH'],config={'scope':s['scope'],'limitations':s['limitations']})
for role,rate in s['throughputs'].items():run.summary[role+'/tokens_per_second']=rate
for role,comp in s['comparisons'].items():
 for base,v in comp.items():run.summary[role+'/gain_vs_'+base]=v['gain_percent']
art=wandb.Artifact('draft-v50-screen','exploratory-audit')
for p in [out,Path(__file__),r/'analyze_draft_v50.py',r/'benchmark_draft_v50.py',r/'draft_v50_plan.json',r/'draft_v50_manifest.json',r/'draft_v50_ar-reference.json']+sorted(r.glob('draft_v50_*-round*.json')):art.add_file(str(p))
run.log_artifact(art);(r/'draft_v50_audit_run.json').write_text(json.dumps({'run_url':run.url})+'\n');run.finish()
