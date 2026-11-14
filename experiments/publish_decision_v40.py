import os,json
from pathlib import Path
import wandb
from analyze_decision_v40 import analyze
r=Path(__file__).resolve().parent;s=analyze(r);out=r/'decision_v40_summary.json';out.write_text(json.dumps(s,indent=2)+'\n')
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='decision-v40-exploration',name='decision-headroom-audit',dir=os.environ['SERVING_SCRATCH'],config={'scope':s['scope'],'limitations':s['limitations']})
for domain, report in s['reports'].items():
 for key in ['steps','head_vs_full_proxy_percent','oracle_vs_head_proxy_percent']:run.summary[domain+'/'+key]=report[key]
art=wandb.Artifact('decision-v40-audit','exploratory-audit')
for p in [out,r/'decision_v40_results.json',r/'analyze_decision_v40.py',r/'decision_v40_plan.json']:art.add_file(str(p))
run.log_artifact(art);(r/'decision_v40_audit_run.json').write_text(json.dumps({'run_url':run.url})+'\n');run.finish()
