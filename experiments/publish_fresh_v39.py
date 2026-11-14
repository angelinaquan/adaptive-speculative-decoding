import os,json
from pathlib import Path
import wandb
from analyze_fresh_v39 import analyze
r=Path(__file__).resolve().parent;s=analyze(r);out=r/'fresh_v39_summary.json';out.write_text(json.dumps(s,indent=2)+'\n')
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='fresh-v39-exploration',name='fresh-confidence-audit',dir=os.environ['SERVING_SCRATCH'],config={'scope':s['scope'],'limitations':s['limitations']})
for k in ['gain_percent','passed_exploratory_gate','budget_replay_passed','diagnostic_steps','changed_budgets']:run.summary[k]=s[k]
art=wandb.Artifact('fresh-v39-audit','exploratory-audit')
for p in [out,r/'fresh_v39_results.json',r/'analyze_fresh_v39.py',r/'fresh_v39_plan.json']:art.add_file(str(p))
run.log_artifact(art);(r/'fresh_v39_audit_run.json').write_text(json.dumps({'run_url':run.url})+'\n');run.finish()
