import os,json
from pathlib import Path
import wandb
from analyze_runtime_cost_v33 import analyze as runtime
from analyze_dual_cost_v34 import analyze as dual
r=Path(__file__).resolve().parent
for name,fn in [('runtime_cost_v33_summary.json',runtime),('dual_cost_v34_summary.json',dual)]:
 s=fn(r);assert s['passed'];(r/name).write_text(json.dumps(s,indent=2)+'\n')
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='cost-followup-audits',name='runtime-diagnostic-and-dual-cost-negative',dir=os.environ['SERVING_SCRATCH'])
run.summary['all_audits_passed']=True
art=wandb.Artifact('cost-followup-v33-v34','diagnostic-and-exploratory-audit')
for n in ['runtime_cost_v33.json','runtime_cost_v33_summary.json','analyze_runtime_cost_v33.py','dual_cost_v34_results.json','dual_cost_v34_summary.json','analyze_dual_cost_v34.py','cost_v29_domain_oracles.json']:art.add_file(str(r/n))
run.log_artifact(art);(r/'followup_audits_v34_run.json').write_text(json.dumps({'run_url':run.url})+'\n');run.finish()
