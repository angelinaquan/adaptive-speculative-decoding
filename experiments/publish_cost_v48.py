import json,os
from pathlib import Path
import wandb
from analyze_cost_v48 import analyze
r=Path(__file__).resolve().parent
s=analyze(r);out=r/'cost_v48_summary.json';out.write_text(json.dumps(s,indent=2)+'\n')
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='cost-v48-batch128-exploration',name='complete-selection-audit',dir=os.environ['SERVING_SCRATCH'],config={'scope':s['scope'],'limitation':s['limitation']})
for k,v in s['measured_gain_percent'].items():run.summary['gain_vs_'+k]=v
run.summary['meets_exploratory_5_percent_gate']=s['meets_exploratory_5_percent_gate']
art=wandb.Artifact('cost-v48-selection','exploratory-audit')
for p in [out,r/'cost_v48_plan.json',r/'analyze_cost_v48.py']+[r/f'cost_v48_{m}.json' for m in s['runs']]:art.add_file(str(p))
run.log_artifact(art);(r/'cost_v48_audit_run.json').write_text(json.dumps({'run_url':run.url})+'\n');run.finish()
