import os,json
from pathlib import Path
import wandb
from analyze_cost_v38 import analyze
r=Path(__file__).resolve().parent;s=analyze(r);out=r/'cost_v38_summary.json';out.write_text(json.dumps(s,indent=2)+'\n')
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='cost-v38-heldout',name='complete-frozen-confirmation-audit',dir=os.environ['SERVING_SCRATCH'],config={'scope':s['scope'],'limitations':s['limitations']})
for k in ['passed_statistical_criteria','passed_5_percent_practical_target','passed_primary_success']:run.summary[k]=s[k]
for m,v in s['comparisons'].items():
 run.summary[m+'/gain_percent']=v['gain_percent'];run.summary[m+'/ci95_low']=v['batch_and_round_bootstrap_95'][0];run.summary[m+'/ci95_high']=v['batch_and_round_bootstrap_95'][1]
art=wandb.Artifact('cost-v38-heldout-audit','heldout-audit')
for p in [out,r/'analyze_cost_v38.py',r/'cost_v38_manifest.json',r/'cost_v38_plan.json',r/'cost_v38_ar-reference.json']+sorted(r.glob('cost_v38_*-round*.json')):art.add_file(str(p))
run.log_artifact(art);(r/'cost_v38_audit_run.json').write_text(json.dumps({'run_url':run.url})+'\n');run.finish()
