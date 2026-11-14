import os,json
from pathlib import Path
import wandb
from analyze_dynamic_v58 import analyze
r=Path(__file__).resolve().parent;s=analyze(r);out=r/'dynamic_v58_summary.json';out.write_text(json.dumps(s,indent=2)+'\n')
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='dynamic-v58-recency-baseline',name='complete-recency-baseline-audit',dir=os.environ['SERVING_SCRATCH'],config={'scope':s['scope'],'limitations':s['limitations']})
run.summary['existing_recency_baseline_not_novel_method']=True
for role,rate in s['throughputs'].items():run.summary[role+'/tokens_per_second']=rate
for role,comp in s['comparisons'].items():
 for base,v in comp.items():run.summary[role+'/gain_vs_'+base]=v['gain_percent']
art=wandb.Artifact('dynamic-v58-baseline-audit','exploratory-audit')
for p in [out,Path(__file__),r/'analyze_dynamic_v58.py',r/'benchmark_dynamic_v58.py',r/'dynamic_v58_plan.json',r/'oracle_v57_manifest.json',r/'dynamic_v58_ar-reference.json',r/'active_vocab_v55.py',r/'context_v55_priors.json',r/'select_context_v55.py',r/'dynamic_vocab_v58.py',r/'dynamic_v58_checks.json',r/'check_dynamic_v58.py']+sorted(r.glob('dynamic_v58_*-round*.json')):art.add_file(str(p))
run.log_artifact(art);(r/'dynamic_v58_audit_run.json').write_text(json.dumps({'run_url':run.url})+'\n');run.finish()
