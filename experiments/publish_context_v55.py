import os,json
from pathlib import Path
import wandb
from analyze_context_v55 import analyze
r=Path(__file__).resolve().parent;s=analyze(r);out=r/'context_v55_summary.json';out.write_text(json.dumps(s,indent=2)+'\n')
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='context-v55-headroom',name='complete-prefix-vocabulary-audit',dir=os.environ['SERVING_SCRATCH'],config={'scope':s['scope'],'limitations':s['limitations']})
for role,rate in s['throughputs'].items():run.summary[role+'/tokens_per_second']=rate
for role,comp in s['comparisons'].items():
 for base,v in comp.items():run.summary[role+'/gain_vs_'+base]=v['gain_percent']
art=wandb.Artifact('context-v55-screen','exploratory-audit')
for p in [out,Path(__file__),r/'analyze_context_v55.py',r/'benchmark_context_v55.py',r/'context_v55_plan.json',r/'vocab_v54_manifest.json',r/'context_v55_ar-reference.json',r/'active_vocab_v55.py',r/'context_v55_priors.json',r/'select_context_v55.py']+sorted(r.glob('context_v55_*-round*.json')):art.add_file(str(p))
run.log_artifact(art);(r/'context_v55_audit_run.json').write_text(json.dumps({'run_url':run.url})+'\n');run.finish()
