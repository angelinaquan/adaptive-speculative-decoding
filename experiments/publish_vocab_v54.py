import os,json
from pathlib import Path
import wandb
from analyze_vocab_v54 import analyze
r=Path(__file__).resolve().parent;s=analyze(r);out=r/'vocab_v54_summary.json';out.write_text(json.dumps(s,indent=2)+'\n')
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='vocab-v54-headroom',name='complete-prefix-vocabulary-audit',dir=os.environ['SERVING_SCRATCH'],config={'scope':s['scope'],'limitations':s['limitations']})
for role,rate in s['throughputs'].items():run.summary[role+'/tokens_per_second']=rate
for role,comp in s['comparisons'].items():
 for base,v in comp.items():run.summary[role+'/gain_vs_'+base]=v['gain_percent']
art=wandb.Artifact('vocab-v54-screen','exploratory-audit')
for p in [out,Path(__file__),r/'analyze_vocab_v54.py',r/'benchmark_vocab_v54.py',r/'vocab_v54_plan.json',r/'vocab_v54_manifest.json',r/'vocab_v54_ar-reference.json',r/'active_vocab_v54.py',r/'prefix_v52_fit_summary.json']+sorted(r.glob('vocab_v54_*-round*.json')):art.add_file(str(p))
run.log_artifact(art);(r/'vocab_v54_audit_run.json').write_text(json.dumps({'run_url':run.url})+'\n');run.finish()
