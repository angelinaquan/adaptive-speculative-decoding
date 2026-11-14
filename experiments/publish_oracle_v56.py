import os,json
from pathlib import Path
import wandb
from analyze_oracle_v56 import analyze
r=Path(__file__).resolve().parent;s=analyze(r);out=r/'oracle_v56_summary.json';out.write_text(json.dumps(s,indent=2)+'\n')
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='oracle-v56-DIAGNOSTIC-NOT-A-METHOD',name='NONDEPLOYABLE-oracle-audit',dir=os.environ['SERVING_SCRATCH'],config={'scope':s['scope'],'limitations':s['limitations']})
run.summary['NONDEPLOYABLE_ORACLE_DIAGNOSTIC']=True
for role,rate in s['throughputs'].items():run.summary[role+'/tokens_per_second']=rate
for role,comp in s['comparisons'].items():
 for base,v in comp.items():run.summary[role+'/gain_vs_'+base]=v['gain_percent']
art=wandb.Artifact('oracle-v56-DIAGNOSTIC','exploratory-audit')
for p in [out,Path(__file__),r/'analyze_oracle_v56.py',r/'benchmark_oracle_v56.py',r/'oracle_v56_plan.json',r/'vocab_v54_manifest.json',r/'oracle_v56_ar-reference.json',r/'active_vocab_v55.py',r/'context_v55_priors.json',r/'select_context_v55.py',r/'oracle_v56_vocabularies.json']+sorted(r.glob('oracle_v56_*-round*.json')):art.add_file(str(p))
run.log_artifact(art);(r/'oracle_v56_audit_run.json').write_text(json.dumps({'run_url':run.url})+'\n');run.finish()
