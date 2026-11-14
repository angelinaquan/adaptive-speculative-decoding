import os,json,hashlib
from pathlib import Path
import wandb
r=Path(__file__).resolve().parent;data=json.loads((r/'history_v31_exploratory.json').read_text());assert all(data['checks'].values())
for domain,digest in data['source_hashes'].items():assert hashlib.sha256((r/f'shift_v8_{domain}_fit.jsonl').read_bytes()).hexdigest()==digest
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='history-v31-exploratory',name='causal-history-old-trace-screen',dir=os.environ['SERVING_SCRATCH'],config={'scope':data['scope'],'limitations':data['limitations']})
for a,domains in data['results'].items():
 for d,v in domains.items():run.summary[f'alpha_{a}/{d}/brier']=v
art=wandb.Artifact('history-v31-exploratory','exploratory-validation')
for name in ['history_v31_exploratory.json','explore_history_v31.py','shift_v8_alpaca_fit.jsonl','shift_v8_gsm8k_fit.jsonl']:art.add_file(str(r/name))
run.log_artifact(art);(r/'history_v31_run.json').write_text(json.dumps({'run_url':run.url})+'\n');run.finish()
