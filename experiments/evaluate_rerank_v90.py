import json,hashlib
from pathlib import Path
import numpy as np
from rerank_v90 import prepare,evaluate
r=Path(__file__).resolve().parent;fit=json.loads((r/'rerank_v90_fit.json').read_text());plan=json.loads((r/'rerank_v90_plan.json').read_text());raw=json.loads((r/'score_v90_checks.json').read_text());assert fit['passed'] and raw['passed']
assert fit['input_sha256']==hashlib.sha256((r/'path_v89_checks.json').read_bytes()).hexdigest()
assert fit['models_sha256']==hashlib.sha256((r/'rerank_v90_models.npz').read_bytes()).hexdigest()
assert fit['plan_sha256']==hashlib.sha256((r/'rerank_v90_plan.json').read_bytes()).hexdigest()
assert raw['manifest_sha256']==plan['manifest_sha256']==hashlib.sha256((r/'score_v90_manifest.json').read_bytes()).hexdigest()
assert raw['source_sha256']==hashlib.sha256((r/'check_score_v90.py').read_bytes()).hexdigest()
old=json.loads((r/'path_v89_manifest.json').read_text());new=json.loads((r/'score_v90_manifest.json').read_text());assert not {x['source_task_index'] for x in old}&{x['source_task_index'] for x in new};assert not {' '.join(x['text'].split()) for x in old}&{' '.join(x['text'].split()) for x in new}
models=np.load(r/'rerank_v90_models.npz');results=[];records=[]
for width in [3,8]:
 data=prepare(raw,width);zero=np.zeros(data[0]['features'][0].shape[-1]);expected=[p for row in raw['records'] if row['anchors'] for p in row['widths'][str(width)]['proposed_ids']];native=evaluate(zero,data);assert [x['tokens'] for x in native]==expected
 base=sum(x['reward'] for x in native);summary={}
 for name,theta in [('native',zero),('ce',models[f'w{width}_ce']),('prefix',models[f'w{width}_prefix'])]:
  rows=evaluate(theta,data);total=sum(x['reward'] for x in rows);summary[name]={'reward':total,'gain_vs_native_percent':100*(total/base-1),'domains':{d:sum(x['reward'] for x in rows if x['domain']==d) for d in ['alpaca','gsm8k','mbpp']}};records.extend([dict(width=width,method=name,**row) for row in rows])
 results.append({'width':width,'states':len(data),'models':summary,'gate_passed':summary['prefix']['reward']>=1.1*base and summary['prefix']['reward']>summary['ce']['reward']})
out={'passed':True,'results':results,'scope':plan['scope'],'models_sha256':fit['models_sha256'],'heldout_sha256':hashlib.sha256((r/'score_v90_checks.json').read_bytes()).hexdigest(),'limitations':['Historical training-pool workloads; not final fresh test','Reward only; no runtime overhead or closed-loop evaluation','Acceptance-oriented training and learned correction have prior art']};(r/'rerank_v90_summary.json').write_text(json.dumps(out,indent=2)+'\n');(r/'rerank_v90_records.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(out,indent=2))
