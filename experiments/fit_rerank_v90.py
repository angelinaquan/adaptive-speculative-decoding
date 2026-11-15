import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,hashlib
from pathlib import Path
import numpy as np
from rerank_v90 import prepare,fit,evaluate
r=Path(__file__).resolve().parent;raw=json.loads((r/'path_v89_checks.json').read_text());assert raw['passed'];plan=json.loads((r/'rerank_v90_plan.json').read_text());models={};summary=[]
for width in [3,8]:
 data=prepare(raw,width);d=data[0]['features'][0].shape[-1];native=sum(x['reward'] for x in evaluate(np.zeros(d),data))
 for objective in ['ce','prefix']:
  candidates=[{'penalty':None,'reward':native,'fits':[]}]
  for penalty in plan['penalties']:
   score=0;fits=[]
   for fold in range(3):
    train=[x for x in data if x['prompt']%3!=fold];valid=[x for x in data if x['prompt']%3==fold]
    assert not {x['prompt'] for x in train}&{x['prompt'] for x in valid}
    theta,status=fit(train,objective,penalty);assert status['success'],status;fits.append(status);score+=sum(x['reward'] for x in evaluate(theta,valid))
   candidates.append({'penalty':penalty,'reward':score,'fits':fits})
  selected=max(candidates,key=lambda c:c['reward']);penalty=selected['penalty']
  if penalty is None:theta=np.zeros(d);status={'success':True,'message':'Native fallback chosen on training folds'}
  else:theta,status=fit(data,objective,penalty);assert status['success'],status
  key=f'w{width}_{objective}';models[key]=theta;summary.append({'width':width,'objective':objective,'parameters':d,'selected_penalty':penalty,'training_fold_candidates':candidates,'final_fit':status})
np.savez_compressed(r/'rerank_v90_models.npz',**models)
out={'passed':True,'scope':'Fit only; no new held-out outcomes read. Priorv89validation explicitly repurposed into training.','models':summary,'input_sha256':hashlib.sha256((r/'path_v89_checks.json').read_bytes()).hexdigest(),'models_sha256':hashlib.sha256((r/'rerank_v90_models.npz').read_bytes()).hexdigest(),'plan_sha256':hashlib.sha256((r/'rerank_v90_plan.json').read_bytes()).hexdigest()};(r/'rerank_v90_fit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
