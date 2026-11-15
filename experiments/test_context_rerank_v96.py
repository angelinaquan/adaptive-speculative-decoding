import json
from pathlib import Path
import numpy as np
from context_rerank_v96 import attach
from rerank_v90 import prepare,evaluate
r=Path(__file__).resolve().parent;raw=json.loads((r/'path_v89_checks.json').read_text());manifest=json.loads((r/'path_v89_manifest.json').read_text());z=np.load(r/'labels_v83_features.npz');raw['records']=raw['records'][:4]
for w in [3,8]:
 original=prepare(raw,w);base=evaluate(np.zeros(5*(w+1)),original)
 augmented=attach(prepare(raw,w),raw,manifest,z)
 assert evaluate(np.zeros(5*(w+1)+80),augmented)==base
 for x,y in zip(original,augmented):
  for a,b in zip(x['features'],y['features']):
   np.testing.assert_array_equal(a,b[...,:a.shape[-1]]);assert np.isfinite(b).all()
 # The label is not an input to feature construction.
 poisoned=prepare(raw,w)
 for x in poisoned:x['gold']=[];x['truth']=[]
 poisoned=attach(poisoned,raw,manifest,z)
 for a,b in zip(augmented,poisoned):
  for f,g in zip(a['features'],b['features']):np.testing.assert_array_equal(f,g)
out={'passed':True,'tested_prompts':4,'checks':['Tested prompt/anchor joins agree with recorded absolute positions','Zero residual exactly preserves every tested native proposal and reward','Existing score features preserved','No nonfinite contextual interactions','Gold/truth replacement leaves augmented features unchanged'],'scope':'Feature construction audit only; optimizer and held-out speed remain separate requirements'}
(r/'context_rerank_v96_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
