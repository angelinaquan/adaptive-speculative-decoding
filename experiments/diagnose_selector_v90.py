"""Training-pool error anatomy; descriptive, never a new validation score."""
import json
from pathlib import Path
import numpy as np
from rerank_v90 import prepare,evaluate
r=Path(__file__).resolve().parent
raw=json.loads((r/'path_v89_checks.json').read_text())
out={'scope':'Descriptive error analysis on the v90 fitting pool only. Oracle quantities are not deployable gains.','widths':[]}
for w in [3,8]:
 data=prepare(raw,w);native=evaluate(np.zeros(5*(w+1)),data)
 missing=0;recoverable=0;full=0;positions={};margins=[];ranks=[]
 for x,y in zip(data,native):
  j=y['reward']-1
  if j==w:full+=1;continue
  if j>=len(x['gold']):missing+=1;continue
  recoverable+=1;positions[str(j)]=positions.get(str(j),0)+1
  prev=x['gold'][j-1] if j else 0
  scores=x['grids'][j][prev];gold=x['gold'][j]
  margins.append(float(scores.max()-scores[gold]));ranks.append(int(np.sum(scores>scores[gold]))+1)
 out['widths'].append({'width':w,'states':len(data),'fully_accepted':full,'first_error_gold_absent':missing,'first_error_gold_present':recoverable,'recoverable_error_positions_zero_based':positions,'gold_selector_rank_histogram':{str(k):ranks.count(k) for k in sorted(set(ranks))},'recoverable_margin_quantiles':np.quantile(margins,[0,.25,.5,.75,1]).tolist(),'native_reward':sum(y['reward'] for y in native),'candidate_oracle_reward':sum(len(x['gold'])+1 for x in data)})
(r/'selector_v90_diagnostic.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
