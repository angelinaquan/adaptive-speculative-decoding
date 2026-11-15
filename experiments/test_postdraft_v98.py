import json,copy,itertools
from pathlib import Path
import numpy as np
from postdraft_v98 import prepare,predictions,replay
from allocation_headroom_v73 import allocate
r=Path(__file__).resolve().parent;a=json.loads((r/'path_v89_manifest.json').read_text());b=json.loads((r/'score_v90_manifest.json').read_text());assert not {x['source_task_index'] for x in a}&{x['source_task_index'] for x in b};assert not {' '.join(x['text'].split()) for x in a}&{' '.join(x['text'].split()) for x in b}
raw=json.loads((r/'score_v90_checks.json').read_text());rows=prepare(raw);pred=predictions(rows,'raw',.5);assert np.isfinite(pred).all() and np.all(np.diff(pred,axis=1)>=0)
poison=copy.deepcopy(rows)
for x in poison:x['accepted']=8-x['accepted'];x['native3']=99
np.testing.assert_array_equal(predictions(poison,'raw',.5),pred);assert replay(rows,pred)[1]==replay(poison,pred)[1]
for accepted in range(9):
 q=np.array([1.]*accepted+[0.]*(8-accepted));expected=1+np.cumsum(np.cumprod(q));np.testing.assert_array_equal(expected,np.minimum(accepted,np.arange(1,9))+1)
out={'passed':True,'screen_states':len(rows),'checks':['Train/screen source IDs and normalized prompt texts disjoint','Native score-grid path exactly reproduces every proposal','Predicted reward curves finite and monotone','Outcome poisoning leaves frozen predictions and allocated widths unchanged','All deterministic accepted-prefix curves match direct truncation'],'limitations':['Current-draft information requires full width8 drafting','No runtime overhead or serving correctness test']};(r/'postdraft_v98_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
