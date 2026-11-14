"""CPU checks of buffer substitution/restoration, independent of vLLM import."""
import ast,contextlib,json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
r=Path(__file__).resolve().parent
source=ast.parse((r/'fresh_confidence_v39.py').read_text())
function=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='get_num_tokens')

def native(manager,tokens,drafts):
    if getattr(manager,'fail',False):raise RuntimeError('injected native failure')
    capacities={k:len(drafts[k]) for k in tokens};anchors={k:tokens[k]-capacities[k] for k in tokens}
    scores=[]
    for key,cap in capacities.items():
        scores.extend(np.cumprod(manager._stale_confidences[manager._stale_idx].np[manager.req_states.req_id_to_index[key]])[:cap])
    budget=int(np.count_nonzero(np.asarray(scores)>.5))
    manager._batch_budget=(capacities,anchors,budget)
    return sum(anchors.values())+budget

globals_={'np':np,'_original':native,'gpu_sync_allowed':contextlib.nullcontext}
exec(compile(ast.Module(body=[function],type_ignores=[]),'<fresh-buffer-check>','exec'),globals_)
wrapper=globals_['get_num_tokens'];checks=[]
for mode in ['native','fresh','diagnostic']:
 for pending in [[],[3]]:
    old=np.full((5,7),.2,dtype=np.float32);latest=np.full((5,7),.9,dtype=np.float32);current=latest.copy();current[pending]=1.
    m=NS(_v39_mode=mode,_stale_idx=0,_stale_confidences=[NS(np=old),NS(np=latest)],_copy_events=[NS(synchronize=lambda:None),NS(synchronize=lambda:None)],_pending_resets=pending,_v39_trace=[],_max_total_logits=100,num_bonus_tokens=1,
        req_states=NS(req_id_to_index={'a':3,'b':1},num_computed_tokens_np=np.ones(5,dtype=int),prefill_len=NS(np=np.ones(5,dtype=int))),_confidence_probs=NS(cpu=lambda:NS(numpy=lambda:current)))
    tokens={'a':8,'b':4};drafts={'a':list(range(7)),'b':list(range(3))}
    snapshot=old.copy();latest_snapshot=latest.copy();result=wrapper(m,tokens,drafts)
    np.testing.assert_array_equal(old,snapshot);np.testing.assert_array_equal(latest,latest_snapshot)
    if mode=='fresh':assert result>2
    else:assert result==2
    if mode=='diagnostic':assert len(m._v39_trace)==1 and m._v39_trace[0]['fresh_budget']>m._v39_trace[0]['old_budget']
    checks.append(dict(mode=mode,pending_reset=pending,passed=True))
    if mode=='fresh':
        m.fail=True
        try:wrapper(m,tokens,drafts)
        except RuntimeError:pass
        else:raise AssertionError('Expected injected failure')
        np.testing.assert_array_equal(old,snapshot)
report={'checks':checks,'exception_restoration':True,'scope':'CPU control-flow tests only. GPU confidence equality and output parity are checked by live pilot.'}
(r/'fresh_v39_cpu_checks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
