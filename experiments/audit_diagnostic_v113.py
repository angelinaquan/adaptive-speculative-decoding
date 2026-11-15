"""Check candidate precision mitigation; this cannot establish original root cause."""
from pathlib import Path
import json
r=Path(__file__).resolve().parent
ar=json.loads((r/'diagnostic_v113_graph_ar.json').read_text());native=json.loads((r/'diagnostic_v113_graph_native.json').read_text())
assert ar['status']==native['status']=='completed'
assert len(ar['records'])==len(native['records'])==5
canonical=ar['records'][0]['outputs']
for s in [ar,native]:
 assert s['config']['hf_overrides']=={'head_dtype':'float32'}
 for z in s['records']:assert z['outputs']==canonical, 'FP32-head output disagreement; do not proceed'
audit=dict(passed=True,ar_native_comparisons=5,raw_repeats_per_engine=4,scope='Eight-prompt precision-mitigation diagnostic only. Full regression gate still required. Not an original BF16 root-cause or sampler-fix claim.')
(r/'diagnostic_v113_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit))
