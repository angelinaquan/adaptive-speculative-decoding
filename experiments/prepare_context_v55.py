import json,hashlib
from pathlib import Path
import numpy as np
from prefix_vocab_v51 import prefix_weights
r=Path(__file__).resolve().parent
ar=json.loads((r/'prefix_v52_ar-reference.json').read_text())
trace=json.loads((r/'prefix_v52_fixed7_greedy-round0.json').read_text())
assert ar['status']==trace['status']=='completed' and ar['outputs']==trace['outputs']
counts=np.bincount([v for row in ar['outputs'] for v in row],minlength=151936)
prefixes=[z['accepted_prefix'] for t in trace['prefix_traces'] for z in t['records']]
w=prefix_weights(prefixes,151936)
freq=np.lexsort((np.arange(151936),-counts)).tolist()
prefix=np.lexsort((np.arange(151936),-counts,-w)).tolist()
priors=dict(frequency=freq,prefix=prefix,source_sha256=hashlib.sha256((r/'prefix_v52_ar-reference.json').read_bytes()).hexdigest(),trace_sha256=hashlib.sha256((r/'prefix_v52_fixed7_greedy-round0.json').read_bytes()).hexdigest(),scope='Source only; prefix weight first then all-target frequency tie-break. No validation output.')
(r/'context_v55_priors.json').write_text(json.dumps(priors)+'\n')
for k in [4096,8192]:
 (r/f'context_v55_init{k}.json').write_text(json.dumps(dict(token_ids=sorted(freq[:k]),source='All-target source frequency initialization'))+'\n')
roles=['native','static8192','contextfreq4096','contextfreq8192','contextprefix8192']
plan=dict(roles=roles,orders=[roles,list(reversed(roles))],scope='Context coverage screen on192 reusedvalidation prompts; no fresh/significance claim.',manifest_sha256=hashlib.sha256((r/'vocab_v54_manifest.json').read_bytes()).hexdigest(),priors_sha256=hashlib.sha256((r/'context_v55_priors.json').read_bytes()).hexdigest(),selection='Prompt token frequency first; fill remaining slots with source ranking. Same batch prompt inputs only, no output feedback; no state across batches.',timing='Prompt-token counting, selection, synchronous worker RPC and weight gathering included in each generate timing; initial tokenization cached for all methods.',success='Both rounds >10% vsnative, then width/budget retuning andfreshconfirmation. Compareprefix vscontextfreq8192 to isolate objective.',limitations=['Prompt context is a simple control, not NanoSpec reproduction.','Width7 only and one model/GPU; vocabulary budget unretuned.','Static8192 may include zero-source-frequency IDs; report count.'],zero_source_fill8192=int(np.sum(counts[freq[:8192]]==0)))
(r/'context_v55_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
