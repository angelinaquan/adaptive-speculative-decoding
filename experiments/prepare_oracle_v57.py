"""Lower-concurrency feasibility diagnostic; future outputs make this nondeployable."""
import json,hashlib
from pathlib import Path
r=Path(__file__).resolve().parent
read=lambda n:json.loads((r/n).read_text())
sha=lambda n:hashlib.sha256((r/n).read_bytes()).hexdigest()
assert read('oracle_v56_summary.json')['comparisons']['oracle4096']['native']['gain_percent']<10
indices=[i for start in [0,64,128] for i in range(start,start+16)]
manifest=[read('vocab_v54_manifest.json')[i] for i in indices]
(r/'oracle_v57_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
ref=read('vocab_v54_ar-reference.json');rank=read('context_v55_priors.json')['frequency']
outputs=[ref['outputs'][i] for i in indices]
vocabs={}
for start in range(0,48,8):
 selected=set(v for seq in outputs[start:start+8] for v in seq);assert len(selected)<=1024
 selected.update([v for v in rank if v not in selected][:1024-len(selected)])
 vocabs[str(start)]=sorted(selected)
(r/'oracle_v57_vocabularies.json').write_text(json.dumps(dict(vocabularies=vocabs,reference_outputs=outputs,reference_sha256=sha('vocab_v54_ar-reference.json'),indices=indices,scope='NONDEPLOYABLE future-output oracle; not a proposed method.'),indent=2)+'\n')
(r/'oracle_v57_init1024.json').write_text(json.dumps(dict(token_ids=sorted(rank[:1024]),source='Source-only initialization; oracle updates follow'))+'\n')
roles=['native3','oracle3','native7','oracle7']
plan=dict(roles=roles,orders=[roles,list(reversed(roles))],manifest_sha256=sha('oracle_v57_manifest.json'),priors_sha256=sha('context_v55_priors.json'),oracle_sha256=sha('oracle_v57_vocabularies.json'),scope='NONDEPLOYABLE ORACLE atbatch8 on48reusedvalidation prompts; futuretargettokens, notmethodsuccess.',limits=['Width3 and7 only; no all-width tuned result.','48 reused prompts,2reverseorders; no significance.','Future outputs cannot be used by realpolicy.','Oracle is not universal upperbound.'],rationale='V56 b64 oracle only8.03%; test whether lower concurrency offersmaterialheadroom. Width3 is a necessarystrongcontrol from priorlowbatchscreens.',success='Oracle must exceed10%over besttestednativewidth inbothorders tojustify implementingonlinevocab inthisregime.')
(r/'oracle_v57_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
