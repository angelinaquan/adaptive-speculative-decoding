"""DIAGNOSTIC ONLY: future-output vocabulary, never a deployable result."""
import json,hashlib
from pathlib import Path
r=Path(__file__).resolve().parent
ref=json.loads((r/'vocab_v54_ar-reference.json').read_text());assert ref['status']=='completed'
priors=json.loads((r/'context_v55_priors.json').read_text())
vocabs={}
for start in [0,64,128]:
 chosen=set(v for seq in ref['outputs'][start:start+64] for v in seq)
 assert len(chosen)<=4096
 chosen.update([v for v in priors['frequency'] if v not in chosen][:4096-len(chosen)])
 vocabs[str(start)]=sorted(chosen)
out=dict(vocabularies=vocabs,reference_sha256=hashlib.sha256((r/'vocab_v54_ar-reference.json').read_bytes()).hexdigest(),scope='NONDEPLOYABLE ORACLE: contains future target output tokens. Diagnostic feasibility only; never count as method success.')
(r/'oracle_v56_vocabularies.json').write_text(json.dumps(out,indent=2)+'\n')
plan=dict(roles=['native','oracle4096'],orders=[['native','oracle4096'],['oracle4096','native']],manifest_sha256=hashlib.sha256((r/'vocab_v54_manifest.json').read_bytes()).hexdigest(),priors_sha256=hashlib.sha256((r/'context_v55_priors.json').read_bytes()).hexdigest(),oracle_sha256=hashlib.sha256((r/'oracle_v56_vocabularies.json').read_bytes()).hexdigest(),scope=out['scope'],conditional_launch='Only if complete v55 screen fails >10% inbothrounds for all deployable roles.',timing='SameGPU, nativefixed7 vs4096future-output vocabulary; includeRPC/gather costs. Two reverseorders,2warmups2timedpasses, freshARparity.',limits=['Not a formal upper bound on all pruning policies; generation trajectories may change.','Contains future answers; no deployment, novelty or achieved-goal claim.','B64, width7, greedy only.'])
(r/'oracle_v56_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
