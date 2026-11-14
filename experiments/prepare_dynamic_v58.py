import json,hashlib
from pathlib import Path
r=Path(__file__).resolve().parent
read=lambda n:json.loads((r/n).read_text())
sha=lambda n:hashlib.sha256((r/n).read_bytes()).hexdigest()
rank=read('context_v55_priors.json')['frequency']
for k in [1024,4096]:
 (r/f'dynamic_v58_init{k}.json').write_text(json.dumps(dict(token_ids=sorted(rank[:k]),source='Source-only all-target frequency initialization'))+'\n')
roles=['native7','prompt1024','recency1024','recency4096']
plan=dict(roles=roles,orders=[roles,list(reversed(roles))],manifest_sha256=sha('oracle_v57_manifest.json'),priors_sha256=sha('context_v55_priors.json'),scope='Deployable generation-time target-candidate recency BASELINE inspired byNanoSpec; no newmethodclaim.48reusedprompts,b8,width7,2reverseorders.',conditional_launch='Require completedv57diagnostic oracle7>10%vsbothnative3/native7 inbothorders, andallgatespass.',selection='Reset at every promptbatch. Initialize fromprompttokenfrequency plus sourceall-targetfrequency. After unmodifiedverifier, obtain top16targetcandidates onverifiedprefix rows including correction/bonus; exclude rejectedtails. Selectmostrecent1024/4096tokens withtokenID tie-break. No futureoutput data.',timing='Includes initialselection/resetRPC and everyGPU topk/recency/gather update. After-generation auditRPC outsidetiming, no auditduringtimedsteps.',success='Deployable recency mustexceed10%vsnative7 inbothorders beforefurtherwidth/budgettuning andfreshconfirmation.',limitations=['Recency ispriorwork, notproposedprefixmethod.','Synchronousgather andsharedbatchvocabulary differ fromNanoSpec; notfullpaperreproduction.','Onlyb8width7greedyFP32singleGPU; notstochastic.','48reusedprompts,2orders; no significance.'])
(r/'dynamic_v58_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
