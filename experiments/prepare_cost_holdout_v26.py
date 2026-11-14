"""Freeze unseen prompts before any v26 outputs; never overwrite the manifest."""
import json,hashlib,random
from pathlib import Path
r=Path(__file__).resolve().parent;dest=r/'cost_v26_manifest.json';assert not dest.exists()
seen=set();sources={}
def collect(v):
 if isinstance(v,dict):
  if isinstance(v.get('prompt_sha256'),str):seen.add(v['prompt_sha256'])
  if isinstance(v.get('text'),str):seen.add(hashlib.sha256(v['text'].encode()).hexdigest())
  for x in v.values():collect(x)
 elif isinstance(v,list):
  for x in v:collect(x)
for p in sorted(r.glob('*.json')):
 try:collect(json.loads(p.read_text()));sources[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
 except (ValueError,UnicodeError):pass
excluded=len(seen);tasks=[];dataset_hashes={}
for domain in ['alpaca','gsm8k','mbpp']:
 path=r.parent/'speculative-baseline/DeepSpec/eval_datasets'/f'{domain}.jsonl';dataset_hashes[domain]=hashlib.sha256(path.read_bytes()).hexdigest()
 rows=[json.loads(line) for line in path.read_text().splitlines()];order=list(range(len(rows)));random.Random('cost-v26/'+domain).shuffle(order)
 selected=[]
 for i in order:
  text=rows[i]['turns'][0];sha=hashlib.sha256(text.encode()).hexdigest()
  if sha in seen or len(text)>3000:continue
  selected.append(dict(domain=domain,index=i,text=text,prompt_sha256=sha,split='cost_v26_heldout'));seen.add(sha)
  if len(selected)==128:break
 assert len(selected)==128
 tasks.extend(selected)
dest.write_text(json.dumps(tasks,indent=2)+'\n')
plan=dict(status='Frozen before any generation; launch only after validation selection audit',manifest_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),prompt_count=384,domain_counts={d:128 for d in ['alpaca','gsm8k','mbpp']},exclusion_sources=sources,excluded_hash_count=excluded,dataset_sha256=dataset_hashes,eligibility='At most3000characters then explicit tokenizer context-length gate; no silent exclusion aftergeneration',batch=32,output_cap=128,precision='FP32,TF32override0,batch_invariant1,Triton_attention',methods=['ar','validation_selected_fixed','original_cost_identity','measured_cost_identity'],orders=[['ar','fixed','original','measured'],['fixed','original','measured','ar'],['original','measured','ar','fixed'],['measured','ar','fixed','original']],timed_repeats=2,warmup='One correctness pass and one full-workload warmup per engine',selection='Best overall fixed width on all144v25validation prompts; require measured beats fixed, AR and original descriptively before launch',interpretation='Cost-model intervention only; no online calibration novelty or generalization claim from validation',uncertainty='Stratified paired prompt-batch bootstrap conditional on observed4engine rounds; report every round and domain, and engine variability separately. Online updates disabled.',code_domain_change='MBPP replaces exhaustedHumanEval; report this domain shift explicitly')
(r/'cost_v26_plan.json').write_text(json.dumps(plan,indent=2)+'\n');print(plan['manifest_sha256'],excluded)
