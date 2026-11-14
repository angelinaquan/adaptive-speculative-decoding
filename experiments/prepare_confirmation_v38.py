"""Freeze fresh confirmation prompts and decisions before model generation."""
import json,hashlib,random
from pathlib import Path
r=Path(__file__).resolve().parent
assert not (r/'cost_v38_manifest.json').exists() and not (r/'cost_v38_plan.json').exists()
norm=lambda t:' '.join(t.lower().split())
seen=set();hashes=set();sources={}
def collect(v):
 if isinstance(v,dict):
  if isinstance(v.get('text'),str):seen.add(norm(v['text']));hashes.add(hashlib.sha256(v['text'].encode()).hexdigest())
  if isinstance(v.get('prompt_sha256'),str):hashes.add(v['prompt_sha256'])
  for value in v.values():collect(value)
 elif isinstance(v,list):
  for value in v:collect(value)
for p in sorted(r.glob('*.json')):
 if p.name.startswith('mbpp_official_'):continue
 try:collect(json.loads(p.read_text()));sources[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
 except (ValueError,UnicodeError):pass
for p in sorted(r.glob('shift_v8_*_fit.jsonl')):
 for line in p.read_text().splitlines():collect(json.loads(line))
 sources[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
prior_norm=set(seen);prior_hashes=set(hashes)
sanitized=json.loads((r/'mbpp_official_sanitized.json').read_text());full=[json.loads(line) for line in (r/'mbpp_official_full.jsonl').read_text().splitlines()]
excluded_ids={x['task_id'] for x in sanitized if norm(x['prompt']) in seen or hashlib.sha256(x['prompt'].encode()).hexdigest() in hashes}
excluded_ids|={x['task_id'] for x in full if norm(x['text']) in seen or hashlib.sha256(x['text'].encode()).hexdigest() in hashes}
tasks=[];data_sources={}
for domain in ['alpaca','gsm8k','mbpp']:
 if domain=='mbpp':
  p=r/'mbpp_official_full.jsonl';rows=[dict(text=x['text'],index=x['task_id'],task_id=x['task_id']) for x in full if 11<=x['task_id']<=510 and x['task_id'] not in excluded_ids]
 else:
  p=r.parent/'speculative-baseline/DeepSpec/eval_datasets'/f'{domain}.jsonl';rows=[dict(text=json.loads(line)['turns'][0],index=i) for i,line in enumerate(p.read_text().splitlines())]
 data_sources[domain]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 order=list(range(len(rows)));random.Random('cost-v38-fresh/'+domain).shuffle(order);selected=[]
 for i in order:
  row=rows[i];text=row['text'];sha=hashlib.sha256(text.encode()).hexdigest()
  if norm(text) in seen or sha in hashes or len(text)>3000:continue
  selected.append(dict(domain=domain,**row,prompt_sha256=sha,split='cost_v38_heldout'));seen.add(norm(text));hashes.add(sha)
  if len(selected)==256:break
 assert len(selected)==256,(domain,len(selected));tasks.extend(selected)
assert len(tasks)==768 and len({norm(t['text']) for t in tasks})==768
assert not {norm(t['text']) for t in tasks}&prior_norm
assert not {t['prompt_sha256'] for t in tasks}&prior_hashes
assert not {t['task_id'] for t in tasks if t['domain']=='mbpp'}&excluded_ids
manifest=r/'cost_v38_manifest.json';manifest.write_text(json.dumps(tasks,indent=2)+'\n')
plan=dict(status='Frozen before any v38 model generation',scope='Fresh768 prompts, equal256 chat/math/code, batch64, same model pair/H200/backend. Mixed workload is primary; chat is secondary and cannot replace primary success.',manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),prompt_count=768,batch=64,output_cap=128,domain_counts={d:256 for d in ['alpaca','gsm8k','mbpp']},source_datasets=data_sources,excluded_prior_normalized_texts=len(prior_norm),excluded_prior_hashes=len(prior_hashes),excluded_mbpp_task_ids=sorted(excluded_ids),exclusion_sources=sources,eligibility='At most3000characters; tokenizercontextgate beforegeneration; no silent prompt exclusion',precision='FP32 tensors, batch_invariant1, NVIDIA_TF32_OVERRIDE0 requested. Cached Triton TF32 instructions mean strict IEEE arithmetic is NOT established.',roles=['fixed3','fixed7','original','measured'],orders=[['fixed3','fixed7','original','measured'],['fixed7','original','measured','fixed3'],['original','measured','fixed3','fixed7'],['measured','fixed3','fixed7','original']],reference='Separate fresh AR session first; every output in every warmup/timedpass checked against it',timed_repeats=2,warmup='One correctness pass plus one fullworkload warmup perengine',method_selection='Retain minimal target-cost-only v26 intervention. v32onlinecalibration,v34dualcurve,v37lagallocation did not show consistent material incremental improvement. No further tuning on v38.',global_fixed='fixed7',domain_fixed={'alpaca':'fixed3','gsm8k':'fixed7','mbpp':'fixed7'},baseline_selection='Selected from complete v29 allwidth1..7 sweep on separate reusedvalidation data. Both fixed3 andfixed7 run onallfreshprompts; domainoracle assembled from their prespecified domain timings.',validation_gate_note='v29 missed aspirational5% gate: +4.6785% vs globalfixed7. Launch this independent confirmation because the exploratory effect is materially larger than v26, but do NOT relabel v29 as passing. Confirmation retains5% practical target.',primary_success='Mixed-workload measured throughput gain>=5% versus BOTH globalfixed7 and prespecified domain-specific fixed baseline AND original adaptive; all4round gainspositive and paired stratified-batch and batch-plus-round bootstrap95% lowerboundspositive for each.',secondary='Report everydomain andmethod regardlessofoutcome, including chat vsfixed3. A domain-only gain does not pass mixed primarycriterion.',uncertainty='10000 paired stratified batch bootstrap draws,4 batches/domain; resample same promptgroups acrossmethods/rounds; second interval alsoresamples4paired engine rounds. Repeats kept together. One device/model pair only.',code_source_change='EarlierMBPP used sanitized257testsubset; thisconfirmation uses full officialtest tasks11..510, excluding previouslyused task IDs acrossoriginal/edited descriptions. Report this change.',mbpp_provenance={'full_url':'https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl','sanitized_url':'https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json','sanitized_sha256':hashlib.sha256((r/'mbpp_official_sanitized.json').read_bytes()).hexdigest()})
(r/'cost_v38_plan.json').write_text(json.dumps(plan,indent=2)+'\n');print(plan['manifest_sha256'],plan['excluded_prior_normalized_texts'],len(excluded_ids))
