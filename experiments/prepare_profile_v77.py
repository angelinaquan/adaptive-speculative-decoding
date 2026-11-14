"""Freeze disjoint profile/validation/test workloads before any v77 timings."""
import json,hashlib,random
from pathlib import Path
r=Path(__file__).resolve().parent
assert not (r/'profile_v77_manifest.json').exists()
norm=lambda s:' '.join(s.lower().split())
seen=set();hashes=set()
def collect(v):
 if isinstance(v,dict):
  if isinstance(v.get('text'),str):seen.add(norm(v['text']))
  if isinstance(v.get('prompt_sha256'),str):hashes.add(v['prompt_sha256'])
  for z in v.values():collect(z)
 elif isinstance(v,list):
  for z in v:collect(z)
for p in r.glob('*.json'):
 if p.name.startswith('mbpp_official'):continue
 try:collect(json.loads(p.read_text()))
 except (ValueError,UnicodeError):pass
for p in r.glob('*fit.jsonl'):
 for line in p.read_text().splitlines():collect(json.loads(line))
mbpp=[json.loads(l) for l in (r/'mbpp_official_full.jsonl').read_text().splitlines()]
san=json.loads((r/'mbpp_official_sanitized.json').read_text())
old_ids={x['task_id'] for x in mbpp if norm(x['text']) in seen}|{x['task_id'] for x in san if norm(x['prompt']) in seen}
allrows=[];audit={};sources={}
for domain in ['alpaca','gsm8k','mbpp']:
 if domain=='alpaca':
  p=r.parent/'speculative-baseline/DeepSpec/eval_datasets/alpaca.jsonl';rows=[dict(text=json.loads(l)['turns'][0],index=i) for i,l in enumerate(p.read_text().splitlines())]
 elif domain=='gsm8k':
  p=r/'gsm8k_official_train.jsonl';rows=[dict(text=json.loads(l)['question'],index=i) for i,l in enumerate(p.read_text().splitlines())]
 else:
  p=r/'mbpp_official_full.jsonl';rows=[dict(text=x['text'],index=x['task_id'],task_id=x['task_id']) for x in mbpp]
 sources[domain]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 unique={norm(x['text']):x for x in rows if len(x['text'])<=3000};rows=list(unique.values())
 fresh=[x for x in rows if norm(x['text']) not in seen and hashlib.sha256(x['text'].encode()).hexdigest() not in hashes and (domain!='mbpp' or x['task_id'] not in old_ids)]
 random.Random('v77/fresh/'+domain).shuffle(fresh);assert len(fresh)>=192,(domain,len(fresh))
 val,test=fresh[:64],fresh[64:192];held={norm(x['text']) for x in val+test}
 pool=[x for x in rows if norm(x['text']) not in held];random.Random('v77/profile/'+domain).shuffle(pool);profile=pool[:256];assert len(profile)==256
 audit[domain]={'fresh_available':len(fresh),'profile_reused_texts':sum(norm(x['text']) in seen for x in profile),'counts':{'profile':256,'validation':64,'test':128}}
 for split,subset in [('profile',profile),('validation',val),('test',test)]:
  for x in subset:allrows.append(dict(domain=domain,split=split,**x,prompt_sha256=hashlib.sha256(x['text'].encode()).hexdigest()))
assert len(allrows)==1344 and len({norm(x['text']) for x in allrows})==1344
(r/'profile_v77_manifest.json').write_text(json.dumps(allrows,indent=2)+'\n')
plan={'question':'Profiling budget and cross-domain transfer of native fixed draft length selection','manifest_sha256':hashlib.sha256((r/'profile_v77_manifest.json').read_bytes()).hexdigest(),'widths':list(range(1,9)),'batches':[8,64],'rounds':2,'warmups':2,'timed_passes':2,'cap':64,'budgets_prompts_per_source_domain':[64,128,256],'subsets':1000,'seed':1865777,'selection':'Minimize total profiling seconds at each batch size; ties choose smaller width. No test feedback. Profile subsets are nested random 64-prompt blocks (1,2,4). Paired across widths and batch sizes. All widths measured independently.','evaluation':'All 3x3 source-to-target domain pairs at each batch/budget; pooled source with same total budget 192 via one block/domain, and full pooled768. Fixed width8, global fixed width chosen on validation with equal prompt counts across batches/domains, and validation-selected per-batch and per-domain widths. Test oracle is descriptive ceiling only.','metrics':['test tokens/s','gain vs fixed8 and validation baselines','relative throughput loss to hindsight test best','fraction within 1% and 5% of test best','selected width histogram','profiling generation seconds summed across all8 widths','amortization output tokens when savings positive'],'uncertainty':'Selection variability across overlapping profiling subsets is descriptive, not 1000 independent experiments. Report both engine rounds, domain-specific results, and test-block paired bootstrap for frozen full-profile selectors (two64promptblocks/domain; limited precision).','gates':['all outputs exact AR, including warmups','AR agrees across batch sizes','disjoint splits and fresh validation/test audit','source/config/manifest hashes','test-label perturbation leaves selection unchanged','changing unselected-width test outcomes never changes selected width'],'limitations':['One H200/model pair and short greedy generations. No live switch cost.','GSM8K official training questions and MBPP all partitions are inference workloads; no claim unseen in model pretraining.','256 profile prompts/domain; profiling may reuse earlier-run examples, validation/test may not.','Profile cost includes all8 widths; report measured generation cost separately from cold-start/compile cost. Cold-start setup wall time amortization is separate.','No claimed statistical independence of repeated passes or resampled overlapping subsets.']}
(r/'profile_v77_plan.json').write_text(json.dumps(plan,indent=2)+'\n');(r/'profile_v77_manifest_audit.json').write_text(json.dumps({'passed':True,'domains':audit,'sources':sources,'manifest_sha256':plan['manifest_sha256']},indent=2)+'\n');print(audit)
