"""Select fresh workload prompts; exclude prior text and MBPP IDs before generation."""
import json,hashlib,random
from pathlib import Path
r=Path(__file__).resolve().parent;assert not (r/'load_v76_manifest.json').exists()
norm=lambda s:' '.join(s.lower().split())
seen=set();hashes=set();sources={}
def collect(v):
 if isinstance(v,dict):
  if isinstance(v.get('text'),str):seen.add(norm(v['text']));hashes.add(hashlib.sha256(v['text'].encode()).hexdigest())
  if isinstance(v.get('prompt_sha256'),str):hashes.add(v['prompt_sha256'])
  for x in v.values():collect(x)
 elif isinstance(v,list):
  for x in v:collect(x)
for p in sorted(r.glob('*.json')):
 if p.name.startswith('mbpp_official_'):continue
 try:collect(json.loads(p.read_text()));sources[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
 except (ValueError,UnicodeError):pass
for p in sorted(r.glob('shift_v8_*_fit.jsonl')):
 for line in p.read_text().splitlines():collect(json.loads(line))
 sources[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
prior=set(seen);priorhash=set(hashes)
full=[json.loads(line) for line in (r/'mbpp_official_full.jsonl').read_text().splitlines()]
san=json.loads((r/'mbpp_official_sanitized.json').read_text())
excluded_ids={x['task_id'] for x in full if norm(x['text']) in seen or hashlib.sha256(x['text'].encode()).hexdigest() in hashes}
excluded_ids|={x['task_id'] for x in san if norm(x['prompt']) in seen or hashlib.sha256(x['prompt'].encode()).hexdigest() in hashes}
chosen=[];datasets={};available={}
for domain in ['alpaca','gsm8k','mbpp']:
 if domain=='mbpp':
  path=r/'mbpp_official_full.jsonl';rows=[{'text':x['text'],'index':x['task_id'],'task_id':x['task_id']} for x in full if x['task_id'] not in excluded_ids]
 else:
  path=r.parent/'speculative-baseline/DeepSpec/eval_datasets'/f'{domain}.jsonl';rows=[{'text':json.loads(line)['turns'][0],'index':i} for i,line in enumerate(path.read_text().splitlines())]
 datasets[domain]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 pool=[x for x in rows if norm(x['text']) not in seen and hashlib.sha256(x['text'].encode()).hexdigest() not in hashes and len(x['text'])<=3000]
 available[domain]=len(pool);random.Random('load-v76-confirm/'+domain).shuffle(pool);selected=[]
 for x in pool:
  if norm(x['text']) in seen:continue
  sha=hashlib.sha256(x['text'].encode()).hexdigest();selected.append({'domain':domain,**x,'prompt_sha256':sha,'split':'load_v76_confirmation'})
  seen.add(norm(x['text']));hashes.add(sha)
  if len(selected)==256:break
 assert len(selected)==256,(domain,len(selected),available)
 chosen.extend(selected)
assert not {norm(x['text']) for x in chosen}&prior
assert not {x['prompt_sha256'] for x in chosen}&priorhash
(r/'load_v76_manifest.json').write_text(json.dumps(chosen,indent=2)+'\n')
meta={'selected':len(chosen),'available_beforeselection':available,'excluded_prior_texts':len(prior),'excluded_mbpp_ids':sorted(excluded_ids),'sources':sources,'datasets':datasets,'manifest_sha256':hashlib.sha256((r/'load_v76_manifest.json').read_bytes()).hexdigest(),'mbpp_scope':'All officialMBPP partitions eligible, excludingpreviouslyused task IDs; thisisfreshinference-workload evaluation, notclaimofheldoutmodeltrainingdata. Report expandedpool.'}
(r/'load_v76_manifest_audit.json').write_text(json.dumps(meta,indent=2)+'\n');print({k:v for k,v in meta.items() if k not in ['sources','excluded_mbpp_ids','datasets']})
