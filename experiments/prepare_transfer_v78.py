import json,random,hashlib
from pathlib import Path
r=Path(__file__).resolve().parent;seen=set()
norm=lambda x:' '.join(x.lower().split())
def collect(x):
 if isinstance(x,dict):
  if isinstance(x.get('text'),str):seen.add(norm(x['text']))
  for v in x.values():collect(v)
 elif isinstance(x,list):
  for v in x:collect(v)
for p in r.glob('*.json'):
 if p.name.startswith('mbpp_official'):continue
 try:collect(json.loads(p.read_text()))
 except (ValueError,UnicodeError):pass
for p in r.glob('*fit.jsonl'):
 for l in p.read_text().splitlines():collect(json.loads(l))
manifest=[];audit={}
for d,dataset in [('alpaca','alpaca'),('math500','math500'),('lbpp','lbpp')]:
 p=r.parent/'speculative-baseline/DeepSpec/eval_datasets'/f'{dataset}.jsonl';rows=[dict(text=json.loads(l)['turns'][0],index=i) for i,l in enumerate(p.read_text().splitlines())];pool=list({norm(x['text']):x for x in rows if norm(x['text']) not in seen and len(x['text'])<=3000}.values());random.Random('transfer-v78/'+d).shuffle(pool);assert len(pool)>=128,(d,len(pool));audit[d]={'fresh_available':len(pool),'selected':128,'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 for x in pool[:128]:manifest.append(dict(domain=d,split='fresh_test',**x,prompt_sha256=hashlib.sha256(x['text'].encode()).hexdigest()))
assert len({norm(x['text']) for x in manifest})==384
assert not (r/'transfer_v78_manifest.json').exists();(r/'transfer_v78_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
plan={'question':'Frozen profiling-selected widths under new benchmarks and doubled output cap','manifest_sha256':hashlib.sha256((r/'transfer_v78_manifest.json').read_bytes()).hexdigest(),'batches':[8,64],'widths':list(range(1,9)),'cap':128,'rounds':2,'warmup_passes':2,'timed_passes':2,'frozen_selected_widths':{'8':8,'64':3},'selection_provenance':'v77 mixed192 and all64 posthoc mixed subsets chose8/3; locked before v78outputs. No fitting on v78.','primary':'For each batch, frozen width achieves at least95% of best measured fixed-width pooled test throughput.','secondary':'Everydomain and round; width3vs8 gain at64; transfer from64tokencap Alpaca/GSM8K/MBPP profiling to128tokencap freshAlpaca/MATH500/LBPP. Paired bootstrap over2rounds and2blocks/domain for comparisonvs8; oracle gap is descriptive due test winner selection.','freshness':audit,'limitations':['Only128freshprompts/domain,6independent64promptblocks; twoengineorders.','Test all8 widths for hindsightreference, not retuning deployedchoice.','SameH200/modelpair; tests dataset and generationlength transfer, not hardware/model transfer.','Prior profiling subsets overlap; 64 combinations not64 independent datasets.','No claim prompts unseen in releasedmodelpretraining.','Both dataset and outputcap change; this is joint robustness, not isolated causal attribution.']}
(r/'transfer_v78_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
s=(r/'benchmark_profile_v77.py').read_text().replace('profile_v77','transfer_v78').replace('1344','384').replace('len(tok.encode(x))+64','len(tok.encode(x))+128').replace('max_tokens=64','max_tokens=128');(r/'benchmark_transfer_v78.py').write_text(s)
print(audit)
