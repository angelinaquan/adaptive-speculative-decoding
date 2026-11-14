"""Independent reconstruction of every vocabulary from prompt/source data only."""
import json,hashlib,os
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer
import wandb
r=Path(__file__).resolve().parent
read=lambda n:json.loads((r/n).read_text())
plan=read('context_v55_plan.json');priors=read('context_v55_priors.json')
source=read('prefix_v52_ar-reference.json');trace=read('prefix_v52_fixed7_greedy-round0.json')
assert source['status']==trace['status']=='completed' and source['outputs']==trace['outputs']
assert priors['source_sha256']==hashlib.sha256((r/'prefix_v52_ar-reference.json').read_bytes()).hexdigest()
assert priors['trace_sha256']==hashlib.sha256((r/'prefix_v52_fixed7_greedy-round0.json').read_bytes()).hexdigest()
freq=Counter(v for seq in source['outputs'] for v in seq)
weighted=Counter()
for block in trace['prefix_traces']:
 for row in block['records']:
  p=row['accepted_prefix']
  for j,v in enumerate(p):weighted[v]+=len(p)-j
assert priors['frequency']==sorted(range(151936),key=lambda v:(-freq[v],v))
assert priors['prefix']==sorted(range(151936),key=lambda v:(-weighted[v],-freq[v],v))
tasks=read('vocab_v54_manifest.json');sources=read('prefix_v52_manifest.json')
assert not {x['text'].strip() for x in sources}.intersection(x['text'].strip() for x in tasks)
tok=AutoTokenizer.from_pretrained(read('model_paths.json')['Qwen/Qwen3-4B'],local_files_only=True)
prompts=[tok.apply_chat_template([{'role':'user','content':x['text']}],tokenize=False,add_generation_prompt=True,enable_thinking=False) for x in tasks]
ids=[tok.encode(p) for p in prompts]
summary={}
ref=read('context_v55_ar-reference.json')
for role in plan['roles']:
 if role=='native':continue
 k=4096 if role.endswith('4096') else 8192
 for rd in range(2):
  data=read(f'context_v55_{role}-round{rd}.json');assert data['status']=='completed'
  for start in [0,64,128]:
   c=Counter(v for seq in ids[start:start+64] for v in seq) if role.startswith('context') else Counter()
   ranked_prompt=sorted(c,key=lambda v:(-c[v],v))[:k]
   chosen=set(ranked_prompt)
   ranking=priors['prefix' if role=='contextprefix8192' else 'frequency']
   chosen.update([v for v in ranking if v not in chosen][:k-len(chosen)])
   assert len(chosen)==k
   for u in [z for z in data['vocabulary_updates'] if z['start']==start]:
    assert u['token_ids']==sorted(chosen)
    assert u['stats']==dict(unique_prompt_tokens=len(c),included_prompt_tokens=len(ranked_prompt))
   outputs=[v for seq in ref['outputs'][start:start+64] for v in seq]
   summary[f'{role}/{tasks[start]["domain"]}']=dict(prompt_unique=len(c),target_token_coverage=sum(v in chosen for v in outputs)/len(outputs),scope='Post-selection descriptive target coverage, not acceptance or speed.')
result=dict(passed=True,source_only_ranks_verified=True,prompt_only_vocabulary_reconstruction_verified=True,coverage=summary)
out=r/'context_v55_selection_audit.json';out.write_text(json.dumps(result,indent=2)+'\n')
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='context-v55-coverage',name='independent-selection-audit',dir=os.environ['SERVING_SCRATCH'])
run.summary['passed']=True
for k,v in summary.items():run.summary[k+'/target_coverage']=v['target_token_coverage']
a=wandb.Artifact('context-v55-selection-audit','audit');a.add_file(str(out));a.add_file(__file__);run.log_artifact(a)
(r/'context_v55_selection_audit_run.json').write_text(json.dumps(dict(run_url=run.url))+'\n');run.finish()
