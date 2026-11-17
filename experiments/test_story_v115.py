"""CPU harness regression for split isolation, clocks, gates and exact equality."""
import hashlib,importlib.metadata,json,os,runpy,sys,tempfile,time
from pathlib import Path
from types import ModuleType,SimpleNamespace
from unittest.mock import patch
r=Path(__file__).resolve().parent
calls=[];corrupt=False
real_monotonic_ns=time.monotonic_ns
class FakeLLM:
 def __init__(self,**kwargs):self.llm_engine=SimpleNamespace(engine_core=SimpleNamespace(shutdown=lambda **kw:None))
 def get_tokenizer(self):return SimpleNamespace(apply_chat_template=lambda messages,**kw:messages[0]['content'],encode=lambda text:[1])
 def generate(self,prompts,params,**kw):
  calls.extend(prompts)
  return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[int(hashlib.sha256(t.encode()).hexdigest()[:6],16),99 if corrupt else 42])]) for t in prompts]
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp);script=root/'benchmark_story_v115.py';script.write_text((r/script.name).read_text().replace('/tmp/aquan-forespark-v63',str(root/'scratch')))
 for b in [8,64]:
  for role in ['ar0','native3','native7']:
   for sub in ['vllm','inductor','triton']:(root/f'scratch/cache_v114/b{b}_{role}'/sub).mkdir(parents=True)
 rows=[dict(split=s,text=f'{s}-{i}',prompt_sha256=f'{s}-{i}') for s,n in [('profile',192),('validation',96),('test',192)] for i in range(n)]
 raw=json.dumps(rows).encode();(root/'model_split_v103_manifest.json').write_bytes(raw)
 (root/'story_v115_plan.json').write_text(json.dumps(dict(manifest_sha256=hashlib.sha256(raw).hexdigest(),horizons_test_passes=[1,4,16],profile_order_indices=list(reversed(range(192))))))
 (root/'model_v102_paths.json').write_text(json.dumps(dict(passed=True,paths={'Qwen/Qwen3-8B':'fixture','RedHatAI/Qwen3-8B-speculator.dflash2':'drafter'})))
 (root/'matched_v112_plan.json').write_bytes((root/'story_v115_plan.json').read_bytes())
 (root/'benchmark_matched_v112.py').write_bytes(script.read_bytes())
 (root/'correctness_v112_gate.json').write_text(json.dumps(dict(passed=True,plan_sha256=hashlib.sha256((root/'story_v115_plan.json').read_bytes()).hexdigest(),benchmark_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),precision={'backbone':'bfloat16','head':'float32'})))
 fake=ModuleType('vllm');fake.LLM=FakeLLM;fake.SamplingParams=lambda **kw:kw
 def execute(batch,role,phase,tag,width=0,budget=192):
  calls.clear();start=real_monotonic_ns()+9_000_000_000
  args=[str(script),'--budget',str(budget),'--batch',str(batch),'--role',role,'--phase',phase,'--tag',tag,'--width',str(width),'--pipeline-start-ns',str(start)]
  with patch.object(time,'monotonic_ns',side_effect=lambda:real_monotonic_ns()+10_000_000_000),patch.object(sys,'argv',args),patch.dict(sys.modules,{'vllm':fake}),patch.object(importlib.metadata,'version',return_value='fixture'),patch.dict(os.environ,{},clear=False):runpy.run_path(str(script),run_name='__main__')
  return json.loads((root/f'story_v115_{tag}.json').read_text())
 for batch in [8,64]:
  ref=execute(batch,'ar','reference',f'b{batch}_reference');assert len(ref['outputs'])==288
  assert not any(x.startswith('test-') for x in calls)
  execute(batch,'native','profile',f'b{batch}_profile',3);assert len(calls)==384 and all(x.startswith('profile-') for x in calls)
  execute(batch,'native','profile',f'b{batch}_small',3,64);assert calls==[f'profile-{i}' for i in reversed(range(128,192))]*2
  dep=execute(batch,'native','deploy',f'b{batch}_deploy',7);assert len(calls)==1536 and all(x.startswith('validation-') for x in calls)
  assert [x['prompts'] for x in dep['milestones'].values()]==[96,384,1536]
  times=[x['total_pipeline_seconds'] for x in dep['milestones'].values()];assert times==sorted(times) and times[0]>=1
 corrupt=True
 try:execute(8,'native','profile','bad_output',3)
 except AssertionError as e:assert 'Exact AR mismatch' in str(e)
 else:raise AssertionError('Bad output accepted')
 (root/'correctness_v112_gate.json').write_text(json.dumps(dict(passed=False)))
 try:execute(8,'native','profile','failed_gate',3)
 except AssertionError:pass
 else:raise AssertionError('Failed gate accepted')
 assert not calls
 print(json.dumps(dict(passed=True,scope='CPU fake engine only',checks=['Nested64/192 budgets use identical prefix ordering','Declared96 heldout prompts, no legacy test access','Both batches8/64 including partial final batch','All total-time milestones include prior pipeline interval','Output corruption aborts','Failed correctness gate prevents engine construction']),indent=2))
