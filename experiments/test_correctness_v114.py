"""Regression of gate plumbing without GPU: historical prompts and fail-closed checks."""
import hashlib,importlib.metadata,json,os,runpy,sys,tempfile
from pathlib import Path
from types import ModuleType,SimpleNamespace
from unittest.mock import patch
r=Path(__file__).resolve().parent;calls=[];corrupt=False
class FakeLLM:
 def __init__(self,**kw):
  assert kw['hf_overrides']=={'head_dtype':'float32'}
  self.llm_engine=SimpleNamespace(engine_core=SimpleNamespace(shutdown=lambda **kw:None))
 def get_tokenizer(self):return SimpleNamespace(apply_chat_template=lambda x,**kw:x[0]['content'],encode=lambda x:[1])
 def generate(self,prompts,params,**kw):
  calls.extend(prompts)
  return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[int(hashlib.sha256(t.encode()).hexdigest()[:6],16),99 if corrupt else 42])]) for t in prompts]
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp);script=root/'benchmark_correctness_v114.py';script.write_text((r/script.name).read_text().replace('/tmp/aquan-forespark-v63',str(root/'scratch')))
 rows=[dict(split=s,text=f'{s}{i}',domain='fixture') for s in ['profile','validation','test'] for i in range(192)]
 raw=json.dumps(rows).encode();(root/'model_split_v103_manifest.json').write_bytes(raw);(root/'matched_v112_plan.json').write_text(json.dumps(dict(manifest_sha256=hashlib.sha256(raw).hexdigest())))
 (root/'model_v102_metadata.json').write_text('{}');(root/'model_v102_paths.json').write_text(json.dumps(dict(passed=True,paths={'Qwen/Qwen3-8B':'model','RedHatAI/Qwen3-8B-speculator.dflash2':'drafter'},metadata_sha256=hashlib.sha256(b'{}').hexdigest())))
 (root/'vocab_v54_manifest.json').write_text(json.dumps([dict(text=f'old{i}',domain='fixture') for i in range(192)]));(root/'model_v102_plan.json').write_text(json.dumps(dict(indices=list(range(192)))))
 fake=ModuleType('vllm');fake.LLM=FakeLLM;fake.SamplingParams=lambda **kw:kw
 def execute(batch,role,width,order):
  calls.clear();args=[str(script),'--batch',str(batch),'--role',role,'--width',str(width),'--round',str(order)]
  with patch.object(sys,'argv',args),patch.dict(sys.modules,{'vllm':fake}),patch.object(importlib.metadata,'version',return_value='fixture'),patch.dict(os.environ,{},clear=False):runpy.run_path(str(script),run_name='__main__')
  s=json.loads((root/f'correctness_v114_b{batch}_{role}{width}_round{order}.json').read_text());assert len(s['records'])==4*((200+batch-1)//batch)
  assert len(calls)==800 and not any(x.startswith(('validation','test')) for x in calls)
  assert {f'old{i}' for i in [70,97,149,165,0,1,2,3]}<=set(calls)
 for args in [(8,'ar',0,0),(64,'ar',0,0),(8,'ar',0,1),(8,'native',2,0),(64,'native',7,1)]:execute(*args)
 corrupt=True
 try:execute(8,'native',2,1)
 except AssertionError as e:assert 'Exact greedy output mismatch' in str(e)
 else:raise AssertionError('Bad output accepted')
 assert json.loads((root/'correctness_v114_b8_native2_round1.json').read_text())['status']=='failed'
 print(json.dumps(dict(passed=True,scope='CPU fake engine only',checks=['200 regression prompts; no heldout data','Historical failure cases included','4passes per engine at both batch sizes','Fresh AR cross-batch reference and reload','Corrupted output fails immediately']),indent=2))
