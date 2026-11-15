"""CPU harness integration: split isolation, exact-output gate and total-time endpoints."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import time
from types import ModuleType,SimpleNamespace
from unittest.mock import patch

r=Path(__file__).resolve().parent
calls=[];corrupt=False
class FakeLLM:
    def __init__(self,**kw):self.llm_engine=SimpleNamespace(engine_core=SimpleNamespace(shutdown=lambda **kw:None))
    def get_tokenizer(self):
        return SimpleNamespace(apply_chat_template=lambda messages,**kw:messages[0]['content'],encode=lambda text:[1])
    def collective_rpc(self,method,kwargs=None):return [dict(target_unchanged=True,logging_disabled=True)]
    def generate(self,prompts,params,**kw):
        calls.extend(prompts)
        return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[int(hashlib.sha256(t.encode()).hexdigest()[:6],16),99 if corrupt else 42])]) for t in prompts]

with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);script=root/'benchmark_matched_v110.py'
    code=(r/script.name).read_text().replace('/tmp/aquan-forespark-v63',str(root/'scratch'))
    script.write_text(code)
    for name in ['cache_v72/ar0','cache_v72/native3','cache_v105/forced']:
        for sub in ['vllm','inductor','triton']:(root/'scratch'/name/sub).mkdir(parents=True,exist_ok=True)
    rows=[dict(split=s,text=f'{s}-{i}',prompt_sha256=f'{s}-{i}') for s in ['profile','test','validation'] for i in range(192)]
    raw=json.dumps(rows).encode();(root/'model_split_v103_manifest.json').write_bytes(raw)
    (root/'matched_v110_plan.json').write_text(json.dumps(dict(manifest_sha256=hashlib.sha256(raw).hexdigest(),horizons_test_passes=[1,4,16])))
    (root/'export_v68_checks.json').write_text(json.dumps(dict(passed=True,UNTRAINED_HEADS=True,target_path='fixture',checkpoint='wrapper',original_checkpoint='drafter')))
    fake=ModuleType('vllm');fake.LLM=FakeLLM;fake.SamplingParams=lambda **kw:kw
    def execute(role,phase,tag,width=0):
        calls.clear();start=time.monotonic_ns()-1_000_000_000
        args=[str(script),'--role',role,'--phase',phase,'--tag',tag,'--width',str(width),'--pipeline-start-ns',str(start)]
        with patch.object(sys,'argv',args),patch.dict(sys.modules,{'vllm':fake}),patch.object(importlib.metadata,'version',return_value='fixture'),patch.dict(os.environ,{},clear=False):runpy.run_path(str(script),run_name='__main__')
        return json.loads((root/f'matched_v110_{tag}.json').read_text())
    ref=execute('ar','reference','reference');assert len(ref['outputs'])==384
    assert not any(x.startswith('validation-') for x in calls)
    native=execute('native','profile','profile3',3);assert len(calls)==384 and all(x.startswith('profile-') for x in calls)
    screen=execute('screen','profile','screen',8);assert len(calls)==8*384 and len(screen['audits'])==16
    assert all(x.startswith('profile-') for x in calls)
    deploy=execute('native','deploy','deploy3',3);assert len(calls)==16*192 and all(x.startswith('test-') for x in calls)
    assert list(deploy['milestones'])==['1','4','16']
    times=[x['total_pipeline_seconds'] for x in deploy['milestones'].values()]
    assert times==sorted(times) and times[0]>=1
    assert [x['prompts'] for x in deploy['milestones'].values()]==[192,768,3072]
    corrupt=True
    try:execute('native','profile','corrupted',3)
    except AssertionError as error:assert 'Exact AR mismatch' in str(error)
    else:raise AssertionError('Corrupted outputs accepted')
    assert json.loads((root/'matched_v110_corrupted.json').read_text())['status']=='failed'
    print(json.dumps(dict(passed=True,checks=['profile/test/validation isolation','screen8width schedule',
        'all3deployment endpoints count imports/startup via global clock','bad output aborts'],scope='CPU fake engine only'),indent=2))
