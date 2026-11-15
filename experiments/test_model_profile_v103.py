"""CPU integration checks of harness bookkeeping, not GPU/decoding correctness."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
from types import SimpleNamespace, ModuleType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
calls = []
fail_width = None


class FakeLLM:
    def __init__(self, **config):
        self.config = config
        self.llm_engine = SimpleNamespace(engine_core=SimpleNamespace(shutdown=lambda **kw: None))

    def get_tokenizer(self):
        return SimpleNamespace(apply_chat_template=lambda messages, **kw: messages[0]['content'],
                               encode=lambda text: [1, 2])

    def generate(self, prompts, params, **kw):
        width = self.config.get('speculative_config', {}).get('num_speculative_tokens', 0)
        calls.extend(prompts)
        assert all(text.startswith('profile-') for text in prompts), 'Held-out text reached model'
        return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[int(text.split('-')[-1]),
                    999 if width == fail_width else 42])]) for text in prompts]


def main():
    fake = ModuleType('vllm')
    fake.LLM = FakeLLM
    fake.SamplingParams = lambda **kw: kw
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        script = root / 'benchmark_model_profile_v103.py'
        shutil.copy2(ROOT / script.name, script)
        rows = [dict(text=f'profile-{i}', split='profile', domain=['alpaca','gsm8k','mbpp'][i//64]) for i in range(192)]
        rows += [dict(text='DO NOT USE', split=s, domain='mbpp') for s in ['validation','test']]
        raw = json.dumps(rows).encode()
        (root/'model_split_v103_manifest.json').write_bytes(raw)
        (root/'model_split_v103_plan.json').write_text(json.dumps(dict(manifest_sha256=hashlib.sha256(raw).hexdigest())))
        (root/'model_v102_summary.json').write_text('{"passed":true}')
        metadata = b'{"fixture":true}'
        (root/'model_v102_metadata.json').write_bytes(metadata)
        (root/'model_v102_paths.json').write_text(json.dumps(dict(passed=True,
            metadata_sha256=hashlib.sha256(metadata).hexdigest(),
            paths={'Qwen/Qwen3-8B':'fake-target', 'RedHatAI/Qwen3-8B-speculator.dflash2':'fake-drafter'})))

        def execute(batch, role, width, order):
            argv = [str(script),'--batch',str(batch),'--role',role,'--width',str(width),'--round',str(order)]
            with patch.object(sys, 'argv', argv), patch.dict(sys.modules, {'vllm':fake}), \
                 patch.object(importlib.metadata, 'version', return_value='fixture'), \
                 patch.dict(os.environ, {'MODEL_PROFILE_V103_CACHE':str(root/'cache')}, clear=False):
                runpy.run_path(str(script), run_name='__main__')
            return json.loads((root/f'model_profile_v103_b{batch}_{role}{width}_round{order}.json').read_text())

        count = 0
        for batch, role, width, order in [(8,'ar',0,-1),(64,'ar',0,-1)]+[(b,'native',w,o) for b in [8,64] for w in [1,7] for o in [0,1]]:
            result = execute(batch, role, width, order)
            assert result['status']=='completed'
            assert len(result['records'])==4*192//batch
            assert all(x['parity'] and x['tokens']==2*batch for x in result['records'])
            for record in result['records']:
                ids = [[i, 42] for i in range(record['start'], record['start']+batch)]
                assert record['output_sha256']==hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            assert len(result['outputs'])==(192 if role=='ar' else 0)
            count += 1
        assert len(calls)==count*4*192

        global fail_width
        fail_width = 2
        try:
            execute(64,'native',2,0)
        except AssertionError as error:
            assert 'Exact greedy output mismatch' in str(error)
        else:
            raise AssertionError('Injected decoding mismatch was not rejected')
        failed=json.loads((root/'model_profile_v103_b64_native2_round0.json').read_text())
        assert failed['status']=='failed' and 'parity_failure' in failed
        fail_width = None
        refpath=root/'model_profile_v103_b8_ar0_round-1.json'
        ref=json.loads(refpath.read_text());ref['plan_sha256']='stale';refpath.write_text(json.dumps(ref))
        try:
            execute(64,'native',3,0)
        except AssertionError:
            pass
        else:
            raise AssertionError('Stale reference accepted')
        assert json.loads((root/'model_profile_v103_b64_native3_round0.json').read_text())['status']=='failed'
        print(json.dumps(dict(passed=True, completed_mock_sessions=count,
            checks=['held-out prompts never passed to engine','AR batch consistency bookkeeping',
                    'warmup and timing record counts','output hashes and token counts',
                    'injected mismatch aborts','stale reference rejected'],
            scope='CPU fake-engine harness test only; real GPU parity remains untested'),indent=2))


if __name__=='__main__':
    main()
