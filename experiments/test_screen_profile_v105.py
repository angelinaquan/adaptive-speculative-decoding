"""Fake-engine audit of width switching and output checks; no GPU claims."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent


def main():
    switches, emitted_widths = [], []
    inject_mismatch = False

    class LLM:
        def __init__(self, **config):
            self.width = 0
            self.llm_engine = SimpleNamespace(engine_core=SimpleNamespace(shutdown=lambda **kw: None))

        def get_tokenizer(self):
            return SimpleNamespace(apply_chat_template=lambda messages, **kw: messages[0]['content'],
                                   encode=lambda text: [1])

        def collective_rpc(self, method, kwargs=None):
            if method == 'force_begin':
                self.width = kwargs['width']; switches.append(self.width)
                return [dict(width=self.width)]
            assert method == 'force_audit'
            return [dict(width=self.width, target_unchanged=True, logging_disabled=True)]

        def generate(self, prompts, params, **kw):
            emitted_widths.append(self.width)
            return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[int(text),
                       99 if inject_mismatch and self.width else 42])]) for text in prompts]

    fake = ModuleType('vllm'); fake.LLM=LLM; fake.SamplingParams=lambda **kw: kw
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp); script=root/'benchmark_screen_profile_v105.py'
        source=(ROOT/script.name).read_text()
        # Only the scratch directory is substituted; execute the production
        # benchmark body rather than a separate implementation of its loops.
        source=source.replace("Path('/tmp/aquan-forespark-v63/cache_v105')",f'Path({str(root / "cache")!r})')
        script.write_text(source)
        manifest=json.dumps([dict(text=str(i),domain=['alpaca','gsm8k','mbpp'][i//64]) for i in range(192)]).encode()
        (root/'vocab_v54_manifest.json').write_bytes(manifest)
        (root/'screen_profile_v105_plan.json').write_text(json.dumps(dict(indices=list(range(192)),manifest_sha256=hashlib.sha256(manifest).hexdigest())))
        (root/'export_v68_checks.json').write_text(json.dumps(dict(passed=True,UNTRAINED_HEADS=True,
            target_path='fixture-target',checkpoint='fixture-wrapper',original_checkpoint='fixture-drafter')))

        def execute(role):
            with patch.object(sys,'argv',[str(script),'--role',role]), patch.dict(sys.modules,{'vllm':fake}), \
                 patch.object(importlib.metadata,'version',return_value='fixture'), patch.dict(os.environ,{},clear=False):
                runpy.run_path(str(script),run_name='__main__')
            return json.loads((root/f'screen_profile_v105_{role}.json').read_text())

        ar=execute('ar');screen=execute('forced')
        assert ar['status']==screen['status']=='completed'
        assert switches==[w for w in range(1,9) for _ in range(2)]+[w for order in [range(1,9),range(8,0,-1)] for w in order for _ in range(2)]
        assert len(screen['records'])==144 and len(screen['audits'])==48
        for row in screen['records']:
            ids=ar['outputs'][row['start']:row['start']+64]
            assert row['tokens']==128 and row['parity']
            assert row['output_sha256']==hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        # Every RPC-selected width is used for exactly three domain batches.
        assert emitted_widths[18:]==[w for w in switches for _ in range(3)]
        (root/'screen_profile_v105_forced.json').unlink()
        inject_mismatch=True
        try:
            execute('forced')
        except AssertionError as error:
            assert 'Exact greedy output mismatch' in str(error)
        else:
            raise AssertionError('Corrupted output was accepted')
        assert json.loads((root/'screen_profile_v105_forced.json').read_text())['status']=='failed'
        print(json.dumps(dict(passed=True,records_checked=144,width_switches_checked=48,
            checks=['ascending and descending width schedule','two warmups and four timed passes per width',
                    'fresh AR output hashes and token counts','injected output mismatch aborts'],
            scope='CPU fake-engine benchmark harness only; no actual GPU correctness or performance result'),indent=2))


if __name__=='__main__':
    main()
