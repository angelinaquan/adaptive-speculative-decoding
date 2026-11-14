"""Validation-only batched probe. AR reference must precede speculative modes.

Greedy agreement is a necessary gate, not a stochastic sampler proof.
No held-out performance claims may be made from this probe.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['ar', 'adaptive'] + [f'fixed{k}' for k in range(1, 8)], required=True)
    args = parser.parse_args()
    if args.method == 'ar':
        subprocess.run([sys.executable, str(Path(__file__).with_name('check_serving_sampler_v14.py'))], check=True)
    assert json.loads((Path(os.environ['SERVING_SCRATCH'])/'sampler-check.json').read_text())['passed']
    import torch
    import wandb
    from vllm import LLM, SamplingParams
    root = Path(__file__).resolve().parent
    outdir = Path(os.environ['SERVING_SCRATCH']) / 'probe-results-v15'
    outdir.mkdir(exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    paths = json.loads((root / 'model_paths.json').read_text())
    rows = json.loads((root / 'splits.json').read_text())
    domains = ['alpaca', 'gsm8k', 'humaneval']
    by_domain = [[r for r in rows if r['split'] == 'validation' and r['domain'] == d] for d in domains]
    tasks = [by_domain[i % 3][i // 3] for i in range(32)]
    config = dict(model=paths['Qwen/Qwen3-4B'], dtype='float32', max_model_len=2048,
                  max_num_seqs=32, max_num_batched_tokens=4096,
                  gpu_memory_utilization=.65, enable_prefix_caching=False,
                  compilation_config={'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 32, 64, 128, 256]})
    if args.method != 'ar':
        config['speculative_config'] = dict(method='dspark', model=paths['deepseek-ai/dspark_qwen3_4b_block7'],
            num_speculative_tokens=int(args.method[5:]) if args.method.startswith('fixed') else 7,
            draft_sample_method='probabilistic',
            enable_adaptive_verification=args.method == 'adaptive')
    plan = dict(scope='Validation selection across all fixed proposal widths; not held-out evidence', method=args.method,
                batches=[1, 8, 32], repeats=3, max_tokens=128, config=config,
                sampling='greedy; temperature=0',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                fixed_length_scope='Configured proposal width; shorter widths may change the parallel drafter computation, not just truncate verification',
                prompt_sha256=hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest(),
                correctness_scope='Greedy token equality against same-engine AR at matching batch size; stochastic correctness not established',
                timing='Full synchronous generate wall time, including prefill, excluding initialization and two warmups per batch')
    run = wandb.init(entity='aquan-world-model', project='18657-speculative-pilot',
                     group='serving-v15', name=args.method+'-validation', config=plan,
                     dir=os.environ['SERVING_SCRATCH'])
    print('RUN_URL', run.url, flush=True)
    result = dict(plan=plan, run_url=run.url, checks=[], records=[])
    path = outdir / (args.method+'.json')
    try:
        llm = LLM(**config)
        tokenizer = llm.get_tokenizer()
        prompts = [tokenizer.apply_chat_template([{'role':'user','content':r['text']}],
                   tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in tasks]
        assert all(len(tokenizer.encode(p))+128 <= 2048 for p in prompts), 'Prompt exceeds predeclared context limit'
        params = SamplingParams(temperature=0., max_tokens=128)
        def generate_workload(batch):
            outputs = []
            for offset in range(0, len(prompts), batch):
                outputs.extend(llm.generate(prompts[offset:offset+batch], params, use_tqdm=False))
            return outputs
        reference = None
        if args.method != 'ar':
            reference = json.loads((outdir/'ar.json').read_text())
            assert reference['status'] == 'completed_validation_only'
            assert reference['plan']['prompt_sha256'] == plan['prompt_sha256']
        result['outputs'] = {}
        # Finish all correctness gates before any timed measurements.
        for batch in plan['batches']:
            outputs = generate_workload(batch)
            ids = [list(o.outputs[0].token_ids) for o in outputs]
            result['outputs'][str(batch)] = ids
            if reference is not None:
                expected = reference['outputs'][str(batch)]
                assert len(ids) == len(expected) == len(prompts), 'Missing request outputs'
                mismatches = [i for i, (a,b) in enumerate(zip(ids, expected)) if a != b]
                result['checks'].append(dict(batch=batch, passed=not mismatches, mismatches=mismatches))
                assert not mismatches, f'Greedy mismatch: batch={batch}, requests={mismatches}'
        result['greedy_gate_passed'] = True
        for batch in plan['batches']:
            for _ in range(2):
                generate_workload(batch)
            for repeat in range(plan['repeats']):
                start = time.perf_counter()
                outputs = generate_workload(batch)
                seconds = time.perf_counter()-start
                ids = [list(o.outputs[0].token_ids) for o in outputs]
                assert ids == result['outputs'][str(batch)], 'Non-repeatable greedy outputs'
                tokens = sum(len(x) for x in ids)
                record = dict(batch=batch, repeat=repeat, seconds=seconds, tokens=tokens, tokens_per_second=tokens/seconds)
                result['records'].append(record)
                run.log(record)
                print('TIMED', json.dumps(record), flush=True)
        result['status'] = 'completed_validation_only'
    except BaseException as error:
        result['status'] = 'failed'
        result['error'] = repr(error)
        raise
    finally:
        path.write_text(json.dumps(result, indent=2))
        run.summary['status'] = result['status']
        run.summary['checks'] = result['checks']
        artifact = wandb.Artifact('serving-v15-'+args.method, type='validation-probe')
        artifact.add_file(str(path))
        artifact.add_file(__file__)
        run.log_artifact(artifact)
        run.finish()


if __name__ == '__main__':
    main()
