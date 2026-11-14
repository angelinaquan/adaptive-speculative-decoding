"""Fresh fixed-prompt comparison with balanced engine order across three rounds."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=['ar','fixed','adaptive'], required=True)
    parser.add_argument('--batch', type=int, choices=[1,8,32], required=True)
    parser.add_argument('--round', type=int, choices=[0,1,2], required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    scratch = Path(os.environ['SERVING_SCRATCH'])
    outdir = scratch/'heldout-v16'
    outdir.mkdir(exist_ok=True)
    plan = json.loads((root/'serving_v16_plan.json').read_text())
    manifest_bytes = (root/'serving_v16_manifest.json').read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == plan['manifest_sha256']
    tasks = json.loads(manifest_bytes)
    assert len(tasks) == 96 and len({t['prompt_sha256'] for t in tasks}) == 96
    assert json.loads((scratch/'sampler-check.json').read_text())['passed']
    selection_path = scratch/'serving_v15_summary.json'
    selection = json.loads(selection_path.read_text())
    fixed = selection['batches'][str(args.batch)]['selected_fixed']
    method = fixed if args.role == 'fixed' else args.role
    import torch
    import vllm
    import wandb
    from vllm import LLM, SamplingParams
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    paths = json.loads((root/'model_paths.json').read_text())
    config = dict(model=paths['Qwen/Qwen3-4B'], dtype='float32', max_model_len=2048,
                  max_num_seqs=32, max_num_batched_tokens=4096, gpu_memory_utilization=.65,
                  enable_prefix_caching=False,
                  compilation_config={'cudagraph_capture_sizes':[1,2,4,8,16,32,64,128,256]})
    if method != 'ar':
        config['speculative_config'] = dict(method='dspark', model=paths['deepseek-ai/dspark_qwen3_4b_block7'],
            num_speculative_tokens=int(method[5:]) if method.startswith('fixed') else 7,
            draft_sample_method='probabilistic', enable_adaptive_verification=method=='adaptive')
    orders = [['ar','fixed','adaptive'], ['fixed','adaptive','ar'], ['adaptive','ar','fixed']]
    metadata = dict(role=args.role, method=method, batch=args.batch, round=args.round,
        method_order=orders[args.round], slot=orders[args.round].index(args.role),
        manifest_sha256=plan['manifest_sha256'], selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), config=config,
        versions={'torch':torch.__version__,'vllm':vllm.__version__},
        scope='Held-out greedy throughput; published adaptive scheduling, not online calibration novelty',
        timing='Full synchronous generate call per batch, including prefill; initial correctness pass and one full warmup excluded',
        primary_batch_size=8, timing_repeats=2)
    run = wandb.init(entity='aquan-world-model', project='18657-speculative-pilot', group='serving-v16-heldout',
        name=f'{args.role}-b{args.batch}-round{args.round}', config=metadata, dir=str(scratch))
    print('RUN_URL', run.url, flush=True)
    result = dict(metadata=metadata, run_url=run.url, status='initializing', records=[], checks=[])
    output_path = outdir/f'{args.role}-b{args.batch}-round{args.round}.json'
    assert not output_path.exists(), 'Do not overwrite an existing held-out session'
    reference_path = outdir/f'ar-b{args.batch}-round0.json'
    try:
        llm = LLM(**config)
        tokenizer = llm.get_tokenizer()
        prompts = [tokenizer.apply_chat_template([{'role':'user','content':t['text']}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False) for t in tasks]
        assert all(len(tokenizer.encode(p))+128 <= 2048 for p in prompts), 'Context limit; no silent prompt dropping'
        params = SamplingParams(temperature=0., max_tokens=128)
        groups = [list(range(i,i+args.batch)) for i in range(0,len(prompts),args.batch)]
        def generate(indices):
            return llm.generate([prompts[i] for i in indices], params, use_tqdm=False)
        expected = None
        if not (args.role == 'ar' and args.round == 0):
            reference = json.loads(reference_path.read_text())
            assert reference['status'] == 'completed' and reference['metadata']['manifest_sha256'] == plan['manifest_sha256']
            expected = reference['outputs']
        outputs = []
        for indices in groups:
            outputs.extend([list(o.outputs[0].token_ids) for o in generate(indices)])
        assert len(outputs) == len(tasks)
        result['outputs'] = outputs
        if expected is not None:
            mismatches = [i for i,(a,b) in enumerate(zip(outputs,expected)) if a != b]
            result['checks'].append(dict(name='AR_token_equality', passed=not mismatches, mismatches=mismatches))
            assert not mismatches, f'AR token mismatch at prompt indices {mismatches}'
        for indices in groups:
            generate(indices)
        for repeat in range(2):
            # Rotate request batches identically for every method within a round.
            offset = (args.round*3+repeat) % len(groups)
            order = groups[offset:]+groups[:offset]
            for slot, indices in enumerate(order):
                start = time.perf_counter()
                generated = generate(indices)
                elapsed = time.perf_counter()-start
                ids = [list(o.outputs[0].token_ids) for o in generated]
                assert ids == [outputs[i] for i in indices], 'Repeated greedy output mismatch'
                tokens = sum(map(len,ids))
                row = dict(repeat=repeat, group=indices[0]//args.batch, indices=indices,
                    slot=slot, seconds=elapsed, tokens=tokens, tokens_per_second=tokens/elapsed)
                result['records'].append(row)
                run.log({k:v for k,v in row.items() if k!='indices'})
            print('TIMED_REPEAT', args.role, args.batch, args.round, repeat, flush=True)
        result['status'] = 'completed'
    except BaseException as error:
        result['status'] = 'failed'
        result['error'] = repr(error)
        raise
    finally:
        output_path.write_text(json.dumps(result,indent=2)+'\n')
        run.summary['status'] = result['status']
        run.summary['checks'] = result['checks']
        artifact = wandb.Artifact(f'serving-v16-{args.role}-b{args.batch}-round{args.round}', type='heldout-evaluation')
        for path in [output_path, Path(__file__), root/'serving_v16_plan.json', root/'serving_v16_manifest.json', selection_path]:
            artifact.add_file(str(path))
        run.log_artifact(artifact)
        run.finish()


if __name__ == '__main__':
    main()
