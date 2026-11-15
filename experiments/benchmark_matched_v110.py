"""One stage of a measured profiling-plus-inference pipeline; strict output checks."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback

p=argparse.ArgumentParser()
p.add_argument('--role',choices=['ar','native','screen'],required=True)
p.add_argument('--phase',choices=['reference','profile','deploy'],required=True)
p.add_argument('--width',type=int,default=0)
p.add_argument('--order',type=int,choices=[0,1],default=0)
p.add_argument('--tag',required=True)
p.add_argument('--pipeline-start-ns',type=int,default=0)
a=p.parse_args()
assert (a.role=='ar') == (a.phase=='reference')
assert a.role!='native' or a.width in range(1,9)
assert a.phase!='deploy' or (a.role=='native' and a.pipeline_start_ns>0)
r=Path(__file__).resolve().parent
out=r/f'matched_v110_{a.tag}.json';assert not out.exists()
plan=json.loads((r/'matched_v110_plan.json').read_text())
raw=(r/'model_split_v103_manifest.json').read_bytes()
assert hashlib.sha256(raw).hexdigest()==plan['manifest_sha256']
manifest=json.loads(raw)
profile=[x for x in manifest if x['split']=='profile']
test=[x for x in manifest if x['split']=='test']
assert len(profile)==len(test)==192
assert not {x['prompt_sha256'] for x in profile}&{x['prompt_sha256'] for x in test}
tasks=profile+test if a.phase=='reference' else profile if a.phase=='profile' else test
export=json.loads((r/'export_v68_checks.json').read_text())
assert export['passed'] and export['UNTRAINED_HEADS']
os.environ['VLLM_BATCH_INVARIANT']='1'
if a.role=='screen':
    cache=Path('/tmp/aquan-forespark-v63/cache_v105/forced')
    os.environ['FORESPARK_FORE_D']='1';os.environ['FORESPARK_FORE_D_FIXED_WIDTH']='8'
else:
    cache=Path('/tmp/aquan-forespark-v63/cache_v72')/('ar0' if a.role=='ar' else f'native{a.width}')
    os.environ.pop('FORESPARK_FORE_D',None);os.environ.pop('FORESPARK_FORE_D_FIXED_WIDTH',None)
assert cache.exists(), 'Matched cached-start protocol requires existing caches'
os.environ.pop('FORESPARK_FORE_D_LOG',None)
for key,sub in [('VLLM_CACHE_ROOT','vllm'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('TRITON_CACHE_DIR','triton')]:
    path=cache/sub;assert path.exists();os.environ[key]=str(path)
from vllm import LLM,SamplingParams
config=dict(model=export['target_path'],dtype='bfloat16',max_model_len=2048,
    max_num_seqs=64,max_num_batched_tokens=4096,gpu_memory_utilization=.65,
    enable_prefix_caching=False,attention_config={'backend':'TRITON_ATTN'},
    compilation_config={'cudagraph_capture_sizes':[1,2,4,8,16,32,64,128,192,256,384,512,576]})
if a.role!='ar':
    config['speculative_config']=dict(method='dflash',
        model=export['checkpoint'] if a.role=='screen' else export['original_checkpoint'],
        num_speculative_tokens=8 if a.role=='screen' else a.width,
        draft_sample_method='greedy',enable_adaptive_verification=a.role=='screen',
        disable_eagle_block_drop=True,attention_backend='FLASH_ATTN')
if a.role=='screen':config['worker_extension_cls']='forced_policy_v72.ForcedPolicyWorker'
result=dict(status='initializing',role=a.role,phase=a.phase,width=a.width,order=a.order,
    config=config,cache=str(cache),records=[],outputs=[],audits=[],milestones={},
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    plan_sha256=hashlib.sha256((r/'matched_v110_plan.json').read_bytes()).hexdigest(),
    versions={n:importlib.metadata.version(n) for n in ['vllm','torch','speculators']})
def save():out.write_text(json.dumps(result,indent=2)+'\n')
try:
    before=time.perf_counter();llm=LLM(**config);result['initialization_seconds']=time.perf_counter()-before
    tok=llm.get_tokenizer()
    prompts=[tok.apply_chat_template([{'role':'user','content':x['text']}],tokenize=False,
        add_generation_prompt=True,enable_thinking=False) for x in tasks]
    assert all(len(tok.encode(x))+64<=2048 for x in prompts)
    params=SamplingParams(temperature=0.,max_tokens=64)
    expected=None
    if a.role!='ar':
        ref=json.loads((r/'matched_v110_reference.json').read_text())
        assert ref['status']=='completed' and ref['plan_sha256']==result['plan_sha256']
        assert ref['source_sha256']==result['source_sha256'] and ref['versions']==result['versions']
        expected=ref['outputs'][:192] if a.phase=='profile' else ref['outputs'][192:]
        assert len(expected)==len(tasks)

    def evaluate(width,repeat,warmup=False):
        if a.role=='screen':llm.collective_rpc('force_begin',kwargs={'width':width})
        for start in range(0,len(tasks),64):
            before=time.perf_counter()
            generated=llm.generate(prompts[start:start+64],params,use_tqdm=False)
            duration=time.perf_counter()-before
            ids=[list(x.outputs[0].token_ids) for x in generated]
            if expected is None:result['outputs'].extend(ids)
            elif ids!=expected[start:start+64]:
                result['parity_failure']=dict(start=start,width=width,repeat=repeat,
                    actual=ids,expected=expected[start:start+64])
                raise AssertionError('Exact AR mismatch; comparison invalid')
            result['records'].append(dict(start=start,width=width,repeat=repeat,warmup=warmup,
                seconds=duration,tokens=sum(map(len,ids)),parity=True,
                output_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest()))
        if a.role=='screen':result['audits'].append(llm.collective_rpc('force_audit')[0])
        save()

    if a.phase=='reference':evaluate(0,0)
    elif a.phase=='profile':
        widths=list(range(1,9)) if a.role=='screen' else [a.width]
        if a.order:widths.reverse()
        for width in widths:
            evaluate(width,-1,warmup=True)
            evaluate(width,0)
    else:
        # The first test pass is useful inference, not an uncharged warmup.
        for repeat in range(max(plan['horizons_test_passes'])):
            evaluate(a.width,repeat)
            if repeat+1 in plan['horizons_test_passes']:
                result['milestones'][str(repeat+1)]=dict(
                    total_pipeline_seconds=(time.monotonic_ns()-a.pipeline_start_ns)/1e9,
                    cumulative_inference_seconds=sum(x['seconds'] for x in result['records']),
                    output_tokens=sum(x['tokens'] for x in result['records']),
                    prompts=192*(repeat+1))
                save()
    result['status']='completed'
except BaseException as error:
    result.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
finally:
    save()
    if 'llm' in globals():llm.llm_engine.engine_core.shutdown(timeout=10)
