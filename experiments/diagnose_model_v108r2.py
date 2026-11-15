"""Reproduce 8B divergence with log probabilities; never a timing benchmark."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import traceback

p=argparse.ArgumentParser();p.add_argument('--role',choices=['ar','native'],required=True);args=p.parse_args()
r=Path(__file__).resolve().parent
ar=json.loads((r/'model_v102_ar0_round-1.json').read_text())
native=json.loads((r/'model_v102_native3_round0.json').read_text())
assert ar['status']=='completed' and native['status']=='failed'
assert native['parity_failure']['phase']=='warmup' and native['parity_failure']['start']==64
historical=ar if args.role=='ar' else native
os.environ['VLLM_BATCH_INVARIANT']='1'
for key,value in historical['cache_paths'].items():os.environ[key]=value
for key in ['FORESPARK_FORE_D','FORESPARK_FORE_D_FIXED_WIDTH','FORESPARK_FORE_D_LOG']:os.environ.pop(key,None)
assert historical['versions']=={n:importlib.metadata.version(n) for n in historical['versions']}
from vllm import LLM, SamplingParams
out=r/f'model_diagnostic_v108r2_{args.role}.json';assert not out.exists()
manifest=(r/'vocab_v54_manifest.json').read_bytes()
plan=json.loads((r/'model_v102_plan.json').read_text());assert hashlib.sha256(manifest).hexdigest()==plan['manifest_sha256']
tasks=[json.loads(manifest)[i] for i in plan['indices'][:128]]
expected=ar['outputs'][:128] if args.role=='ar' else ar['outputs'][:64]+native['parity_failure']['actual']
focus={}
for i,(actual,reference) in enumerate(zip(native['parity_failure']['actual'],native['parity_failure']['expected'])):
    if actual!=reference:
        pos=next(j for j,(x,y) in enumerate(zip(actual,reference)) if x!=y)
        focus[64+i]=dict(position=pos,ar_token=reference[pos],native_token=actual[pos])
config=dict(historical['config']);config['worker_extension_cls']='tie_audit_v108r2.TieAuditWorker'
result=dict(status='initializing',role=args.role,scope='No-logprob sampler observer; synchronization may affect execution. No speed claim.',
            config=config,outputs=[],focus={},historical_outputs_reproduced=None,
            benchmark_source_sha256=historical['source_sha256'],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
try:
    llm=LLM(**config);tok=llm.get_tokenizer()
    assert llm.collective_rpc('tie_begin')[0]['enabled']
    prompts=[tok.apply_chat_template([{'role':'user','content':x['text']}],tokenize=False,add_generation_prompt=True,enable_thinking=False) for x in tasks]
    params=SamplingParams(temperature=0.,max_tokens=64,logprobs=None)
    for start in [0,64]:
        outputs=llm.generate(prompts[start:start+64],params,use_tqdm=False)
        for i,request in enumerate(outputs):
            completion=request.outputs[0];ids=list(completion.token_ids);result['outputs'].append(ids)
            index=start+i
            if index in focus:
                f=focus[index];pos=f['position'];entry=dict(**f,
                    prefix_matches_historical=ids[:pos]==expected[index][:pos],
                    emitted_token=ids[pos] if pos<len(ids) else None,
                    prompt_text=tasks[index]['text'])
                if pos<len(ids) and completion.logprobs is not None:
                    lp=completion.logprobs[pos]
                    entry['top_logprobs']={str(k):dict(logprob=v.logprob,rank=v.rank,decoded_token=v.decoded_token) for k,v in lp.items()}
                result['focus'][str(index)]=entry
    result['sampler_audit']=llm.collective_rpc('tie_audit')[0]
    result['historical_outputs_reproduced']=result['outputs']==expected
    result['mismatched_historical_request_indices']=[i for i,(a,b) in enumerate(zip(result['outputs'],expected)) if a!=b]
    result['status']='completed'
except BaseException as error:
    result.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
finally:
    out.write_text(json.dumps(result,indent=2)+'\n')
    if 'llm' in globals():llm.llm_engine.engine_core.shutdown(timeout=10)
