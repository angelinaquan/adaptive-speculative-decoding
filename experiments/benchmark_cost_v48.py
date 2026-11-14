"""Exploratory batch128 sweep on reused prompts; no held-out claim."""
import argparse,os,json,time,hashlib
import importlib.metadata
from pathlib import Path
os.environ['VLLM_BATCH_INVARIANT']='1'
p=argparse.ArgumentParser();p.add_argument('--method',required=True,choices=['ar','original','measured']+[f'fixed{i}' for i in range(1,8)]);a=p.parse_args()
import wandb
from vllm import LLM,SamplingParams
r=Path(__file__).resolve().parent;s=Path(os.environ['SERVING_SCRATCH']);out=r/f'cost_v48_{a.method}.json';assert not out.exists()
plan=json.loads((r/'cost_v26_plan.json').read_text());raw=(r/'cost_v26_manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==plan['manifest_sha256'];tasks=json.loads(raw);assert len(tasks)==384
method=a.method
paths=json.loads((r/'model_paths.json').read_text())
config=dict(model=paths['Qwen/Qwen3-4B'],dtype='float32',max_model_len=2048,max_num_seqs=128,max_num_batched_tokens=4096,gpu_memory_utilization=.65,enable_prefix_caching=False,attention_config={'backend':'TRITON_ATTN'},compilation_config={'cudagraph_capture_sizes':[1,2,4,8,16,32,64,128,256,512,1024]})
if method!='ar':config['speculative_config']=dict(method='dspark',model=paths['deepseek-ai/dspark_qwen3_4b_block7'],num_speculative_tokens=int(method[5:]) if method.startswith('fixed') else 7,draft_sample_method='probabilistic',enable_adaptive_verification=method in ['original','measured'])
if method in ['original','measured']:config['worker_extension_cls']='cost_only_v26.CostOnlyWorker'
metadata=dict(method=method,batch=128,engine=config,manifest_sha256=plan['manifest_sha256'],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cost_source_sha256=hashlib.sha256((r/'cost_only_v26.py').read_bytes()).hexdigest(),scope='Exploratory reused v26 prompts; batch128 and capture1024; cost intervention only, native confidence, no per-step diagnostics',batch_invariant=True,versions={name:importlib.metadata.version(name) for name in ['vllm','torch','wandb']},tf32_override=os.environ.get('NVIDIA_TF32_OVERRIDE'),node=os.environ.get('SLURMD_NODENAME'))
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='cost-v48-batch128-exploration',name=method,dir=str(s),config=metadata)
result=dict(status='initializing',metadata=metadata,run_url=run.url,records=[],outputs=[])
try:
 llm=LLM(**config);tok=llm.get_tokenizer();prompts=[tok.apply_chat_template([{'role':'user','content':x['text']}],tokenize=False,add_generation_prompt=True,enable_thinking=False) for x in tasks]
 assert all(len(tok.encode(p))+128<=2048 for p in prompts),'No silent context filtering'
 if method in ['original','measured']:result['costs']=llm.collective_rpc('cost_control_v26',kwargs={'mode':method})[0]
 params=SamplingParams(temperature=0.,max_tokens=128)
 expected=None
 if method!='ar':
  ref=json.loads((r/'cost_v48_ar.json').read_text());assert ref['status']=='completed' and ref['metadata']['manifest_sha256']==plan['manifest_sha256'];expected=ref['outputs']
 for repeat in range(-2,2):
  for start in range(0,384,128):
   before=time.perf_counter();outputs=llm.generate(prompts[start:start+128],params,use_tqdm=False);duration=time.perf_counter()-before
   ids=[list(o.outputs[0].token_ids) for o in outputs]
   if repeat==-2:result['outputs'].extend(ids)
   if expected is not None:assert ids==expected[start:start+128],('Greedy parity',method,repeat,start)
   else:assert ids==result['outputs'][start:start+128]
   if repeat>=0:
    row=dict(repeat=repeat,start=start,domain=tasks[start]['domain'],seconds=duration,tokens=sum(map(len,ids)));result['records'].append(row);run.log(row)
 result['status']='completed';result['throughput']=sum(x['tokens'] for x in result['records'])/sum(x['seconds'] for x in result['records']);run.summary['throughput']=result['throughput']
except BaseException as e:result.update(status='failed',error=repr(e));raise
finally:
 out.write_text(json.dumps(result,indent=2)+'\n');run.summary['status']=result['status'];art=wandb.Artifact(f'cost-v48-{method}','exploratory-session')
 for path in [out,Path(__file__),r/'cost_only_v26.py',r/'cost_v26_manifest.json',r/'cost_v26_plan.json',r/'cost_v48_plan.json']:art.add_file(str(path))
 run.log_artifact(art);run.finish()
 if 'llm' in globals():llm.llm_engine.engine_core.shutdown(timeout=10)
