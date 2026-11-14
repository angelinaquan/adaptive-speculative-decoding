"""Exploratory batch64 sweep on reused prompts; no held-out claim."""
import argparse,os,json,time,hashlib
import importlib.metadata
from pathlib import Path
os.environ['VLLM_BATCH_INVARIANT']='1'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
a=argparse.Namespace(method='fixed3')
import wandb
from vllm import LLM,SamplingParams
r=Path(__file__).resolve().parent;s=Path(os.environ['SERVING_SCRATCH']);out=r/'sampling_v47_results.json';assert not out.exists()
plan=json.loads((r/'cost_v26_plan.json').read_text());sampling_plan=json.loads((r/'stochastic_v42_plan.json').read_text());gate=json.loads((r/'joint_sampler_v41.json').read_text());assert gate['passed'];raw=(r/'cost_v26_manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==plan['manifest_sha256'];tasks=json.loads(raw);assert len(tasks)==384
method=a.method
paths=json.loads((r/'model_paths.json').read_text())
config=dict(async_scheduling=False,model=paths['Qwen/Qwen3-4B'],dtype='float32',max_model_len=2048,max_num_seqs=64,max_num_batched_tokens=4096,gpu_memory_utilization=.65,enable_prefix_caching=False,attention_config={'backend':'TRITON_ATTN'},compilation_config={'cudagraph_capture_sizes':[1,2,4,8,16,32,64,128,256,512]})
if method!='ar':config['speculative_config']=dict(method='dspark',model=paths['deepseek-ai/dspark_qwen3_4b_block7'],num_speculative_tokens=int(method[5:]) if method.startswith('fixed') else 7,draft_sample_method='probabilistic',enable_adaptive_verification=method in ['original','measured'])
config['worker_extension_cls']='sampling_trace_v47.SamplingTraceWorker'
metadata=dict(method=method,batch=64,engine=config,manifest_sha256=plan['manifest_sha256'],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cost_source_sha256=hashlib.sha256((r/'cost_only_v26.py').read_bytes()).hexdigest(),scope='Reproduce seeded stochastic mismatch with4uninstrumented and4instrumented repeats. Synchronous scheduling and in-process engine; no timing evidence.',sampling_plan=sampling_plan,batch_invariant=True,versions={name:importlib.metadata.version(name) for name in ['vllm','torch','wandb']},tf32_override=os.environ.get('NVIDIA_TF32_OVERRIDE'),node=os.environ.get('SLURMD_NODENAME'))
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='sampling-v47-diagnostic',name='fixed3-seeded-repeatability',dir=str(s),config=metadata)
result=dict(status='initializing',metadata=metadata,run_url=run.url,records=[],outputs=[])
try:
 llm=LLM(**config);tok=llm.get_tokenizer();prompts=[tok.apply_chat_template([{'role':'user','content':x['text']}],tokenize=False,add_generation_prompt=True,enable_thinking=False) for x in tasks]
 assert all(len(tok.encode(p))+128<=2048 for p in prompts),'No silent context filtering'
 if method in ['original','measured']:result['costs']=llm.collective_rpc('cost_control_v26',kwargs={'mode':method})[0]
 # First retain the strict greedy model/cache gate used by earlier studies.
 ref=json.loads((r/'cost_v29_ar.json').read_text());assert ref['status']=='completed'
 for start in range(0,384,64):
  actual=llm.generate(prompts[start:start+64],SamplingParams(temperature=0.,max_tokens=128),use_tqdm=False)
  ids=[list(o.outputs[0].token_ids) for o in actual]
  if ids!=ref['outputs'][start:start+64]:
   result['parity_failure']=dict(start=start,actual=ids,expected=ref['outputs'][start:start+64]);raise AssertionError('Greedy gate')
 result['greedy_gate_passed']=True
 # Distribution diagnostic, separate from all timed measurements. Audit seeds
 # differ from timing seeds. Same prompts/seeds for every method, but outputs
 # across different speculative policies need NOT match token-for-token.
 result['sampling_audit']=[]
 for prompt_index in [0,128,256]:
  audit_outputs=[]
  for offset in range(0,512,64):
   parameters=[SamplingParams(temperature=1.,top_p=1.,max_tokens=8,seed=4200000+prompt_index*1000+j) for j in range(offset,offset+64)]
   generated=llm.generate([prompts[prompt_index]]*64,parameters,use_tqdm=False)
   audit_outputs.extend([list(o.outputs[0].token_ids) for o in generated])
  result['sampling_audit'].append(dict(prompt_index=prompt_index,domain=tasks[prompt_index]['domain'],outputs=audit_outputs))
 params=[SamplingParams(temperature=1.,top_p=1.,max_tokens=128,seed=421000000+i) for i in range(384)]
 result['runtime']=llm.collective_rpc('sampling_trace_control_v47',kwargs={'enabled':False})[0]['runtime']
 result['passes']=[]
 for repeat in range(8):
  instrumented=repeat>=4
  llm.collective_rpc('sampling_trace_control_v47',kwargs={'enabled':instrumented})
  ids_all=[]
  for start in range(0,384,64):
   outputs=llm.generate(prompts[start:start+64],params[start:start+64],use_tqdm=False)
   ids_all.extend([list(o.outputs[0].token_ids) for o in outputs])
  trace=llm.collective_rpc('sampling_trace_control_v47',kwargs={'enabled':False})[0]['trace']
  if instrumented:assert trace, 'Sampler trace hook did not observe the watched seed'
  result['passes'].append(dict(repeat=repeat,instrumented=instrumented,outputs=ids_all,trace=trace))
 result['status']='completed'
except BaseException as e:result.update(status='failed',error=repr(e));raise
finally:
 out.write_text(json.dumps(result,indent=2)+'\n');run.summary['status']=result['status'];art=wandb.Artifact('sampling-v47','correctness-diagnostic')
 for path in [out,Path(__file__),r/'sampling_trace_v47.py',r/'cost_only_v26.py',r/'cost_v26_manifest.json',r/'cost_v26_plan.json',r/'stochastic_v42_plan.json',r/'joint_sampler_v41.json']:art.add_file(str(path))
 run.log_artifact(art);run.finish()
 if 'llm' in globals():llm.llm_engine.engine_core.shutdown(timeout=10)
