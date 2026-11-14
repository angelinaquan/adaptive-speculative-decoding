"""Exploratory screen of existing draft-compute optimizations; not a novel method."""
import argparse,os,json,time,hashlib
import importlib.metadata
from pathlib import Path
os.environ['VLLM_BATCH_INVARIANT']='1'
p=argparse.ArgumentParser();p.add_argument('--role',required=True,choices=['ar','native7','prompt1024','recency1024','recency4096']);p.add_argument('--round',required=True,type=int,choices=[-1,0,1]);a=p.parse_args()
import wandb
from vllm import LLM,SamplingParams
r=Path(__file__).resolve().parent;s=Path(os.environ['SERVING_SCRATCH']);out=r/('dynamic_v58_ar-reference.json' if a.role=='ar' else f'dynamic_v58_{a.role}-round{a.round}.json');assert not out.exists();assert (a.role=='ar')==(a.round==-1)
plan=json.loads((r/'dynamic_v58_plan.json').read_text());raw=(r/'oracle_v57_manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==plan['manifest_sha256'];tasks=json.loads(raw);assert len(tasks)==48
method=a.role
from select_context_v55 import select
priors=json.loads((r/'context_v55_priors.json').read_text());assert hashlib.sha256((r/'context_v55_priors.json').read_bytes()).hexdigest()==plan['priors_sha256']
budget=4096 if method.endswith('4096') else 1024
paths=json.loads((r/'model_paths.json').read_text())
config=dict(model=paths['Qwen/Qwen3-4B'],dtype='float32',max_model_len=2048,max_num_seqs=8,max_num_batched_tokens=4096,gpu_memory_utilization=.65,enable_prefix_caching=False,attention_config={'backend':'TRITON_ATTN'},compilation_config={'cudagraph_capture_sizes':[1,2,4,8,16,32,64,128,256,512]})
if method!='ar':
 config['speculative_config']=dict(method='dspark',model=paths['deepseek-ai/dspark_qwen3_4b_block7'],num_speculative_tokens=7,draft_sample_method='greedy',enable_adaptive_verification=False)
 if method!='native7':
  path=r/f'dynamic_v58_init{budget}.json'
  os.environ['DRAFT_ACTIVE_VOCAB']=str(path)
  config['worker_extension_cls']='dynamic_vocab_v58.DynamicVocabularyWorker'
metadata=dict(active_module_sha256=hashlib.sha256((r/'dynamic_vocab_v58.py').read_bytes()).hexdigest(),base_module_sha256=hashlib.sha256((r/'active_vocab_v55.py').read_bytes()).hexdigest(),checks_sha256=hashlib.sha256((r/'dynamic_v58_checks.json').read_bytes()).hexdigest(),priors_sha256=plan['priors_sha256'],selector_sha256=hashlib.sha256((r/'select_context_v55.py').read_bytes()).hexdigest(),role=a.role,method=method,round=a.round,order=plan['orders'][a.round] if a.role!='ar' else ['ar'],engine=config,manifest_sha256=plan['manifest_sha256'],plan_sha256=hashlib.sha256((r/'dynamic_v58_plan.json').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope=plan['scope'],batch_invariant=True,versions={name:importlib.metadata.version(name) for name in ['vllm','torch','wandb']},tf32_override=os.environ.get('NVIDIA_TF32_OVERRIDE'),node=os.environ.get('SLURMD_NODENAME'))
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='dynamic-v58-recency-baseline',name=f'{a.role}-round{a.round}',dir=str(s),config=metadata)
result=dict(status='initializing',metadata=metadata,run_url=run.url,records=[],outputs=[],vocabulary_updates=[])
try:
 llm=LLM(**config)
 if method not in ['ar','native7']:result['active_vocab']=llm.collective_rpc('active_vocab_metadata')[0]
 tok=llm.get_tokenizer();prompts=[tok.apply_chat_template([{'role':'user','content':x['text']}],tokenize=False,add_generation_prompt=True,enable_thinking=False) for x in tasks]
 assert all(len(tok.encode(p))+128<=2048 for p in prompts),'No silent context filtering'
 prompt_ids=[tok.encode(p) for p in prompts]
 params=SamplingParams(temperature=0.,max_tokens=128)
 expected=None
 if a.role!='ar':
  ref=json.loads((r/'dynamic_v58_ar-reference.json').read_text());assert ref['status']=='completed' and ref['metadata']['manifest_sha256']==plan['manifest_sha256'];expected=ref['outputs']
 for repeat in range(-2,2):
  for start in range(0,48,8):
   before=time.perf_counter()
   if method not in ['ar','native7']:
    selected,stats=select(prompt_ids[start:start+8],priors['frequency'],budget,use_context=True)
    update=llm.collective_rpc('reset_dynamic',kwargs={'token_ids':selected,'enabled':method.startswith('recency')})[0]
    assert update['ids_sha256']==hashlib.sha256(json.dumps(selected).encode()).hexdigest()
    result['vocabulary_updates'].append(dict(repeat=repeat,start=start,token_ids=selected,stats=stats,**update))
   outputs=llm.generate(prompts[start:start+8],params,use_tqdm=False);duration=time.perf_counter()-before
   if method not in ['ar','native7']:
    result['vocabulary_updates'][-1]['dynamic_audit']=llm.collective_rpc('dynamic_audit',kwargs={'check':repeat<0})[0]
   ids=[list(o.outputs[0].token_ids) for o in outputs]
   if repeat==-2:result['outputs'].extend(ids)
   wanted=expected[start:start+8] if expected is not None else result['outputs'][start:start+8]
   if ids!=wanted:
    result['parity_failure']=dict(repeat=repeat,start=start,actual=ids,expected=wanted)
    raise AssertionError(('Greedy parity',a.role,a.round,repeat,start))
   if repeat>=0:
    row=dict(repeat=repeat,start=start,domain=tasks[start]['domain'],seconds=duration,tokens=sum(map(len,ids)));result['records'].append(row);run.log(row)
 result['status']='completed';result['throughput']=sum(x['tokens'] for x in result['records'])/sum(x['seconds'] for x in result['records']);run.summary['throughput']=result['throughput']
except BaseException as e:result.update(status='failed',error=repr(e));raise
finally:
 out.write_text(json.dumps(result,indent=2)+'\n');run.summary['status']=result['status'];art=wandb.Artifact(f'dynamic-v58-{a.role}-round{a.round}','exploratory-session')
 for path in [out,Path(__file__),r/'oracle_v57_manifest.json',r/'dynamic_v58_plan.json',r/'dynamic_vocab_v58.py']+[r/'context_v55_priors.json',r/'select_context_v55.py',r/'active_vocab_v55.py',r/'dynamic_v58_checks.json',r/'check_dynamic_v58.py']:art.add_file(str(path))
 run.log_artifact(art);run.finish()
 if 'llm' in globals():llm.llm_engine.engine_core.shutdown(timeout=10)
