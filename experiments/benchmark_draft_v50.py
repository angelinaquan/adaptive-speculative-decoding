"""Exploratory screen of existing draft-compute optimizations; not a novel method."""
import argparse,os,json,time,hashlib
import importlib.metadata
from pathlib import Path
os.environ['VLLM_BATCH_INVARIANT']='1'
p=argparse.ArgumentParser();p.add_argument('--role',required=True,choices=['ar', 'fixed3_prob', 'fixed7_prob', 'adaptive_prob', 'fixed3_greedy', 'fixed7_greedy', 'adaptive_greedy', 'adaptive_top64', 'adaptive_top256']);p.add_argument('--round',required=True,type=int,choices=[-1,0,1]);a=p.parse_args()
import wandb
from vllm import LLM,SamplingParams
r=Path(__file__).resolve().parent;s=Path(os.environ['SERVING_SCRATCH']);out=r/('draft_v50_ar-reference.json' if a.role=='ar' else f'draft_v50_{a.role}-round{a.round}.json');assert not out.exists();assert (a.role=='ar')==(a.round==-1)
plan=json.loads((r/'draft_v50_plan.json').read_text());raw=(r/'draft_v50_manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==plan['manifest_sha256'];tasks=json.loads(raw);assert len(tasks)==192
method=a.role
paths=json.loads((r/'model_paths.json').read_text())
config=dict(model=paths['Qwen/Qwen3-4B'],dtype='float32',max_model_len=2048,max_num_seqs=64,max_num_batched_tokens=4096,gpu_memory_utilization=.65,enable_prefix_caching=False,attention_config={'backend':'TRITON_ATTN'},compilation_config={'cudagraph_capture_sizes':[1,2,4,8,16,32,64,128,256,512]})
if method!='ar':
 adaptive=method.startswith('adaptive')
 width=7 if adaptive else int(method[5])
 spec=dict(method='dspark',model=paths['deepseek-ai/dspark_qwen3_4b_block7'],num_speculative_tokens=width,draft_sample_method='probabilistic' if method.endswith('_prob') else 'greedy',enable_adaptive_verification=adaptive)
 if '_top' in method:spec['dspark_draft_topk']=int(method.split('_top')[1])
 config['speculative_config']=spec
metadata=dict(role=a.role,method=method,round=a.round,order=plan['orders'][a.round] if a.role!='ar' else ['ar'],engine=config,manifest_sha256=plan['manifest_sha256'],plan_sha256=hashlib.sha256((r/'draft_v50_plan.json').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope=plan['scope'],batch_invariant=True,versions={name:importlib.metadata.version(name) for name in ['vllm','torch','wandb']},tf32_override=os.environ.get('NVIDIA_TF32_OVERRIDE'),node=os.environ.get('SLURMD_NODENAME'))
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='draft-v50-headroom',name=f'{a.role}-round{a.round}',dir=str(s),config=metadata)
result=dict(status='initializing',metadata=metadata,run_url=run.url,records=[],outputs=[])
try:
 llm=LLM(**config);tok=llm.get_tokenizer();prompts=[tok.apply_chat_template([{'role':'user','content':x['text']}],tokenize=False,add_generation_prompt=True,enable_thinking=False) for x in tasks]
 assert all(len(tok.encode(p))+128<=2048 for p in prompts),'No silent context filtering'
 params=SamplingParams(temperature=0.,max_tokens=128)
 expected=None
 if a.role!='ar':
  ref=json.loads((r/'draft_v50_ar-reference.json').read_text());assert ref['status']=='completed' and ref['metadata']['manifest_sha256']==plan['manifest_sha256'];expected=ref['outputs']
 for repeat in range(-2,2):
  for start in range(0,192,64):
   before=time.perf_counter();outputs=llm.generate(prompts[start:start+64],params,use_tqdm=False);duration=time.perf_counter()-before
   ids=[list(o.outputs[0].token_ids) for o in outputs]
   if repeat==-2:result['outputs'].extend(ids)
   wanted=expected[start:start+64] if expected is not None else result['outputs'][start:start+64]
   if ids!=wanted:
    result['parity_failure']=dict(repeat=repeat,start=start,actual=ids,expected=wanted)
    raise AssertionError(('Greedy parity',a.role,a.round,repeat,start))
   if repeat>=0:
    row=dict(repeat=repeat,start=start,domain=tasks[start]['domain'],seconds=duration,tokens=sum(map(len,ids)));result['records'].append(row);run.log(row)
 result['status']='completed';result['throughput']=sum(x['tokens'] for x in result['records'])/sum(x['seconds'] for x in result['records']);run.summary['throughput']=result['throughput']
except BaseException as e:result.update(status='failed',error=repr(e));raise
finally:
 out.write_text(json.dumps(result,indent=2)+'\n');run.summary['status']=result['status'];art=wandb.Artifact(f'draft-v50-{a.role}-round{a.round}','exploratory-session')
 for path in [out,Path(__file__),r/'draft_v50_manifest.json',r/'draft_v50_plan.json']:art.add_file(str(path))
 run.log_artifact(art);run.finish()
 if 'llm' in globals():llm.llm_engine.engine_core.shutdown(timeout=10)
