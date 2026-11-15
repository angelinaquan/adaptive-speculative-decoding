"""Pinned8B native correctness and cost smoke test; no proposed method or best-width claim."""
import os,json,time,hashlib,traceback,argparse,importlib.metadata
from pathlib import Path
os.environ['VLLM_BATCH_INVARIANT']='1'
p=argparse.ArgumentParser();p.add_argument('--role',choices=['ar','native'],required=True);p.add_argument('--width',type=int,required=True);p.add_argument('--round',type=int,required=True);a=p.parse_args();assert (a.role=='ar' and a.width==0 and a.round==-1) or (a.role!='ar' and a.width in [3,7] and a.round in [0,1])
cache=Path('/tmp/aquan-forespark-v63/cache_v102')/f'{a.role}{a.width}'
for key,sub in [('VLLM_CACHE_ROOT','vllm'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('TRITON_CACHE_DIR','triton')]:
 path=cache/sub;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
r=Path(__file__).resolve().parent;out=r/f'model_v102_{a.role}{a.width}_round{a.round}.json';assert not out.exists()
plan=json.loads((r/'model_v102_plan.json').read_text())
raw=(r/'vocab_v54_manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==plan['manifest_sha256']
all_tasks=json.loads(raw);tasks=[all_tasks[i] for i in plan['indices']]
export=json.loads((r/'model_v102_paths.json').read_text());assert export['passed'];paths=export['paths'];assert export['metadata_sha256']==plan['metadata_sha256']
os.environ.pop('FORESPARK_FORE_D',None);os.environ.pop('FORESPARK_FORE_D_FIXED_WIDTH',None)
os.environ.pop('FORESPARK_FORE_D_LOG',None)
from vllm import LLM,SamplingParams
config=dict(model=paths['Qwen/Qwen3-8B'],dtype='bfloat16',max_model_len=2048,max_num_seqs=64,max_num_batched_tokens=4096,gpu_memory_utilization=.65,enable_prefix_caching=False,attention_config={'backend':'TRITON_ATTN'},compilation_config={'cudagraph_capture_sizes':[1,2,4,8,16,32,64,128,192,256,384,512,576]})
if a.role!='ar':
 config['speculative_config']=dict(method='dflash',model=paths['RedHatAI/Qwen3-8B-speculator.dflash2'],num_speculative_tokens=a.width,draft_sample_method='greedy',enable_adaptive_verification=False,disable_eagle_block_drop=True,attention_backend='FLASH_ATTN')
res=dict(cache_paths={key:os.environ[key] for key in ['VLLM_CACHE_ROOT','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR']},status='initializing',role=a.role,width=a.width,round=a.round,config=config,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),plan_sha256=hashlib.sha256((r/'model_v102_plan.json').read_bytes()).hexdigest(),versions={n:importlib.metadata.version(n) for n in ['torch','vllm','speculators']},records=[],audits=[],outputs=[])
def save():out.write_text(json.dumps(res,indent=2)+'\n')
try:
 llm=LLM(**config);tok=llm.get_tokenizer()
 prompts=[tok.apply_chat_template([{'role':'user','content':x['text']}],tokenize=False,add_generation_prompt=True,enable_thinking=False) for x in tasks]
 assert all(len(tok.encode(x))+64<=2048 for x in prompts)
 params=SamplingParams(temperature=0.,max_tokens=64)
 expected=None
 if a.role!='ar':
  ref=json.loads((r/'model_v102_ar0_round-1.json').read_text());assert ref['status']=='completed';expected=ref['outputs']
 def evaluate(width,phase,order,repeat):
  for start in range(0,len(tasks),64):
   before=time.perf_counter();outputs=llm.generate(prompts[start:start+64],params,use_tqdm=False);duration=time.perf_counter()-before
   ids=[list(x.outputs[0].token_ids) for x in outputs]
   if expected is None and phase=='warmup' and repeat==0:res['outputs'].extend(ids)
   wanted=expected[start:start+64] if expected is not None else res['outputs'][start:start+64]
   if ids!=wanted:
    res['parity_failure']=dict(width=width,phase=phase,start=start,actual=ids,expected=wanted)
    raise AssertionError('Exact greedy output mismatch; timing claims prohibited')
   res['records'].append(dict(width=width,phase=phase,order=order,repeat=repeat,start=start,domain=tasks[start]['domain'],seconds=duration,tokens=sum(map(len,ids)),parity=True,output_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest()))
  save()
 for repeat in range(2):evaluate(a.width,'warmup',a.round,repeat)
 res['all_widths_parity_passed']=True;save()
 for repeat in range(2):evaluate(a.width,'timed',a.round,repeat)
 res['status']='completed'
except BaseException as e:res.update(status='failed',error=repr(e),traceback=traceback.format_exc());raise
finally:
 save()
 if 'llm' in globals():llm.llm_engine.engine_core.shutdown(timeout=10)
