"""Independent-prompt confirmation of load-dependent native draft-width effects."""
import os,json,time,hashlib,traceback,argparse,importlib.metadata
from pathlib import Path
os.environ['VLLM_BATCH_INVARIANT']='1'
p=argparse.ArgumentParser();p.add_argument('--batch',type=int,choices=[8,64],required=True);p.add_argument('--width',type=int,choices=list(range(9)),required=True);p.add_argument('--round',type=int,required=True);a=p.parse_args();assert (a.width==0 and a.round==-1) or (a.width in range(1,9) and a.round in range(2))
role='ar' if a.width==0 else 'native'
cache=(Path('/tmp/aquan-forespark-v63/cache_v72')/f'{role}{a.width}') if a.batch==64 else Path('/tmp/aquan-forespark-v63/cache_v77')/f'b8_{role}{a.width}'
for key,sub in [('VLLM_CACHE_ROOT','vllm'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('TRITON_CACHE_DIR','triton')]:
 path=cache/sub;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
r=Path(__file__).resolve().parent;out=r/f'transfer_v78_b{a.batch}_w{a.width}_r{a.round}.json';assert not out.exists()
plan=json.loads((r/'transfer_v78_plan.json').read_text());raw=(r/'transfer_v78_manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==plan['manifest_sha256'];tasks=json.loads(raw);assert len(tasks)==384
export=json.loads((r/'export_v68_checks.json').read_text());assert export['passed']
for key in ['FORESPARK_FORE_D','FORESPARK_FORE_D_FIXED_WIDTH','FORESPARK_FORE_D_LOG']:os.environ.pop(key,None)
from vllm import LLM,SamplingParams
config=dict(model=export['target_path'],dtype='bfloat16',max_model_len=2048,max_num_seqs=a.batch,max_num_batched_tokens=4096,gpu_memory_utilization=.65,enable_prefix_caching=False,attention_config={'backend':'TRITON_ATTN'},compilation_config={'cudagraph_capture_sizes':[1,2,4,8,16,32,64,128,192,256,384,512,576]})
if a.width:config['speculative_config']=dict(method='dflash',model=export['original_checkpoint'],num_speculative_tokens=a.width,draft_sample_method='greedy',enable_adaptive_verification=False,disable_eagle_block_drop=True,attention_backend='FLASH_ATTN')
wall_start=time.perf_counter()
res=dict(status='initializing',batch=a.batch,width=a.width,round=a.round,config=config,cache_paths={key:os.environ[key] for key in ['VLLM_CACHE_ROOT','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR']},source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),plan_sha256=hashlib.sha256((r/'transfer_v78_plan.json').read_bytes()).hexdigest(),manifest_sha256=plan['manifest_sha256'],versions={n:importlib.metadata.version(n) for n in ['vllm','torch','speculators']},outputs=[],records=[])
def save():out.write_text(json.dumps(res,indent=2)+'\n')
try:
 expected=None
 if a.width:
  ref=json.loads((r/f'transfer_v78_b{a.batch}_w0_r-1.json').read_text());other=json.loads((r/f'transfer_v78_b{72-a.batch}_w0_r-1.json').read_text())
  assert ref['status']==other['status']=='completed' and ref['outputs']==other['outputs'],'Cross-load reference mismatch'
  expected=ref['outputs']
 llm=LLM(**config);res['engine_init_seconds']=time.perf_counter()-wall_start;tok=llm.get_tokenizer();prompts=[tok.apply_chat_template([{'role':'user','content':x['text']}],tokenize=False,add_generation_prompt=True,enable_thinking=False) for x in tasks]
 assert all(len(tok.encode(x))+128<=2048 for x in prompts),'No silent filtering'
 params=SamplingParams(temperature=0.,max_tokens=128)
 for repeat in [-2,-1,0,1]:
  for start in range(0,384,a.batch):
   before=time.perf_counter();outputs=llm.generate(prompts[start:start+a.batch],params,use_tqdm=False);duration=time.perf_counter()-before
   ids=[list(x.outputs[0].token_ids) for x in outputs]
   if expected is None and repeat==-2:res['outputs'].extend(ids)
   wanted=(expected if expected is not None else res['outputs'])[start:start+a.batch]
   if ids!=wanted:res['parity_failure']=dict(start=start,repeat=repeat,actual=ids,expected=wanted);raise AssertionError('Exactgreedyparityfailed')
   res['records'].append(dict(repeat=repeat,start=start,domain=tasks[start]['domain'],split=tasks[start]['split'],tokens=sum(map(len,ids)),seconds=duration,parity=True,output_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest()))
  save()
 res['status']='completed';res['wall_seconds_before_shutdown']=time.perf_counter()-wall_start
except BaseException as e:res.update(status='failed',error=repr(e),traceback=traceback.format_exc());raise
finally:
 save()
 if 'llm' in globals():llm.llm_engine.engine_core.shutdown(timeout=10)
