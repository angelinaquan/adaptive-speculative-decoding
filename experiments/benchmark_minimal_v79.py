"""One warmup and one measured profiling pass; no test feedback."""
import os,json,hashlib,time,argparse,traceback
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--panel',type=int,choices=range(3),required=True);p.add_argument('--width',type=int,choices=range(1,9),required=True);a=p.parse_args();r=Path(__file__).resolve().parent;out=r/f'minimal_v79_p{a.panel}_w{a.width}.json';assert not out.exists()
os.environ['VLLM_BATCH_INVARIANT']='1'
cache=Path('/tmp/aquan-forespark-v63/cache_v72')/f'native{a.width}'
for key,sub in [('VLLM_CACHE_ROOT','vllm'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('TRITON_CACHE_DIR','triton')]:os.environ[key]=str(cache/sub)
for key in ['FORESPARK_FORE_D','FORESPARK_FORE_D_FIXED_WIDTH','FORESPARK_FORE_D_LOG']:os.environ.pop(key,None)
plan=json.loads((r/'minimal_v79_plan.json').read_text());source=json.loads((r/'profile_v77_manifest.json').read_text());assert hashlib.sha256((r/'profile_v77_manifest.json').read_bytes()).hexdigest()==plan['source_manifest_sha256']
indices=plan['panels'][a.panel];tasks=[source[i] for i in indices];assert len(tasks)==192 and all(x['split']=='profile' for x in tasks)
ref=json.loads((r/'profile_v77_b64_w0_r-1.json').read_text());assert ref['status']=='completed';expected=[ref['outputs'][i] for i in indices]
export=json.loads((r/'export_v68_checks.json').read_text());assert export['passed']
config=dict(model=export['target_path'],dtype='bfloat16',max_model_len=2048,max_num_seqs=64,max_num_batched_tokens=4096,gpu_memory_utilization=.65,enable_prefix_caching=False,attention_config={'backend':'TRITON_ATTN'},compilation_config={'cudagraph_capture_sizes':[1,2,4,8,16,32,64,128,192,256,384,512,576]},speculative_config=dict(method='dflash',model=export['original_checkpoint'],num_speculative_tokens=a.width,draft_sample_method='greedy',enable_adaptive_verification=False,disable_eagle_block_drop=True,attention_backend='FLASH_ATTN'))
res={'status':'initializing','panel':a.panel,'width':a.width,'records':[],'config':config,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'plan_sha256':hashlib.sha256((r/'minimal_v79_plan.json').read_bytes()).hexdigest()};start=time.perf_counter()
try:
 from vllm import LLM,SamplingParams
 llm=LLM(**config);res['import_and_engine_init_seconds']=time.perf_counter()-start;tok=llm.get_tokenizer();prompts=[tok.apply_chat_template([{'role':'user','content':x['text']}],tokenize=False,add_generation_prompt=True,enable_thinking=False) for x in tasks];params=SamplingParams(temperature=0.,max_tokens=64)
 for rep in [-1,0]:
  for j in range(0,192,64):
   t=time.perf_counter();outputs=llm.generate(prompts[j:j+64],params,use_tqdm=False);seconds=time.perf_counter()-t;ids=[list(x.outputs[0].token_ids) for x in outputs];assert ids==expected[j:j+64],'Exact profiling output mismatch'
   res['records'].append({'repeat':rep,'start':j,'seconds':seconds,'tokens':sum(map(len,ids)),'parity':True,'output_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest()})
 res['status']='completed'
except BaseException as e:res.update(status='failed',error=repr(e),traceback=traceback.format_exc());raise
finally:
 res['wall_seconds_before_shutdown']=time.perf_counter()-start;out.write_text(json.dumps(res,indent=2)+'\n')
 if 'llm' in globals():llm.llm_engine.engine_core.shutdown(timeout=10)
