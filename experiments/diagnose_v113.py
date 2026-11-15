"""Reproduce v103 first-batch discrepancy without modifying decoding semantics."""
import argparse,json,os,hashlib,traceback
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--role',choices=['ar','native'],required=True);p.add_argument('--mode',choices=['graph','eager'],required=True);a=p.parse_args()
r=Path(__file__).resolve().parent
historical=json.loads((r/f'model_profile_v103_b8_{"ar0_round-1" if a.role=="ar" else "native2_round1"}.json').read_text())
for k,sub in [('VLLM_CACHE_ROOT','vllm'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('TRITON_CACHE_DIR','triton')]:
 path=Path('/tmp/aquan-forespark-v63/cache_v113')/a.role/sub;path.mkdir(parents=True,exist_ok=True);os.environ[k]=str(path)
os.environ['VLLM_BATCH_INVARIANT']='1'
for k in ['FORESPARK_FORE_D','FORESPARK_FORE_D_FIXED_WIDTH','FORESPARK_FORE_D_LOG']:os.environ.pop(k,None)
from vllm import LLM,SamplingParams
config=dict(historical['config']);config['enforce_eager']=False;config['hf_overrides']={'head_dtype':'float32'}
tasks=[x for x in json.loads((r/'model_split_v103_manifest.json').read_text()) if x['split']=='profile'][:8]
ref=json.loads((r/'model_profile_v103_b8_ar0_round-1.json').read_text())['outputs'][:8]
out=r/f'diagnostic_v113_{a.mode}_{a.role}.json';assert not out.exists()
result=dict(status='initializing',config=config,role=a.role,mode=a.mode,records=[],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
try:
 llm=LLM(**config);tok=llm.get_tokenizer();prompts=[tok.apply_chat_template([{'role':'user','content':x['text']}],tokenize=False,add_generation_prompt=True,enable_thinking=False) for x in tasks]
 for repeat in range(5):
  params=SamplingParams(temperature=0,max_tokens=64,logprobs=20 if repeat==4 else None)
  outputs=llm.generate(prompts,params,use_tqdm=False);ids=[list(x.outputs[0].token_ids) for x in outputs]
  entry=dict(repeat=repeat,logprobs=params.logprobs,outputs=ids,mismatch_indices=[i for i,(x,y) in enumerate(zip(ids,ref)) if x!=y])
  if repeat==4:
   comp=outputs[6].outputs[0];entry['focus_text']=comp.text
   entry['focus_logprobs']={str(pos):{str(k):dict(logprob=v.logprob,rank=v.rank,decoded_token=v.decoded_token) for k,v in comp.logprobs[pos].items()} for pos in range(55,min(60,len(comp.token_ids)))}
  result['records'].append(entry);out.write_text(json.dumps(result,indent=2)+'\n')
 result['status']='completed'
except BaseException as e:result.update(status='failed',error=repr(e),traceback=traceback.format_exc());raise
finally:
 out.write_text(json.dumps(result,indent=2)+'\n')
 if 'llm' in globals():llm.llm_engine.engine_core.shutdown(timeout=10)
