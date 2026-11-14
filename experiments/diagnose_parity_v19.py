"""Investigate v16 parity failure; diagnostic data, not an efficacy benchmark."""
import argparse,json,os,hashlib,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--method',choices=['ar','fixed3'],required=True);a=p.parse_args()
import torch,wandb,vllm
from vllm import LLM,SamplingParams
r=Path(__file__).resolve().parent;s=Path(os.environ['SERVING_SCRATCH'])
paths=json.loads((r/'model_paths.json').read_text())
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
config=dict(model=paths['Qwen/Qwen3-4B'],dtype='float32',max_model_len=2048,max_num_seqs=32,max_num_batched_tokens=4096,gpu_memory_utilization=.65,enable_prefix_caching=False,compilation_config={'cudagraph_capture_sizes':[1,2,4,8,16,32,64,128,256]})
if a.method=='fixed3':config['speculative_config']=dict(method='dspark',model=paths['deepseek-ai/dspark_qwen3_4b_block7'],num_speculative_tokens=3,draft_sample_method='probabilistic',enable_adaptive_verification=False)
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='parity-diagnostic-v19',name=a.method,dir=str(s),config=config)
result=dict(status='initializing',method=a.method,run_url=run.url,config=config,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),records=[])
out=r/f'parity_diagnostic_v19_{a.method}.json'
assert not out.exists()
try:
 llm=LLM(**config);tok=llm.get_tokenizer()
 tasks=json.loads((r/'serving_v16_manifest.json').read_text())
 prompts=[tok.apply_chat_template([{'role':'user','content':t['text']}],tokenize=False,add_generation_prompt=True,enable_thinking=False) for t in tasks]
 # Reproduce full ordered correctness pass; repeat without changing the workload.
 for repeat in range(2):
  for start in range(0,96,8):
   outputs=llm.generate(prompts[start:start+8],SamplingParams(temperature=0.,max_tokens=128,logprobs=5),use_tqdm=False)
   for offset,o in enumerate(outputs):
    q=o.outputs[0]
    record=dict(index=start+offset,repeat=repeat,ids=list(q.token_ids))
    if start+offset==88:
     record['text']=q.text
     record['logprobs']=[{str(k):dict(logprob=v.logprob,rank=v.rank,decoded_token=v.decoded_token) for k,v in step.items()} for step in q.logprobs]
    result['records'].append(record)
 result['status']='completed'
except BaseException as e:result.update(status='failed',error=repr(e));raise
finally:
 out.write_text(json.dumps(result,indent=2)+'\n');run.summary['status']=result['status']
 art=wandb.Artifact('parity-diagnostic-v19-'+a.method,'correctness-diagnostic');art.add_file(str(out));art.add_file(str(Path(__file__)));run.log_artifact(art);run.finish()
