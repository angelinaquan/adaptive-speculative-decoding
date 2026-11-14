"""Validation selection against all static widths in matched numerical mode."""
import argparse,os,json,time,hashlib
from pathlib import Path
os.environ['VLLM_BATCH_INVARIANT']='1'
p=argparse.ArgumentParser();p.add_argument('--method',required=True,choices=['ar','original','measured']+[f'fixed{i}' for i in range(1,8)]);a=p.parse_args()
import wandb
from vllm import LLM,SamplingParams
r=Path(__file__).resolve().parent;s=Path(os.environ['SERVING_SCRATCH']);outdir=s/'cost-v25';outdir.mkdir(exist_ok=True);out=outdir/(a.method+'.json');assert not out.exists()
paths=json.loads((r/'model_paths.json').read_text());manifest=json.loads((r/'head_calibration_v18_manifest.json').read_text());ref=json.loads((r/'head_calibration_v18_ar.json').read_text());assert ref['status']=='completed'
config=dict(model=paths['Qwen/Qwen3-4B'],dtype='float32',max_model_len=2048,max_num_seqs=32,max_num_batched_tokens=4096,gpu_memory_utilization=.65,enable_prefix_caching=False,attention_config={'backend':'TRITON_ATTN'},compilation_config={'cudagraph_capture_sizes':[1,2,4,8,16,32,64,128,256]})
if a.method!='ar':config['speculative_config']=dict(method='dspark',model=paths['deepseek-ai/dspark_qwen3_4b_block7'],num_speculative_tokens=int(a.method[5:]) if a.method.startswith('fixed') else 7,draft_sample_method='probabilistic',enable_adaptive_verification=a.method in ['original','measured'])
if a.method in ['original','measured']:config['worker_extension_cls']='cost_policy_v24.CostPolicyWorker'
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='cost-v25-static-selection',name=a.method,dir=str(s),config={'engine':config,'batch':32,'scope':'Validation selection only; separate engine order not balanced; 144 reused prompts','batch_invariant':True})
result=dict(status='initializing',method=a.method,run_url=run.url,records=[],config=config,manifest_sha256=hashlib.sha256((r/'head_calibration_v18_manifest.json').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
try:
 llm=LLM(**config);tok=llm.get_tokenizer();prompts=[tok.apply_chat_template([{'role':'user','content':x['text']}],tokenize=False,add_generation_prompt=True,enable_thinking=False) for x in manifest['validation']]
 if a.method in ['original','measured']:
  result['costs']=llm.collective_rpc('cost_control_v24',kwargs={'mode':a.method})[0]
  llm.collective_rpc('calibration_control_v17',kwargs={'learning':False,'slope':1.,'intercepts':[0.]*7,'reset_statistics':True})
 params=SamplingParams(temperature=0.,max_tokens=128)
 for repeat in range(-2,3):
  for start in range(0,len(prompts),32):
   before=time.perf_counter();outputs=llm.generate(prompts[start:start+32],params,use_tqdm=False);duration=time.perf_counter()-before
   ids=[list(o.outputs[0].token_ids) for o in outputs];assert ids==ref['outputs'][start:start+32],('Greedy parity',a.method,repeat,start)
   if repeat>=0:
    row=dict(repeat=repeat,start=start,seconds=duration,tokens=sum(map(len,ids)));result['records'].append(row);run.log(row)
 result['status']='completed';result['throughput']=sum(x['tokens'] for x in result['records'])/sum(x['seconds'] for x in result['records']);run.summary['throughput']=result['throughput']
except BaseException as e:result.update(status='failed',error=repr(e));raise
finally:
 out.write_text(json.dumps(result,indent=2)+'\n');(r/('cost_v25_'+a.method+'.json')).write_text(json.dumps(result,indent=2)+'\n');run.summary['status']=result['status']
 art=wandb.Artifact('cost-v25-'+a.method,'validation-selection')
 for path in [out,Path(__file__),r/'cost_policy_v24.py',r/'head_calibration_v18_manifest.json']:art.add_file(str(path))
 run.log_artifact(art);run.finish()
 if 'llm' in globals():llm.llm_engine.engine_core.shutdown(timeout=10)
