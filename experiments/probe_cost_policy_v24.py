"""Exploratory matched-engine cost-model ablation; no held-out claim."""
import os,json,time,hashlib
from pathlib import Path
os.environ['VLLM_BATCH_INVARIANT']='1'
import wandb
from vllm import LLM,SamplingParams
r=Path(__file__).resolve().parent;s=Path(os.environ['SERVING_SCRATCH'])
paths=json.loads((r/'model_paths.json').read_text());manifest=json.loads((r/'head_calibration_v18_manifest.json').read_text())
reference=json.loads((r/'head_calibration_v18_ar.json').read_text());assert reference['status']=='completed'
prior=json.loads((r/'head_calibration_v18_calibration.json').read_text());assert prior['status']=='completed'
fit=prior['source_fit'];expected=reference['outputs'];tasks=manifest['validation']
config=dict(model=paths['Qwen/Qwen3-4B'],dtype='float32',max_model_len=2048,max_num_seqs=32,max_num_batched_tokens=4096,gpu_memory_utilization=.65,enable_prefix_caching=False,attention_config={'backend':'TRITON_ATTN'},worker_extension_cls='cost_policy_v24.CostPolicyWorker',compilation_config={'cudagraph_capture_sizes':[1,2,4,8,16,32,64,128,256]},speculative_config=dict(method='dspark',model=paths['deepseek-ai/dspark_qwen3_4b_block7'],num_speculative_tokens=7,draft_sample_method='probabilistic',enable_adaptive_verification=True))
methods=['original_identity','measured_identity','measured_frozen','measured_online100']
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='cost-policy-v24',name='batch32-validation',dir=str(s),config={'engine':config,'batch':32,'methods':methods,'scope':'Validation only; cost intervention and head calibration ablations; matched instrumentation overhead'})
result=dict(status='initializing',run_url=run.url,records=[],states=[],costs={},source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_fit=fit)
out=r/'cost_policy_v24_results.json';assert not out.exists()
try:
 llm=LLM(**config);tok=llm.get_tokenizer();prompts=[tok.apply_chat_template([{'role':'user','content':x['text']}],tokenize=False,add_generation_prompt=True,enable_thinking=False) for x in tasks]
 params=SamplingParams(temperature=0.,max_tokens=128)
 def rpc(method,**kw):return llm.collective_rpc(method,kwargs=kw)[0]
 def configure(method):
  costmode='original' if method=='original_identity' else 'measured'
  result['costs'][costmode]=rpc('cost_control_v24',mode=costmode)
  fitted=method in ['measured_frozen','measured_online100']
  rpc('calibration_control_v17',learning=method=='measured_online100',slope=fit['slope'][0] if fitted else 1.,intercepts=fit['intercepts'] if fitted else [0.]*7,reset_statistics=True,refit_interval=100)
  rpc('budget_state_v22',reset=True)
 def generate(start):
  before=time.perf_counter();outputs=llm.generate(prompts[start:start+32],params,use_tqdm=False);duration=time.perf_counter()-before
  ids=[list(o.outputs[0].token_ids) for o in outputs]
  assert ids==expected[start:start+32],('Greedy mismatch',start)
  return duration,sum(map(len,ids))
 for method in methods:
  configure(method)
  for start in range(0,len(prompts),32):generate(start)
 for rd in range(4):
  order=methods[rd:]+methods[:rd]
  for method in order:
   configure(method)
   for start in range(0,len(prompts),32):
    duration,tokens=generate(start);row=dict(round=rd,method=method,start=start,count=min(32,len(prompts)-start),seconds=duration,tokens=tokens,tokens_per_second=tokens/duration)
    result['records'].append(row);run.log(row)
   result['states'].append(dict(round=rd,method=method,calibration=rpc('calibration_control_v17'),budgets=rpc('budget_state_v22')['rows']))
 result['status']='completed'
except BaseException as e:result.update(status='failed',error=repr(e));raise
finally:
 out.write_text(json.dumps(result,indent=2)+'\n');run.summary['status']=result['status'];art=wandb.Artifact('cost-policy-v24','validation-pilot')
 for path in [out,Path(__file__),r/'cost_policy_v24.py',r/'diagnostic_budget_v22.py',r/'head_calibration_v17.py',r/'head_calibration_v18_manifest.json']:art.add_file(str(path))
 run.log_artifact(art);run.finish()
