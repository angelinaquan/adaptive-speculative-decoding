"""Exploratory calibration intervention at batch64, matched extension overhead."""
import os,json,hashlib,copy,time
from pathlib import Path
os.environ['VLLM_BATCH_INVARIANT']='1'
import wandb
from vllm import LLM,SamplingParams
r=Path(__file__).resolve().parent;s=Path(os.environ['SERVING_SCRATCH']);out=r/'head_cost_v32_results.json';assert not out.exists()
reference=json.loads((r/'cost_v29_ar.json').read_text());assert reference['status']=='completed';expected=reference['outputs']
original=json.loads((r/'cost_v29_original.json').read_text());assert original['status']=='completed'
config=copy.deepcopy(original['metadata']['engine']);config['worker_extension_cls']='head_cost_v32.HeadCostWorker'
raw=(r/'cost_v26_manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==original['metadata']['manifest_sha256'];tasks=json.loads(raw)
prior_bytes=(r/'head_calibration_v18_calibration.json').read_bytes();prior=json.loads(prior_bytes);assert prior['status']=='completed';fit=prior['source_fit']
plan=json.loads((r/'head_cost_v32_plan.json').read_text())
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='head-cost-v32-batch64',name='matched-overhead-calibration-pilot',dir=str(s),config={'engine':config,'plan':plan,'source_fit':fit,'source_fit_file_sha256':hashlib.sha256(prior_bytes).hexdigest()})
result=dict(status='initializing',run_url=run.url,records=[],states=[],config=config,source_fit=fit,source_fit_file_sha256=hashlib.sha256(prior_bytes).hexdigest(),manifest_sha256=hashlib.sha256(raw).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),module_sha256=hashlib.sha256((r/'head_cost_v32.py').read_bytes()).hexdigest(),plan=plan)
try:
 llm=LLM(**config);tok=llm.get_tokenizer();prompts=[tok.apply_chat_template([{'role':'user','content':x['text']}],tokenize=False,add_generation_prompt=True,enable_thinking=False) for x in tasks];assert all(len(tok.encode(p))+128<=2048 for p in prompts)
 params=SamplingParams(temperature=0.,max_tokens=128)
 result['costs']=llm.collective_rpc('cost_control_v26',kwargs={'mode':'measured'})[0]
 def configure(method):
  return llm.collective_rpc('calibration_control_v17',kwargs=dict(learning=method=='online100',slope=1. if method=='identity' else fit['slope'][0],intercepts=[0.]*7 if method=='identity' else fit['intercepts'],reset_statistics=True,refit_interval=100))[0]
 def generate(start):
  before=time.perf_counter();outputs=llm.generate(prompts[start:start+64],params,use_tqdm=False);seconds=time.perf_counter()-before
  ids=[list(x.outputs[0].token_ids) for x in outputs];assert ids==expected[start:start+64],('Greedy parity',start)
  return seconds,sum(map(len,ids))
 # Exercise every path before timing. Reset fitted states again before every timed stream.
 for method in plan['methods']:
  configure(method)
  for _ in range(2):
   for start in range(0,384,64):generate(start)
 for rd,order in enumerate(plan['orders']):
  for method in order:
   initial=configure(method)
   for start in range(0,384,64):
    seconds,tokens=generate(start);row=dict(round=rd,method=method,start=start,domain=tasks[start]['domain'],seconds=seconds,tokens=tokens);result['records'].append(row);run.log(row)
   state=llm.collective_rpc('calibration_control_v17')[0];result['states'].append(dict(round=rd,method=method,initial=initial,final=state))
 result['status']='completed'
except BaseException as e:result.update(status='failed',error=repr(e));raise
finally:
 out.write_text(json.dumps(result,indent=2)+'\n');run.summary['status']=result['status'];art=wandb.Artifact('head-cost-v32','exploratory-calibration')
 for path in [out,Path(__file__),r/'head_cost_v32.py',r/'head_calibration_v17.py',r/'cost_only_v26.py',r/'head_cost_v32_plan.json',r/'cost_v26_manifest.json',r/'head_calibration_v18_calibration.json']:art.add_file(str(path))
 run.log_artifact(art);run.finish()
 if 'llm' in globals():llm.llm_engine.engine_core.shutdown(timeout=10)
