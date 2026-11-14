"""Exploratory lag-aware allocation under unchanged native total budget."""
import os,json,hashlib,copy,time
from pathlib import Path
os.environ['VLLM_BATCH_INVARIANT']='1'
import wandb
from vllm import LLM,SamplingParams
r=Path(__file__).resolve().parent;s=Path(os.environ['SERVING_SCRATCH']);out=r/'lag_v37_results.json';assert not out.exists()
ref=json.loads((r/'cost_v29_ar.json').read_text());prior=json.loads((r/'cost_v29_original.json').read_text());assert ref['status']==prior['status']=='completed'
config=copy.deepcopy(prior['metadata']['engine']);config['worker_extension_cls']='lag_allocation_v37.LagWorker'
raw=(r/'cost_v26_manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==prior['metadata']['manifest_sha256'];tasks=json.loads(raw)
plan=json.loads((r/'lag_v37_plan.json').read_text());methods=plan['methods']
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='lag-v37-exploration',name='lag-aware-allocation-fixed-budget',dir=str(s),config={'engine':config,'plan':plan})
result=dict(status='initializing',run_url=run.url,records=[],costs={},config=config,plan=plan,manifest_sha256=hashlib.sha256(raw).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),module_sha256=hashlib.sha256((r/'lag_allocation_v37.py').read_bytes()).hexdigest())
try:
 llm=LLM(**config);tok=llm.get_tokenizer();prompts=[tok.apply_chat_template([{'role':'user','content':x['text']}],tokenize=False,add_generation_prompt=True,enable_thinking=False) for x in tasks];assert all(len(tok.encode(p))+128<=2048 for p in prompts)
 params=SamplingParams(temperature=0.,max_tokens=128)
 result['allocation_checks']=llm.collective_rpc('lag_checks_v37')[0];assert result['allocation_checks']['passed']
 result['target_costs']=llm.collective_rpc('cost_control_v26',kwargs={'mode':'measured'})[0]
 def configure(method):return llm.collective_rpc('lag_control_v37',kwargs={'strength':{'identity':0.,'lag1':1.,'lag4':4.}[method]})[0]
 def generate(start):
  before=time.perf_counter();outputs=llm.generate(prompts[start:start+64],params,use_tqdm=False);seconds=time.perf_counter()-before
  ids=[list(x.outputs[0].token_ids) for x in outputs];assert ids==ref['outputs'][start:start+64],('Greedy parity',start)
  return seconds,sum(map(len,ids))
 for method in methods:
  result['costs'][method]=configure(method)
  for _ in range(2):
   for start in range(0,384,64):generate(start)
 for rd,order in enumerate(plan['orders']):
  for method in order:
   configure(method)
   for start in range(0,384,64):
    seconds,tokens=generate(start);row=dict(round=rd,method=method,start=start,domain=tasks[start]['domain'],seconds=seconds,tokens=tokens);result['records'].append(row);run.log(row)
 result['status']='completed'
except BaseException as e:result.update(status='failed',error=repr(e));raise
finally:
 out.write_text(json.dumps(result,indent=2)+'\n');run.summary['status']=result['status'];art=wandb.Artifact('lag-v37','exploratory-cost-ablation')
 for path in [out,Path(__file__),r/'lag_allocation_v37.py',r/'cost_only_v26.py',r/'lag_v37_plan.json',r/'cost_v26_manifest.json']:art.add_file(str(path))
 run.log_artifact(art);run.finish()
 if 'llm' in globals():llm.llm_engine.engine_core.shutdown(timeout=10)
