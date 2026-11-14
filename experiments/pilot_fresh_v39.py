"""Exploratory freshness of confidence inputs to CPU budget selection."""
import os,json,hashlib,copy,time
from pathlib import Path
os.environ['VLLM_BATCH_INVARIANT']='1'
import wandb
from vllm import LLM,SamplingParams
r=Path(__file__).resolve().parent;s=Path(os.environ['SERVING_SCRATCH']);out=r/'fresh_v39_results.json';assert not out.exists()
ref=json.loads((r/'cost_v29_ar.json').read_text());prior=json.loads((r/'cost_v29_original.json').read_text());assert ref['status']==prior['status']=='completed'
config=copy.deepcopy(prior['metadata']['engine']);config['worker_extension_cls']='fresh_confidence_v39.FreshWorker'
raw=(r/'cost_v26_manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==prior['metadata']['manifest_sha256'];tasks=json.loads(raw)
plan=json.loads((r/'fresh_v39_plan.json').read_text());methods=plan['methods']
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='fresh-v39-exploration',name='fresh-confidence-budget',dir=str(s),config={'engine':config,'plan':plan})
result=dict(status='initializing',run_url=run.url,records=[],parity_checks=[],costs={},config=config,plan=plan,manifest_sha256=hashlib.sha256(raw).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),module_sha256=hashlib.sha256((r/'fresh_confidence_v39.py').read_bytes()).hexdigest())
try:
 llm=LLM(**config);tok=llm.get_tokenizer();prompts=[tok.apply_chat_template([{'role':'user','content':x['text']}],tokenize=False,add_generation_prompt=True,enable_thinking=False) for x in tasks];assert all(len(tok.encode(p))+128<=2048 for p in prompts)
 params=SamplingParams(temperature=0.,max_tokens=128)
 result['target_costs']=llm.collective_rpc('cost_control_v26',kwargs={'mode':'measured'})[0]
 def configure(method):return llm.collective_rpc('fresh_control_v39',kwargs={'mode':method})[0]
 def generate(start):
  before=time.perf_counter();outputs=llm.generate(prompts[start:start+64],params,use_tqdm=False);seconds=time.perf_counter()-before
  ids=[list(x.outputs[0].token_ids) for x in outputs]
  if ids!=ref['outputs'][start:start+64]:
   result['parity_failure']=dict(start=start,actual=ids,expected=ref['outputs'][start:start+64]);raise AssertionError(('Greedy parity',start))
  result['parity_checks'].append(dict(start=start,output_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest()))
  return seconds,sum(map(len,ids))
 configure('diagnostic')
 for start in range(0,384,64):generate(start)
 result['diagnostic']=configure('native')['trace']
 assert result['diagnostic']
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
 out.write_text(json.dumps(result,indent=2)+'\n');run.summary['status']=result['status'];art=wandb.Artifact('fresh-v39','exploratory-cost-ablation')
 for path in [out,Path(__file__),r/'fresh_confidence_v39.py',r/'cost_only_v26.py',r/'fresh_v39_plan.json',r/'cost_v26_manifest.json']:art.add_file(str(path))
 run.log_artifact(art);run.finish()
 if 'llm' in globals():llm.llm_engine.engine_core.shutdown(timeout=10)
