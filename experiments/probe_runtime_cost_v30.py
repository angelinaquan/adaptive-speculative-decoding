"""Compare startup costs with observed decode steps; no speed conclusions."""
import os,json,hashlib,copy
from pathlib import Path
os.environ['VLLM_BATCH_INVARIANT']='1'
import wandb
from vllm import LLM,SamplingParams
r=Path(__file__).resolve().parent;s=Path(os.environ['SERVING_SCRATCH']);out=r/'runtime_cost_v30.json';assert not out.exists()
selection=json.loads((r/'cost_v29_summary.json').read_text())
assert not selection['meets_exploratory_5_percent_gate'],'Material gain should receive fresh confirmation before more tuning'
original=json.loads((r/'cost_v29_original.json').read_text());ref=json.loads((r/'cost_v29_ar.json').read_text());assert original['status']==ref['status']=='completed'
config=copy.deepcopy(original['metadata']['engine']);config['worker_extension_cls']='diagnostic_runtime_cost_v30.RuntimeCostWorker'
raw=(r/'cost_v26_manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==original['metadata']['manifest_sha256'];tasks=json.loads(raw)
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='runtime-cost-v30-diagnostic',name='startup-vs-actual-decode-cost',dir=str(s),config={'scope':'Reused prompts, instrumented diagnostic only; not speed evidence','engine':config})
result=dict(status='initializing',run_url=run.url,costs={},traces=[],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),extension_sha256=hashlib.sha256((r/'diagnostic_runtime_cost_v30.py').read_bytes()).hexdigest())
active=False
try:
 llm=LLM(**config);tok=llm.get_tokenizer();prompts=[tok.apply_chat_template([{'role':'user','content':x['text']}],tokenize=False,add_generation_prompt=True,enable_thinking=False) for x in tasks]
 assert all(len(tok.encode(p))+128<=2048 for p in prompts)
 params=SamplingParams(temperature=0.,max_tokens=128)
 def generate(start):
  outputs=llm.generate(prompts[start:start+64],params,use_tqdm=False)
  assert [list(x.outputs[0].token_ids) for x in outputs]==ref['outputs'][start:start+64],('Greedy parity',start)
 for mode in ['original','measured']:
  result['costs'][mode]=llm.collective_rpc('cost_control_v26',kwargs={'mode':mode})[0]
  for _ in range(2):
   for start in range(0,384,64):generate(start)
  for start in range(0,384,64):
   llm.collective_rpc('runtime_cost_control_v30',kwargs={'start':True});active=True
   generate(start)
   trace=llm.collective_rpc('runtime_cost_control_v30',kwargs={'start':False})[0];active=False
   assert trace['samples'] and trace['budgets']
   result['traces'].append(dict(mode=mode,start=start,domain=tasks[start]['domain'],**trace))
   out.write_text(json.dumps(result,indent=2)+'\n')
 result['status']='completed'
except BaseException as e:result.update(status='failed',error=repr(e));raise
finally:
 if active:llm.collective_rpc('runtime_cost_control_v30',kwargs={'start':False})
 out.write_text(json.dumps(result,indent=2)+'\n');run.summary['status']=result['status'];art=wandb.Artifact('runtime-cost-v30','diagnostic')
 for path in [out,Path(__file__),r/'diagnostic_runtime_cost_v30.py',r/'cost_only_v26.py',r/'cost_v26_manifest.json']:art.add_file(str(path))
 run.log_artifact(art);run.finish()
 if 'llm' in globals():llm.llm_engine.engine_core.shutdown(timeout=10)
