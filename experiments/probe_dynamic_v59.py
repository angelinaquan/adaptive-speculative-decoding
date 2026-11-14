import os,json,hashlib,copy,argparse
from pathlib import Path
os.environ['VLLM_BATCH_INVARIANT']='1'
p=argparse.ArgumentParser();p.add_argument('--role',choices=['identity','recency4096'],required=True);a=p.parse_args()
from vllm import LLM,SamplingParams
import wandb
from select_context_v55 import select
r=Path(__file__).resolve().parent;s=Path(os.environ['SERVING_SCRATCH']);out=r/f'diagnostic_v59_{a.role}.json';assert not out.exists()
ref=json.loads((r/'dynamic_v58_ar-reference.json').read_text());assert ref['status']=='completed'
base=json.loads((r/'dynamic_v58_native7-round0.json').read_text());assert base['status']=='completed' and base['outputs']==ref['outputs']
config=copy.deepcopy(base['metadata']['engine']);config['worker_extension_cls']='diagnose_dynamic_v59.DiagnosticWorker'
os.environ['DRAFT_ACTIVE_VOCAB']=str(r/('vocab_v54_identity.json' if a.role=='identity' else 'dynamic_v58_init4096.json'))
rank=json.loads((r/'context_v55_priors.json').read_text())['frequency']
tasks=json.loads((r/'oracle_v57_manifest.json').read_text())
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='diagnostic-v59-update-vs-acceptance',name=a.role,dir=str(s),config={'scope':'INSTRUMENTED_DIAGNOSTIC_NOT_SPEED_EVIDENCE','engine':config})
result=dict(status='initializing',run_url=run.url,role=a.role,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),module_sha256=hashlib.sha256((r/'diagnose_dynamic_v59.py').read_bytes()).hexdigest(),traces=[],scope='Instrumented GPU event spans and sampler-accepted prefixes beforeEOS/cap clipping; not speed evidence.')
try:
 llm=LLM(**config);tok=llm.get_tokenizer();prompts=[tok.apply_chat_template([{'role':'user','content':x['text']}],tokenize=False,add_generation_prompt=True,enable_thinking=False) for x in tasks];promptids=[tok.encode(p) for p in prompts]
 params=SamplingParams(temperature=0.,max_tokens=128)
 for rep in range(-2,1):
  for start in range(0,48,8):
   selected=list(range(151936)) if a.role=='identity' else select(promptids[start:start+8],rank,4096)[0]
   llm.collective_rpc('reset_dynamic',kwargs={'token_ids':selected,'enabled':a.role=='recency4096'})
   if rep==0:llm.collective_rpc('diagnostic_control',kwargs={'start':True})
   outputs=llm.generate(prompts[start:start+8],params,use_tqdm=False)
   assert [list(o.outputs[0].token_ids) for o in outputs]==ref['outputs'][start:start+8]
   if rep==0:
    trace=llm.collective_rpc('diagnostic_control',kwargs={'start':False})[0]
    audit=llm.collective_rpc('dynamic_audit',kwargs={'check':True})[0]
    result['traces'].append(dict(start=start,domain=tasks[start]['domain'],audit=audit,**trace))
 result['status']='completed'
except BaseException as e:result.update(status='failed',error=repr(e));raise
finally:
 out.write_text(json.dumps(result,indent=2)+'\n');run.summary['status']=result['status'];artifact=wandb.Artifact(f'diagnostic-v59-{a.role}','diagnostic')
 for f in [out,Path(__file__),r/'diagnose_dynamic_v59.py',r/'dynamic_vocab_v58.py',r/'active_vocab_v55.py',r/'oracle_v57_manifest.json']:artifact.add_file(str(f))
 run.log_artifact(artifact);run.finish()
 if 'llm' in globals():llm.llm_engine.engine_core.shutdown(timeout=10)
