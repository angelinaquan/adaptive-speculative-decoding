"""Independent-seed AR reference for v42 distribution tests; not timing data."""
import os,json,hashlib,copy
from pathlib import Path
os.environ['VLLM_BATCH_INVARIANT']='1'
import wandb
from vllm import LLM,SamplingParams
r=Path(__file__).resolve().parent;out=r/'stochastic_v42_independent_ar.json';assert not out.exists()
prior=json.loads((r/'stochastic_v42_ar.json').read_text());assert prior['status']=='completed'
config=copy.deepcopy(prior['metadata']['engine']);assert 'speculative_config' not in config
manifest=(r/'cost_v26_manifest.json').read_bytes();assert hashlib.sha256(manifest).hexdigest()==prior['metadata']['manifest_sha256'];tasks=json.loads(manifest)
amendment=json.loads((r/'stochastic_v42_audit_amendment.json').read_text());assert amendment['new_ar_seed_base']==1700000000
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='stochastic-v42-exploration',name='independent-seed-ar-distribution-reference',dir=os.environ['SERVING_SCRATCH'],config={'engine':config,'audit_amendment':amendment})
result=dict(status='initializing',run_url=run.url,sampling_audit=[],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(manifest).hexdigest(),amendment=amendment,scope=__doc__)
try:
 llm=LLM(**config);tok=llm.get_tokenizer();old=json.loads((r/'cost_v29_ar.json').read_text())
 for index in [0,128,256]:
  prompt=tok.apply_chat_template([{'role':'user','content':tasks[index]['text']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  greedy=llm.generate([prompt],SamplingParams(temperature=0.,max_tokens=128),use_tqdm=False)
  assert list(greedy[0].outputs[0].token_ids)==old['outputs'][index],('Reference greedy gate',index)
  outputs=[]
  for offset in range(0,512,64):
   params=[SamplingParams(temperature=1.,top_p=1.,max_tokens=8,seed=1700000000+index*1000+j) for j in range(offset,offset+64)]
   outputs.extend([list(o.outputs[0].token_ids) for o in llm.generate([prompt]*64,params,use_tqdm=False)])
  result['sampling_audit'].append(dict(prompt_index=index,domain=tasks[index]['domain'],outputs=outputs))
 result['status']='completed';result['greedy_gate_passed']=True
except BaseException as e:result.update(status='failed',error=repr(e));raise
finally:
 out.write_text(json.dumps(result,indent=2)+'\n');run.summary['status']=result['status'];art=wandb.Artifact('stochastic-v42-independent-reference','correctness')
 for path in [out,Path(__file__),r/'stochastic_v42_audit_amendment.json']:art.add_file(str(path))
 run.log_artifact(art);run.finish()
 if 'llm' in globals():llm.llm_engine.engine_core.shutdown(timeout=10)
