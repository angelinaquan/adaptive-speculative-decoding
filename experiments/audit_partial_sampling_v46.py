"""Distribution diagnostics for completed audit samples, no speed comparisons."""
import os,json,hashlib,inspect
from pathlib import Path
import wandb
from analyze_stochastic_v42 import prefix_test
r=Path(__file__).resolve().parent;out=r/'partial_sampling_v46.json';assert not out.exists()
reference=json.loads((r/'stochastic_v42_independent_ar.json').read_text());assert reference['status']=='completed'
assert reference['source_sha256']==hashlib.sha256((r/'reference_sampling_v42.py').read_bytes()).hexdigest()
assert reference['amendment']['new_ar_seed_base']==1700000000
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='sampling-v43-diagnostic',name='independent-reference-partial-distribution-audit',dir=os.environ['SERVING_SCRATCH'])
report=dict(status='initializing',tests=[],run_url=run.url,reference_run=reference['run_url'],scope=__doc__,family_threshold=.01/27,limitations=['Only fixed1/2/3 have completed512-sample audits; fixed3 failed a later seeded-repeatability gate. This does not complete or pass v42.','512 samples on one prompt/domain has limited power; no rejection is not equivalence.','No inference of throughput, no online-calibration benefit, no fresh heldout claim.'])
try:
 for method in ['fixed1','fixed2','fixed3']:
  x=json.loads((r/f'stochastic_v42_{method}.json').read_text());assert x['greedy_gate_passed']
  assert x['metadata']['source_sha256']==hashlib.sha256((r/'benchmark_stochastic_v42.py').read_bytes()).hexdigest()
  assert len(x['sampling_audit'])==3
  for i in range(3):
   a=reference['sampling_audit'][i];b=x['sampling_audit'][i];assert a['prompt_index']==b['prompt_index'] and len(a['outputs'])==len(b['outputs'])==512
   result=prefix_test(a['outputs'],b['outputs']);result.update(method=method,domain=a['domain'],prompt_index=a['prompt_index']);report['tests'].append(result)
 from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
 source=inspect.getsource(RejectionSampler);path=r/'installed_rejection_sampler_v46.py';path.write_text(source)
 report['installed_sampler_class_sha256']=hashlib.sha256(source.encode()).hexdigest()
 report['status']='completed';report['detected_differences']=[z for z in report['tests'] if z['permutation_p']<=report['family_threshold']]
except BaseException as e:report.update(status='failed',error=repr(e));raise
finally:
 out.write_text(json.dumps(report,indent=2)+'\n');run.summary['status']=report['status'];run.summary['detected_difference_count']=len(report.get('detected_differences',[]))
 for z in report['tests']:run.summary[z['method']+'/'+z['domain']+'/p']=z['permutation_p']
 art=wandb.Artifact('partial-sampling-v46','correctness-diagnostic')
 for p in [out,Path(__file__),r/'analyze_stochastic_v42.py',r/'stochastic_v42_independent_ar.json',r/'stochastic_v42_audit_amendment.json',r/'installed_rejection_sampler_v46.py']:
  if p.exists():art.add_file(str(p))
 run.log_artifact(art);run.finish()
