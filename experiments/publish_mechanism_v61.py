import os,json
from pathlib import Path
import wandb
from analyze_mechanism_v61 import analyze
r=Path(__file__).resolve().parent;s=analyze(r);out=r/'mechanism_v61_summary.json';out.write_text(json.dumps(s,indent=2)+'\n')
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='mechanism-v61-ORACLE-NOT-METHOD',name='audited-component-diagnostic',dir=os.environ['SERVING_SCRATCH'],config={'scope':s['scope']})
for role,info in s['roles'].items():
 for domain,z in info['domains'].items():
  for k in ['mean_accepted_prefix','mean_update_span_ms','steps']:run.summary[f'{role}/{domain}/{k}']=z[k]
a=wandb.Artifact('mechanism-v61-components','diagnostic')
for f in [out,Path(__file__),r/'analyze_mechanism_v61.py',r/'mechanism_v61_plan.json',r/'diagnose_dynamic_v59.py',r/'probe_mechanism_v61.py',r/'mechanism_v61_identity.json',r/'mechanism_v61_oracle1024.json']:a.add_file(str(f))
run.log_artifact(a);(r/'mechanism_v61_audit_run.json').write_text(json.dumps(dict(run_url=run.url))+'\n');run.finish()
