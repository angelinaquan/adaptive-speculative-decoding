import json,os,subprocess,hashlib
from pathlib import Path
import wandb
r=Path(__file__).resolve().parent
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='robust-v80-equal-budget',name='posthoc-minimax-vs-pooled-vs-single-domain',dir=os.environ['SERVING_SCRATCH'],config={'budget_unique_prompts':192,'widths':list(range(1,9)),'data':'v77 profiling with v77/v78 test outcomes','design':'Post-hoc deterministic replay of saved complete fixed-width runs. No new GPU timing.','robust_rule':'argmin_k max_domain (profile_time[k,d]/min_j profile_time[j,d]-1)','pooled_rule':'argmin_k sum_domain profile_time[k,d]','single_domain_rule':'192 prompts from one domain, minimize profiling time','paired_rule_comparison':'robust andpooled use same64mixed panels; single domains use all4three-block subsets','timing_sensitivity':'mean rounds; select0/test1; select1/test0'})
status='failed'
try:
 subprocess.run(['/tmp/aquan-forespark-v63/ForeSpark/.venv/bin/python',str(r/'compare_robust_v80.py')],check=True,stdout=subprocess.DEVNULL)
 s=json.loads((r/'robust_v80_summary.json').read_text());assert s['passed'];run.summary['paired']=s['paired'];run.summary['limitations']=s['limitations']
 for name in ['summary','paired']:
  rows=s[name];cols=list(rows[0]);run.log({name:wandb.Table(columns=cols,data=[[json.dumps(row[c]) if isinstance(row[c],(list,dict)) else row[c] for c in cols] for row in rows])})
 hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for pattern in ['profile_v77_b*.json','transfer_v78_b*.json'] for p in r.glob(pattern)}
 (r/'robust_v80_input_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n');status='completed'
finally:
 run.summary['status']=status;a=wandb.Artifact('robust-v80-equal-budget','posthoc-rule-comparison')
 for pattern in ['*robust_v80.py','robust_v80*.json']:
  for p in r.glob(pattern):a.add_file(str(p))
 run.log_artifact(a);(r/'robust_v80_run.json').write_text(json.dumps({'url':run.url,'status':status},indent=2)+'\n');run.finish()
