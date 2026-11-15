import os,json
from pathlib import Path
import wandb
r=Path(__file__).resolve().parent
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='profile-v77-budget-transfer',name='posthoc-exhaustive-equal-budget-robustness',dir=os.environ['SERVING_SCRATCH'])
s=json.loads((r/'profile_v77_robustness.json').read_text());run.summary['scope']=s['scope'];run.summary['analysis']=s['summary']
a=wandb.Artifact('profile-v77-subset-robustness','posthoc-analysis')
for n in ['robust_profile_v77.py','profile_v77_robustness.json']:a.add_file(str(r/n))
run.log_artifact(a);(r/'profile_v77_robustness_run.json').write_text(json.dumps({'url':run.url},indent=2)+'\n');run.finish()
