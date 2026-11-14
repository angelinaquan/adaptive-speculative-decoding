"""Independent structural/token audit of the real-label smoke; no speed inference."""
import json,hashlib
from pathlib import Path

def prefix_equal(a,b):
 n=0
 for x,y in zip(a,b):
  if x!=y:break
  n+=1
 return n

def analyze(root):
 x=json.loads((root/'labels_v67_checks.json').read_text())
 assert x['status']=='completed' and x['passed']
 for key,name in [('source_sha256','check_labels_v67.py'),('plan_sha256','labels_v67_plan.json'),('manifest_sha256','oracle_v57_manifest.json'),('feature_sha256','labels_v67_features.pt')]:
  assert x[key]==hashlib.sha256((root/name).read_bytes()).hexdigest()
 assert x['revision']=='27aa911ddb040c61adeeade08e92949e24c9e17b' and x['forespark_commit']=='5d457ddab73a76f28db34acc577abfbd9e7c6eee'
 assert len(x['records'])==3 and [r['task_index'] for r in x['records']]==[0,16,32]
 tasks=json.loads((root/'oracle_v57_manifest.json').read_text())
 rows=[]
 for r in x['records']:
  assert r['domain']==tasks[r['task_index']]['domain']
  assert r['future_perturbation_invariant'] and r['target_greedy_replay_exact'] and r['features_identical_across_widths']
  assert r['anchors']==[len(r['input_ids'])+k for k in [0,8,16]]
  assert set(r['widths'])==set(map(str,range(1,9)))
  sequence=r['input_ids']+r['output_ids']
  means={}
  for width in range(1,9):
   w=r['widths'][str(width)]
   assert len(w['proposed_ids'])==len(w['accepted'])==3
   for a,pred,accepted in zip(r['anchors'],w['proposed_ids'],w['accepted']):
    assert len(pred)==width and all(0<=v<151936 for v in pred)
    assert accepted==prefix_equal(pred,sequence[a+1:a+width+1])
   assert len(w['independent_target_ids'])==width
   assert w['independent_first_anchor_accepted']==w['accepted'][0]==prefix_equal(w['proposed_ids'][0],w['independent_target_ids'])
   means[str(width)]=sum(w['accepted'])/3
  rows.append({'domain':r['domain'],'task_index':r['task_index'],'mean_accepted_by_width':means})
 return {'passed':True,'scope':__doc__,'prompts':3,'anchors':9,'widths':8,'independent_target_verifications':24,'records':rows,'limitations':['Only3reusedprompts and9anchors, no calibration efficacy orspeed claim.','Different widths are freshly drafted onthe sameanchorcontexts; doesnotestablishclosed-loopservingcorrectness.','Targetgreedycached/fullreplay andfutureperturbation gatespassed forthesesamples.']}
if __name__=='__main__':
 import sys
 print(json.dumps(analyze(Path(sys.argv[1])),indent=2))
