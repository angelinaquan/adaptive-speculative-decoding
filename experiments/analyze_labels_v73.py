"""Token audit and hindsight equal-budget acceptance diagnostic; not inference speed."""
import json,hashlib,sys
from pathlib import Path
from allocation_headroom_v73 import allocate

def prefix(a,b):
 n=0
 for x,y in zip(a,b):
  if x!=y:break
  n+=1
 return n

def analyze(r):
 x=json.loads((r/'labels_v73_checks.json').read_text());assert x['passed'] and x['status']=='completed'
 for key,name in [('source_sha256','check_labels_v73.py'),('plan_sha256','labels_v73_plan.json'),('manifest_sha256','oracle_v57_manifest.json'),('feature_sha256','labels_v73_features.pt')]:assert x[key]==hashlib.sha256((r/name).read_bytes()).hexdigest()
 assert len(x['records'])==48 and [row['task_index'] for row in x['records']]==list(range(48))
 tasks=json.loads((r/'oracle_v57_manifest.json').read_text());groups={};skipped=[];count=0
 for row in x['records']:
  assert row['domain']==tasks[row['task_index']]['domain'] and row['target_greedy_replay_exact']
  n=len(row['output_ids']);p=len(row['input_ids']);offsets=[a for a in [0,8,16,24,32,40] if a+8<n]
  assert row['anchors']==[p+a for a in offsets]
  if not offsets:assert row['skipped'];skipped.append(row['task_index']);continue
  assert row['features_identical_across_widths'] and row['future_perturbation_invariant']
  assert set(row['widths'])==set(map(str,range(1,9)))
  sequence=row['input_ids']+row['output_ids']
  for width in range(1,9):
   data=row['widths'][str(width)];assert len(data['accepted'])==len(data['proposed_ids'])==len(offsets)
   for anchor,pred,length in zip(row['anchors'],data['proposed_ids'],data['accepted']):
    assert len(pred)==width and all(0<=t<151936 for t in pred)
    assert length==prefix(pred,sequence[anchor+1:anchor+1+width])
   assert len(data['independent_target_ids'])==width
   assert data['independent_first_anchor_accepted']==data['accepted'][0]==prefix(data['proposed_ids'][0],data['independent_target_ids'])
  for i,offset in enumerate(offsets):
   groups.setdefault(offset,[]).append([row['widths'][str(w)]['accepted'][i]+1 for w in range(1,9)]);count+=1
 diagnostics=[]
 for offset,rows in sorted(groups.items()):
  n=len(rows);choice=allocate(rows,3*n);fixed3=sum(row[2] for row in rows)
  best_uniform=max(sum(row[w-1] for row in rows) for w in [1,2,3])
  permuted=sum(sum(row[w-1] for row in rows) for w in choice['widths'])/n
  diagnostics.append({'offset':offset,'states':n,'budget':3*n,'oracle':choice,'fixed3_reward':fixed3,'best_uniform_reward_within_budget':best_uniform,'permuted_oracle_widths_expected_reward':permuted,'oracle_vs_fixed3_reward_percent':100*(choice['reward']/fixed3-1)})
 total_oracle=sum(d['oracle']['reward'] for d in diagnostics);total_fixed=sum(d['fixed3_reward'] for d in diagnostics)
 return {'passed':True,'scope':__doc__,'prompts':48,'anchors':count,'skipped_short_prompts':skipped,'independent_target_verifications':8*(48-len(skipped)),'oracle_vs_fixed3_reward_percent':100*(total_oracle/total_fixed-1),'groups':diagnostics,'limitations':['Hindsightoracle usesfuturetokens, notdeployable.','Teachertrajectory anchors are notclosedloopsteps fromselectedpolicies.','Equal totaldraftbudget doesnotguaranteeequallatency, attentioncost,orverificationcost.','HF FP32labels differfromBF16vLLMserving; no throughputclaim.','Reusedprompts andEOS-censoredanchors; noindependentconfirmation orconfidenceinterval.']}
if __name__=='__main__':
 r=Path(sys.argv[1]);summary=analyze(r);(r/'labels_v73_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
