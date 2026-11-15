import json,hashlib
from pathlib import Path
import torch
import numpy as np
r=Path(__file__).resolve().parent
s=json.loads((r/'labels_v83_checks.json').read_text());assert s['passed'] and len(s['records'])==480
assert hashlib.sha256((r/'labels_v83_features.pt').read_bytes()).hexdigest()==s['feature_sha256']
plan=json.loads((r/'labels_v83_plan.json').read_text());assert s['manifest_sha256']==plan['manifest_sha256']
tasks=json.loads((r/'labels_v83_manifest.json').read_text());features=torch.load(r/'labels_v83_features.pt',map_location='cpu',weights_only=True)
contexts=[];anchors=[];ids=[];offsets=[];labels=[];remaining=[];splits=[];domains=[]
for f in features:
 i=f['task_index'];record=s['records'][i];assert i==record['task_index'];n=len(record['anchors']);assert f['anchors'].tolist()==record['anchors']
 assert record['split']==tasks[i]['split'] and record['features_identical_across_widths'] and record['future_perturbation_invariant'] and record['target_greedy_replay_exact']
 contexts.extend(f['context'].reshape(n,-1).numpy());anchors.extend(f['anchor_embedding'].reshape(n,-1).numpy())
 sequence=record['input_ids']+record['output_ids']
 for j,a in enumerate(record['anchors']):
  row=[]
  for w in range(1,9):
   proposed=record['widths'][str(w)]['proposed_ids'][j];accepted=0
   for k,t in enumerate(proposed):
    if t!=sequence[a+1+k]:break
    accepted+=1
   assert accepted==record['widths'][str(w)]['accepted'][j]
   if j==0:assert accepted==record['widths'][str(w)]['independent_first_anchor_accepted']
   row.append(accepted+1)
  ids.append(i);offsets.append(a-len(record['input_ids']));labels.append(row);remaining.append(len(sequence)-a-1);splits.append(record['split']);domains.append(record['domain'])
np.savez_compressed(r/'labels_v83_features.npz',context=contexts,anchor=anchors,ids=ids,offsets=offsets,labels=labels,remaining=remaining,splits=splits,domains=domains)
summary={'passed':True,'prompts':480,'usable_prompts':len(features),'states':len(ids),'splits':{sp:{'prompts':sum(t['split']==sp for t in tasks),'states':splits.count(sp)} for sp in ['train','validation']},'censored_prompts':[x['task_index'] for x in s['records'] if not x['anchors']],'scope':'Training/validation labels only; no speed or method success claim.','feature_npz_sha256':hashlib.sha256((r/'labels_v83_features.npz').read_bytes()).hexdigest()}
(r/'labels_v83_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
