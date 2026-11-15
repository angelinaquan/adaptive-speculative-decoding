"""Context-conditioned score adapter screen. No novelty or throughput claim."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,hashlib
from pathlib import Path
import numpy as np
from rerank_v90 import prepare,fit,evaluate

def attach(data,raw,manifest,z):
 # Random projection fixed before labels; preserves current causal pre-draft context only.
 rng=np.random.default_rng(1865796);P=rng.normal(size=(z['context'].shape[1],16))/4
 lookup={}
 for record in raw['records']:
  source=manifest[record['task_index']]['source_task_index'];ix=np.flatnonzero(z['ids']==source);ix=ix[np.argsort(z['offsets'][ix])]
  assert len(ix)==len(record['anchors'])
  for anchor,i in zip(record['anchors'],ix):
   assert anchor==len(record['input_ids'])+int(z['offsets'][i])
  lookup[record['task_index']]=list(ix)
 cursor={}
 for x in data:
  pid=x['prompt'];j=cursor.get(pid,0);cursor[pid]=j+1;i=lookup[pid][j]
  v=np.asarray(z['context'][i],float);v=v/max(np.linalg.norm(v),1e-12);context=v@P*4
  x['features']=[np.concatenate([f,(f[...,:5,None]*context).reshape(*f.shape[:-1],80)],axis=-1) for f in x['features']]
 return data

def run():
 r=Path(__file__).resolve().parent;raw=json.loads((r/'path_v89_checks.json').read_text());manifest=json.loads((r/'path_v89_manifest.json').read_text());z=np.load(r/'labels_v83_features.npz');audit=json.loads((r/'labels_v83_summary.json').read_text());assert audit['feature_npz_sha256']==hashlib.sha256((r/'labels_v83_features.npz').read_bytes()).hexdigest()
 results=[]
 for w in [3,8]:
  data=attach(prepare(raw,w),raw,manifest,z);fitting=[x for x in data if x['prompt']%2==0];screen=[x for x in data if x['prompt']%2==1];base=sum(y['reward'] for y in evaluate(np.zeros(data[0]['features'][0].shape[-1]),screen));models={}
  for objective in ['ce','prefix']:
   candidates=[{'penalty':None,'reward':sum(x['reward'] for x in evaluate(np.zeros(data[0]['features'][0].shape[-1]),fitting)),'fits':[]}]
   for penalty in [.1,.01,1.]:
    reward=0;fits=[]
    for fold in range(3):
     tr=[x for x in fitting if (x['prompt']//2)%3!=fold];va=[x for x in fitting if (x['prompt']//2)%3==fold];theta,info=fit(tr,objective,penalty);assert info['success'],info;fits.append(info);reward+=sum(y['reward'] for y in evaluate(theta,va))
    candidates.append({'penalty':penalty,'reward':reward,'fits':fits})
   selected=max(candidates,key=lambda x:x['reward'])['penalty']
   if selected is None:theta=np.zeros(data[0]['features'][0].shape[-1]);info={'native_fallback':True}
   else:theta,info=fit(fitting,objective,selected);assert info['success'],info
   value=sum(x['reward'] for x in evaluate(theta,screen));models[objective]={'selected_penalty':selected,'reward':value,'gain_vs_native_percent':100*(value/base-1),'candidates':candidates,'fit':info}
  results.append({'width':w,'parameters':data[0]['features'][0].shape[-1],'fitting_states':len(fitting),'screen_states':len(screen),'native_reward':base,'models':models})
 out={'scope':__doc__,'results':results,'limitations':['Reused exposed v89 even/odd prompt split: exploratory only','Random projection uses causal pre-draft context, joined by audited prompt/anchor identity','Contextual correction already exists in DCOCO; this is a lightweight adaptation screen','Reward only; no runtime or independent validation claim']};(r/'context_rerank_v96_summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':run()
