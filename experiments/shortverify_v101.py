"""Width3 drafting with shortened verification: exploratory reward/cost frontier."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json
from pathlib import Path
import numpy as np
from scipy.special import expit,logsumexp
from scipy.optimize import minimize
from allocation_headroom_v73 import allocate

def prepare(raw):
 rows=[]
 for record in raw['records']:
  seq=record['input_ids']+record['output_ids'];entry=record['widths'].get('3',{})
  for a,anchor in enumerate(record['anchors']):
   prev=0;F=[];prob=[]
   for j,grid in enumerate(entry['score_grids'][a]):
    scores=np.asarray(grid)[prev];p=np.exp(scores-logsumexp(scores));order=np.sort(scores);prev=int(np.argmax(scores));top=float(p[prev]);entropy=-float((p*np.log(np.maximum(p,1e-30))).sum())/np.log(len(p))
    assert entry['candidate_ids'][a][j][prev]==entry['proposed_ids'][a][j]
    F.append([1.,j/2,np.log(np.clip(top,1e-6,1-1e-6)/(1-np.clip(top,1e-6,1-1e-6)))/5,entropy,np.clip(order[-1]-order[-2],0,10)/5]);prob.append(top)
   truth=seq[anchor+1:anchor+4];proposed=entry['proposed_ids'][a];accepted=0
   for x,y in zip(proposed,truth):
    if x!=y:break
    accepted+=1
   assert accepted==entry['accepted'][a]
   rows.append({'prompt':record['task_index'],'anchor':a,'F':np.array(F),'p':np.array(prob),'accepted':accepted,'native3':record['widths']['3']['accepted'][a]+1})
 return rows

def fit(rows,penalty):
 F=[];y=[]
 for x in rows:
  n=min(x['accepted']+1,3);F.extend(x['F'][:n]);y.extend([1.]*x['accepted']+([0.] if x['accepted']<3 else []))
 F=np.array(F);y=np.array(y);mask=np.array([0,1,1,1,1])
 def fun(theta):
  z=F@theta;return np.mean(np.logaddexp(0,z)-y*z)+penalty/2*np.sum(mask*theta**2),F.T@(expit(z)-y)/len(y)+penalty*mask*theta
 ans=minimize(fun,np.array([0,0,5,0,0.]),jac=True,method='L-BFGS-B',bounds=[(-20,20)]*5);assert ans.success;return ans.x

def predictions(rows,mode,param):
 out=[]
 for x in rows:
  if mode=='calibrated':q=expit(x['F']@param)
  elif mode=='raw':q=expit(np.log(np.clip(x['p'],1e-9,1-1e-9)/(1-np.clip(x['p'],1e-9,1-1e-9)))/param)
  else:q=np.asarray(param)
  out.append(1+np.cumsum(np.cumprod(q)))
 return np.array(out)

def allocate_zero(curves,budget):
 # Conditional survival decreases by depth, so sorted marginals preserve prefixes.
 curves=np.asarray(curves);marginals=np.diff(np.concatenate([np.ones((len(curves),1)),curves],axis=1),axis=1)
 assert np.all(marginals>=-1e-12) and np.all(np.diff(marginals,axis=1)<=1e-12)
 ranked=sorted([(-float(g),j,i) for i,row in enumerate(marginals) for j,g in enumerate(row)])[:budget]
 widths=[0]*len(curves)
 for _,j,i in ranked:
  assert widths[i]==j
  widths[i]+=1
 return {'widths':widths}

def replay(rows,pred):
 groups={pid:j//16 for j,pid in enumerate(sorted({x['prompt'] for x in rows}))};value=0;choices=[]
 for anchor in sorted({x['anchor'] for x in rows}):
  for group in sorted(set(groups.values())):
   ix=[i for i,x in enumerate(rows) if x['anchor']==anchor and groups[x['prompt']]==group]
   if not ix:continue
   widths=allocate_zero(pred[ix],2*len(ix))['widths'];assert sum(widths)<=2*len(ix)
   for i,w in zip(ix,widths):value+=min(rows[i]['accepted'],w)+1;choices.append((i,w))
 return value,choices

def run():
 r=Path(__file__).resolve().parent;tr=prepare(json.loads((r/'path_v89_checks.json').read_text()));te=prepare(json.loads((r/'score_v90_checks.json').read_text()));results={}
 # Three request folds select calibration penalty by out-of-fold reward.
 candidates=[]
 for lam in [.001,.01,.1]:
  pred=np.empty((len(tr),3))
  for fold in range(3):
   fitrows=[x for x in tr if x['prompt']%3!=fold];ix=[i for i,x in enumerate(tr) if x['prompt']%3==fold];theta=fit(fitrows,lam);pred[ix]=predictions([tr[i] for i in ix],'calibrated',theta)
  candidates.append((replay(tr,pred)[0],lam))
 lam=max(candidates,key=lambda x:x[0])[1];theta=fit(tr,lam)
 # Raw logit-temperature baseline has the same three-parameter-selection budget.
 temps=[.5,1,2];raw_scores=[replay(tr,predictions(tr,'raw',t))[0] for t in temps];temp=temps[int(np.argmax(raw_scores))]
 for name,param in [('raw',temp),('calibrated',theta)]:
  reward,choices=replay(te,predictions(te,name,param));results[name]={'reward':reward,'parameter':param.tolist() if hasattr(param,'tolist') else param,'gain_vs_native3_percent':100*(reward/sum(x['native3'] for x in te)-1)}
 out={'scope':__doc__,'training_states':len(tr),'screen_states':len(te),'native3_reward':sum(x['native3'] for x in te),'fixed_draft3_verify2_reward':sum(min(x['accepted'],2)+1 for x in te),'calibration_oof_candidates':candidates,'raw_fitting_scores':raw_scores,'results':results,'limitations':['Reused exposed historical fitting-pool data, not final independent validation','Draft3 is matched; verification budget2*n vs native3*n. Reward loss must be offset by measured runtime savings.','Raw temperature selected on fitting reward; calibration selected on grouped OOF reward','Conditional hazard fit includes first rejection and censors later positions','Verification widths0..3 allowed; zero means target emits the correction/bonus token without verifying a draft','No novelty claim; dynamic verification and calibration are prior art']};(r/'shortverify_v101_summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':run()
