"""Idealized budget sensitivity using v34's same-engine startup costs.
This is a policy-model illustration, not a measured throughput experiment.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 repo=Path(__file__).resolve().parents[1]
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=repo/'results/v34/dual_cost_v34_results.json');p.add_argument('--output-prefix',type=Path,default=repo/'docs/figures/decision_sensitivity');a=p.parse_args()
 raw=a.source.read_bytes();data=json.loads(raw);assert data['status']=='completed'
 costs=data['costs'];batch=64;steps=7;budget=np.arange(batch*steps+1)
 curves={m:np.asarray(costs[m]['cost_tables'][1])[batch+budget]+costs[m]['cost_tables'][0][batch] for m in ['original','target_only']}
 ps=np.linspace(.05,.99,401);selected={m:[] for m in curves}
 for probability in ps:
  prefix=np.power(probability,np.arange(1,steps+1));scores=np.sort(np.tile(prefix,batch))[::-1]
  reward=np.r_[batch,batch+np.cumsum(scores)]
  for method,curve in curves.items():selected[method].append(int(np.argmax(reward/curve)))
 assert len(set(selected['original']))==1 and selected['original'][0]==448
 report=dict(scope='Idealized sensitivity only; 64 identical requests with constant conditional acceptance p. Costs from one v34 engine. No wall-clock speed claim.',source_sha256=hashlib.sha256(raw).hexdigest(),probabilities=ps.tolist(),selected_total_draft_budgets=selected,assumptions=['Conditionally identical positions and requests; real workloads are heterogeneous.','Uses startup cost model, not full serving latency including prefill/host overhead.','No simulated choice is fed into the fresh v38 experiment.'])
 a.output_prefix.parent.mkdir(parents=True,exist_ok=True);a.output_prefix.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
 fig,ax=plt.subplots(figsize=(8.5,4.4))
 for method,label,color in [('original','Monotone cost envelope','#777777'),('target_only','Measured target-cost curve','#386cb0')]:ax.step(ps,np.asarray(selected[method])/batch,where='mid',label=label,color=color,lw=2)
 ax.set(xlabel='Conditional acceptance probability p (same for every draft)',ylabel='Mean admitted draft tokens per request',xlim=(0,1),ylim=(0,7.6),yticks=np.arange(8))
 ax.set_title('Idealized decision sensitivity — not measured throughput',fontsize=12)
 ax.legend(loc='lower right',frameon=False);ax.spines[['top','right']].set_visible(False)
 fig.text(.5,.02,'64 identical requests, 7 available drafts each · same-engine v34 startup costs\nConfidence changes matter only when they move the decision across a budget boundary.',ha='center',fontsize=8)
 fig.tight_layout(rect=[0,.10,1,1])
 for ext in ['png','pdf']:fig.savefig(a.output_prefix.with_suffix('.'+ext),dpi=180)
 plt.close(fig);print({m:sorted(set(v)) for m,v in selected.items()})
if __name__=='__main__':main()
