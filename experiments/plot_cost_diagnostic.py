"""Plot completed profiling and validation evidence; no held-out claims."""
from pathlib import Path
import json,statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
r=Path(__file__).resolve().parents[1]
output=r/'docs/figures';output.mkdir(parents=True,exist_ok=True)
profile=json.loads((r/'results/v23/cost_v23_results.json').read_text())['profile']
summary=json.loads((r/'results/v25/cost_v25_summary_local.json').read_text())
points=profile['initial'];sizes=sorted({x['tokens'] for x in points if x['tokens']<=256})
y=np.array([statistics.median(x['forward_ms'] for x in points if x['tokens']==n) for n in sizes])
fig,(ax,bx)=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
ax.plot(sizes,y,'o-',label='Measured median',color='#24608e',lw=2)
ax.plot(sizes,np.maximum.accumulate(y),'--',label='Nondecreasing envelope',color='#b85c18',lw=2)
ax.set_xscale('log',base=2);ax.set_xticks([1,4,16,64,256],labels=['1','4','16','64','256']);ax.set_ylim(0,24)
ax.set(xlabel='Target tokens in profiling step',ylabel='Profiled target time (ms)',title='Cost envelope removes measured structure')
ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=9,loc='lower left')
methods=['original','fixed7','measured'];values=[summary['methods'][m]['tokens_per_second'] for m in methods]
bx.barh(['Original cost','Best fixed (7)','Measured cost'],values,color=['#999999','#777777','#24608e']);bx.invert_yaxis();bx.set_xlim(0,4100)
for i,v in enumerate(values):bx.text(v+45,i,f'{v:,.0f}',va='center',fontsize=10)
bx.set(xlabel='Generated tokens per second',title='Validation gain over best fixed: 0.90%');bx.grid(axis='x',alpha=.2)
fig.suptitle('Measured verification costs and validation throughput',fontsize=13,y=.99)
fig.get_layout_engine().set(rect=(0,.12,1,.78))
fig.text(.02,.025,'H200 · Qwen3-4B + DSpark · FP32, batch-invariant · validation batch 32\nProfiling is synthetic; validation engine order is unbalanced. Fresh confirmation is pending.',fontsize=8,color='#444444')
for ext in ['png','pdf']:fig.savefig(output/f'cost_diagnostic_validation.{ext}',dpi=180)
