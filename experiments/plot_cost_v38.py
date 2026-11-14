"""Plot the completed v38 audit without hiding zero or negative changes."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 repo=Path(__file__).resolve().parents[1]
 p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,default=repo/'results/v38/cost_v38_summary.json');p.add_argument('--output-prefix',type=Path,default=repo/'docs/figures/cost_v38_confirmation');a=p.parse_args()
 s=json.loads(a.summary.read_text());assert s['all_correctness_and_provenance_checks_passed']
 keys=['fixed7','domain_fixed','original'];labels=['Global fixed width 7','Domain-specific fixed width','Original adaptive policy'];colors=['#386cb0','#238b45','#7556a4']
 fig,(left,right)=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={'width_ratios':[1.1,1.]})
 extent=[0,5]
 for i,(key,label,color) in enumerate(zip(keys,labels,colors)):
  v=s['comparisons'][key];low,high=v['batch_and_round_bootstrap_95'];gain=v['gain_percent'];y=2-i
  left.plot([low,high],[y,y],color=color,lw=3);left.plot(gain,y,'o',color=color,ms=7)
  left.annotate(f'{gain:+.2f}% [{low:+.2f}, {high:+.2f}]',(gain,y),xytext=(0,12),textcoords='offset points',ha='center',fontsize=9)
  extent.extend([low,high,gain])
 left.set_yticks([2,1,0],labels);left.set_ylim(-.55,2.65);left.axvline(0,color='#777777',lw=.8)
 left.axvline(5,color='#777777',ls='--',lw=1,label='5% practical target');left.legend(loc='lower right',frameon=False,fontsize=8)
 lo,hi=min(extent),max(extent);pad=max(1.,.16*(hi-lo));left.set_xlim(lo-pad,hi+pad)
 left.set_title('Mixed workload\n95% paired batch-and-round intervals',fontsize=11)
 domains=['alpaca','gsm8k','mbpp'];x=np.arange(3);all_values=[0.]
 for i,(key,label,color) in enumerate(zip(keys,labels,colors)):
  values=[s['domain_gain_percent'][d][key] for d in domains];all_values.extend(values)
  bars=right.bar(x+(i-1)*.24,values,width=.22,color=color,label=label)
  for bar,value in zip(bars,values):
   right.annotate(f'{value:+.1f}',(bar.get_x()+bar.get_width()/2,value),xytext=(0,3 if value>=0 else -3),textcoords='offset points',ha='center',va='bottom' if value>=0 else 'top',fontsize=8)
 right.axhline(0,color='#777777',lw=.8);right.set_xticks(x,['Chat','Math','Code']);right.set_title('Domain breakdown\nDescriptive differences',fontsize=11);right.set_ylabel('Throughput change (%)')
 lo,hi=min(all_values),max(all_values);pad=max(.7,.2*(hi-lo));right.set_ylim(lo-pad,hi+pad)
 left.set_xlabel('Throughput change (%)')
 for ax in [left,right]:
  ax.spines[['top','right']].set_visible(False)
 fig.suptitle('Measured verification costs: fresh 768-prompt confirmation',fontsize=13)
 fig.text(.5,.025,'Qwen3-4B + DSpark · H200 · batch 64 · equal prompt counts by domain\nFP32 tensors with batch-invariant backend; strict IEEE arithmetic not established.',ha='center',fontsize=8,color='#444444')
 fig.tight_layout(rect=[0,.11,1,.93]);a.output_prefix.parent.mkdir(parents=True,exist_ok=True)
 for ext in ['png','pdf']:fig.savefig(a.output_prefix.with_suffix('.'+ext),dpi=180)
 plt.close(fig)
if __name__=='__main__':main()
