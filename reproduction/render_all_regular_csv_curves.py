"""Render every held-out regular CSV separately from prespecified repeat1 export."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reproduction'))
import plot_full_paper_results_v4 as plot

REC=ROOT/'outputs/full_reproduction/cloud_emergency_recovered/workspace'
OUT=ROOT/'outputs/full_reproduction/regular_per_CSV_curves_repeat1_v4'


def main():
    if OUT.exists() and any(OUT.iterdir()):raise ValueError('Use new empty output')
    OUT.mkdir(exist_ok=True)
    data=plot.Inputs(REC/'pinn4soh_full_report_20261110',REC/'pinn4soh_full_evidence_20261110')
    plot.apply_figure_style()
    curves,metadata={},{}
    for r in data.repeats:
        if r['family']=='regular' and r['evaluation']=='final' and r['repeat']==1:
            arm=r['loss_variant'] if r['model']=='PINN' else r['model']
            metadata[r['task_id']]=(r['paper_group'],arm)
    grouped=defaultdict(dict)
    for r in data.curves:
        if r['task_id'] not in metadata or r['evaluation']!='final':continue
        group,arm=metadata[r['task_id']]
        truth=data.truths[r['truth_id']];key=(group,truth['battery'])
        if arm in grouped[key]:raise ValueError('Duplicate CSV curve arm')
        grouped[key][arm]=(truth,r)
    assert len(grouped)==82 and all(len(v)==4 for v in grouped.values())
    qa,manifest=[],[]
    arms=[('published_label_monotonicity','Label-only PINN','#197c80'),
          ('paper_prediction_monotonicity','Prediction-mono PINN','#877a23'),
          ('MLP','MLP','#b96537'),('CNN','CNN','#655897')]
    for (group,battery),records in sorted(grouped.items()):
        truth=next(iter(records.values()))[0]
        assert all(x[0]==truth for x in records.values())
        x=np.array(truth['relative_retained_point_indices']);y=np.array(truth['measured_SOH'])*100
        fig,axes=plt.subplots(1,2,figsize=(8.3,3.6),gridspec_kw={'width_ratios':[1.6,1]})
        fig.subplots_adjust(left=.08,right=.98,bottom=.25,top=.75,wspace=.35)
        short=battery.removeprefix('data/');fig.suptitle(group+' | '+short,x=.08,y=.98,ha='left',fontsize=9)
        handles=[]
        handles += axes[0].plot(x,y,color='#252525',label='Measured',linewidth=1)
        for arm,label,color in arms:
            r=records[arm][1];pred=np.array(r['predicted_SOH'])*100
            assert len(pred)==len(x) and np.isfinite(pred).all()
            handles += axes[0].plot(x,pred,color=color,label=label,linewidth=.8,alpha=.9)
            axes[1].plot(x,pred-y,color=color,linewidth=.8,alpha=.9)
        axes[0].set(xlabel='Retained point within CSV',ylabel='SOH (%)')
        axes[1].set(xlabel='Retained point within CSV',ylabel='Prediction - measured (pp)')
        axes[1].axhline(0,color='#777777',linewidth=.6)
        for ax in axes:ax.margins(.04);ax.grid(axis='y',alpha=.16)
        fig.legend(handles,[h.get_label() for h in handles],loc='upper left',bbox_to_anchor=(.075,.91),ncol=3,fontsize=7)
        fig.text(.08,.05,'Prespecified repeat 1; identical CSV truth for all four arms; uniform display sample <=256 points.\n'
            'Full-array metrics are separate. '+('MIT CSV5 mixes physical identities; CSV is the evaluation unit.' if group=='MIT' else 'CSV filename identifies the released evaluation sequence.'),fontsize=7)
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        from matplotlib.text import Text
        visible=[(t,t.get_window_extent(renderer)) for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
        # Suppress ticks beyond limits, as in the main plotter.
        suppress=set()
        for ax in axes:
            for axis in [ax.xaxis,ax.yaxis]:
                lo,hi=sorted(axis.get_view_interval())
                for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                    if not lo<=tick.get_loc()<=hi:suppress.update([id(tick.label1),id(tick.label2)])
        visible=[(t,b) for t,b in visible if id(t) not in suppress]
        outside=[t.get_text() for t,b in visible if not fig.bbox.contains(b.x0,b.y0) or not fig.bbox.contains(b.x1,b.y1)]
        overlaps=[]
        for i,(t,b) in enumerate(visible):
            for u,c in visible[i+1:]:
                if b.overlaps(c):
                    w=min(b.x1,c.x1)-max(b.x0,c.x0);h=min(b.y1,c.y1)-max(b.y0,c.y0)
                    if w>1 and h>1:overlaps.append([t.get_text(),u.get_text()])
        name=group.replace('/','_')+'__'+battery.removeprefix('data/').replace('/','__').replace(' ','_').removesuffix('.csv')
        for ext in ['png','pdf']:fig.savefig(OUT/(name+'.'+ext),dpi=300)
        plt.close(fig)
        qa.append(dict(group=group,battery=battery,outside=outside,overlaps=overlaps,passed=not outside and not overlaps))
        manifest.append(dict(group=group,battery=battery,n_full_retained_points=truth['n_full_retained_points'],
            n_display_points=len(x),png=name+'.png',pdf=name+'.pdf',truth_id=truth['truth_id']))
    (OUT/'curve_index.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (OUT/'figure_qa.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (OUT/'README.md').write_text('# 逐发布CSV常规预测曲线\n\n82个测试CSV各一图，四方法/损失臂共享同一实测序列，均事先固定repeat1；两个PINN实现分开标识。每图包括预测和误差面板，显示点最多256，指标仍来自完整数组。MIT按发布文件而非纯物理电池解释。\n',encoding='utf8')
    result={'curves':len(manifest),'png':82,'pdf':82,'all_geometry_passed':all(x['passed'] for x in qa),
            'failed':[x for x in qa if not x['passed']],'input_sha256':data.hashes}
    (OUT/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:result[k] for k in ['curves','png','pdf','all_geometry_passed','failed']},ensure_ascii=False))


if __name__=='__main__':main()
