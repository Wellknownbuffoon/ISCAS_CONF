"""Render previews from the same CSVs used by the MATLAB code; no MATLAB execution."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Polygon

ROOT = Path(__file__).resolve().parent
METHODS = ['PCM-HWA', 'PCM-DAR-HWA', 'PCM-NSR-HWA']
TRAINING = ['HWA', 'DAR-HWA', 'NSR-HWA']
COMPS = ['none', 'GDC-safe', 'DS-B1', 'DS-val']
TIMES = np.array([20.,60.,1800.,3600.,86400.,432000.,2592000.,15552000.,
                  31557600.,63115200.,94672800.,157788000.])
LABELS = ['20 s','1 min','30 min','1 h','1 d','5 d','1 mo','6 mo','1 y','2 y','3 y','5 y']
fold = pd.read_csv(ROOT/'source_fold_means.csv')
selected = fold[fold.method.isin(METHODS)].copy()
assert not ((selected.method=='PCM-DAR-HWA') & selected.compensation.str.startswith('DS-')).any()
assert len(selected) == 360
assert not selected.duplicated(['method','compensation','t_index','fold']).any()
for key, g in selected.groupby(['method','compensation','t_index']):
    assert len(g) == 3 and set(g.fold) == {0,1,2}
    assert g.t_s.eq(TIMES[key[2]]).all()
    assert np.isfinite(g.accuracy).all() and g.accuracy.between(0,100).all()
summary = selected.groupby(['method','compensation','t_index','t_s']).accuracy.agg(
    mean_accuracy='mean',sd_across_folds='std',fold_count='size').reset_index()
summary['mean_test_error'] = 100-summary.mean_accuracy
summary.to_csv(ROOT/'expected_matlab_aggregated_source.csv',index=False)

def curve(method, comp):
    return summary[summary.method.eq(method)&summary.compensation.eq(comp)].sort_values('t_index')

def endpoint(candidate, control):
    a, b = curve(*candidate), curve(*control)
    return a.mean_accuracy.to_numpy()[[0,-1]]-b.mean_accuracy.to_numpy()[[0,-1]]

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
                     'axes.labelsize':11,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42})

def save(fig, name):
    for ext in ['png','pdf']:
        fig.savefig(ROOT/f'{name}.{ext}',dpi=240,bbox_inches='tight',facecolor='white')
    plt.close(fig)

panels = [
    ('(a) Training only: no compensation', [
        ('PCM-HWA','none','HWA (no compensation)','#111111'),
        ('PCM-DAR-HWA','none','DAR only (no compensation)','#884EA0'),
        ('PCM-NSR-HWA','none','NSR-HWA (no compensation)','#D55E00')]),
    ('(b) Pure DAR vs compensated pipelines', [
        ('PCM-HWA','GDC-safe','HWA + GDC-safe (baseline)','#111111'),
        ('PCM-HWA','DS-B1','HWA + DriftSlice B=1','#0072B2'),
        ('PCM-HWA','DS-val','HWA + DriftSlice selected B','#009E73'),
        ('PCM-DAR-HWA','none','DAR only (no compensation)','#884EA0'),
        ('PCM-NSR-HWA','DS-B1','NSR-HWA + DriftSlice B=1','#D55E00')])]
fig, axes = plt.subplots(1,2,figsize=(12.6,4.9),layout='constrained',sharey=True)
plot_rows = []
for ax, (title, curves) in zip(axes, panels):
    for method, comp, label, color in curves:
        a = curve(method, comp)
        x = a.t_index.to_numpy()
        y = a.mean_test_error.to_numpy()
        sd = a.sd_across_folds.to_numpy()
        ax.fill_between(x,y-2*sd,y+2*sd,color=color,alpha=.14,lw=0,zorder=1)
        marker = 's' if method=='PCM-DAR-HWA' else '^' if method=='PCM-NSR-HWA' and ax is axes[1] else 'o'
        ax.plot(x,y,marker+'-',color=color,lw=1.8,ms=6 if marker=='s' else 4,
                markerfacecolor='none' if marker=='s' else color,label=label,zorder=3)
        for row in a.itertuples(index=False):
            plot_rows.append(dict(panel=title,method=method,compensation=comp,
                t_index=row.t_index,t_s=row.t_s,mean_test_error_pct=row.mean_test_error,
                sd_across_training_repetitions_pp=row.sd_across_folds,
                band_lower_pct=row.mean_test_error-2*row.sd_across_folds,
                band_upper_pct=row.mean_test_error+2*row.sd_across_folds))
    ax.set_xticks(range(12),LABELS,rotation=45,ha='right')
    ax.set(title=title,xlabel='Time after programming',ylim=(0,100))
    ax.grid(alpha=.25,linestyle=':'); ax.set_axisbelow(True)
    ax.legend(loc='upper left',fontsize=8.5,framealpha=.95)
axes[0].set_ylabel('Test error (%)')
fig.suptitle('ResNet-32 / CIFAR-10: test error under PCM drift',fontsize=13)
fig.supxlabel('Mean ± 2 SD across 3 training repetitions; 30 device realizations per repetition\n'
              'Timestamps are equally spaced · bands describe variability, not 95% confidence intervals',fontsize=9)
save(fig,'figure1_pure_DAR_benchmark')
pd.DataFrame(plot_rows).to_csv(ROOT/'figure1_plotted_values.csv',index=False)

# A separate close-up of the formerly circled right panel, redrawn from CSVs.
fig, ax = plt.subplots(figsize=(8.2,5.3),layout='constrained')
for method, comp, label, color in panels[1][1]:
    a = curve(method,comp)
    x,y,sd = a.t_index.to_numpy(),a.mean_test_error.to_numpy(),a.sd_across_folds.to_numpy()
    marker = 's' if method=='PCM-DAR-HWA' else '^' if method=='PCM-NSR-HWA' else 'o'
    ax.fill_between(x,y-2*sd,y+2*sd,color=color,alpha=.14,lw=0,zorder=1)
    ax.plot(x,y,marker+'-',color=color,lw=1.8,ms=7 if marker=='s' else 4.5,
            markerfacecolor='none' if marker=='s' else color,label=label,zorder=3)
ax.set_xticks(range(12),LABELS,rotation=45,ha='right')
ax.set(title='Pure DAR (no compensation) versus compensated pipelines',
       xlabel='Time after programming',ylabel='Test error (%)',ylim=(0,100))
ax.grid(alpha=.25,linestyle=':');ax.set_axisbelow(True)
ax.legend(loc='upper left',fontsize=9,framealpha=.95)
fig.supxlabel('Mean ± 2 sample SD across 3 training repetitions; 30 realizations per repetition\n'
              'Training varies across curves; differences compare complete pipelines.',fontsize=8.5)
save(fig,'figure1b_pure_DAR_comparison')


fig, ax = plt.subplots(figsize=(8.2,5.3),layout='constrained')
gdc_rows = []
for method,comp,label,color in [
    ('PCM-HWA','GDC-safe','HWA + GDC-safe (baseline)','#111111'),
    ('PCM-DAR-HWA','GDC-safe','DAR + GDC-safe (no DriftSlice)','#884EA0'),
    ('PCM-HWA','DS-B1','HWA + DriftSlice B=1','#0072B2'),
    ('PCM-HWA','DS-val','HWA + DriftSlice selected B','#009E73'),
    ('PCM-NSR-HWA','DS-B1','NSR-HWA + DriftSlice B=1','#D55E00')]:
    a=curve(method,comp)
    x,y,sd=a.t_index.to_numpy(),a.mean_test_error.to_numpy(),a.sd_across_folds.to_numpy()
    marker='s' if method=='PCM-DAR-HWA' else '^' if method=='PCM-NSR-HWA' else 'o'
    assert np.max(y+2*sd)<=18
    ax.fill_between(x,y-2*sd,y+2*sd,color=color,alpha=.14,lw=0,zorder=1)
    ax.plot(x,y,marker+'-',color=color,lw=1.8,ms=7 if marker=='s' else 4.5,
        markerfacecolor='none' if marker=='s' else color,label=label,zorder=3)
    for row in a.itertuples(index=False):
        gdc_rows.append(dict(method=method,compensation=comp,t_index=row.t_index,
            t_s=row.t_s,mean_test_error_pct=row.mean_test_error,sd_across_folds_pp=row.sd_across_folds))
ax.set_xticks(range(12),LABELS,rotation=45,ha='right')
ax.set(title='Separate GDC-supported DAR benchmark: no DAR + DriftSlice',
    xlabel='Time after programming',ylabel='Test error (%)',ylim=(9.1,18))
ax.grid(alpha=.25,linestyle=':');ax.set_axisbelow(True);ax.legend(loc='upper left',fontsize=9)
fig.supxlabel('Mean ± 2 sample SD across 3 training repetitions.\nDAR + GDC-safe is distinct from DAR without compensation.',fontsize=8.5)
save(fig,'figure3_DAR_GDC_safe_separate_benchmark')
pd.DataFrame(gdc_rows).to_csv(ROOT/'figure3_plotted_values.csv',index=False)

left_initial, left_final = np.full((2,4),np.nan), np.full((2,4),np.nan)
right_initial, right_final = np.zeros((2,2)), np.zeros((2,2))
heat_rows = []
for mi, method in enumerate(METHODS[1:]):
    for ci, comp in enumerate(COMPS):
        if method=='PCM-DAR-HWA' and comp.startswith('DS-'):continue
        a,b = endpoint((method,comp),('PCM-HWA',comp))
        left_initial[mi,ci],left_final[mi,ci] = a,b
        heat_rows.append(dict(panel='Training minus HWA (same compensation)',
            candidate_method=method,candidate_compensation=comp,control_method='PCM-HWA',
            control_compensation=comp,delta_accuracy_20s_pp=a,delta_accuracy_5y_pp=b))
for mi, method in enumerate(['PCM-HWA','PCM-NSR-HWA']):
    for ci, comp in enumerate(COMPS[2:]):
        a,b = endpoint((method,comp),(method,'GDC-safe'))
        right_initial[mi,ci],right_final[mi,ci] = a,b
        heat_rows.append(dict(panel='DriftSlice minus GDC-safe (same training)',
            candidate_method=method,candidate_compensation=comp,control_method=method,
            control_compensation='GDC-safe',delta_accuracy_20s_pp=a,delta_accuracy_5y_pp=b))
heat = pd.DataFrame(heat_rows)
heat.to_csv(ROOT/'expected_matlab_heatmap_source.csv',index=False)
limit = max(abs(heat.delta_accuracy_20s_pp).max(),abs(heat.delta_accuracy_5y_pp).max())
cmap = LinearSegmentedColormap.from_list('gain_loss',['#B2182B','#F7F7F7','#1B7837'])
norm = TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
fig, axes = plt.subplots(1,2,figsize=(12.5,5.6),layout='constrained',gridspec_kw={'width_ratios':[4,2]})
for ax, initial, final, rows, cols, title in [
    (axes[0],left_initial,left_final,['DAR − HWA','NSR − HWA'],
     ['None','GDC-safe','DS-B1','DS-val'],'Training minus HWA (same compensation)'),
    (axes[1],right_initial,right_final,['HWA','NSR-HWA'],['DS-B1','DS-val'],
     'DriftSlice minus GDC-safe (same training)')]:
    for r in range(initial.shape[0]):
        for c in range(initial.shape[1]):
            if not np.isfinite(initial[r,c]):
                ax.add_patch(Polygon([(c,r),(c+1,r),(c+1,r+1),(c,r+1)],facecolor='.9',edgecolor='#555555',lw=.75))
                ax.text(c+.5,r+.5,'Excluded\nno DAR + DriftSlice',ha='center',va='center',fontsize=8)
                continue
            for value, verts, xy in [
                (initial[r,c],[(c,r),(c+1,r),(c+1,r+1)],(c+.72,r+.26)),
                (final[r,c],[(c,r),(c,r+1),(c+1,r+1)],(c+.28,r+.74))]:
                ax.add_patch(Polygon(verts,facecolor=cmap(norm(value)),edgecolor='#555555',lw=.75))
                ax.text(*xy,f'{value:+.2f}',ha='center',va='center',fontsize=10,
                        color='white' if abs(value)>.7*limit else '#111111')
    ax.set_xlim(0,len(cols));ax.set_ylim(len(rows),0);ax.set_aspect('equal')
    ax.set_xticks(np.arange(len(cols))+.5,cols);ax.set_yticks(np.arange(len(rows))+.5,rows)
    ax.tick_params(length=0,pad=8);ax.set_title(title,pad=14,fontsize=10)
    for spine in ax.spines.values():spine.set_visible(False)
fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=list(axes),fraction=.025,pad=.025,
             label='Accuracy change (percentage points)')
fig.suptitle('Full-run accuracy differences: initial versus five-year retention',fontsize=13)
fig.supxlabel('Upper triangle: 20 s · Lower triangle: 5 y · Green: improvement · Red: degradation\n'
              'Differences of means across 3 repetitions; colors do not indicate statistical significance.',fontsize=9)
save(fig,'figure2_no_DAR_DriftSlice_heatmap')

base = curve('PCM-HWA','GDC-safe').set_index('t_index').mean_accuracy
table_rows = []
row_definitions = [('PCM-HWA','HWA','GDC-safe'),('PCM-HWA','HWA','DS-B1'),
    ('PCM-HWA','HWA','DS-val'),('PCM-DAR-HWA','DAR-HWA','none'),
    ('PCM-DAR-HWA','DAR-HWA','GDC-safe'),('PCM-NSR-HWA','NSR-HWA','GDC-safe'),
    ('PCM-NSR-HWA','NSR-HWA','DS-B1'),('PCM-NSR-HWA','NSR-HWA','DS-val')]
for method,training,comp in row_definitions:
    a = curve(method,comp).set_index('t_index').mean_accuracy
    delta = a-base
    is_base = method=='PCM-HWA' and comp=='GDC-safe'
    table_rows.append(dict(model='ResNet-32',training=training,compensation=comp,
        final_test_accuracy=np.nan,initial_20s_accuracy=a.iloc[0],final_5y_accuracy=a.iloc[-1],
        max_drop_pp=a.iloc[0]-a.min(),improvement_1h_pp=np.nan if is_base else delta.loc[3],
        improvement_1d_pp=np.nan if is_base else delta.loc[4],
        improvement_6mo_pp=np.nan if is_base else delta.loc[7],
        improvement_5y_pp=np.nan if is_base else delta.loc[11]))
table = pd.DataFrame(table_rows)
table.to_csv(ROOT/'table_III_no_DAR_DriftSlice.csv',index=False)

def comp_label(comp, baseline=False):
    return {'DS-B1':'DriftSlice B=1','DS-val':'DriftSlice selected B','none':'None (pure DAR)'}.get(
        comp,'GDC-safe (baseline)' if baseline else 'GDC-safe')

def format_row(i,row, latex=False):
    values = [row.model,row.training,comp_label(row.compensation,i==0)]
    for j,c in enumerate(table.columns[3:],start=3):
        v = row[c]
        text = ('--' if latex else '—') if not np.isfinite(v) else f'{v:+.2f}' if j>=7 else f'{v:.2f}'
        best = ((j==4 and v==table.initial_20s_accuracy.max()) or
                (j==5 and v==table.final_5y_accuracy.max()) or
                (j==6 and v==table.max_drop_pp.min()))
        values.append(r'\textbf{'+text+'}' if latex and best else text)
    return values

fig = plt.figure(figsize=(16.5,6.4),facecolor='white')
ax = fig.add_axes([.015,.18,.97,.64]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
widths = np.array([1,1,1.8,.85,1,1,.9,.65,.65,.65,.65])
edges = np.r_[0,np.cumsum(widths)/widths.sum()];centers = (edges[:-1]+edges[1:])/2
headers = ['Model','Training','Compensation','Final Test','Initial 20s','Final 5Y','Max-drop','1h','1d','6M','5Y']
kw = dict(ha='center',va='center',fontfamily='DejaVu Serif',fontsize=11)
for c in range(3):ax.text(centers[c],.90,headers[c],weight='bold',**kw)
ax.text((edges[3]+edges[7])/2,.96,'Test Accuracy (%)',weight='bold',**kw)
ax.text((edges[7]+edges[11])/2,.96,'Gain vs HWA + GDC-safe (pp)',weight='bold',**kw)
for c in range(3,11):ax.text(centers[c],.85,headers[c],weight='bold',**kw)
for y in [1,.79,.035]:ax.plot([0,1],[y,y],color='k',lw=.8)
for a,b in [(3,7),(7,11)]:ax.plot(edges[[a,b]],[.91,.91],color='k',lw=.6)
for i,row in table.iterrows():
    y = .73-i*.09
    for c,text in enumerate(format_row(i,row)):
        v = row.iloc[c] if c>=3 else np.nan
        best = ((c==4 and v==table.initial_20s_accuracy.max()) or
                (c==5 and v==table.final_5y_accuracy.max()) or
                (c==6 and v==table.max_drop_pp.min()))
        ax.text(centers[c],y,text,weight='bold' if best else 'normal',**kw)
    if i in [2,4]:ax.plot([0,1],[y-.045,y-.045],color='.6',lw=.5)
fig.text(.5,.97,'TABLE III',ha='center',va='center',fontsize=12)
fig.text(.5,.895,'DAR BENCHMARKS WITHOUT DRIFTSLICE; HWA + GDC-SAFE REMAINS THE BASELINE',ha='center',va='center',fontsize=12)
for y,text in zip([.12,.075,.03],[
    'All gains use HWA + GDC-safe as the baseline. Means across 3 repetitions, each with 30 device realizations.',
    'Training and compensation vary across rows; matched-training comparisons isolate compensation.',
    'Max-drop = initial 20s accuracy minus the minimum over 12 timestamps. Final Test before PCM is unavailable.']):
    fig.text(.5,y,text,ha='center',va='center',fontsize=10)
save(fig,'table_III_no_DAR_DriftSlice')

lines = [r'\begin{table*}[t]',r'\centering',
    r'\caption{DAR without compensation and DAR + GDC-safe are reported separately; neither uses DriftSlice. All improvements are relative to HWA + GDC-safe. Accuracy is in percent; gains are percentage points.}',
    r'\label{tab:all-training-driftslice}',r'\resizebox{\textwidth}{!}{%',
    r'\begin{tabular}{lllrrrrrrrr}',r'\toprule',
    r'Model & Training & Compensation & \multicolumn{4}{c}{Test Accuracy (\%)} & \multicolumn{4}{c}{Gain vs HWA + GDC-safe (pp)} \\',
    r'\cmidrule(lr){4-7}\cmidrule(lr){8-11}',
    r' & & & Final Test & Initial 20s & Final 5Y & Max-drop & 1h & 1d & 6M & 5Y \\',r'\midrule']
for i,row in table.iterrows():
    lines.append(' & '.join(format_row(i,row,latex=True))+r' \\')
    if i in [2,4]:lines.append(r'\midrule')
lines += [r'\bottomrule',r'\end{tabular}}',
    r'\par\smallskip\footnotesize Means across three training repetitions, each with 30 device realizations. Training and compensation vary across rows; matched-training comparisons isolate compensation. Max-drop is initial 20s accuracy minus the minimum over 12 retention times. Final Test before PCM is unavailable in the retention CSVs.',
    r'\end{table*}']
(ROOT/'table_III_no_DAR_DriftSlice.tex').write_text('\n'.join(lines)+'\n')
print(table.to_string(index=False))
print('Rendered pure-DAR comparisons, a separate DAR + GDC-safe figure, 20 heatmap triangle values and the eight-row table.')
