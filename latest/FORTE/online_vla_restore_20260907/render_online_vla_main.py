"""Render publication artifacts only from the complete verified online main table."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import read,write,sha

METHODS=('FIXED_3','FIXED_4','FIXED_5','ACTIVEFORCING')
LABELS=('Fixed-3','Fixed-4','Fixed-5','ActiveForcing')
COLORS=('#94a3b8','#64748b','#334155','#c45b1b')

def render(source,destination):
    data=read(source)
    if (data.get('role')!='FINAL_ONLINE_FROZEN_VLA_MAIN_EVIDENCE' or data.get('branches')!=192
            or data.get('contexts')!=48 or data.get('all_branch_sources')!='ONLINE_VLA'
            or data.get('scripted_results_included') is not False):
        raise RuntimeError('Only complete verified online final data may be rendered')
    rows=data['rows'];roots=data['roots']
    if len(rows)!=192 or len(set(roots))!=4:
        raise RuntimeError('Final denominator mismatch')
    for p,d in data['source_hashes'].items():
        if sha(p)!=d:raise RuntimeError('Source admission changed')
    for r in rows:
        if sha(Path(r['job'])/'BRANCH_RESULT.json')!=r['result_sha256']:
            raise RuntimeError('Source outcome changed')
    counts={m:sum(r['full_task_success'] for r in rows if r['method']==m) for m in METHODS}
    for m in METHODS:
        group=[r for r in rows if r['method']==m]
        if len(group)!=48 or abs(counts[m]/48-data['main'][m]['FULL_TASK_SR'])>1e-12:
            raise RuntimeError('Aggregated success disagrees with source rows')
        if abs(np.mean([r['selected_force_N'] for r in group])-data['main'][m]['MEAN_SELECTED_SETPOINT'])>1e-12:
            raise RuntimeError('Force aggregation mismatch')
    destination.mkdir(exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.0),constrained_layout=True)
    for ax,key,title,scale,ylim in (
        (axes[0],'FULL_TASK_SR','Full-task success (%)',100,(0,110)),
        (axes[1],'MEAN_SELECTED_SETPOINT','Selected squeeze setpoint (N)',1,(0,5.6))):
        values=[data['main'][m][key]*scale for m in METHODS]
        ax.bar(range(4),values,color=COLORS,width=.65,alpha=.8)
        for i,m in enumerate(METHODS):
            rv=[data['by_root'][str(root)][m][key]*scale for root in roots]
            ax.scatter(i+np.linspace(-.15,.15,4),rv,c='black',s=15,zorder=3)
            label=f'{counts[m]}/48' if key=='FULL_TASK_SR' else f'{values[i]:.2f}'
            ax.text(i,values[i]+(3 if scale==100 else .12),label,ha='center',fontsize=8)
        ax.set_xticks(range(4),LABELS,rotation=15);ax.set_ylabel(title);ax.set_ylim(*ylim)
        ax.yaxis.grid(alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Online frozen VLA; common established grasp and probe',fontsize=10)
    for suffix in ('pdf','png'):fig.savefig(destination/f'online_vla_main.{suffix}',dpi=220)
    plt.close(fig)
    # The four rows per task are the four independent reset roots, not repeated timesteps.
    af=[r for r in rows if r['method']=='ACTIVEFORCING']
    fig,axes=plt.subplots(1,4,figsize=(7.1,2.7),sharey=True,constrained_layout=True)
    for ax,task in zip(axes,(0,1,5,6)):
        for root in roots:
            group={r['friction']:r for r in af if r['root']==root and r['task']==task}
            vals=[group[b]['selected_force_N'] for b in ('LOW','MID','HIGH')]
            ax.plot(range(3),vals,color='#c45b1b',alpha=.5,linewidth=1)
            for i,b in enumerate(('LOW','MID','HIGH')):
                ax.scatter(i,vals[i],s=22,marker='o' if group[b]['full_task_success'] else 'x',
                           color='#c45b1b' if group[b]['full_task_success'] else 'black')
        ax.set_title(f'Task {task}');ax.set_xticks(range(3),('Low','Mid','High'))
        ax.set_xlabel('Friction stratum');ax.set_ylim(2.9,5.1);ax.grid(alpha=.2)
    axes[0].set_ylabel('AF selected force (N)')
    for suffix in ('pdf','png'):fig.savefig(destination/f'online_vla_force_by_context.{suffix}',dpi=220)
    plt.close(fig)
    header=['method','full_success_n','contexts','full_sr','lift_sr','drop_rate','selected_force_N','measured_squeeze_N','contact_conditional_squeeze_N']
    table=[]
    for method,label in zip(METHODS,LABELS):
        a=data['main'][method]
        table.append([label,counts[method],48,a['FULL_TASK_SR'],a['LIFT_SR'],a['DROP_RATE'],
            a['MEAN_SELECTED_SETPOINT'],a['MEASURED_BILATERAL_SQUEEZE'],a['BILATERAL_CONTACT_CONDITIONAL_SQUEEZE']])
    with (destination/'main_table.csv').open('x',newline='') as f:
        writer=csv.writer(f);writer.writerow(header);writer.writerows(table)
    tex=[r'\begin{table*}[t]',r'\centering',r'\caption{Online frozen VLA evaluation from common established grasps. All main methods use the same probe. Each method has 48 contexts grouped into four fresh reset roots.}',
        r'\label{tab:online_vla_main}',r'\begin{tabular}{lrrrrrr}',r'\hline',
        r'Method & Full task & Lift (\%) & Drop (\%) & Setpoint (N) & Squeeze (N) & Contact squeeze (N) \\',r'\hline']
    for row in table:
        label,n,_,sr,lift,drop,setpoint,measured,contact=row
        contact_text='--' if contact is None else f'{contact:.2f}'
        tex.append(f'{label} & {n}/48 & {100*lift:.1f} & {100*drop:.1f} & {setpoint:.2f} & {measured:.2f} & {contact_text} '+r'\\')
    tex.extend([r'\hline',r'\end{tabular}',r'\end{table*}'])
    (destination/'main_table.tex').write_text('\n'.join(tex)+'\n')
    saving=100*data['FORCE_SAVING_VS_FIXED5']
    prose=(f'We evaluate a frozen pretrained Tabero $\\pi_0$ policy with Fixed-3, Fixed-4, Fixed-5, and ActiveForcing on 48 matched contexts: four fresh simulator reset roots, four object-to-basket tasks, and three friction strata. All methods start from a common established-grasp state and execute the same physical probe. The native VLA predicts 50 actions, executes 10, and requeries during each rollout. Only the grip-force strategy differs.\n\n'
        f'Full-task successes are {counts["FIXED_3"]}/48, {counts["FIXED_4"]}/48, {counts["FIXED_5"]}/48, and {counts["ACTIVEFORCING"]}/48, respectively (Table~\\ref{{tab:online_vla_main}}). ActiveForcing selects a mean setpoint of {data["main"]["ACTIVEFORCING"]["MEAN_SELECTED_SETPOINT"]:.3f} N, corresponding to {saving:.2f}\\% setpoint saving relative to Fixed-5. This quantity does not use post-drop zero-contact samples.\n\n'
        'Full-task success requires the frozen whole-mesh containment, release, support, and lift conditions. We separately report native or measured pre-release drop, allowing explicit reporting of any recovered drop. Bilateral squeeze is twice the smaller normal contact force; the nonrelease and bilateral-contact-conditional means are both reported. Each branch contributes equally to its method mean. The four reset roots are the independent grouping units; branches and timesteps are not treated as independent samples.\n\n'
        'The evaluation establishes downstream adaptation after an established grasp, not initial grasp planning, unseen object-category generalization, real-robot perception, or hardware hard-real-time performance. Existing controlled-motion supervision remains scripted auxiliary data; no scripted rollout enters this online-VLA main table.\n')
    (destination/'main_results_section.tex').write_text(prose)
    (destination/'FIGURE_CAPTIONS.md').write_text(
        'online_vla_main: Bars show means across 48 contexts per method. Black dots show the four root-level means, each based on 12 contexts; they are not confidence intervals. All arm actions are generated online by the same frozen VLA.\n\n'
        'online_vla_force_by_context: Each line follows one fresh reset root within a task. Circles denote full-task success and crosses failure. Friction strata share object/task identity; they are paired contexts, not independent objects.\n')
    artifacts={str(p):sha(p) for p in destination.iterdir() if p.is_file()}
    write(destination/'RENDER_MANIFEST.json',dict(source=str(source),source_sha256=sha(source),
        implementation_sha256=sha(__file__),artifacts=artifacts,verified_online_branches=192,
        source_runtime_manifest_sha256=data['runtime_manifest_sha256'],scripted_rows_included=False,
        ablations_or_secondary_results_not_inferred=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True);a=p.parse_args();render(a.source,a.destination)
