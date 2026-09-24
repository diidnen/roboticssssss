"""Export an observed-force/geometry development comparison; no SR inference."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--context',required=True);p.add_argument('--figure',required=True);a=p.parse_args()
    fig,axes=plt.subplots(3,1,figsize=(9,7),sharex=True,constrained_layout=True)
    for method,color in [('ACTIVEFORCING','#16705b'),('FIXED_3','#c34b43'),('FIXED_5','#3d65b2')]:
        job=Path(a.out)/'branches'/(a.context+'__'+method)
        rows=json.loads((job/'BRANCH_TRACE.json').read_text());decision=json.loads((job/'DECISION_METADATA.json').read_text())
        force=json.loads((job/'PLANNER_DECISION.json').read_text())['executed_force_N'];time=np.arange(1,len(rows)+1)*.05
        axes[0].plot(time,[r['measured_bilateral_squeeze'] for r in rows],label=f'{method} ({force:g} N)',color=color,lw=1.2)
        axes[1].plot(time,[100*(r['object_position_m'][2]-decision['object_pose'][2]) for r in rows],color=color,lw=1.2)
        axes[2].plot(time,[100*np.linalg.norm(r['object_relative_translation_m']) for r in rows],color=color,lw=1.2)
    axes[0].set_ylabel('Measured squeeze (N)');axes[0].legend(loc='upper right',ncol=3)
    axes[1].set_ylabel('Object lift (cm)');axes[2].set_ylabel('Object–gripper distance (cm)');axes[2].set_xlabel('Downstream simulated time (s)')
    for ax in axes:ax.grid(alpha=.2);ax.set_xlim(0,17.5)
    axes[0].set_title(f'Development only: {a.context} | phase-free feasibility + online frozen VLA')
    fig.savefig(a.figure,dpi=160)

if __name__=='__main__':main()
