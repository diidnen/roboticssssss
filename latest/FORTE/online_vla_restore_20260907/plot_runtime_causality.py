"""Publication vector diagram of the implemented causal runtime."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

HERE=Path(__file__).resolve().parent
fig,ax=plt.subplots(figsize=(10,3.5));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
nodes={
 'grasp':(.07,.73,.12,.22,'Common\nestablished grasp','#f5f7fa'),
 'probe':(.225,.73,.13,.22,'Same physical\nprobe for all\nmain methods','#e4f2ec'),
 'obs':(.39,.73,.13,.22,'Post-probe\nobservation /\ndecision state','#f5f7fa'),
 'belief':(.57,.86,.15,.19,'58D evidence\nPosterior q(z)','#e4f2ec'),
 'vla':(.39,.28,.18,.27,'Frozen pi0 / Tabero\nONLINE inference\nPredict 50 / execute 10','#e6eefb'),
 'feas':(.655,.60,.18,.23,'Phase-free feasibility\nFirst 8 chunk steps\n+ state + q(z)','#e4f2ec'),
 'force':(.865,.60,.17,.23,'Force F* selection\nExpected utility\n0.05 N grid, [3,5] N','#e4f2ec'),
 'arbit':(.655,.23,.18,.27,'VLA arm unchanged\nAF squeeze control\nNative open intent kept','#fff1db'),
 'env':(.865,.23,.17,.27,'Task execution\nFull geometric\nsuccess evaluation','#f5f7fa'),
}
for x,y,w,h,label,color in nodes.values():
 ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.005,rounding_size=0.012',facecolor=color,edgecolor='#526477',linewidth=.8))
 ax.text(x,y,label,ha='center',va='center',fontsize=7.5)
def arrow(start,end,label=None,at=None,rad=0):
 ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=9,color='#526477',linewidth=.8,connectionstyle=f'arc3,rad={rad}'))
 if label:ax.text(*at,label,fontsize=6.8,ha='center',va='center',backgroundcolor='white')
arrow((.13,.73),(.155,.73));arrow((.29,.73),(.32,.73))
arrow((.46,.77),(.49,.86));arrow((.57,.76),(.61,.72))
arrow((.45,.68),(.56,.63),'state',(.50,.68))
arrow((.39,.61),(.39,.425),'first online query',(.45,.51))
arrow((.48,.35),(.56,.53),'first chunk,\nbefore action',(.54,.42))
arrow((.75,.60),(.775,.60))
arrow((.865,.48),(.71,.37),'F* once; force\nfeedback continuously',(.84,.42))
arrow((.485,.25),(.56,.25),'arm +\nopen intent',(.52,.32))
arrow((.75,.23),(.775,.23))
arrow((.87,.087),(.39,.135),'new live observation every 10 control steps',(.60,.055),rad=-.13)
fig.subplots_adjust(left=.015,right=.985,top=.99,bottom=.02)
fig.savefig(HERE/'RUNTIME_CAUSAL_DIAGRAM.pdf',bbox_inches='tight')
fig.savefig(HERE/'RUNTIME_CAUSAL_DIAGRAM.png',dpi=180,bbox_inches='tight')
plt.close(fig)
