"""Compose source-backed scene/sequence figures; never synthesize task evidence."""
from pathlib import Path
import json,hashlib,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
P=Path(__file__).resolve().parent
V=P/'HELDOUT_TASK_FORM_VISUAL_AUDIT';V.mkdir(exist_ok=True)
sources=[
 ('FORM_C_CANDIDATE','Book → back compartment','libero_10_5','preflight_gpu/libero_10_5/NATIVE_RESET_FRAME.png'),
 ('FORM_C_CANDIDATE','Wine bottle → rack','libero_goal_9','preflight_gpu/libero_goal_9/NATIVE_RESET_FRAME.png'),
 ('FORM_D_CANDIDATE','Bowl → out of open drawer','libero_spatial_4','preflight_open_drawer/libero_spatial_4/NATIVE_RESET_FRAME.png')]
receipts=[]
def preserve(src,dst,role):
 dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
 receipts.append(dict(source=str(src),artifact=str(dst.relative_to(P)),sha256=hashlib.sha256(src.read_bytes()).hexdigest(),role=role))
fig,axes=plt.subplots(1,3,figsize=(15,6.3),facecolor='#f5f3ef')
for ax,(form,title,key,rel) in zip(axes,sources):
 src=P/rel;preserve(src,V/form/key/'SCENE.png','native scene, not successful rollout')
 ax.imshow(mpimg.imread(src));ax.axis('off')
 ax.set_title(('C • Constrained-placement candidate' if form.startswith('FORM_C') else 'D • Extraction candidate')+'\n'+title,fontsize=13,loc='left',pad=12)
 ax.text(0,-.045,'Fixed-5 screen: 0/2; four contexts unrun',transform=ax.transAxes,fontsize=11,color='#6a3530')
 episode=P/'qualification'/f'{key}_r5100_mu0.30'
 frames=[episode/f'FRAME_{i:04d}.png' for i in (1,100,200)]
 seq,axs=plt.subplots(1,3,figsize=(12,4.8),facecolor='#f5f3ef')
 for sa,frame,i in zip(axs,frames,(1,100,200)):
  preserve(frame,V/form/key/frame.name,'negative native-reset qualification sequence')
  sa.imshow(mpimg.imread(frame));sa.axis('off');sa.set_title(f'VLA step {i}',fontsize=11)
 seq.suptitle(title+' — Fixed-5 qualification, root 5100, friction 0.30',fontsize=14)
 seq.text(.04,.035,'No sustained bilateral grasp detected by step 200. This is not a successful task-form demonstration.',fontsize=10)
 seq.subplots_adjust(top=.85,bottom=.12,wspace=.025)
 seq.savefig(V/form/key/'QUALIFICATION_SEQUENCE.png',dpi=160,facecolor=seq.get_facecolor());plt.close(seq)
fig.suptitle('Available candidate scenes • no qualified held-out task forms',x=.035,y=.98,ha='left',fontsize=20,fontweight='bold')
fig.text(.035,.905,'Repository-native scenes show potential constraints; downstream task-form execution remains unverified.',fontsize=12,color='#444444')
fig.text(.035,.06,'Candidate scenes only. No AF branches were executed. Native-reset screens do not test supplied-grasp competence.',fontsize=11,color='#6a3530')
fig.subplots_adjust(left=.035,right=.985,top=.77,bottom=.14,wspace=.045)
fig.savefig(P/'TASK_FORM_OVERVIEW_CONTACT_SHEET.png',dpi=180,facecolor=fig.get_facecolor());plt.close(fig)
preserve(P/'preflight_gpu/libero_spatial_4/NATIVE_RESET_FRAME.png',V/'RESET_DIAGNOSTIC'/'ORIGINAL_INVALID_CLOSED_DRAWER.png','invalid extraction reset, diagnostic only')
(V/'SOURCE_MANIFEST.json').write_text(json.dumps(receipts,indent=2)+'\n')
(V/'README.md').write_text('''# Visual task-breadth audit

There are **no qualified final task forms**. The overview presents repository-native candidate scenes, grouped by intended form. It is not evidence of successful insertion, extraction, or AF transfer. Three-frame sequences preserve the actual negative Fixed-5 native-reset screen at development root 5100 and friction 0.30; no images were generated or task outcomes reconstructed.

The bowl scene uses the source-specified open top drawer, verified before extraction VLA screening. The original incorrect closed-drawer reset is retained separately as diagnostic evidence. Book/rack constraints still require stronger geometric evaluator validation before they could count as genuine held-out forms. No arrow or composite depicts an unobserved successful motion.

`SOURCE_MANIFEST.json` gives original paths and SHA256 digests for every copied frame. The assembly script is `../build_visual_audit.py`.
''')
print(json.dumps({'preserved_frames':len(receipts),'final_qualified_forms':0}))
