"""Independent outcome/temporal audit of the bounded repaired placement pilot.

Source artifacts are read-only. Writes only derived audits in this run's output.
No training, old720 labels, posterior update, selector or physics launch.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import numpy as np
import torch
import run_placement_contract_repair_20260905 as run
from activeforcing_execution_snapshot import max_difference
from activeforcing_placement_contract import Geometry, CONFIG, PHASE_COUNTS, transform, rotation
from activeforcing_placement_stability_audit import audit_stability
from activeforcing_placement_label_v2 import evaluate as evaluate_v2

OUT=run.OUT
BANDS=('high','mid','low')


def read(path):return json.loads(Path(path).read_text())
def csvread(path):
    with Path(path).open() as stream:return list(csv.DictReader(stream))
def csvwrite(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def direct_ray_surface(geometry,xy):
    """Independent analytic vertical triangle intersections, no raster lookup."""
    t=geometry.data['basket_vertices'][geometry.data['basket_triangles']]
    a=t[:,0];b=t[:,1]-a;c=t[:,2]-a;den=b[:,0]*c[:,1]-b[:,1]*c[:,0]
    valid=np.abs(den)>1e-15;out=[]
    for x,y in xy:
        u=np.divide((x-a[:,0])*c[:,1]-(y-a[:,1])*c[:,0],den,out=np.zeros_like(den),where=valid)
        v=np.divide(b[:,0]*(y-a[:,1])-b[:,1]*(x-a[:,0]),den,out=np.zeros_like(den),where=valid)
        hit=valid&(u>=0)&(v>=0)&(u+v<=1)
        z=a[:,2]+u*b[:,2]+v*c[:,2]
        out.append(float(z[hit].max()) if hit.any() else np.nan)
    return np.array(out)


def independently_label(trace,counts,z0,geometry):
    tail=trace[-20:]
    checks=[geometry.containment(t['object_pose_w'],t['basket_pose_w']) for t in tail]
    positions=np.array([c['object_center_basket_frame'] for c in checks])
    complete=counts==PHASE_COUNTS
    stable=bool(max(np.linalg.norm(t['object_velocity_w'][:3]) for t in tail)<=.02
        and max(np.linalg.norm(t['object_velocity_w'][3:]) for t in tail)<=.2
        and np.linalg.norm(np.ptp(positions,axis=0))<=.005)
    unheld=all(max(t['finger_object_contact_norms_N'])<=.05 for t in tail)
    place=int(complete and len(tail)==20 and all(c['inside'] for c in checks) and stable and unheld)
    lift=int(max(t['object_position_m'][2]-z0 for t in trace)>=.03)
    dropped=int(any(t['terminations']['object_1_dropped'] for t in trace))
    return (int(lift and place and not dropped),lift,place,dropped)


def audit(partial=False):
    protocol=run.protocol();geometry=Geometry(OUT/'ASSET_GEOMETRY.npz')
    v2_manifest=read(OUT/'LABEL_V2_AUDIT_MANIFEST.json')
    for path,value in v2_manifest['sources_sha256'].items():assert run.sha256(path)==value
    rows=[];checks=[];phase_rows=[];repeat_gates=[];motion_rows=[];velocity_sources=[];probe_info={};maxdiff=0.
    for band in BANDS:
        ref=run.reference_dir(band)
        probe_info[band]={'probe':read(ref/'PROBE_RESULT.json'),'posterior':read(ref/'POSTERIOR.json')}
        x=np.load(ref/'FROZEN_PREACTION_FEATURES.npy');assert x.shape==(9,8,71)
        for m in range(3):
            for i,f in enumerate((3.,4.,5.)):
                residual=np.delete(x[3*i+m]-x[m],17,axis=-1)
                np.testing.assert_array_equal(residual,np.zeros_like(residual))
                np.testing.assert_array_equal(x[3*i+m,:,17],np.full(8,f/8))
                maxdiff=max(maxdiff,float(np.abs(residual).max()))
        snapshot=torch.load(ref/'DECISION_STATE.pt',map_location='cpu',weights_only=False)
        assert snapshot['candidate_actions_already_executed']==0
        z0=float(snapshot['state']['rigid_object']['alphabet_soup_1']['root_pose'][0,2])
        b0=np.asarray(snapshot['state']['rigid_object']['basket_1']['root_pose'][0,:3])
        for force,repeat in ((3.,0),(4.,0),(5.,0),(4.,1)):
            job=run.job_dir('branch',band,force,repeat)
            if not (job/'RESULT.json').exists():
                if partial:continue
                raise ValueError('Incomplete pilot: '+str(job))
            r=read(job/'RESULT.json');trace=read(job/'BRANCH_TRACE.json')
            native=csvread(job/'NATIVE_BRANCH_TRACE_DIAGNOSTIC.csv')
            counts=dict(Counter(t['phase'] for t in native))
            assert r['passed'] and len(trace)==len(native)==350 and counts==PHASE_COUNTS
            assert read(job/'PREFIX_RECONSTRUCTION_GATE.json')['passed']
            assert read(job/'FEATURE_GATE.json')['passed']
            assert read(job/'DECISION_FEATURES.json')['no_candidate_executed_before_extraction']
            assert r['decision_snapshot_sha256']==run.sha256(ref/'DECISION_STATE.pt')
            np.testing.assert_array_equal(np.load(job/'FROZEN_PREACTION_FEATURES.npy'),x)
            labels=independently_label(trace,counts,z0,geometry)
            assert labels==tuple(r[k] for k in ('full_task_success_y','lift_success','place_success','dropped'))
            final=trace[-1];inside=geometry.containment(final['object_pose_w'],final['basket_pose_w'])
            motion=audit_stability(trace)
            final_snapshot=torch.load(job/'FINAL_SCENE_STATE.pt',map_location='cpu',weights_only=False)
            objects=final_snapshot['objects'];prefix='asset_data/alphabet_soup_1'
            timestamp=objects[prefix]['_sim_timestamp']
            cache_times={k:objects[prefix+'/'+k]['timestamp'] for k in ('_root_state_w','_root_link_pose_w','_root_com_vel_w')}
            direct_difference=float(max(np.max(np.abs(np.array(t['object_velocity'])-np.array(t['object_velocity_w']))) for t in trace))
            assert all(value==timestamp for value in cache_times.values()) and direct_difference==0
            velocity_sources.append({'context':band.upper(),'candidate_F':force,'repeat':repeat,
                'asset_sim_timestamp':timestamp,'cache_timestamps':cache_times,
                'direct_PhysX_vs_root_state_velocity_max_diff':direct_difference,
                'application_cache_stale':False,'unproven':'No claim that solver velocities are stale; sub-step velocity need not represent net 20Hz pose motion'})
            v2=evaluate_v2(trace,counts,geometry)
            v2_label=int(labels[1] and v2['place_success'] and not labels[3])
            run.write(job/'OFFLINE_LABEL_V2_AUDIT.json',v2)
            motion_rows.append({'context':band.upper(),'candidate_F':force,'repeat':repeat,**motion})
            points=(transform(geometry.object,final['object_pose_w'])-np.array(final['basket_pose_w'])[:3])@rotation(final['basket_pose_w'][3:])
            ray=direct_ray_surface(geometry,points[:,:2]);surface=geometry.top_surface(points[:,:2])
            finite=np.isfinite(ray)&np.isfinite(surface)
            ray_floor_pass=bool(np.isfinite(ray).all() and (points[:,2]-ray).min()>=-.003)
            # A positive must also pass direct mesh rays, independent of the
            # raster envelope. Negative results are not promoted by this check.
            if (labels[2] or v2['place_success']) and not ray_floor_pass:raise AssertionError('Raster-only positive not confirmed by analytic mesh rays')
            basket_shift=np.array(final['basket_pose_w'][:3])-b0
            before_release=[t for t,n in zip(trace,native) if n['phase'] not in ('release','settle')]
            row={'context':band.upper(),'candidate_F':force,'repeat':repeat,
                'full_task_success':labels[0],'lift_success':labels[1],'place_success':labels[2],'dropped':labels[3],
                'all_phases_completed':True,'steps':len(trace),
                'final_inside_geometric':inside['inside'],'final_outer_xy_margin_m':inside['outer_xy_margin_m'],
                'basket_xy_displacement_m':float(np.linalg.norm(basket_shift[:2])),
                'max_basket_contact_before_release_N':max(t['basket_contact_force_N'] for t in before_release),
                'mean_true_normal_per_finger_held250_N':float(np.mean([np.mean(t['normal_force_N']) for t in before_release])),
                'mean_object_filtered_local_normal_per_finger_held250_N':float(np.mean([float(n['F_obj_avg_per_finger_n']) for n in native if n['phase'] not in ('release','settle')])),
                'final_native_goal':final['terminations']['success'],'native_contact_proxy_label':r['native_runner_label'],
                'final_basket_contact_N':final['basket_contact_force_N'],
                'unheld':r['geometric_evaluation']['unheld'],'settled':r['geometric_evaluation']['settled'],
                'label_adjudication_pending':motion['instantaneous_velocity_only_rejection'],
                'offline_v2_full_task_success':v2_label,'offline_v2_place_success':v2['place_success'],
                'label_v2_is_post_start_revision':True,
                'geometric_completion_without_stability':int(labels[1] and not labels[3] and r['geometric_evaluation']['all_final_window_inside'] and r['geometric_evaluation']['unheld']),
                'independent_label_matches':True,'formal_training_admitted':False,
                'source':str(job/'RESULT.json'),'source_sha256':run.sha256(job/'RESULT.json')}
            rows.append(row)
            checks.append({'context':band,'candidate_F':force,'repeat':repeat,'all_checks_passed':True,
                'independent_label_matches':True,'analytic_ray_floor_pass':ray_floor_pass,
                'minimum_raster_minus_analytic_height_m':float((surface[finite]-ray[finite]).min()) if finite.any() else None,
                'final_geometry':inside,'reference_features_exact':True})
            for phase in PHASE_COUNTS:
                subset=[t for t,n in zip(trace,native) if n['phase']==phase]
                normals=np.array([t['normal_force_N'] for t in subset])
                phase_rows.append({'context':band.upper(),'candidate_F':force,'repeat':repeat,'phase':phase,
                    'mean_left_normal_N':float(normals[:,0].mean()),'mean_right_normal_N':float(normals[:,1].mean()),
                    'max_basket_contact_N':max(t['basket_contact_force_N'] for t in subset),
                    'max_basket_xy_displacement_m':float(max(np.linalg.norm(np.array(t['basket_pose_w'][:2])-b0[:2]) for t in subset)),
                    'max_object_linear_speed_m_s':float(max(np.linalg.norm(t['object_velocity_w'][:3]) for t in subset))})
            if repeat:
                a=read(run.job_dir('branch',band,force,0)/'BRANCH_TRACE.json')
                diff=max_difference(a,trace);assert diff==0
                assert read(job/'FULL_BRANCH_REPLAY_GATE.json')['passed']
                repeat_gates.append({'context':band,'candidate_F':force,'steps':350,'all_trace_fields_max_diff':diff,'passed':True})
    primary=[r for r in rows if r['repeat']==0]
    all_complete=len(rows)==12 and len(repeat_gates)==3
    boundary={b:len({r['full_task_success'] for r in primary if r['context']==b.upper()})==2 for b in BANDS}
    unresolved=[r for r in primary if r['label_adjudication_pending']]
    geometric_boundary={b:len({r['geometric_completion_without_stability'] for r in primary if r['context']==b.upper()})==2 for b in BANDS}
    v2_boundary={b:len({r['offline_v2_full_task_success'] for r in primary if r['context']==b.upper()})==2 for b in BANDS}
    minimum={b:min([r['candidate_F'] for r in primary if r['context']==b.upper() and r['full_task_success']] or [float('inf')]) for b in BANDS}
    csvwrite(OUT/'EMPIRICAL_FORCE_SUCCESS_TABLE.csv',primary)
    csvwrite(OUT/'EMPIRICAL_FORCE_SUCCESS_TABLE_V2.csv',[{
        'context':r['context'],'candidate_F':r['candidate_F'],'full_task_success':r['offline_v2_full_task_success'],
        'lift_success':r['lift_success'],'place_success':r['offline_v2_place_success'],'dropped':r['dropped'],
        'raw_frozen_v1_full_task_success':r['full_task_success'],'label_version':'GEOMETRIC_POSE_WINDOW_V2',
        'post_start_label_revision':True,'formal_training_admitted':False,
        'source':r['source'],'source_sha256':r['source_sha256']} for r in primary])
    csvwrite(OUT/'ALL_BRANCH_OUTCOMES.csv',rows)
    csvwrite(OUT/'PHASE_FORCE_DIAGNOSTICS.csv',phase_rows)
    run.write(OUT/'LABEL_AND_TEMPORAL_AUDIT.json',checks)
    run.write(OUT/'FULL_TASK_RECONSTRUCTION_PARITY.json',repeat_gates)
    run.write(OUT/'STABILITY_CONSISTENCY_AUDIT.json',{'method':'read-only post-run diagnostic; frozen thresholds/labels untouched',
        'source':str(Path(__file__).parent/'activeforcing_placement_stability_audit.py'),
        'source_sha256':run.sha256(Path(__file__).parent/'activeforcing_placement_stability_audit.py'),
        'cases':motion_rows,'unresolved_unique_cases':len(unresolved),'labels_promoted':0})
    velocity_source=Path('/media/volume/newdata/exouser/tabero/IsaacLab/source/isaaclab/isaaclab/assets/rigid_object/rigid_object_data.py')
    run.write(OUT/'VELOCITY_SOURCE_AUDIT.json',{'source':str(velocity_source),'sha256':run.sha256(velocity_source),
        'root_state_semantics':'actor-frame pose plus world-frame center-of-mass linear/angular velocity',
        'source_inspected_after_run_started':True,'cases':velocity_sources})
    starts=list(OUT.glob('branches/*/*/STARTED.json'))
    visual_path=OUT/'VISUAL_REVIEW.json';visual=read(visual_path) if visual_path.exists() else {'complete':False}
    geometry_audit_complete=bool(all_complete and visual.get('complete',False) and visual.get('agrees_with_containment',False))
    status={'FINAL_STATUS':'PARTIAL_REPAIRED_PLACEMENT_PILOT' if not all_complete else 'REPAIRED_PLACEMENT_PILOT_COMPLETE',
        'PROBE_CONTRACT_FIXED':True,'POSTERIOR_INPUT_CONTRACT_FIXED':True,
        'CURRENT_PROBE_STEPS_HIGH_MID_LOW':[probe_info[b]['probe']['total_steps'] for b in BANDS],
        'CURRENT_PROBE_OUTWARD_STEPS_HIGH_MID_LOW':[probe_info[b]['probe']['outward_steps'] for b in BANDS],
        'CURRENT_POSTERIOR_HIGH_MID_LOW':[probe_info[b]['posterior']['mean'] for b in BANDS],
        'DECISION_STATE_DEFINED':True,'TEMPORAL_FEATURE_PARITY':True,'CANDIDATE_PREACTION_STATE_EQUALITY':True,
        'MAX_FEATURE_DIFF':maxdiff,'EXECUTION_CONTRACT_PARITY':all_complete,
        'EXECUTION_PARITY_SCOPE':'fresh-process exact pre-action prefix plus 350-step same-force repeats, not literal hidden PhysX snapshot restoration',
        'DIRECT_SNAPSHOT_RESTORE_VALIDATED':False,'PRODUCTION_RUNTIME_SWITCHED':False,
        'PILOT_PHYSICS_EXECUTED':bool(starts),'PILOT_CONTEXTS':['HIGH','MID','LOW'],'PILOT_ROOTS':[5100],
        'PILOT_CANDIDATE_FORCES':[3.,4.,5.],'PHYSICAL_PREFIX_ATTEMPTS':len(starts),'COMPLETED_BRANCHES':len(rows),
        'UNIQUE_CONTEXT_FORCE_CELLS':len(primary),'SAME_FORCE_REPEAT_GATES':len(repeat_gates),
        'DOWNSTREAM_NOMINAL_PATH_CHANGED':True,'FORCE_CONTROLLER_CHANGED':False,'PROBE_CHANGED':False,
        'POSTERIOR_CHANGED':False,'UTILITY_CHANGED':False,'FORCE_SUPPORT_CHANGED':False,
        'GEOMETRIC_LABEL_IMPLEMENTED':True,'GEOMETRIC_LABEL_PILOT_VALIDATED':geometry_audit_complete,
        'FULL_TASK_LABEL_CONTRACT_VALIDATED':bool(geometry_audit_complete and not unresolved),
        'FORCE_BOUNDARY_EXISTS':bool(all_complete and not unresolved and any(boundary.values())),
        'FORCE_BOUNDARY_VERDICT':'NOT_ESTABLISHED_LABEL_ADJUDICATION_PENDING' if unresolved else 'FIXED_GRID_EMPIRICAL_ONLY',
        'RAW_FROZEN_LABEL_WITHIN_CONTEXT_BOUNDARY':boundary,
        'GEOMETRIC_COMPLETION_WITHIN_CONTEXT_BOUNDARY':geometric_boundary,
        'OFFLINE_LABEL_V2_WITHIN_CONTEXT_BOUNDARY':v2_boundary,
        'OFFLINE_LABEL_V2_POSITIVES':sum(r['offline_v2_full_task_success'] for r in primary),
        'OFFLINE_LABEL_V2_NEGATIVES':sum(1-r['offline_v2_full_task_success'] for r in primary),
        'OFFLINE_LABEL_V2_POST_START_REVISION':True,
        'OFFLINE_LABEL_V2_FORMAL_TRAINING_ADMITTED':False,
        'STABILITY_LABEL_ADJUDICATION_PENDING_CASES':[{'context':r['context'],'candidate_F':r['candidate_F']} for r in unresolved],
        'MINIMUM_OBSERVED_SUCCESSFUL_SETPOINT_RAW_V1':{b:(None if not np.isfinite(v) else v) for b,v in minimum.items()},
        'MINIMUM_OBSERVED_SUCCESSFUL_SETPOINT_OFFLINE_V2':{b:min([r['candidate_F'] for r in primary if r['context']==b.upper() and r['offline_v2_full_task_success']] or [None]) for b in BANDS},
        'CONTEXT_DEPENDENT_FORCE_BOUNDARY':bool(all_complete and not unresolved and any(boundary.values()) and len(set(minimum.values()))>1),
        'FORMAL_CONTINUOUS_POSTERIOR_VALIDATED':False,'READY_FOR_FULL_DATA_COLLECTION':False,
        'MODEL_TRAINED':False,'OLD720_LABELS_USED':False,'FORMAL_TRAINING_ROWS':0,
        'DROPPED_TERM_SEMANTICS':'object_1_dropped is object z below -0.05m, not every slip; geometric placement separately rejects table/outside landings',
        'REMAINING_BLOCKERS':(['Pilot not complete'] if not all_complete else [])+
            ([] if geometry_audit_complete else ['Final image/geometry adjudication not complete'])+
            (['Frozen instantaneous velocity stability check conflicts with nearly stationary 20Hz pose traces; do not treat these label failures as force boundary'] if unresolved else [])+
            (['No within-context force-success boundary under offline pose-window v2 in the fixed 3/4/5 pilot'] if all_complete and not any(v2_boundary.values()) else [])+
            ['Formal continuous posterior integration remains unvalidated; existing interface unchanged',
             'Only one task0 root tested; task1/task5/task6 not qualified'],
        'NOTE':'Empirical deterministic single-branch outcomes plus exact repeats are not estimated population success rates; no final checkpoint eligibility claim'}
    run.write(OUT/'FINAL_STATUS.json',status)
    run.write(OUT/'PILOT_PROVENANCE.json',{'protocol_sha256':run.sha256(OUT/'PROTOCOL.json'),
        'asset_geometry_sha256':run.sha256(OUT/'ASSET_GEOMETRY.npz'),
        'audit_builder':str(Path(__file__)),'audit_builder_sha256':run.sha256(Path(__file__)),
        'unique_preaction_contexts':len(BANDS),'result_sources':[{'path':r['source'],'sha256':r['source_sha256']} for r in rows],
        'formal_training_eligible':False,'snapshot_reconstruction_not_claimed_literal_restore':True,
        'frozen_config':CONFIG,'old_artifacts_modified':False})
    if all_complete:
        lines=['# 放置 contract 修复：限定 task0 pilot', '',
            '已将旧的低空搬运／固定旧篮子目标替换为几何间隙驱动的抬升、搬运、放置目标；没有修改原始 P5、force controller、probe、posterior、utility 或 force support。新路径只在独立 pilot adapter 中生效，尚未切换生产 runtime。', '',
            '## 实测结果（冻结v1标签，未事后改写）', '',
            '| Context | Setpoint | Full task | Lift | Place | Dropped |',
            '|---|---:|---:|---:|---:|---:|']
        lines += ['| '+ ' | '.join(str(r[k]) for k in ('context','candidate_F','full_task_success','lift_success','place_success','dropped'))+' |' for r in primary]
        lines += ['', '9 个独立 context-force 单元，外加 3 个 4N 精确重放检查；共执行 12 次真实 probe 前缀和 12 个完整 350-step branch。重放检查不是独立随机重复，不能据此估计总体成功率。', '',
            '## 修复与判据', '',
            '- 新路径：按实际 basket mesh、物体 authored Cube collision geometry、7 个夹爪刚体 collision bounds 计算高度。搬运间隙 30mm，释放前间隙 10mm；每个搬运/放置阶段刷新篮子位姿，并使被抓物体而非 EEF 原点对准篮子。',
            '- 各阶段时长及 inner force/action loop 未改，仍为 20/50/110/30/40/50/50，共350步。',
            '- 新 place 判据：完整阶段；最后20步内物体碰撞体采样面处于当前篮子坐标系的保守 mesh free-column 体积、低于篮沿、脱离双指、稳定。原 native success/contact 标签只作诊断。',
            '- 几何容差3mm、线速度上限0.02m/s、角速度上限0.2rad/s、相对位置范围5mm、双指接触上限0.05N；全部在 pilot 前冻结。保守离散几何不是任意形状的通用精确包含测试。',
            '- 三个已知旧篮外案例均被回归测试拒绝；独立解析三角面竖直射线复算验证正例没有仅由栅格算法制造。最终图像审计见 VISUAL_REVIEW.json。', '',
            '## 输入时序、执行和 provenance', '',
            '- 相同 context 的3/4/5候选在显式 F/8 条件外逐元素一致，MAX_FEATURE_DIFF=0。模型输入和posterior全部先于candidate执行冻结。',
            '- 12个前缀均与既有 reference 的 actions、58D evidence、exposed DECISION_STATE 完全一致；HIGH/MID/LOW 的4N 350步重复全字段差异为0。',
            '- 使用 fresh-process exact prefix reconstruction；没有把失败的 literal hidden-PhysX snapshot restore 写成通过。生产 runtime 没有自动迁移。',
            '- 本轮未重训任何模型，未混入 old720 或 reconstructed labels，正式训练行数0。完整哈希在 PROTOCOL.json、ASSET_GEOMETRY_PROVENANCE.json、PILOT_PROVENANCE.json。', '',
            '## Boundary 与下一步', '',
            '固定候选上的 within-context mixed outcome：`'+json.dumps(boundary)+'`。',
            '以上是冻结v1标签的数值差异，不等于合格物理边界。需要稳定性定义审核的单元：`'+json.dumps(status['STABILITY_LABEL_ADJUDICATION_PENDING_CASES'])+'`。其逐帧位姿几乎静止，但瞬时角速度字段超过原阈值；详见 STABILITY_CONSISTENCY_AUDIT.json。未改标签、未放宽阈值。',
            '只看几何入篮、释放、lift/drop的 mixed outcome：`'+json.dumps(geometric_boundary)+'`；该拆分只定位失败原因，不替代full-task标签。',
            '另外保存 GEOMETRIC_POSE_WINDOW_V2：相同几何、释放、20步窗口和数值阈值，仅以已验证时钟上的相邻位姿计算净运动。v2 mixed outcome：`'+json.dumps(v2_boundary)+'`；正/负计数：'+str(status['OFFLINE_LABEL_V2_POSITIVES'])+'/'+str(status['OFFLINE_LABEL_V2_NEGATIVES'])+'。见 EMPIRICAL_FORCE_SUCCESS_TABLE_V2.csv。',
            'v2在8个单元已完成后声明，是透明的事后label-semantic修复，不是整轮预注册结果；raw v1没有被覆盖，没有物理重跑来挑结果，没有正式训练入库。新规则及声明时点、哈希见 LABEL_V2_AUDIT_MANIFEST.json。',
            '最低已观测成功 setpoint：冻结v1（含争议门槛）`'+json.dumps(status['MINIMUM_OBSERVED_SUCCESSFUL_SETPOINT_RAW_V1'])+'`；离线v2 `'+json.dumps(status['MINIMUM_OBSERVED_SUCCESSFUL_SETPOINT_OFFLINE_V2'])+'`。这不是范围以下的真实物理阈值，也不是 planner F*；不可用v1门槛争议制造自适应结论。',
            '释放前最大物体—篮子接触：'+str(max(r['max_basket_contact_before_release_N'] for r in primary))+' N；最终篮子最大 XY 位移：'+str(max(r['basket_xy_displacement_m'] for r in primary))+' m。', '',
            '候选是 continuous force setpoint，不是 exact instantaneous Newton force。ALL_BRANCH_OUTCOMES.csv另列250步持物阶段的逐指true-normal平均及object-filtered局部normal平均，用于检查候选是否实际产生不同接触力。', '',
            'READY_FOR_FULL_DATA_COLLECTION=NO。当前限制：']
        lines += ['- '+b for b in status['REMAINING_BLOCKERS']]
        lines += ['', '如果固定3/4/5全成功：只能说明这三个task0 context中当前范围没跨过成功边界；不能用这种全正数据宣称恢复了adaptive selection，也不能通过改utility/label/force范围制造边界。下一步需单独审查代表性边界覆盖和正式posterior接口；本轮到此停止。', '',
            '## 复算', '',
            '`/usr/bin/python3 /home/exouser/FORTE/audit_placement_contract_repair_20260905.py`', '',
            'READ_ONLY_AUDIT.ipynb 是不启动物理、不改源数据的可执行核验伴随文件。数据质量审计将接触代理标签与真正几何放置分开，并阻止未通过边界/接口门槛的pilot进入正式训练。', '']
        (OUT/'PLACEMENT_CONTRACT_REPAIR_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'status':status,'table':primary},indent=2))
    return status,rows


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');a=ap.parse_args()
    audit(a.partial)
