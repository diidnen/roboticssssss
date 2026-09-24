"""Assemble source-backed recovery evidence; never trains or runs physics."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import e1_verified_inference as rt

ROOT=rt.ROOT
OUT=ROOT/'analysis/results/e1_verified_recovery_revision_20260905'
PREV=ROOT/'analysis/results/e1_pipeline_recovery_and_current_rerun_20260905'

def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,default=str)+'\n')

def main():
    parity=json.loads((OUT/'CHAINED_PARITY.json').read_text())
    d=pd.read_csv(OUT/'CHAINED_ALL_DECISIONS.csv');point=d[d.variant=='point'].copy()
    base=pd.read_csv(ROOT/'activeforcing_residual_utility_20260901_055605/POOLED_OOF_UTILITY_DATASET.csv')
    minimum=base.groupby(['context_id','repeat']).force_N.min()
    point['at_lowest_candidate']=[bool(abs(r.selected_setpoint-minimum.loc[(r.context_id,r.repeat)])<1e-10) for r in point.itertuples()]
    point['friction_group']=point.context_id.str.extract(r'_(low|mid|high)_')[0]
    agg={'parity':parity,'mean_selected_setpoint':point.selected_setpoint.mean(),
         'lowest_available_candidate_rate':point.at_lowest_candidate.mean(),
         'lowest_available_candidate_count':int(point.at_lowest_candidate.sum()),
         'selected_distribution':point.selected_setpoint.describe().to_dict(),
         'task_friction_means':point.groupby(['task','friction_group']).selected_setpoint.mean().rename('mean').reset_index().to_dict('records'),
         'candidate_domain':'five saved stratified continuous candidates per context/repeat; not uniform fine grid'}
    dump('E1_AGGREGATE_REPLAY.json',agg)
    # Ten distinct golden contexts spanning tasks, friction groups and forces.
    g=point.sort_values('selected_setpoint').drop_duplicates('context_id')
    ids=np.linspace(0,len(g)-1,10).round().astype(int)
    golden=g.iloc[ids].copy()
    probe=pd.read_csv(ROOT/'activeforcing_probe_conditioned_wm_20260901_064627/POOLED_OOF_PROBE_PREDICTIONS.csv')
    p0=probe[probe.seed.astype(str)=='0'].set_index('context_id')
    golden['historical_probe_source']=[p0.loc[c,'raw_path'] for c in golden.context_id]
    golden.to_csv(OUT/'E1_GOLDEN_INFERENCE_CASES.csv',index=False)
    point.to_csv(OUT/'E1_EXACT_INFERENCE_REPLAY.csv',index=False)
    provenance=json.loads((PREV/'E1_PIPELINE_PROVENANCE.json').read_text())
    provenance.update(E1_label_definition='Archived prospective full-task terminal success; task1 includes terminal-height reconstruction, NOT current lift_hold_success_y',
        posterior_construction='Primary E1: average 3 OOF mu predictions then one condition. Separate Posterior ablation: average probabilities at 3 member mu supports.',
        probe_normalization='Raw46 -> each OOF checkpoint train-fitted mean/std exactly once. Transferred NORMALIZATION_46D supplies schema only.',
        status='CHAINED_RAW_PROBE_REPLAY_PASS', verified_runtime=str(ROOT/'e1_verified_inference.py'))
    dump('E1_PIPELINE_PROVENANCE.json',provenance)
    eligibility={'E1_USES_BUGGY_58_ROW_LABELS':'NO','E1_LABELS_CLEAN':'PARTIAL: separate task1 terminal reconstruction caveat',
        'E1_POSTERIOR_CONDITIONED':'NO for primary; YES for separate Posterior ablation',
        'E1_CONTINUOUS_FORCE_INPUT':'YES','E1_METHOD_COMPATIBLE':'PARTIAL','E1_ELIGIBLE_FOR_FINAL_METHOD':'PARTIAL',
        'FINAL_MODEL_PROVENANCE_ACCEPTABLE':False,
        'reason':'E1 main point estimate is not final uncertainty-marginalized method; posterior ablation lacks verified transfer to current probes. Do not deploy an OOF fold merely because historical replay passes.',
        'final_feasibility_retrain_required':'NOT_ESTABLISHED; no training authorized or performed',
        'clean_alternatives':['final_probe_continuous_posterior_rebuild_20260904 (lift+hold)', 'final_fulltask_posterior_feasibility_20260905 (existing clean-label candidate; manifest explicitly DO_NOT_DEPLOY; checkpoint hashes independently verified)']}
    dump('E1_MODEL_ELIGIBILITY.json',eligibility)
    diff={'physical_belief':'E1 grouped-root OOF ProbeGRU (3 per held fold, 9 total) versus E6/E7 locked handoff members',
        'belief_preprocessing':'E1 per-checkpoint mean/std versus current transferred common normalization; different checkpoints, not automatically double normalization in current smoke',
        'feasibility':'E1 prospective Direct OOF full-task versus old720 clean lift+hold full-population model',
        'feature_contract':'E1 strict preprobe step190 masked force state versus clean model branch-start/post-probe measured state',
        'candidate_normalization':'F/8 in both; not the identified regression',
        'marginalization':'primary E1 point vs current probability average across member mus',
        'utility':'unchanged intended formula; E1 tie is atol1e-12 low-force',
        'domain':'historical five saved stratified values vs task0 fine 3..5 by .01',
        'first_observed_input_difference':'raw probe baseline tangential force and outward termination differ before learned inference',
        'first_causal_difference':'No single uniquely identified upstream cause. Matched weights-only ablation establishes feasibility weights are sufficient to change selections on fixed diagnostic inputs; collector source is identical.',
        'checkpoint_only_ablation':'same E1 normalization (identical across seeds verified), context, members, candidates, utility; only weights change. E1 variable, clean all3.',
        'existing_adapter_repairs':'activeforcing_feasibility_features.py already corrects phases, per-finger forces, joints and missing measurement masks; preserved, not overwritten',
        'raw_input_transfer_gate':'current206/1 outward vs historical215/10 outward; E1 members include nonpositive mu'}
    dump('E1_VS_CURRENT_PIPELINE_DIFF.json',diff)
    cur=pd.read_csv(OUT/'CURRENT_DIAGNOSTIC_SELECTIONS.csv')
    cur.to_csv(OUT/'E1_VS_CURRENT_CHECKPOINT_ONLY_ABLATION.csv',index=False)
    dump('CURRENT_TASK0_E1_STYLE_SELECTION.json',{'status':'DIAGNOSTIC_ONLY_NOT_DEPLOYABLE','results':cur.to_dict('records')})
    # Verify complete historical probe support, not just the three task0 cases.
    archive=Path('/media/volume/newdata/exouser/ACTIVEFORCING_DISK_ARCHIVE_20260902/Tabero/analysis/results/gnp_style_visual_context_prospective_20260831_011000')
    evidence=[]
    for cid,r in p0.iterrows():
        path=Path(str(r.raw_path).replace('/home/exouser/Tabero/analysis/results/gnp_style_visual_context_prospective_20260831_011000',str(archive)))
        raw=pd.read_csv(path)
        evidence.append({'context':cid,'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':len(raw),'outward_rows':int((raw.probe_phase=='probe_out').sum())})
    pd.DataFrame(evidence).to_csv(OUT/'HISTORICAL_PROBE_SUPPORT.csv',index=False)
    emap={r['context']:r for r in evidence}
    golden['resolved_probe_path']=[emap[c]['path'] for c in golden.context_id]
    golden['resolved_probe_sha256']=[emap[c]['sha256'] for c in golden.context_id]
    golden.to_csv(OUT/'E1_GOLDEN_INFERENCE_CASES.csv',index=False)
    assert all(r['rows']==215 and r['outward_rows']==10 for r in evidence)
    p4=Path('/home/exouser/Tabero/analysis/results/p4_contact_conditioned_probe_20260822_184213/scripts/p4_collect_probe.py')
    archived=archive.parent/'p4_contact_conditioned_probe_20260822_184213/scripts/p4_collect_probe.py'
    assert p4.read_bytes()==archived.read_bytes()
    dump('PROBE_SOURCE_PARITY.json',{'current':str(p4),'archive':str(archived),'identical':True,'sha256':hashlib.sha256(p4.read_bytes()).hexdigest(),'historical_context_count':len(evidence),'cap':.08})
    paths=list(rt.PROBE.glob('*.pt'))+list(rt.DIRECT.glob('*.pt'))+[ROOT/n for n in ['e1_verified_inference.py','e1_saved_probe_inference.py','e1_recovery_verified_audit.py','activeforcing_feasibility_features.py','test_e1_verified_inference.py']]
    dump('FINAL_INFERENCE_RUNTIME_MANIFEST.json',{'status':'NOT_APPROVED_FOR_PHYSICAL_RUN','historical_parity':True,'current_support_gate':False,'controller_changed':False,'utility_changed':False,'probe_changed':False,'training_run':False,'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}})
    bug=ROOT/'analysis/results/feasibility_runtime_bugfix_reproduction_20260905'
    separate=pd.read_csv(bug/'HISTORICAL_REPRODUCTION.csv')
    separate[separate.model=='FROZEN_DIRECT'].to_csv(OUT/'SEPARATE_325_375_461_REPLAY_SOURCES.csv',index=False)
    report=f'''# E1 历史推理恢复：核验版

## 结论

历史结果真实存在，而且已经从原始 probe 端到端复现；当前 pipeline 尚未恢复到可物理部署状态。
E1 主结果 144/144、Posterior 对照 144/144 选点完全一致，最大选点误差 0。
raw probe 成员误差上限 {parity['raw_probe_member_max_error']:.12g}；选中候选概率误差上限 {parity['selected_probability_max_error']:.12g}。
主结果 mean setpoint={parity['point_mean']:.15f}，Posterior mean={parity['posterior_mean']:.15f}。
这是离线推理复现，不是重新获得144次物理成功。历史 SR=0.9305555556，原 E1 结果见 provenance 中 E1_report。

## 原样恢复了什么

9 个 OOF probe checkpoint（每个 held fold 三成员），9 个 OOF Direct checkpoint；原始46维 probe 单次 checkpoint normalization；FeasibilityOnly 17+54 特征；F/8；原候选集；Expected Utility 及低力 tie-break。
历史 E1 主结果使用三成员 mu 的均值，不等同于对 posterior 积分。单独的 Posterior ablation 也已从原始输入复现。
源码运行 commit 未保存，不能把今天 HEAD 当历史 commit。绝对路径与 SHA256 已锁定在 provenance/manifest。
72 contexts 的所有 probe 均为215行、向外10步，逐文件SHA与行数见 HISTORICAL_PROBE_SUPPORT.csv。

## 历史 adaptive behavior

{point.groupby(['task','friction_group']).selected_setpoint.mean().to_string()}

最低可用候选占比={agg['lowest_available_candidate_rate']:.8f}。这里比较各 episode 自己的最小候选，不能用 F<=3 冒充 lower-bound rate。
10个 golden contexts 与全部288个决策/1440条曲线均有 CSV。
3.25、3.75、4.61不是这张E1表中的三个精确值：另一路已有复现分别来自 E5 physical-probe point、E5 no-query prior、root7703 no-probe prior。
它们的原 JSON 与概率误差见 SEPARATE_325_375_461_REPLAY_SOURCES.csv；不能把 prior case 包装成 physical-probe 成果。

## 当前问题及已实施修复

已新增 CPU-only E1 runtime，严格区分 checkpoint family、point/posterior、preprobe/postprobe feature contract；未知模型名不再静默降级，非法概率拒绝选点。
已新增 saved-probe 接口；短 probe 或非正/非有限 mu 不给物理选点。6项单元测试通过，真实 current-high refusal 测试以exit2正确拒绝。
工作区已有 clean-runtime feature 修复（phase、单指力、带符号关节、速度与mask），本轮保留并阅读其720行 feature parity 测试，不覆盖。
旧脚本中的state bug与E1 preprobe约定不是一回事；不能把E1 masked-state特征塞给clean模型就叫正式修复。这里该组合仅作受控消融。

## 当前 trace 证明了什么

P4 collector 当前与归档字节完全一致，rho=norm(sum finger xy)/(2*min(abs(finger z)))，cap=.08。
current high/mid/low 首向外步 rho约.356034/.347090/.317510，历史约.025；当前只向外1步即shear_ratio_cap，206行。
偏移在probe前hold已存在，不是 selector 引起。上游为什么出现更大切向力仍需核验传感坐标/接触运行状态，不能凭现有结果定为某一控制bug。
E1 current-high成员为[1.78772,-.00293,.13287]。短序列本身不是数学上不能跑GRU，但不在已验证历史支持内，且产生非物理mu，不能作为可信posterior。

## 当前离线选点（诊断，不是部署）

使用当前保存的command prefix及E1严格preprobe特征：point high/mid/low=4.02/3.99/4.00；posterior=5.00/3.50/5.00。
两次已有repeat一致。E1不全选3，但没有恢复可信的低摩擦高setpoint排序。
冻结所有输入、normalization、utility后，仅换clean权重全选3.00。说明权重变化足以造成这组决策变化，不代表它是唯一原因。
已有 NEW_FULLTASK + 修复后 current feature 结果也仍全选3；不能只因名字是fulltask就宣称修好了。

## 模型合法性

E1不是58-row缓存label bug那套数据；另有task1终点高度重构的140行需单独保留溯源。
主E1为point、候选为五个归档值；不能直接等同当前最终posterior连续选择方法。
已有clean lift+hold与另一个fulltask checkpoint可继续审计；本轮没有训练，不武断宣布必须重训。

## 物理门禁与下一步

E1 parity通过；current posterior可靠性与最终model资格未通过；因此18分支不启动。用户已有条件授权，阻碍是证据门禁，不是缺少许可。
下一步：核验现有fulltask模型的来源及raw-probe-transfer失配，尽量用现有保存信号完成可证实修复。不要调utility、补造probe步骤或用GT替换posterior。

## 对上一轮审计的更正

先前4.07/4.52/4.19只是另一套输入/后验诊断，不能宣称adaptivity recovered。
此前所谓checkpoint-only同时换normalization，本版新增固定normalization的权重消融。
此前probe读档失败归因archive mismatch不成立：审计脚本曾double-normalize；这不是current E6 normalization自动有错的证据。
本版把fresh raw probe成员实际串入feasibility，不再分别验证后把saved mu当推理输入。
以上更正另存此目录，未覆盖历史报告。
'''
    (OUT/'E1_PIPELINE_RECOVERY_REPORT.md').write_text(report)
    if (OUT/'CURRENT_CORRECTED_CLEAN_MODEL_COMPARISON.csv').exists():
        cc=pd.read_csv(OUT/'CURRENT_CORRECTED_CLEAN_MODEL_COMPARISON.csv')
        with (OUT/'E1_PIPELINE_RECOVERY_REPORT.md').open('a') as f:
            f.write('\n## 补充：本轮独立复核已修复 current 接口\n\n')
            f.write('使用每个repeat自己的raw probe与matched snapshot，保持当前E6 physical belief不变。两个现有clean模型都全选3，但概率不再为先前的约.999。不能继续引用旧的饱和值描述修复后接口。\n\n')
            f.write(cc[cc['repeat']==1][['context','model','p_at_3','p_at_5','selected_setpoint']].to_string(index=False)+'\n')
            f.write('\nfulltask checkpoint三个SHA已与原manifest逐一核对；原manifest明确CANDIDATE_ONLY_DO_NOT_DEPLOY。无训练、无新physics。\n')
    if (OUT/'ADDITIONAL_CHAINED_REPLAY.csv').exists():
        extra=pd.read_csv(OUT/'ADDITIONAL_CHAINED_REPLAY.csv')
        with (OUT/'E1_PIPELINE_RECOVERY_REPORT.md').open('a') as f:
            f.write('\n## 补充：3.25 / 3.75 / 4.61本轮独立推理复现\n\n')
            f.write('不只引用已有报告。本轮重新加载实际FROZEN_DIRECT模型与normalization；3.25重新从raw probe经历史单模型FRICTION_GRU得到mu，mu误差0，然后接入Direct。其他两个为原prior，未冒充probe结果。\n\n')
            f.write(extra[['case','selected_setpoint','max_probability_error','raw_probe_mu_error']].to_string(index=False)+'\n')
    print(json.dumps(agg,indent=2))

if __name__=='__main__':main()
