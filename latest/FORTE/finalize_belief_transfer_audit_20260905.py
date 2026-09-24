"""Persist evidence and test results for the current transfer diagnosis."""
import difflib
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path('/home/exouser/FORTE')
OUT=ROOT/'analysis/results/physical_belief_transfer_root_cause_20260905'

def main():
    import e1_verified_inference as rt
    parity=json.loads((OUT/'E6_REPLAY_PARITY.json').read_text())
    raw=pd.read_csv(OUT/'E6_PROBE_SUPPORT.csv')
    current=pd.read_csv(OUT/'CURRENT_E6_TRANSFER.csv')
    control=pd.read_csv(OUT/'MU_POSITIVE_CONTROL.csv')
    support=pd.read_csv(OUT/'CORRECTED_CURRENT_STATE_SUPPORT.csv')
    applied=[]
    for band,mu in [('high','0.940189'),('mid','0.450580'),('low','0.293710')]:
        cid=f'p5s0c_train_t0_r00_s5100_{band}_mu{mu}'
        folder=ROOT/'analysis/results/activeforcing_e2e_task0_smoke_20260905/PROBE_TELEMETRY'
        source=folder/f'{cid}_probe_result.json'
        if source.exists():
            q=json.loads(source.read_text())
            error=float(np.max(abs(rt.raw_features(folder/f'{cid}.csv')-rt.raw_features(folder/f'{cid}_repeat1.csv'))))
            applied.append({'context_id':cid,'source':str(source),'object_static_friction_readback':q['probe_record']['friction_applied'],'base_vs_repeat1_raw46_max_error':error,'scope':'base readback; repeat raw features identical; not full effective contact-pair friction'})
        else:
            applied.append({'context_id':cid,'source':'MISSING_BASE_PROBE_RESULT','scope':'repeat selector bundles do not contain material readback'})
    pd.DataFrame(applied).to_csv(OUT/'APPLIED_FRICTION_READBACK_AUDIT.csv',index=False)
    old=ROOT/'analysis/results/final_fulltask_posterior_feasibility_20260905/TRAINING_FEATURES.npz'
    data=np.load(old);labels=pd.read_csv(ROOT/'analysis/results/final_probe_continuous_posterior_rebuild_20260904/FULL_TASK_LABEL_DATASET.csv')
    np.testing.assert_array_equal(data['branch_id'],labels.branch_id)
    row0=[]
    for r in labels.itertuples():
        line=pd.read_csv(r.trace_path,nrows=1).iloc[0]
        row0.append({'context_id':r.context_id,'branch_id':r.branch_id,'task':r.task,'candidate_setpoint':r.requested_force_N,
            'first_record_step':int(line.step),'first_record_phase':line.phase,'left_normal_N':float(line.left_normal_force_N),
            'right_normal_N':float(line.right_normal_force_N),'source_trace':r.trace_path})
    table=pd.DataFrame(row0)
    np.testing.assert_allclose(table.left_normal_N,data['x'][:,0,25],atol=1e-5,rtol=0)
    np.testing.assert_allclose(table.right_normal_N,data['x'][:,0,26],atol=1e-5,rtol=0)
    table.to_csv(OUT/'TRAIN_FEATURE_OBSERVATION_TIME_AUDIT.csv',index=False)
    t=subprocess.run([sys.executable,'-m','unittest','test_e1_verified_inference','test_activeforcing_belief_contract','-v'],cwd=ROOT,capture_output=True,text=True)
    assert t.returncode==0,t.stderr
    (OUT/'TEST_RESULTS.txt').write_text(t.stdout+t.stderr)
    before=(OUT/'activeforcing_e2e_task0_smoke.before_belief_guard.py').read_text()
    after=(ROOT/'activeforcing_e2e_task0_smoke_20260905.py').read_text()
    (OUT/'BELIEF_GUARD_CHANGE.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='before',tofile='after')))
    paths=[ROOT/'activeforcing_e2e_task0_smoke_20260905.py',ROOT/'activeforcing_belief_contract.py',ROOT/'test_activeforcing_belief_contract.py',
           Path('/home/exouser/Tabero/analysis/results/p4_contact_conditioned_probe_20260822_184213/scripts/p4_collect_probe.py'),
           Path('/home/exouser/Tabero/source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/force_position_action.py')]
    result={'status':'E6_REPLAY_PASS_CURRENT_TRANSFER_BLOCKED','E1_PARITY_REMAINS_PASS':True,
        'E6_raw_replay_contexts':96,'E6_replay_member_error':parity['member_mu_max_error'],'normalization_error':0,
        'current_input_support_verified':False,'current_context_sensitive_selection_recovered':False,'physical_rerun_allowed':False,
        'oracle_clean_models_also_select_all3':bool((control[(control.mu_source=='ORACLE_ANALYSIS_ONLY')&(control.model!='E1_OOF')].selected_setpoint==3).all()),
        'clean_model_training_condition_observation':'first branch row AFTER candidate action; not a saved predecision probe state',
        'training_rows_with_first_record_step1':int((table.first_record_step==1).sum()),
        'skill_influence':'Data-quality audit separated label cleanliness, input availability, train/runtime support and decision validity.',
        'repairs':['explicit epistemic versus total uncertainty metadata','actual selector rejects unsupported probe before writing selection','diagnostics require separate output directory'],
        'tests_passed':12,'controller_changed':False,'probe_changed':False,'utility_changed':False,'training_performed':False,'physics_performed':False,
        'next_required_authority':'If current physical transfer must be repaired, authorize narrowly instrumented probe-only diagnosis before the currently closed physical-selection gate, or supply already-saved equivalent telemetry.',
        'hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (OUT/'TRANSFER_DIAGNOSIS_SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    report='''# Current physical-belief transfer: source-backed diagnosis

## 结论

E1历史推理已经恢复。当前自适应仍未恢复，不能通过改utility、塞GT、补造probe步骤或修改历史artifact来宣布完成。
本轮96个E6 TRAIN/DEV原始probe全部重放；normalization误差0，成员mu误差约2.2e-16，总variance误差约9.7e-17。
这证明今天使用的E6 checkpoint/adapter能复现它自己的历史结果；不能继续把异常归因为加载错模型或double normalization。

## 输入信号发生了什么

同三个context：历史E6 mean high/mid/low约.9317/.5080/.3300，当前约.7126/.6709/.7046；精确值见CURRENT_E6_TRANSFER.csv。
96个历史probe均215行、向外10步；当前206行、向外1步。P4代码与archive完全一致，.08 shear-ratio cap未改。
当前预探测hold已有异常大的横向力，第一向外步rho=.356/.347/.318触发原门限。其上游物理原因尚未通过材料readback/接触观测验证。
不能把单纯序列长度差异定义成物理失败：这里拒绝的是未经验证的模型输入支持，不是P4-B自适应停止本身。

## 当前feasibility还存在另一个独立问题

显式oracle诊断用分析参考mu替换posterior，只用于定位、绝不输出为deployable decision：
E1 strict-preprobe模型 high/mid/low选4.03/4.09/4.66；当前clean lift+hold和现有clean fulltask模型仍全部选3.00。
因此belief修复不是使clean模型恢复variable selection的充分条件。Utility没有实现错误；当前概率曲线本身支持3.0。

从原始720条branch CSV追回训练feature：输入state取branch第1步记录，而该记录在执行candidate action之后。
task0训练left-normal约25.4811–34.5824N，right-normal约27.1007–35.3873N；当前post-probe对应值约1.39–2.56N。
旧isolated runner的branch handoff初始化d_pred=D_CLOSED，随后env.step才观测并记录。代码路径：
`/home/exouser/Tabero_old720_exact_80ab/analysis/p5s0c_paired_boundary_probe_value.py` lines431/448/498。
现有fulltask refit代码明确使用branch row zero。标签clean并不能证明这些feature在决策前可获得，也不能证明其分布适用于当前执行。
这不是说25–35N等于selected setpoint，也不是否定old720的setpoint ordering历史证据；它限制的是这些模型向当前推理的合法转接。
本轮没有改训练数据、没有重训。是否必须重训尚未确立；不能仅凭当前三点皆3就要求任意改模型。

## 已修复的工程问题

smoke旧std只是member均值的sample SD。现保留std兼容字段并明确命名，额外恢复历史sigma-head aleatoric + epistemic + TRAIN scale的total SD。
不改变member均值、posterior候选support、Expected Utility或force domain。Total SD仅纠正报告语义，没有偷偷换积分算法。
实际infer_curves默认拒绝未经验证的probe输入；明确诊断模式必须写入独立目录，不得覆盖历史smoke结果。
12项测试通过，包括旧数值完全不变、实际entrypoint拒绝current短probe、历史支持样本可进入输入检查但不自动获得最终部署资格。
已有phase/per-finger-force/signed-joint/mask修复保留；未改controller、probe、checkpoint。

## 仍缺什么

检查的repeat selector bundle没有保留probe_record。另在high/low早期base probe_result.json追回object material静摩擦readback：high=.9401893615722656，low=.29371020197868347；P4确实在set_material_properties后调用get_material_properties。不能说所有applied-mu证据都丢失，也不能因此宣称完整接触对的effective friction已验证。mid对应base结果缺失。
已保存的三份probe不能补出未执行的9步剪切，也不能通过软件重放证明当前physics会产生训练支持内的信号。
进一步修复物理转接需要额外已保存的可审计信号，或获得probe-only仪表化诊断授权；现有门禁禁止在这些条件未通过时启动18分支。

## 文件

- E6_REPLAY_PARITY.json / E6_HISTORICAL_RAW_REPLAY.csv：原始输入与模型parity。
- E6_PROBE_SUPPORT.csv / CURRENT_RAW_FEATURE_SUPPORT.csv：96个样本支持及当前异常字段。
- MU_POSITIVE_CONTROL.csv / MU_POSITIVE_CONTROL_CURVES.csv：oracle/共同mu/current posterior对照，全部不可部署。
- TRAIN_FEATURE_OBSERVATION_TIME_AUDIT.csv：720行原trace首步与训练tensor核验。
- BELIEF_GUARD_CHANGE.patch / TEST_RESULTS.txt：最小接口修复与测试。
'''
    (OUT/'PHYSICAL_BELIEF_TRANSFER_REPORT.md').write_text(report)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
