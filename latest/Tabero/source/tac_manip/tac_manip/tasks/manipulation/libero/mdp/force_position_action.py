from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

import isaaclab.utils.math as math_utils
import torch
from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.envs.mdp.actions.task_space_actions import (
    DifferentialInverseKinematicsAction,
)
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .observations import contact_force_in_gripper_frame
from .terminations import object_filtered_gripper_force_local

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer


@configclass
class ForcePositionActionCfg(ActionTermCfg):
    """混合力–位控制 Action 的配置（指令 13 维：6D EEF 位姿(轴角) + 1D 夹爪 + 左右指各 3D 目标力）。

    该 ActionTerm 本身不直接暴露给策略使用 IK 的细节，而是：

    - 内部包装一个 IsaacLab 自带的 `DifferentialInverseKinematicsAction`（只控制机械臂关节）；
    - 使用左右手指的 3D 目标力 + 传感器测得的当前力：
      - 合成对外的 EEF wrench，作为 OSC 的 `wrench_abs` 输入；
      - 计算挤压内力误差，用增量式公式更新夹爪开合度：
        `d_cmd = d_curr + squeeze_kp * (f_sq_curr - f_sq_target)`。

    对齐数据录制类 `AbsEEFPoseAxisAngleAbsGripperWithForceActionStateRecorder` 的约定，
    输入动作的 13 维含义为：

    - 0:6  ->  EEF 绝对位姿，基座坐标系下的 (x, y, z, ax, ay, az)，其中 (ax, ay, az) 为轴角
    - 6:7  ->  绝对 gripper 开合度（当前实现中不直接使用，夹爪由力闭环增量式控制）
    - 7:10 ->  左指目标力（指头局部坐标系，fx, fy, fz）
    - 10:13->  右指目标力（指头局部坐标系，fx, fy, fz）

    内部会把 (x, y, z, ax, ay, az) 转成 (x, y, z, qw, qx, qy, qz) 喂给 DiffIK 的绝对位姿命令，
    并根据左右指目标力与实际指尖力构造成混合位姿：

        P_pos_hybrid = P_pos_target + K_pos * (F_target^b - F_measured^b)
    """

    # 嵌套的 IK action 配置，由 env cfg 填好（asset_name/joint_names/body_name/offset 等）
    ik_cfg: DifferentialInverseKinematicsActionCfg = MISSING

    # 传感器和坐标系名称（需在 env cfg 里已经创建）
    ee_frame_name: str = "ee_frame"
    left_gripper_frame_name: str = "left_gripper_frame"
    right_gripper_frame_name: str = "right_gripper_frame"
    contact_sensor_name: str = "contact_gripper"
    history_length: int = 1

    # Optional JSON-configured fixed squeeze target.  Missing/disabled values
    # preserve the original policy-driven force behavior exactly.
    target_contact_squeeze_enabled: bool = False
    target_contact_sensor_name: str = ""
    target_contact_object_name: str = ""
    target_contact_single_finger_normal_force_n: float = 0.0
    target_contact_threshold_n: float = 0.0
    target_contact_activation: str = "first_contact_latched"

    # 增益
    # 位置混合增益：可以是标量，表示 xyz 共用一个 K，也可以是 (kx, ky, kz)
    pos_kp: float | tuple[float, float, float] = 0.0
    squeeze_kp: float = 0.001  # 夹爪开合度的力误差增益（squeeze 定义×2 后，在控制律中对误差做 0.5 缩放以保持等效幅度）
    squeeze_deadzone: float = 0.1  # 挤压力误差死区（等效）：按位置修正量 |Δd| 判断，阈值取 |squeeze_kp|*squeeze_deadzone（兼容旧配置）

    # 实测挤压力滤波（仅用于夹爪 squeeze 闭环，不影响 obs/数据录制，也不滤波左右指 3D 力）
    # EMA: s_filt = alpha * s_curr + (1-alpha) * s_filt_prev
    # - alpha=1.0: 不滤波（默认保持原行为）
    # - alpha 越小：滤波越强（响应越慢）
    meas_force_filter_alpha: float = 0.2

    # Optional FORTE authoritative object-filtered true-force inner loop.
    # Disabled by default so native Tabero behavior is unchanged.  When
    # enabled by the FORTE runtime bridge, F_des is supplied at the outer
    # environment rate while this action term updates the gripper target at
    # every physics application.
    authoritative_true_force_inner_loop_enabled: bool = False
    authoritative_true_force_sensor_name: str = ""
    authoritative_true_force_contact_threshold_n: float = 0.15
    authoritative_true_force_filter_alpha: float = 0.50
    authoritative_true_force_filter_reference_dt_s: float = 0.05
    authoritative_true_force_kp_m_per_n_s: float = 0.004
    # Dynamic root7703 needs to unload roughly 3.85 mm of contact compression
    # within roughly the lift/hold transition.  A 1.2 mm/s ceiling made the
    # otherwise correct 60 Hz feedback saturate for the entire hold, while a
    # 3.0 mm/s diagnostic crossed the contact boundary before the filtered
    # force settled.  The frozen 2.4 mm/s bound is the predeclared midpoint
    # diagnostic; the force-error gain itself remains unchanged and
    # dt-normalized.
    authoritative_true_force_open_rate_mps: float = 0.0024
    authoritative_true_force_close_rate_mps: float = 0.0030
    authoritative_true_force_deadzone_n: float = 0.25
    authoritative_true_force_raw_open_guard_enabled: bool = True
    # Optional actual-state velocity-resolved FORTE controller.  This is a
    # separate, default-off execution mode: the native Tabero path and the
    # earlier stateful-position FORTE diagnostics remain unchanged unless a
    # FORTE runner explicitly opts in.  The velocity gain is derived online
    # as 1 / (local force-aperture slope * settling horizon), so its units are
    # m / (N s) and the position target is always one physics step ahead of
    # the measured aperture rather than accumulated from an old command.
    authoritative_true_force_velocity_resolved_enabled: bool = False
    authoritative_true_force_velocity_slope_n_per_m: float = 9400.0
    authoritative_true_force_velocity_settling_time_s: float = 0.20
    authoritative_true_force_actual_velocity_filter_alpha: float = 0.50
    authoritative_true_force_velocity_contact_safety_filter_enabled: bool = False
    authoritative_true_force_velocity_friction_aux_guard_enabled: bool = True
    authoritative_true_force_velocity_weak_guard_width_n: float = 0.75
    # Final strict-Newton engineering attempt: preserve the loaded position
    # drive's virtual-equilibrium offset and regulate only that offset.  This
    # remains independently default-off and never changes native Tabero.
    authoritative_true_force_offset_servo_enabled: bool = False
    authoritative_true_force_offset_slope_n_per_m: float = 9400.0
    authoritative_true_force_offset_settling_time_s: float = 0.20
    authoritative_true_force_eq_offset_min_m: float = -0.0057
    authoritative_true_force_eq_offset_max_m: float = -0.0019
    authoritative_true_force_eq_release_rate_mps: float = 0.0024
    authoritative_true_force_eq_close_rate_mps: float = 0.0030
    authoritative_true_force_offset_contact_safety_filter_enabled: bool = False
    authoritative_true_force_offset_friction_aux_guard_enabled: bool = True
    # Optional FORTE dynamic release law.  This remains disabled by default
    # and affects only over-force opening.  Closing and contact reacquisition
    # retain the already validated physics-rate controller semantics.
    authoritative_true_force_adaptive_release_enabled: bool = False
    authoritative_true_force_release_slope_n_per_m: float = 9400.0
    authoritative_true_force_release_alpha: float = 0.20
    authoritative_true_force_opening_backlog_soft_start_m: float = 0.0012124786153435707
    authoritative_true_force_opening_backlog_limit_m: float = 0.0020110527984797955
    authoritative_true_force_contact_guard_width_n: float = 0.75
    # Optional predictive contact-safety supervisor for the FORTE-only
    # authoritative true-force loop.  It is independently feature flagged and
    # defaults to shadow-only so neither native Tabero nor the already frozen
    # release command changes without an explicit runner opt-in.
    authoritative_true_force_contact_safety_supervisor_enabled: bool = False
    authoritative_true_force_contact_safety_shadow_only: bool = True
    authoritative_true_force_mu_safe: float = 0.5726061820983886
    authoritative_true_force_rho_cautious: float = 0.70
    authoritative_true_force_rho_critical: float = 0.90
    authoritative_true_force_weak_prediction_horizon_s: float = 0.05
    authoritative_true_force_weak_derivative_steps: int = 3
    authoritative_true_force_warning_confirm_steps: int = 2
    authoritative_true_force_actuator_following_factor: float = 0.0023732437353127787
    authoritative_true_force_contact_preservation_reflex_enabled: bool = True
    authoritative_true_force_contact_preservation_reclose_rate_mps: float = 0.0006
    authoritative_true_force_contact_preservation_reclose_limit_m: float = 0.00005
    authoritative_true_force_contact_preservation_force_bound_n: float = 5.0
    authoritative_true_force_reacquire_step_limit_m: float = 0.00005
    authoritative_true_force_reacquire_cumulative_limit_m: float = 0.00015
    authoritative_true_force_safety_limit_n: float = 8.0

    # squeeze 前馈补偿（用于防滑/随“预测挤压力”增压）：把目标挤压力提升为
    #   f_sq_target_eff = f_sq_target + squeeze_ff_k_load_z * f_sq_target
    # 等价于：f_sq_target_eff = (1 + squeeze_ff_k_load_z) * f_sq_target
    # 默认系数为 0，不改变现有行为。
    squeeze_ff_k_load_z: float = 0.9
    squeeze_ff_contact_threshold: float = 1.0  # >0 时，仅当 f_sq_meas_raw >= threshold 才启用前馈

    # 由 manager 使用的 ActionTerm 类型（提供一个非 MISSING 的默认值，后续在模块末尾覆盖）
    class_type: type[ActionTerm] = ActionTerm


class ForcePositionAction(ActionTerm):
    """基于 OSC 的力–位混合控制 ActionTerm。

    输入 action: (N, 13)
        - 0:6  ->  EEF 绝对位姿 (x, y, z, ax, ay, az) 基座坐标系下，旋转为轴角
        - 6:7  ->  绝对 gripper 值（当前实现中仅对齐维度，不直接控制）
        - 7:10 ->  左指目标力（指头局部坐标系，fx, fy, fz）
        - 10:13->  右指目标力（指头局部坐标系，fx, fy, fz）

    作用：
    - 利用左右指目标力：
        - 合成对外 EEF wrench（在 base frame 下），喂给内部的 OSC (`wrench_abs`);
        - 通过当前测得的挤压力，增量式更新夹爪开合度：
          `d_cmd = d_curr + squeeze_kp * (f_sq_curr - f_sq_target)`。
    - EEF 位姿部分目前直接作为 OSC 的 `pose_abs` 目标（可在 cfg.eef_kp > 0 时加力反馈外环）。
    """

    cfg: ForcePositionActionCfg

    def __init__(self, cfg: ForcePositionActionCfg, env: ManagerBasedRLEnv) -> None:
        # 初始化基类（解析 asset_name -> robot articulation）
        super().__init__(cfg, env)

        self._env: ManagerBasedRLEnv = env
        self._device = env.device

        # 机器人
        self._robot: Articulation = self._asset

        # 内部 DiffIK ActionTerm（只控制机械臂关节）
        self._ik_term = DifferentialInverseKinematicsAction(cfg.ik_cfg, env)

        # 帧 & 传感器
        self._ee_frame: FrameTransformer = env.scene[cfg.ee_frame_name]
        self._left_frame: FrameTransformer = env.scene[cfg.left_gripper_frame_name]
        self._right_frame: FrameTransformer = env.scene[cfg.right_gripper_frame_name]
        # InteractiveScene 不实现 dict.get，用与 observations 相同的检查逻辑
        self._contact_sensor = (
            env.scene[cfg.contact_sensor_name]
            if cfg.contact_sensor_name in env.scene.keys()
            and env.scene[cfg.contact_sensor_name] is not None
            else None
        )
        self._target_contact_sensor = None
        if cfg.target_contact_squeeze_enabled:
            if cfg.target_contact_activation != "first_contact_latched":
                raise ValueError(
                    "[ForcePositionAction] unsupported target-contact activation "
                    f"mode: {cfg.target_contact_activation!r}."
                )
            if cfg.target_contact_single_finger_normal_force_n <= 0.0:
                raise ValueError(
                    "[ForcePositionAction] target-contact single-finger force "
                    "must be positive."
                )
            if cfg.target_contact_threshold_n < 0.0:
                raise ValueError(
                    "[ForcePositionAction] target-contact threshold must be "
                    "non-negative."
                )
            if not cfg.target_contact_sensor_name:
                raise ValueError(
                    "[ForcePositionAction] target-contact sensor name is required "
                    "when the squeeze override is enabled."
                )
            if cfg.target_contact_sensor_name not in env.scene.keys():
                raise RuntimeError(
                    "[ForcePositionAction] target-contact sensor "
                    f"{cfg.target_contact_sensor_name!r} is missing from the scene."
                )
            self._target_contact_sensor = env.scene[
                cfg.target_contact_sensor_name
            ]

        # 解析夹爪关节（平行夹爪，两个 finger）
        if not hasattr(env.cfg, "gripper_joint_names"):
            raise RuntimeError(
                "[ForcePositionAction] env.cfg 中缺少 gripper_joint_names，"
                "无法根据力误差更新夹爪开合度。"
            )
        self._gripper_joint_ids, self._gripper_joint_names = self._robot.find_joints(
            env.cfg.gripper_joint_names
        )
        if len(self._gripper_joint_ids) != 2:
            raise RuntimeError(
                f"[ForcePositionAction] 期望平行夹爪有 2 个 finger 关节，实际解析到 {len(self._gripper_joint_ids)} 个。"
            )

        # raw / processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self._device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # 缓存拆分后的目标量
        self._eef_pos_cmd = torch.zeros(self.num_envs, 3, device=self._device)
        self._eef_aa_cmd = torch.zeros(self.num_envs, 3, device=self._device)
        self._gripper_abs_cmd = torch.zeros(self.num_envs, 1, device=self._device)
        self._fL_target_local = torch.zeros(self.num_envs, 3, device=self._device)
        self._fR_target_local = torch.zeros(self.num_envs, 3, device=self._device)

        # 实测挤压力 EMA 滤波状态（标量）
        self._f_sq_meas_ema = torch.zeros(self.num_envs, device=self._device)
        self._f_sq_meas_ema_initialized = False

        # Per-environment state for the first-contact latch.  The step counter
        # advances in process_actions(), which is called once per environment
        # step (rather than once per decimated simulation step).
        self._target_contact_detected = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self._device
        )
        self._target_contact_latched = torch.zeros_like(
            self._target_contact_detected
        )
        self._target_contact_force_norm = torch.zeros(
            self.num_envs, device=self._device
        )
        self._target_contact_activation_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self._device
        )
        self._environment_step_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self._device
        )

        # 调试信息缓存（仅用于可视化，不影响控制逻辑）
        self._debug: dict[str, torch.Tensor] = {}
        self._last_d_cmd = torch.zeros(self.num_envs, device=self._device)

        # Runtime-only FORTE inner-loop state.  These fields are deliberately
        # separate from the native squeeze loop and remain inactive unless the
        # FORTE bridge explicitly enables them after the handoff.
        self._authoritative_true_force_enabled = False
        self._authoritative_true_force_target = torch.zeros(
            self.num_envs, device=self._device
        )
        self._authoritative_true_force_filter: torch.Tensor | None = None
        self._authoritative_true_force_d_cmd = torch.zeros(
            self.num_envs, device=self._device
        )
        self._authoritative_true_force_d_cmd_initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self._device
        )
        self._authoritative_true_force_loss_streak = torch.zeros(
            self.num_envs, dtype=torch.long, device=self._device
        )
        self._authoritative_true_force_recovery_streak = torch.zeros(
            self.num_envs, dtype=torch.long, device=self._device
        )
        self._authoritative_true_force_reacquire_accum = torch.zeros(
            self.num_envs, device=self._device
        )
        self._authoritative_true_force_release_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self._device
        )
        self._authoritative_true_force_release_command_reference = torch.zeros(
            self.num_envs, device=self._device
        )
        self._authoritative_true_force_release_actual_reference = torch.zeros(
            self.num_envs, device=self._device
        )
        self._authoritative_true_force_weak_history = torch.zeros(
            self.num_envs, 4, device=self._device
        )
        self._authoritative_true_force_weak_history_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self._device
        )
        self._authoritative_true_force_warning_streak = torch.zeros(
            self.num_envs, dtype=torch.long, device=self._device
        )
        self._authoritative_true_force_preservation_reclose_accum = torch.zeros(
            self.num_envs, device=self._device
        )
        self._authoritative_true_force_previous_actual_aperture = torch.zeros(
            self.num_envs, device=self._device
        )
        self._authoritative_true_force_actual_velocity_filter = torch.zeros(
            self.num_envs, device=self._device
        )
        self._authoritative_true_force_actual_velocity_initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self._device
        )
        self._authoritative_true_force_eq_offset = torch.zeros(
            self.num_envs, device=self._device
        )
        self._authoritative_true_force_eq_offset_initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self._device
        )
        self._authoritative_true_force_eq_offset_min = torch.zeros(
            self.num_envs, device=self._device
        )
        self._authoritative_true_force_eq_offset_max = torch.zeros(
            self.num_envs, device=self._device
        )
        self._authoritative_true_force_physics_trace: list[dict[str, Any]] = []

    # --------------------------------------------------------------------- #
    # Properties
    # --------------------------------------------------------------------- #

    @property
    def action_dim(self) -> int:
        # 6 (eef pose: pos+axis-angle) + 1 (gripper) + 3 (left force) + 3 (right force)
        return 13

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        # 这里只是简单缓存，实际控制逻辑在 apply_actions 中完成
        return self._processed_actions

    @property
    def debug_info(self) -> dict:
        """返回当前 step 的调试信息（env0），用于可视化."""
        if not self._debug:
            return {}
        out: dict[str, object] = {}
        for k, v in self._debug.items():
            if isinstance(v, torch.Tensor):
                # 只取 env 0，并搬到 CPU/numpy，方便 matplotlib 使用
                out[k] = v[0].detach().cpu().numpy()
            else:
                out[k] = v
        return out

    @property
    def last_d_cmd(self) -> torch.Tensor:
        """最近一次计算得到的夹爪目标开合度 d_cmd（每 env 一个标量）."""
        return self._last_d_cmd

    def reset(self, env_ids: Sequence[int] | slice | torch.Tensor | None = None) -> None:
        """Reset actions, filters, and target-contact latch state."""

        ids = slice(None) if env_ids is None else env_ids
        self._ik_term.reset(env_ids=ids)
        self._raw_actions[ids] = 0.0
        self._processed_actions[ids] = 0.0
        self._f_sq_meas_ema[ids] = 0.0
        # EMA initialization is currently shared by all environments; forcing
        # reinitialization is safe for partial resets and avoids stale force.
        self._f_sq_meas_ema_initialized = False
        self._target_contact_detected[ids] = False
        self._target_contact_latched[ids] = False
        self._target_contact_force_norm[ids] = 0.0
        self._target_contact_activation_step[ids] = -1
        self._environment_step_count[ids] = 0
        self._authoritative_true_force_d_cmd_initialized[ids] = False
        self._authoritative_true_force_loss_streak[ids] = 0
        self._authoritative_true_force_recovery_streak[ids] = 0
        self._authoritative_true_force_reacquire_accum[ids] = 0.0
        self._authoritative_true_force_release_active[ids] = False
        self._authoritative_true_force_release_command_reference[ids] = 0.0
        self._authoritative_true_force_release_actual_reference[ids] = 0.0
        self._authoritative_true_force_weak_history[ids] = 0.0
        self._authoritative_true_force_weak_history_count[ids] = 0
        self._authoritative_true_force_warning_streak[ids] = 0
        self._authoritative_true_force_preservation_reclose_accum[ids] = 0.0
        self._authoritative_true_force_previous_actual_aperture[ids] = 0.0
        self._authoritative_true_force_actual_velocity_filter[ids] = 0.0
        self._authoritative_true_force_actual_velocity_initialized[ids] = False
        self._authoritative_true_force_eq_offset[ids] = 0.0
        self._authoritative_true_force_eq_offset_initialized[ids] = False
        self._authoritative_true_force_eq_offset_min[ids] = 0.0
        self._authoritative_true_force_eq_offset_max[ids] = 0.0
        # A new episode must explicitly re-enter the FORTE phase at its
        # handoff.  Never carry a target, filtered measurement, or aperture
        # command across resets (including partial vectorized resets).
        self._authoritative_true_force_enabled = False
        self.cfg.authoritative_true_force_inner_loop_enabled = False
        self._authoritative_true_force_filter = None
        self._authoritative_true_force_physics_trace.clear()
        if self._target_contact_sensor is not None:
            self._target_contact_sensor.reset(ids)
        self._debug = {}

    def enable_authoritative_true_force_inner_loop(
        self,
        F_des: float | torch.Tensor,
        *,
        sensor_name: str,
        initial_aperture: torch.Tensor | float | None = None,
    ) -> None:
        """Enable the FORTE object-filtered true-force loop at physics rate.

        The outer runner owns the continuous physical target.  This method
        only installs that target and initializes a bounded aperture command;
        it never changes the arm command or native Tabero defaults.
        """

        if not sensor_name:
            raise ValueError("sensor_name is required for true-force inner loop")
        self._authoritative_true_force_enabled = True
        self.cfg.authoritative_true_force_inner_loop_enabled = True
        self.cfg.authoritative_true_force_sensor_name = str(sensor_name)
        target = torch.as_tensor(
            F_des, dtype=torch.float32, device=self._device
        ).reshape(-1)
        if target.numel() == 1:
            target = target.expand(self.num_envs)
        if target.numel() != self.num_envs:
            raise ValueError("F_des must be scalar or have one value per environment")
        self._authoritative_true_force_target[:] = target
        self._authoritative_true_force_filter = None
        self._authoritative_true_force_loss_streak.zero_()
        self._authoritative_true_force_recovery_streak.zero_()
        self._authoritative_true_force_reacquire_accum.zero_()
        self._authoritative_true_force_release_active.zero_()
        self._authoritative_true_force_weak_history.zero_()
        self._authoritative_true_force_weak_history_count.zero_()
        self._authoritative_true_force_warning_streak.zero_()
        self._authoritative_true_force_preservation_reclose_accum.zero_()
        actual_now = self._robot.data.joint_pos[
            :, self._gripper_joint_ids
        ].mean(dim=-1)
        self._authoritative_true_force_previous_actual_aperture[:] = actual_now
        self._authoritative_true_force_actual_velocity_filter.zero_()
        self._authoritative_true_force_actual_velocity_initialized[:] = True
        if initial_aperture is None:
            # Preserve the actuator's currently applied preload command at
            # handoff.  In contact, joint position can differ by millimetres
            # from its target; initializing from the physical position would
            # create an unintended large opening target jump before the first
            # bounded force correction is evaluated.
            self._authoritative_true_force_d_cmd[:] = self._last_d_cmd
        else:
            aperture = torch.as_tensor(
                initial_aperture, dtype=torch.float32, device=self._device
            ).reshape(-1)
            if aperture.numel() == 1:
                aperture = aperture.expand(self.num_envs)
            if aperture.numel() != self.num_envs:
                raise ValueError(
                    "initial_aperture must be scalar or have one value per environment"
                )
            self._authoritative_true_force_d_cmd[:] = aperture
        self._authoritative_true_force_d_cmd_initialized[:] = True
        initial_eq_offset = self._authoritative_true_force_d_cmd - actual_now
        configured_eq_min = float(
            self.cfg.authoritative_true_force_eq_offset_min_m
        )
        configured_eq_max = float(
            self.cfg.authoritative_true_force_eq_offset_max_m
        )
        if configured_eq_min >= configured_eq_max:
            raise ValueError("equilibrium offset min must be below max")
        self._authoritative_true_force_eq_offset[:] = initial_eq_offset
        self._authoritative_true_force_eq_offset_min[:] = torch.minimum(
            torch.full_like(initial_eq_offset, configured_eq_min),
            initial_eq_offset,
        )
        self._authoritative_true_force_eq_offset_max[:] = torch.maximum(
            torch.full_like(initial_eq_offset, configured_eq_max),
            initial_eq_offset,
        )
        self._authoritative_true_force_eq_offset_initialized[:] = True
        self._authoritative_true_force_release_command_reference[:] = (
            self._authoritative_true_force_d_cmd
        )
        self._authoritative_true_force_release_actual_reference[:] = (
            self._robot.data.joint_pos[:, self._gripper_joint_ids].mean(dim=-1)
        )
        self._authoritative_true_force_physics_trace.clear()

    def disable_authoritative_true_force_inner_loop(self) -> None:
        """Disable the optional FORTE inner loop and clear its transient state."""

        self._authoritative_true_force_enabled = False
        self.cfg.authoritative_true_force_inner_loop_enabled = False
        self._authoritative_true_force_filter = None
        self._authoritative_true_force_d_cmd_initialized.zero_()
        self._authoritative_true_force_loss_streak.zero_()
        self._authoritative_true_force_recovery_streak.zero_()
        self._authoritative_true_force_reacquire_accum.zero_()
        self._authoritative_true_force_release_active.zero_()
        self._authoritative_true_force_release_command_reference.zero_()
        self._authoritative_true_force_release_actual_reference.zero_()
        self._authoritative_true_force_weak_history.zero_()
        self._authoritative_true_force_weak_history_count.zero_()
        self._authoritative_true_force_warning_streak.zero_()
        self._authoritative_true_force_preservation_reclose_accum.zero_()
        self._authoritative_true_force_previous_actual_aperture.zero_()
        self._authoritative_true_force_actual_velocity_filter.zero_()
        self._authoritative_true_force_actual_velocity_initialized.zero_()
        self._authoritative_true_force_eq_offset.zero_()
        self._authoritative_true_force_eq_offset_initialized.zero_()
        self._authoritative_true_force_eq_offset_min.zero_()
        self._authoritative_true_force_eq_offset_max.zero_()

    def consume_authoritative_true_force_physics_trace(self) -> list[dict[str, Any]]:
        """Return and clear per-physics-step FORTE telemetry."""

        rows = self._authoritative_true_force_physics_trace
        self._authoritative_true_force_physics_trace = []
        return rows

    @property
    def authoritative_true_force_inner_loop_enabled(self) -> bool:
        return bool(self._authoritative_true_force_enabled)

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _split_squeeze_and_applied_from_lr_local(
        fL_local: torch.Tensor,
        fR_local: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """基于左右指局部系 3D 力，计算挤压力标量和「加持力」3D 向量（局部系）.

        约定（与 LIBERO Hybrid（force-position）/ 触觉环境保持一致）：
        - 左右指局部坐标系满足：
          - x 轴：均指向世界系的「上」方向；
          - y 轴：均指向世界系的「前」方向；
          - z 轴：均指向夹爪闭合方向（两指 +z 一致）。
        - 传感器测得的是「环境对指尖」的接触力。

        定义：
        - 挤压力标量（squeeze）：
              f_sq = 2 * min(|fL_z|, |fR_z|)
          物理含义：两指在挤压方向上成对出现的那一部分对向/同向力的公共模长。
        - 加持力（applied force）：在局部系下的 3D 向量 F_app = (Fx, Fy, Fz)，其中：
          - Fx, Fy：左右指对应分量直接相加（保持正负号）：
                Fx = fL_x + fR_x
                Fy = fL_y + fR_y
          - Fz：在 z 轴上减去成对出现的「挤压力」部分，仅保留不被两指相互抵消的剩余载荷：
                a = fL_z, b = fR_z
                common = min(|a|, |b|)
                Fz = a + b - common * (sign(a) + sign(b))
          这样：
          - 若为纯对向挤压（a ≈ -b），则 sign(a)+sign(b)≈0，Fz≈a+b≈0，仅通过 f_sq 体现挤压力；
          - 若为同向加载（a, b 同号），则去掉公共部分，仅保留差值，体现「多出来」的净载荷。
        """
        # 3D force components（finger 局部坐标系）
        fL_x, fL_y, fL_z = fL_local[:, 0], fL_local[:, 1], fL_local[:, 2]
        fR_x, fR_y, fR_z = fR_local[:, 0], fR_local[:, 1], fR_local[:, 2]

        # 挤压力：左右指 z 轴分量绝对值的较小者（乘 2 视为两指合计）
        abs_fL_z = torch.abs(fL_z)
        abs_fR_z = torch.abs(fR_z)
        squeeze = 2.0 * torch.minimum(abs_fL_z, abs_fR_z)  # (N,)

        # 加持力 x/y：直接相加
        Fx = fL_x + fR_x
        Fy = fL_y + fR_y

        # 加持力 z：减掉「成对出现」的挤压力部分，只保留不被抵消的剩余
        signL = torch.sign(fL_z)
        signR = torch.sign(fR_z)
        common = torch.minimum(abs_fL_z, abs_fR_z)
        Fz = fL_z + fR_z - common * (signL + signR)

        F_app_local = torch.stack([Fx, Fy, Fz], dim=-1)  # (N, 3)
        return squeeze, F_app_local

    # --------------------------------------------------------------------- #
    # Core logic
    # --------------------------------------------------------------------- #

    def process_actions(self, actions: torch.Tensor):
        """缓存原始 action，并拆分为 EEF 位姿(轴角)、夹爪和 finger 目标力。"""
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions

        if self.cfg.target_contact_squeeze_enabled:
            force_matrix_w = self._target_contact_sensor.data.force_matrix_w
            if force_matrix_w is None:
                raise RuntimeError(
                    "[ForcePositionAction] target-contact sensor has no "
                    "force_matrix_w; finger filter prims are required."
                )
            if force_matrix_w.ndim != 4 or force_matrix_w.shape[-1] != 3:
                raise RuntimeError(
                    "[ForcePositionAction] target-contact sensor must provide "
                    f"(N,B,M,3) force_matrix_w, got {tuple(force_matrix_w.shape)}."
                )
            if force_matrix_w.shape[0] != self.num_envs:
                raise RuntimeError(
                    "[ForcePositionAction] target-contact sensor environment "
                    f"count mismatch: expected {self.num_envs}, got "
                    f"{force_matrix_w.shape[0]}."
                )
            contact_norm = torch.linalg.vector_norm(force_matrix_w, dim=-1)
            contact_norm = contact_norm.amax(dim=(1, 2))
            if not torch.all(torch.isfinite(contact_norm)):
                raise RuntimeError(
                    "[ForcePositionAction] target-contact sensor produced "
                    "non-finite force values."
                )
            detected = contact_norm >= float(
                self.cfg.target_contact_threshold_n
            )
            # PhysX can expose the previous episode's final contact view until
            # one post-reset simulation step has completed.  The Task 0 reset
            # starts with the gripper away from the object, so arm the gate only
            # after that first refresh to prevent a stale step-0 latch.
            detected &= self._environment_step_count > 0
            newly_latched = detected & ~self._target_contact_latched
            self._target_contact_activation_step[newly_latched] = (
                self._environment_step_count[newly_latched]
            )
            self._target_contact_detected[:] = detected
            self._target_contact_latched.logical_or_(detected)
            self._target_contact_force_norm[:] = contact_norm

        self._environment_step_count.add_(1)

        # 0:3 -> 位置, 3:6 -> 轴角, 6:7 -> gripper, 7:10/10:13 -> 左/右指目标力
        self._eef_pos_cmd[:] = actions[:, 0:3]
        self._eef_aa_cmd[:] = actions[:, 3:6]
        self._gripper_abs_cmd[:] = actions[:, 6:7]
        self._fL_target_local[:] = actions[:, 7:10]
        self._fR_target_local[:] = actions[:, 10:13]

    def apply_actions(self):
        """在每个仿真步被调用：更新夹爪开合 + 调用内部 DiffIK。"""
        # ------------------------------
        # 1) finger 级当前力观测（局部系）
        # ------------------------------
        if self._contact_sensor is not None:
            # 使用已有的观测函数把世界系力转为 finger 局部系
            force_hist_local = contact_force_in_gripper_frame(
                self._env,
                contact_sensor_name=self.cfg.contact_sensor_name,
                history_length=self.cfg.history_length,
            )  # (N, H, 2, 3)
            # 取最近一帧
            force_curr_local = force_hist_local[:, -1, :, :]  # (N, 2, 3)
            fL_meas_local_raw = force_curr_local[:, 0, :]  # (N, 3)
            fR_meas_local_raw = force_curr_local[:, 1, :]  # (N, 3)
        else:
            fL_meas_local_raw = torch.zeros_like(self._fL_target_local)
            fR_meas_local_raw = torch.zeros_like(self._fR_target_local)

        # NOTE: 不对左右指 3D 力做滤波；只在后续对 squeeze 标量做滤波
        fL_meas_local = fL_meas_local_raw
        fR_meas_local = fR_meas_local_raw

        # ------------------------------
        # 2) 基于 finger 局部系计算挤压力与「加持力」
        # ------------------------------
        f_sq_meas_raw, F_app_meas_local = self._split_squeeze_and_applied_from_lr_local(fL_meas_local, fR_meas_local)
        f_sq_target, F_app_target_local = self._split_squeeze_and_applied_from_lr_local(
            self._fL_target_local, self._fR_target_local
        )

        # 可选：只对“实测挤压力标量”做 EMA，抑制 min() 切换导致的高频锯齿
        alpha = float(getattr(self.cfg, "meas_force_filter_alpha", 1.0))
        if 0.0 < alpha < 1.0:
            if not self._f_sq_meas_ema_initialized:
                self._f_sq_meas_ema[:] = f_sq_meas_raw
                self._f_sq_meas_ema_initialized = True
            else:
                self._f_sq_meas_ema.mul_(1.0 - alpha).add_(f_sq_meas_raw, alpha=alpha)
            f_sq_meas = self._f_sq_meas_ema
        else:
            f_sq_meas = f_sq_meas_raw

        # squeeze 目标前馈补偿（用于防滑/随负载增压）：基于 applied force 的 z 轴净载荷
        f_sq_target_eff = f_sq_target
        if self.cfg.squeeze_ff_k_load_z != 0.0:
            if self.cfg.squeeze_ff_contact_threshold > 0.0:
                enable_ff = f_sq_meas_raw >= float(self.cfg.squeeze_ff_contact_threshold)
            else:
                enable_ff = torch.ones_like(f_sq_meas_raw, dtype=torch.bool)
            ff = float(self.cfg.squeeze_ff_k_load_z) * torch.abs(f_sq_target)
            f_sq_target_eff = torch.where(enable_ff, f_sq_target + ff, f_sq_target)

        # Apply the fixed target after feed-forward compensation so a configured
        # 13 N per finger remains exactly 26 N in the controller's two-finger
        # squeeze convention.  The policy's x/y and net applied-force targets
        # remain untouched.
        if self.cfg.target_contact_squeeze_enabled:
            override_target = torch.full_like(
                f_sq_target_eff,
                2.0
                * float(
                    self.cfg.target_contact_single_finger_normal_force_n
                ),
            )
            f_sq_target_eff = torch.where(
                self._target_contact_latched,
                override_target,
                f_sq_target_eff,
            )

        # ------------------------------
        # 3) 「加持力」从 finger 局部系 -> 世界 -> base（用于位置外环混合）
        # ------------------------------
        # 使用左指局部系作为代表性抓取坐标系（两指局部轴向已在 cfg 中对齐）
        left_quat_w = self._left_frame.data.target_quat_w[:, 0, :]  # (N, 4)
        F_app_target_w = math_utils.quat_apply(left_quat_w, F_app_target_local)
        F_app_meas_w = math_utils.quat_apply(left_quat_w, F_app_meas_local)

        # 世界 -> base
        root_quat_w = self._robot.data.root_quat_w  # (N,4)
        F_app_pred_b = math_utils.quat_apply_inverse(root_quat_w, F_app_target_w)
        F_app_meas_b = math_utils.quat_apply_inverse(root_quat_w, F_app_meas_w)

        # ------------------------------
        # 4) finger 挤压力 -> 基于“预测开合度 + 力误差”更新夹爪开合度
        # ------------------------------
        # 录制时的 abs gripper 视为“预测开合度”：d_pred
        # 注意：即使 squeeze_kp == 0，我们也需要 d_pred 用于 debug 字段，避免 UnboundLocalError。
        d_pred = self._gripper_abs_cmd.squeeze(-1)  # (N,)

        authoritative_inner_trace: dict[str, Any] | None = None
        if self._authoritative_true_force_enabled:
            sensor_name = self.cfg.authoritative_true_force_sensor_name
            if not sensor_name or sensor_name not in self._env.scene.keys():
                raise RuntimeError(
                    "authoritative true-force inner loop requires an existing "
                    f"object-filtered sensor, got {sensor_name!r}"
                )
            true_sensor = self._env.scene[sensor_name]
            force_matrix_w = true_sensor.data.force_matrix_w
            if force_matrix_w is None or force_matrix_w.ndim != 4:
                shape = None if force_matrix_w is None else tuple(force_matrix_w.shape)
                raise RuntimeError(
                    f"{sensor_name!r} must expose force_matrix_w (N,2,M,3); got {shape}"
                )
            force_lr_w = force_matrix_w.sum(dim=2)
            true_local = object_filtered_gripper_force_local(
                self._env,
                contact_sensor_name=sensor_name,
                left_gripper_frame_name=self.cfg.left_gripper_frame_name,
                right_gripper_frame_name=self.cfg.right_gripper_frame_name,
            )
            left_true = torch.abs(true_local[:, 0, 2])
            right_true = torch.abs(true_local[:, 1, 2])
            left_true_tangential = torch.linalg.vector_norm(
                true_local[:, 0, 0:2], dim=-1
            )
            right_true_tangential = torch.linalg.vector_norm(
                true_local[:, 1, 0:2], dim=-1
            )
            true_raw = 2.0 * torch.minimum(left_true, right_true)
            left_contact = torch.linalg.vector_norm(force_lr_w[:, 0, :], dim=-1) >= float(
                self.cfg.authoritative_true_force_contact_threshold_n
            )
            right_contact = torch.linalg.vector_norm(force_lr_w[:, 1, :], dim=-1) >= float(
                self.cfg.authoritative_true_force_contact_threshold_n
            )
            bilateral = left_contact & right_contact

            dt = float(self._env.cfg.sim.dt)
            alpha_reference = float(
                self.cfg.authoritative_true_force_filter_alpha
            )
            reference_dt = float(
                self.cfg.authoritative_true_force_filter_reference_dt_s
            )
            if not 0.0 < alpha_reference <= 1.0 or reference_dt <= 0.0:
                raise ValueError(
                    "authoritative true-force filter requires alpha in (0,1] "
                    "and a positive reference dt"
                )
            # Preserve the old outer-loop EMA time constant when the same
            # filter is evaluated at physics rate.  This is a time-base
            # conversion, not a retuned alpha.
            alpha_true = 1.0 - (1.0 - alpha_reference) ** (dt / reference_dt)
            if self._authoritative_true_force_filter is None:
                self._authoritative_true_force_filter = true_raw.clone()
            else:
                self._authoritative_true_force_filter.mul_(1.0 - alpha_true).add_(
                    true_raw, alpha=alpha_true
                )
            true_filtered = self._authoritative_true_force_filter

            # A short, reversible loss/recovery guard runs at this same
            # physics cadence.  It never substitutes a zero force error for a
            # lost contact and it never allows an unbounded closing nudge.
            lost = ~bilateral
            # Capture the previous state before updating the streak.  The
            # earlier ordering reset the streak on a bilateral sample before
            # testing it, making ``recovered`` impossible to assert.
            was_lost = self._authoritative_true_force_loss_streak > 0
            self._authoritative_true_force_loss_streak = torch.where(
                lost,
                self._authoritative_true_force_loss_streak + 1,
                torch.zeros_like(self._authoritative_true_force_loss_streak),
            )
            recovered = bilateral & was_lost
            actual_now = self._robot.data.joint_pos[:, self._gripper_joint_ids].mean(dim=-1)
            velocity_alpha = float(
                self.cfg.authoritative_true_force_actual_velocity_filter_alpha
            )
            if not 0.0 < velocity_alpha <= 1.0:
                raise ValueError(
                    "actual-aperture velocity filter alpha must be in (0,1]"
                )
            actual_velocity_raw = torch.where(
                self._authoritative_true_force_actual_velocity_initialized,
                (
                    actual_now
                    - self._authoritative_true_force_previous_actual_aperture
                )
                / dt,
                torch.zeros_like(actual_now),
            )
            self._authoritative_true_force_actual_velocity_filter.mul_(
                1.0 - velocity_alpha
            ).add_(actual_velocity_raw, alpha=velocity_alpha)
            actual_velocity_filtered = (
                self._authoritative_true_force_actual_velocity_filter
            )
            self._authoritative_true_force_previous_actual_aperture[:] = actual_now
            self._authoritative_true_force_actual_velocity_initialized[:] = True
            if bool(torch.any(recovered)):
                self._authoritative_true_force_filter[recovered] = true_raw[recovered]
                true_filtered = self._authoritative_true_force_filter
                self._authoritative_true_force_reacquire_accum[recovered] = 0.0
                self._authoritative_true_force_actual_velocity_filter[recovered] = 0.0
                actual_velocity_filtered = (
                    self._authoritative_true_force_actual_velocity_filter
                )
                # Re-arm from a contact-safe reference.  In a stiff
                # position-controlled grasp the physical joint position can
                # sit millimetres above the applied target.  Copying that
                # displaced position into the actuator target creates a large
                # opening jump and immediately loses the recovered contact.
                # Use actual aperture as an upper (opening) bound while
                # preserving a currently applied, more-closed reacquire
                # target.  Normal bounded force feedback supersedes it below.
                self._authoritative_true_force_d_cmd[recovered] = torch.minimum(
                    self._authoritative_true_force_d_cmd[recovered],
                    actual_now[recovered],
                )
                self._authoritative_true_force_d_cmd_initialized[recovered] = True
                self._authoritative_true_force_release_active[recovered] = False
                self._authoritative_true_force_release_command_reference[recovered] = (
                    self._authoritative_true_force_d_cmd[recovered]
                )
                self._authoritative_true_force_release_actual_reference[recovered] = (
                    actual_now[recovered]
                )
            self._authoritative_true_force_recovery_streak = torch.where(
                bilateral,
                self._authoritative_true_force_recovery_streak + 1,
                torch.zeros_like(self._authoritative_true_force_recovery_streak),
            )
            self._authoritative_true_force_loss_streak = torch.where(
                recovered,
                torch.zeros_like(self._authoritative_true_force_loss_streak),
                self._authoritative_true_force_loss_streak,
            )

            if not bool(torch.all(self._authoritative_true_force_d_cmd_initialized)):
                uninitialized = ~self._authoritative_true_force_d_cmd_initialized
                self._authoritative_true_force_d_cmd[uninitialized] = actual_now[uninitialized]
                self._authoritative_true_force_d_cmd_initialized[uninitialized] = True

            d_previous_command = self._authoritative_true_force_d_cmd.clone()
            true_error = self._authoritative_true_force_target - true_filtered
            kp_step = float(self.cfg.authoritative_true_force_kp_m_per_n_s) * dt
            open_step = float(self.cfg.authoritative_true_force_open_rate_mps) * dt
            close_step = float(self.cfg.authoritative_true_force_close_rate_mps) * dt
            deadzone = float(self.cfg.authoritative_true_force_deadzone_n)
            normal_correction = -kp_step * true_error
            normal_correction = torch.where(
                torch.abs(true_error) >= deadzone,
                normal_correction,
                torch.zeros_like(normal_correction),
            )
            normal_correction = torch.clamp(
                normal_correction, min=-close_step, max=open_step
            )
            force_excess = torch.clamp(-true_error, min=0.0)
            opening_backlog = torch.zeros_like(true_error)
            backlog_scale = torch.ones_like(true_error)
            contact_margin_scale = bilateral.to(dtype=true_error.dtype)
            contact_margin_state = torch.where(
                bilateral,
                torch.full_like(true_error, 2, dtype=torch.long),
                torch.zeros_like(true_error, dtype=torch.long),
            )
            delta_d_needed = torch.zeros_like(true_error)
            adaptive_release_proposed = torch.zeros_like(true_error)
            adaptive_release_before_rate_limit = torch.zeros_like(true_error)
            adaptive_release_rate_limited = torch.zeros_like(true_error)
            adaptive_release_enabled = bool(
                self.cfg.authoritative_true_force_adaptive_release_enabled
            )
            if adaptive_release_enabled:
                slope = float(
                    self.cfg.authoritative_true_force_release_slope_n_per_m
                )
                alpha_release = float(
                    self.cfg.authoritative_true_force_release_alpha
                )
                backlog_limit = float(
                    self.cfg.authoritative_true_force_opening_backlog_limit_m
                )
                backlog_soft_start = float(
                    self.cfg.authoritative_true_force_opening_backlog_soft_start_m
                )
                contact_guard_width = float(
                    self.cfg.authoritative_true_force_contact_guard_width_n
                )
                if slope <= 0.0 or not 0.0 < alpha_release < 1.0:
                    raise ValueError(
                        "adaptive true-force release requires positive slope "
                        "and alpha in (0,1)"
                    )
                if (
                    backlog_limit <= 0.0
                    or backlog_soft_start < 0.0
                    or backlog_soft_start >= backlog_limit
                    or contact_guard_width <= 0.0
                ):
                    raise ValueError(
                        "adaptive true-force release requires 0 <= backlog "
                        "soft start < hard limit and a positive contact guard"
                    )

                release_demand = bilateral & (force_excess >= deadzone)
                release_start = release_demand & (
                    ~self._authoritative_true_force_release_active
                )
                if bool(torch.any(release_start)):
                    self._authoritative_true_force_release_command_reference[
                        release_start
                    ] = d_previous_command[release_start]
                    self._authoritative_true_force_release_actual_reference[
                        release_start
                    ] = actual_now[release_start]
                self._authoritative_true_force_release_active = release_demand

                opening_backlog = torch.clamp(
                    (
                        d_previous_command
                        - self._authoritative_true_force_release_command_reference
                    )
                    - (
                        actual_now
                        - self._authoritative_true_force_release_actual_reference
                    ),
                    min=0.0,
                )
                opening_backlog = torch.where(
                    release_demand, opening_backlog, torch.zeros_like(opening_backlog)
                )
                backlog_scale = torch.clamp(
                    (backlog_limit - opening_backlog)
                    / (backlog_limit - backlog_soft_start),
                    min=0.0,
                    max=1.0,
                )

                # F_raw is the same authoritative object-filtered bilateral
                # squeeze metric as F_filtered, but it reacts without EMA lag.
                # Taper release continuously to zero from F_des+1N down to
                # F_des+deadzone; lost contact always has zero release scale.
                contact_margin_scale = torch.clamp(
                    (
                        true_raw
                        - self._authoritative_true_force_target
                        - deadzone
                    )
                    / contact_guard_width,
                    min=0.0,
                    max=1.0,
                )
                contact_margin_scale = torch.where(
                    bilateral,
                    contact_margin_scale,
                    torch.zeros_like(contact_margin_scale),
                )
                contact_margin_state = torch.where(
                    ~bilateral,
                    torch.zeros_like(contact_margin_state),
                    torch.where(
                        contact_margin_scale < 0.999999,
                        torch.ones_like(contact_margin_state),
                        torch.full_like(contact_margin_state, 2),
                    ),
                )
                delta_d_needed = force_excess / slope
                adaptive_release_proposed = alpha_release * delta_d_needed
                adaptive_release_before_rate_limit = (
                    adaptive_release_proposed
                    * backlog_scale
                    * contact_margin_scale
                )
                adaptive_release_rate_limited = torch.clamp(
                    adaptive_release_before_rate_limit,
                    min=0.0,
                    max=open_step,
                )
                normal_correction = torch.where(
                    release_demand,
                    adaptive_release_rate_limited,
                    normal_correction,
                )
            else:
                self._authoritative_true_force_release_active.zero_()
            # The EMA is the nominal feedback signal, but during rapid
            # unloading it can remain above target for a few physics samples
            # after raw contact force has already reached the target band.
            # Do not let that stale over-force estimate open the fingers past
            # contact.  This guard only suppresses opening; it never turns a
            # raw sample into an unbounded correction.
            raw_open_guard = (
                bool(self.cfg.authoritative_true_force_raw_open_guard_enabled)
                & (normal_correction > 0.0)
                & (
                    true_raw
                    <= self._authoritative_true_force_target + deadzone
                )
            )
            normal_correction = torch.where(
                raw_open_guard,
                torch.zeros_like(normal_correction),
                normal_correction,
            )

            # Optional FORTE-only contact safety supervisor.  In shadow mode
            # every signal and decision below is recorded but normal_correction
            # is left bit-for-bit unchanged.  The force decomposition is valid
            # only because true_local came from the object-filtered force
            # matrix and was rotated into each finger's own frame above.
            safety_supervisor_enabled = bool(
                self.cfg.authoritative_true_force_contact_safety_supervisor_enabled
            )
            safety_shadow_only = bool(
                self.cfg.authoritative_true_force_contact_safety_shadow_only
            )
            mu_safe = float(self.cfg.authoritative_true_force_mu_safe)
            rho_cautious = float(self.cfg.authoritative_true_force_rho_cautious)
            rho_critical = float(self.cfg.authoritative_true_force_rho_critical)
            prediction_horizon = float(
                self.cfg.authoritative_true_force_weak_prediction_horizon_s
            )
            derivative_steps = int(
                self.cfg.authoritative_true_force_weak_derivative_steps
            )
            warning_confirm_steps = int(
                self.cfg.authoritative_true_force_warning_confirm_steps
            )
            beta_follow = float(
                self.cfg.authoritative_true_force_actuator_following_factor
            )
            if safety_supervisor_enabled and (
                mu_safe <= 0.0
                or not 0.0 < rho_cautious < rho_critical
                or rho_critical > 1.0
                or prediction_horizon <= 0.0
                or derivative_steps != 3
                or warning_confirm_steps < 1
                or not 0.0 <= beta_follow <= 1.0
            ):
                raise ValueError(
                    "contact safety supervisor requires positive mu/horizon, "
                    "0<rho_cautious<rho_critical<=1, the audited three-step "
                    "derivative, positive confirmation, and beta in [0,1]"
                )

            eps_force = 1.0e-6
            left_margin = mu_safe * left_true - left_true_tangential
            right_margin = mu_safe * right_true - right_true_tangential
            left_rho = left_true_tangential / torch.clamp(
                mu_safe * left_true, min=eps_force
            )
            right_rho = right_true_tangential / torch.clamp(
                mu_safe * right_true, min=eps_force
            )
            rho_max = torch.maximum(left_rho, right_rho)
            min_margin = torch.minimum(left_margin, right_margin)
            weak_force = torch.minimum(left_true, right_true)
            force_imbalance = torch.abs(left_true - right_true) / torch.clamp(
                left_true + right_true, min=eps_force
            )

            oldest_weak = self._authoritative_true_force_weak_history[:, 0].clone()
            derivative_valid = (
                self._authoritative_true_force_weak_history_count
                >= derivative_steps
            )
            self._authoritative_true_force_weak_history[:, :-1] = (
                self._authoritative_true_force_weak_history[:, 1:].clone()
            )
            self._authoritative_true_force_weak_history[:, -1] = weak_force
            self._authoritative_true_force_weak_history_count.add_(1)
            weak_force_derivative = torch.where(
                derivative_valid,
                (weak_force - oldest_weak) / (derivative_steps * dt),
                torch.zeros_like(weak_force),
            )
            projected_weak_force = (
                weak_force + weak_force_derivative * prediction_horizon
            )
            weakening_warning_raw = (
                derivative_valid
                & bilateral
                & (weak_force_derivative < 0.0)
                & (
                    projected_weak_force
                    < 0.5 * self._authoritative_true_force_target
                )
            )
            friction_warning_raw = bilateral & (rho_max >= rho_cautious)
            safety_warning_raw = weakening_warning_raw | friction_warning_raw
            self._authoritative_true_force_warning_streak = torch.where(
                safety_warning_raw,
                self._authoritative_true_force_warning_streak + 1,
                torch.zeros_like(self._authoritative_true_force_warning_streak),
            )
            safety_warning = bilateral & (
                self._authoritative_true_force_warning_streak
                >= warning_confirm_steps
            )

            # Optional actual-state velocity-resolved controller.  Positive
            # aperture velocity opens the fingers.  Unlike the legacy FORTE
            # path, this controller never advances an old position command:
            # its eventual target is recomputed from actual_now every physics
            # step.  The local slope and settling horizon define a continuous
            # admittance gain with units m/(N s).
            velocity_resolved_enabled = bool(
                self.cfg.authoritative_true_force_velocity_resolved_enabled
            )
            velocity_contact_filter_enabled = bool(
                self.cfg.authoritative_true_force_velocity_contact_safety_filter_enabled
            )
            velocity_friction_aux_enabled = bool(
                self.cfg.authoritative_true_force_velocity_friction_aux_guard_enabled
            )
            velocity_slope = float(
                self.cfg.authoritative_true_force_velocity_slope_n_per_m
            )
            velocity_settling_time = float(
                self.cfg.authoritative_true_force_velocity_settling_time_s
            )
            velocity_weak_guard_width = float(
                self.cfg.authoritative_true_force_velocity_weak_guard_width_n
            )
            if velocity_resolved_enabled and (
                velocity_slope <= 0.0
                or velocity_settling_time <= 0.0
                or velocity_weak_guard_width <= 0.0
            ):
                raise ValueError(
                    "velocity-resolved true-force control requires positive "
                    "slope, settling horizon, and weak-side guard width"
                )
            velocity_gain = 1.0 / max(
                velocity_slope * velocity_settling_time, 1.0e-12
            )
            force_excess_signed = true_filtered - self._authoritative_true_force_target
            velocity_force_unclamped = velocity_gain * force_excess_signed
            velocity_force_unclamped = torch.where(
                torch.abs(force_excess_signed) >= deadzone,
                velocity_force_unclamped,
                torch.zeros_like(velocity_force_unclamped),
            )
            velocity_nominal = torch.clamp(
                velocity_force_unclamped,
                min=-float(self.cfg.authoritative_true_force_close_rate_mps),
                max=float(self.cfg.authoritative_true_force_open_rate_mps),
            )
            # If the measured aperture is already moving faster in the same
            # direction than the force law requests, command zero lead for
            # this sample.  This uses actual velocity as an overspeed brake;
            # it does not integrate velocity or predict actuator state.
            velocity_same_direction_overspeed = (
                (velocity_nominal * actual_velocity_filtered > 0.0)
                & (
                    torch.abs(actual_velocity_filtered)
                    > torch.abs(velocity_nominal)
                )
            )
            velocity_nominal = torch.where(
                velocity_same_direction_overspeed,
                torch.zeros_like(velocity_nominal),
                velocity_nominal,
            )
            velocity_nominal = torch.where(
                raw_open_guard & (velocity_nominal > 0.0),
                torch.zeros_like(velocity_nominal),
                velocity_nominal,
            )

            weak_target = 0.5 * self._authoritative_true_force_target
            weak_guard_floor = torch.clamp(
                weak_target - velocity_weak_guard_width,
                min=float(self.cfg.authoritative_true_force_contact_threshold_n),
            )
            velocity_weak_scale = torch.clamp(
                (projected_weak_force - weak_guard_floor)
                / torch.clamp(
                    weak_target - weak_guard_floor,
                    min=1.0e-6,
                ),
                min=0.0,
                max=1.0,
            )
            velocity_friction_scale = torch.clamp(
                (rho_critical - rho_max)
                / max(rho_critical - rho_cautious, 1.0e-6),
                min=0.0,
                max=1.0,
            )
            if not velocity_friction_aux_enabled:
                velocity_friction_scale = torch.ones_like(
                    velocity_friction_scale
                )
            velocity_safety_scale = torch.ones_like(true_error)
            if velocity_contact_filter_enabled:
                warning_scale = torch.minimum(
                    velocity_weak_scale, velocity_friction_scale
                )
                velocity_safety_scale = torch.where(
                    safety_warning,
                    warning_scale,
                    velocity_safety_scale,
                )
                velocity_safety_scale = torch.where(
                    bilateral,
                    velocity_safety_scale,
                    torch.zeros_like(velocity_safety_scale),
                )
            velocity_safe = torch.where(
                velocity_nominal > 0.0,
                velocity_nominal * velocity_safety_scale,
                velocity_nominal,
            )
            if velocity_resolved_enabled:
                # The contact-loss branch below remains the sole bounded
                # reacquisition authority.  While bilateral contact exists,
                # the normal correction is exactly one-step velocity motion.
                normal_correction = velocity_safe * dt

            # Offset-preserving virtual-equilibrium servo.  Negative
            # equilibrium offset means the applied position target is on the
            # closing side of the measured aperture and therefore carries
            # preload.  Positive v_eq releases that preload; negative v_eq
            # increases it.  Only the offset changes at the bounded velocity;
            # the loaded target/actual separation is never reset to zero.
            offset_servo_enabled = bool(
                self.cfg.authoritative_true_force_offset_servo_enabled
            )
            if offset_servo_enabled and velocity_resolved_enabled:
                raise ValueError(
                    "offset servo and direct actual-state velocity mode are "
                    "mutually exclusive"
                )
            offset_contact_filter_enabled = bool(
                self.cfg.authoritative_true_force_offset_contact_safety_filter_enabled
            )
            offset_friction_aux_enabled = bool(
                self.cfg.authoritative_true_force_offset_friction_aux_guard_enabled
            )
            offset_slope = float(
                self.cfg.authoritative_true_force_offset_slope_n_per_m
            )
            offset_settling_time = float(
                self.cfg.authoritative_true_force_offset_settling_time_s
            )
            if offset_servo_enabled and (
                offset_slope <= 0.0 or offset_settling_time <= 0.0
            ):
                raise ValueError(
                    "offset servo requires positive slope and settling horizon"
                )
            offset_gain = 1.0 / max(
                offset_slope * offset_settling_time, 1.0e-12
            )
            offset_velocity_unclamped = offset_gain * force_excess_signed
            offset_velocity_unclamped = torch.where(
                torch.abs(force_excess_signed) >= deadzone,
                offset_velocity_unclamped,
                torch.zeros_like(offset_velocity_unclamped),
            )
            offset_velocity_nominal = torch.clamp(
                offset_velocity_unclamped,
                min=-float(self.cfg.authoritative_true_force_eq_close_rate_mps),
                max=float(self.cfg.authoritative_true_force_eq_release_rate_mps),
            )
            offset_velocity_nominal = torch.where(
                raw_open_guard & (offset_velocity_nominal > 0.0),
                torch.zeros_like(offset_velocity_nominal),
                offset_velocity_nominal,
            )
            offset_friction_scale = velocity_friction_scale
            if not offset_friction_aux_enabled:
                offset_friction_scale = torch.ones_like(offset_friction_scale)
            offset_safety_scale = torch.ones_like(true_error)
            if offset_contact_filter_enabled:
                offset_warning_scale = torch.minimum(
                    velocity_weak_scale, offset_friction_scale
                )
                offset_safety_scale = torch.where(
                    safety_warning,
                    offset_warning_scale,
                    offset_safety_scale,
                )
                offset_safety_scale = torch.where(
                    bilateral,
                    offset_safety_scale,
                    torch.zeros_like(offset_safety_scale),
                )
            offset_velocity_safe = torch.where(
                offset_velocity_nominal > 0.0,
                offset_velocity_nominal * offset_safety_scale,
                offset_velocity_nominal,
            )
            if offset_servo_enabled:
                normal_correction = offset_velocity_safe * dt

            candidate_release = torch.clamp(normal_correction, min=0.0)
            candidate_command = d_previous_command + candidate_release
            if velocity_resolved_enabled or offset_servo_enabled:
                # The rejected one-step actuator predictor is not part of the
                # velocity controller or its safety filter.  Keep neutral
                # telemetry placeholders for schema compatibility.
                predicted_actual_no_release = actual_now
                predicted_actual_candidate = actual_now
            else:
                predicted_actual_no_release = actual_now + beta_follow * (
                    d_previous_command - actual_now
                )
                predicted_actual_candidate = actual_now + beta_follow * (
                    candidate_command - actual_now
                )
            predictor_slope = float(
                self.cfg.authoritative_true_force_release_slope_n_per_m
            )
            predicted_force_no_release = true_raw - predictor_slope * (
                predicted_actual_no_release - actual_now
            )
            predicted_force_candidate = true_raw - predictor_slope * (
                predicted_actual_candidate - actual_now
            )
            predicted_force_improves = (
                torch.abs(
                    predicted_force_candidate
                    - self._authoritative_true_force_target
                )
                < torch.abs(
                    predicted_force_no_release
                    - self._authoritative_true_force_target
                )
            )
            predicted_normal_delta = 0.5 * (
                predicted_force_candidate - true_raw
            )
            predicted_left_normal = torch.clamp(
                left_true + predicted_normal_delta, min=eps_force
            )
            predicted_right_normal = torch.clamp(
                right_true + predicted_normal_delta, min=eps_force
            )
            predicted_left_rho = left_true_tangential / torch.clamp(
                mu_safe * predicted_left_normal, min=eps_force
            )
            predicted_right_rho = right_true_tangential / torch.clamp(
                mu_safe * predicted_right_normal, min=eps_force
            )
            predicted_rho_max = torch.maximum(
                predicted_left_rho, predicted_right_rho
            )
            safety_release_allowed = (
                bilateral
                & (candidate_release > 0.0)
                & predicted_force_improves
                & (~safety_warning)
                & (predicted_rho_max < rho_cautious)
            )
            critical_safety = bilateral & safety_warning & (
                (predicted_rho_max >= rho_critical)
                | (
                    projected_weak_force
                    < 0.5 * self._authoritative_true_force_target
                )
            )

            reflex_enabled = bool(
                self.cfg.authoritative_true_force_contact_preservation_reflex_enabled
            )
            reclose_rate = float(
                self.cfg.authoritative_true_force_contact_preservation_reclose_rate_mps
            )
            reclose_limit = float(
                self.cfg.authoritative_true_force_contact_preservation_reclose_limit_m
            )
            reclose_force_bound = float(
                self.cfg.authoritative_true_force_contact_preservation_force_bound_n
            )
            reclose_remaining = torch.clamp(
                reclose_limit
                - self._authoritative_true_force_preservation_reclose_accum,
                min=0.0,
            )
            safety_reclose_step = torch.minimum(
                torch.full_like(reclose_remaining, reclose_rate * dt),
                reclose_remaining,
            )
            safety_reclose_requested = (
                critical_safety
                & reflex_enabled
                & (candidate_release <= 0.0)
                & (weak_force_derivative < 0.0)
                & (true_raw <= reclose_force_bound)
                & (reclose_remaining > 0.0)
            )
            self._authoritative_true_force_preservation_reclose_accum += torch.where(
                safety_reclose_requested,
                safety_reclose_step,
                torch.zeros_like(safety_reclose_step),
            )
            self._authoritative_true_force_preservation_reclose_accum = torch.where(
                safety_warning,
                self._authoritative_true_force_preservation_reclose_accum,
                torch.zeros_like(
                    self._authoritative_true_force_preservation_reclose_accum
                ),
            )

            safety_action = torch.zeros_like(true_error, dtype=torch.long)
            safety_action = torch.where(
                safety_release_allowed,
                torch.ones_like(safety_action),
                safety_action,
            )
            safety_action = torch.where(
                (candidate_release > 0.0) & (~safety_release_allowed),
                torch.full_like(safety_action, 2),
                safety_action,
            )
            safety_action = torch.where(
                safety_reclose_requested,
                torch.full_like(safety_action, 3),
                safety_action,
            )
            shadow_command_delta = torch.where(
                safety_reclose_requested,
                -safety_reclose_step,
                torch.where(
                    safety_release_allowed,
                    candidate_release,
                    torch.zeros_like(candidate_release),
                ),
            )
            safety_supervisor_applied = (
                safety_supervisor_enabled
                and not safety_shadow_only
                and not velocity_resolved_enabled
                and not offset_servo_enabled
            )
            if safety_supervisor_applied:
                normal_correction = torch.where(
                    (candidate_release > 0.0) & (~safety_release_allowed),
                    torch.zeros_like(normal_correction),
                    normal_correction,
                )
                normal_correction = torch.where(
                    safety_reclose_requested,
                    -safety_reclose_step,
                    normal_correction,
                )

            unilateral_peak = torch.maximum(left_true, right_true)
            safety_release = (~bilateral) & (
                unilateral_peak >= float(self.cfg.authoritative_true_force_safety_limit_n)
            )
            # During bilateral loss, use only the bounded reacquire budget;
            # if one finger is already overloaded, opening is the only allowed
            # correction.  Normal true-force feedback resumes on recovery.
            remaining = torch.clamp(
                float(self.cfg.authoritative_true_force_reacquire_cumulative_limit_m)
                - self._authoritative_true_force_reacquire_accum,
                min=0.0,
            )
            reacquire_step = torch.minimum(
                torch.full_like(remaining, float(self.cfg.authoritative_true_force_reacquire_step_limit_m)),
                remaining,
            )
            correction = torch.where(
                bilateral,
                normal_correction,
                torch.where(safety_release, torch.full_like(normal_correction, open_step), -reacquire_step),
            )
            self._authoritative_true_force_reacquire_accum += torch.where(
                (~bilateral) & (~safety_release), -torch.minimum(correction, torch.zeros_like(correction)), torch.zeros_like(correction)
            )

            delta_d_raw = normal_correction
            delta_d_after_deadzone = torch.where(
                torch.abs(true_error) >= deadzone,
                delta_d_raw,
                torch.zeros_like(delta_d_raw),
            )
            eq_offset_previous = self._authoritative_true_force_eq_offset.clone()
            eq_offset_unclamped = eq_offset_previous
            eq_offset_clamped = eq_offset_previous
            eq_offset_clamp_active = torch.zeros_like(bilateral)
            if offset_servo_enabled:
                if not bool(torch.all(self._authoritative_true_force_eq_offset_initialized)):
                    eq_uninitialized = ~self._authoritative_true_force_eq_offset_initialized
                    self._authoritative_true_force_eq_offset[eq_uninitialized] = (
                        d_previous_command[eq_uninitialized]
                        - actual_now[eq_uninitialized]
                    )
                    self._authoritative_true_force_eq_offset_initialized[eq_uninitialized] = True
                eq_offset_previous = self._authoritative_true_force_eq_offset.clone()
                eq_offset_unclamped = eq_offset_previous + correction
                eq_offset_clamped = torch.minimum(
                    torch.maximum(
                        eq_offset_unclamped,
                        self._authoritative_true_force_eq_offset_min,
                    ),
                    self._authoritative_true_force_eq_offset_max,
                )
                eq_offset_clamp_active = (
                    torch.abs(eq_offset_clamped - eq_offset_unclamped) > 1.0e-12
                )
                self._authoritative_true_force_eq_offset[:] = eq_offset_clamped
                d_command_unclamped = actual_now + eq_offset_clamped
            elif velocity_resolved_enabled:
                d_command_unclamped = actual_now + correction
            else:
                d_command_unclamped = d_previous_command + correction
            d_command_rate_limited = d_command_unclamped
            d_command_clamped = torch.clamp(
                d_command_rate_limited,
                0.0,
                float(getattr(self._env.cfg, "gripper_open_val", 0.04)),
            )
            self._authoritative_true_force_d_cmd[:] = d_command_clamped
            d_cmd = d_command_clamped
            d_cmd_two = torch.stack([d_cmd, d_cmd], dim=-1)
            self._robot.set_joint_position_target(
                d_cmd_two, joint_ids=self._gripper_joint_ids
            )
            self._last_d_cmd = d_cmd.detach().clone()
            joint_target_buffer = self._robot.data.joint_pos_target[
                :, self._gripper_joint_ids
            ].detach().clone()
            finger_position = self._robot.data.joint_pos[
                :, self._gripper_joint_ids
            ].detach().clone()
            finger_velocity = self._robot.data.joint_vel[
                :, self._gripper_joint_ids
            ].detach().clone()
            finger_effort_limit = self._robot.data.joint_effort_limits[
                :, self._gripper_joint_ids
            ].detach().clone()
            finger_velocity_limit = self._robot.data.joint_vel_limits[
                :, self._gripper_joint_ids
            ].detach().clone()
            finger_applied_torque = self._robot.data.applied_torque[
                :, self._gripper_joint_ids
            ].detach().clone()
            finger_computed_torque = self._robot.data.computed_torque[
                :, self._gripper_joint_ids
            ].detach().clone()
            authoritative_inner_trace = {
                "F_des": self._authoritative_true_force_target.detach().clone(),
                "F_meas_raw": true_raw.detach().clone(),
                "F_meas_filtered": true_filtered.detach().clone(),
                "force_error": true_error.detach().clone(),
                "force_excess": force_excess.detach().clone(),
                "left_true_normal_force": left_true.detach().clone(),
                "right_true_normal_force": right_true.detach().clone(),
                "left_true_tangential_force": left_true_tangential.detach().clone(),
                "right_true_tangential_force": right_true_tangential.detach().clone(),
                "left_true_local_fx": true_local[:, 0, 0].detach().clone(),
                "left_true_local_fy": true_local[:, 0, 1].detach().clone(),
                "left_true_local_fz": true_local[:, 0, 2].detach().clone(),
                "right_true_local_fx": true_local[:, 1, 0].detach().clone(),
                "right_true_local_fy": true_local[:, 1, 1].detach().clone(),
                "right_true_local_fz": true_local[:, 1, 2].detach().clone(),
                "left_contact": left_contact.detach().clone(),
                "right_contact": right_contact.detach().clone(),
                "bilateral_contact": bilateral.detach().clone(),
                "contact_recovered_this_step": recovered.detach().clone(),
                "d_reference": d_pred.detach().clone(),
                "d_previous_command": d_previous_command.detach().clone(),
                "d_actual_before": self._robot.data.joint_pos[:, self._gripper_joint_ids].mean(dim=-1).detach().clone(),
                "left_finger_position": self._robot.data.joint_pos[:, self._gripper_joint_ids[0]].detach().clone(),
                "right_finger_position": self._robot.data.joint_pos[:, self._gripper_joint_ids[1]].detach().clone(),
                "delta_d_raw": delta_d_raw.detach().clone(),
                "delta_d_after_deadzone": delta_d_after_deadzone.detach().clone(),
                "delta_d_after_rate_limit": correction.detach().clone(),
                "d_command_unclamped": d_command_unclamped.detach().clone(),
                "d_command_clamped": d_command_clamped.detach().clone(),
                "d_joint_target": d_cmd.detach().clone(),
                "left_finger_joint_target_buffer": joint_target_buffer[:, 0],
                "right_finger_joint_target_buffer": joint_target_buffer[:, 1],
                "left_joint_position_error": joint_target_buffer[:, 0] - finger_position[:, 0],
                "right_joint_position_error": joint_target_buffer[:, 1] - finger_position[:, 1],
                "d_actual_after": self._robot.data.joint_pos[:, self._gripper_joint_ids].mean(dim=-1).detach().clone(),
                "d_actual_velocity": self._robot.data.joint_vel[:, self._gripper_joint_ids].mean(dim=-1).detach().clone(),
                "actual_aperture_velocity_raw": actual_velocity_raw.detach().clone(),
                "actual_aperture_velocity_filtered": actual_velocity_filtered.detach().clone(),
                "left_finger_velocity": self._robot.data.joint_vel[:, self._gripper_joint_ids[0]].detach().clone(),
                "right_finger_velocity": self._robot.data.joint_vel[:, self._gripper_joint_ids[1]].detach().clone(),
                "left_finger_applied_torque": finger_applied_torque[:, 0],
                "right_finger_applied_torque": finger_applied_torque[:, 1],
                "left_finger_computed_torque": finger_computed_torque[:, 0],
                "right_finger_computed_torque": finger_computed_torque[:, 1],
                "left_finger_effort_limit": finger_effort_limit[:, 0],
                "right_finger_effort_limit": finger_effort_limit[:, 1],
                "left_finger_velocity_limit": finger_velocity_limit[:, 0],
                "right_finger_velocity_limit": finger_velocity_limit[:, 1],
                "reacquire_nudge": ((~bilateral) & (~safety_release)).detach().clone(),
                "reacquire_close_applied_m": torch.where(
                    (~bilateral) & (~safety_release), -correction, torch.zeros_like(correction)
                ).detach().clone(),
                "reacquire_force_safety_clamped": safety_release.detach().clone(),
                "raw_open_guard_active": raw_open_guard.detach().clone(),
                "velocity_resolved_enabled": torch.full_like(
                    bilateral, velocity_resolved_enabled, dtype=torch.bool
                ),
                "velocity_force_aperture_slope_n_per_m": torch.full_like(
                    true_error, velocity_slope
                ),
                "velocity_settling_time_s": torch.full_like(
                    true_error, velocity_settling_time
                ),
                "velocity_gain_m_per_n_s": torch.full_like(
                    true_error, velocity_gain
                ),
                "velocity_force_unclamped_mps": velocity_force_unclamped.detach().clone(),
                "velocity_nominal_mps": velocity_nominal.detach().clone(),
                "velocity_same_direction_overspeed": velocity_same_direction_overspeed.detach().clone(),
                "velocity_contact_safety_filter_enabled": torch.full_like(
                    bilateral, velocity_contact_filter_enabled, dtype=torch.bool
                ),
                "velocity_friction_aux_guard_enabled": torch.full_like(
                    bilateral, velocity_friction_aux_enabled, dtype=torch.bool
                ),
                "velocity_weak_side_scale": velocity_weak_scale.detach().clone(),
                "velocity_friction_scale": velocity_friction_scale.detach().clone(),
                "velocity_safety_scale": velocity_safety_scale.detach().clone(),
                "velocity_safe_mps": velocity_safe.detach().clone(),
                "velocity_position_target_m": d_command_clamped.detach().clone(),
                "velocity_position_backlog_m": (
                    d_command_clamped - actual_now
                ).detach().clone(),
                "offset_servo_enabled": torch.full_like(
                    bilateral, offset_servo_enabled, dtype=torch.bool
                ),
                "equilibrium_offset_definition": torch.full_like(
                    true_error, 1.0
                ),
                "equilibrium_offset_previous_m": eq_offset_previous.detach().clone(),
                "equilibrium_offset_unclamped_m": eq_offset_unclamped.detach().clone(),
                "equilibrium_offset_state_m": eq_offset_clamped.detach().clone(),
                "equilibrium_offset_min_m": self._authoritative_true_force_eq_offset_min.detach().clone(),
                "equilibrium_offset_max_m": self._authoritative_true_force_eq_offset_max.detach().clone(),
                "equilibrium_offset_clamp_active": eq_offset_clamp_active.detach().clone(),
                "equilibrium_velocity_gain_m_per_n_s": torch.full_like(
                    true_error, offset_gain
                ),
                "equilibrium_velocity_unclamped_mps": offset_velocity_unclamped.detach().clone(),
                "equilibrium_velocity_nominal_mps": offset_velocity_nominal.detach().clone(),
                "equilibrium_contact_safety_filter_enabled": torch.full_like(
                    bilateral, offset_contact_filter_enabled, dtype=torch.bool
                ),
                "equilibrium_friction_aux_guard_enabled": torch.full_like(
                    bilateral, offset_friction_aux_enabled, dtype=torch.bool
                ),
                "equilibrium_safety_scale": offset_safety_scale.detach().clone(),
                "equilibrium_velocity_safe_mps": offset_velocity_safe.detach().clone(),
                "equilibrium_target_from_actual_m": (
                    actual_now + eq_offset_clamped
                ).detach().clone(),
                "adaptive_release_enabled": torch.full_like(
                    bilateral, adaptive_release_enabled, dtype=torch.bool
                ),
                "adaptive_local_slope_n_per_m": torch.full_like(
                    true_error,
                    float(self.cfg.authoritative_true_force_release_slope_n_per_m),
                ),
                "adaptive_delta_d_needed_m": delta_d_needed.detach().clone(),
                "adaptive_release_alpha": torch.full_like(
                    true_error,
                    float(self.cfg.authoritative_true_force_release_alpha),
                ),
                "opening_backlog_m": opening_backlog.detach().clone(),
                "opening_backlog_scale": backlog_scale.detach().clone(),
                "contact_margin_state_code": contact_margin_state.detach().clone(),
                "contact_margin_scale": contact_margin_scale.detach().clone(),
                "adaptive_release_proposed_m": adaptive_release_proposed.detach().clone(),
                "adaptive_release_before_rate_limit_m": adaptive_release_before_rate_limit.detach().clone(),
                "adaptive_release_after_rate_limit_m": adaptive_release_rate_limited.detach().clone(),
                "adaptive_release_suppressed_by_backlog": (
                    self._authoritative_true_force_release_active
                    & (backlog_scale < 0.999999)
                ).detach().clone(),
                "adaptive_release_suppressed_by_contact_margin": (
                    self._authoritative_true_force_release_active
                    & (contact_margin_scale < 0.999999)
                ).detach().clone(),
                "safety_supervisor_enabled": torch.full_like(
                    bilateral, safety_supervisor_enabled, dtype=torch.bool
                ),
                "safety_supervisor_shadow_only": torch.full_like(
                    bilateral, safety_shadow_only, dtype=torch.bool
                ),
                "safety_supervisor_applied": torch.full_like(
                    bilateral, safety_supervisor_applied, dtype=torch.bool
                ),
                "mu_safe": torch.full_like(true_error, mu_safe),
                "left_friction_margin": left_margin.detach().clone(),
                "right_friction_margin": right_margin.detach().clone(),
                "minimum_friction_margin": min_margin.detach().clone(),
                "left_friction_utilization": left_rho.detach().clone(),
                "right_friction_utilization": right_rho.detach().clone(),
                "maximum_friction_utilization": rho_max.detach().clone(),
                "weak_force": weak_force.detach().clone(),
                "weak_force_derivative": weak_force_derivative.detach().clone(),
                "projected_weak_force": projected_weak_force.detach().clone(),
                "force_imbalance": force_imbalance.detach().clone(),
                "weakening_warning_raw": weakening_warning_raw.detach().clone(),
                "friction_warning_raw": friction_warning_raw.detach().clone(),
                "safety_warning_raw": safety_warning_raw.detach().clone(),
                "safety_warning_confirmed": safety_warning.detach().clone(),
                "actuator_following_factor": torch.full_like(
                    true_error, beta_follow
                ),
                "predicted_actual_no_release": predicted_actual_no_release.detach().clone(),
                "predicted_actual_candidate": predicted_actual_candidate.detach().clone(),
                "predicted_force_no_release": predicted_force_no_release.detach().clone(),
                "predicted_force_candidate": predicted_force_candidate.detach().clone(),
                "predicted_force_error_improves": predicted_force_improves.detach().clone(),
                "predicted_left_friction_utilization": predicted_left_rho.detach().clone(),
                "predicted_right_friction_utilization": predicted_right_rho.detach().clone(),
                "predicted_maximum_friction_utilization": predicted_rho_max.detach().clone(),
                "safety_release_allowed": safety_release_allowed.detach().clone(),
                "safety_critical": critical_safety.detach().clone(),
                "safety_action_code": safety_action.detach().clone(),
                "safety_shadow_command_delta_m": shadow_command_delta.detach().clone(),
                "safety_reclose_requested": safety_reclose_requested.detach().clone(),
                "safety_reclose_step_m": safety_reclose_step.detach().clone(),
                "safety_reclose_accum_m": self._authoritative_true_force_preservation_reclose_accum.detach().clone(),
                "rate_limit_active": (torch.abs(correction) >= torch.where(correction >= 0, open_step, close_step) - 1e-12).detach().clone(),
                "joint_limit_active": ((d_command_clamped <= 0.0) | (d_command_clamped >= float(getattr(self._env.cfg, "gripper_open_val", 0.04)))).detach().clone(),
                "position_target_clamp_active": (torch.abs(d_command_clamped - d_command_unclamped) > 1e-12).detach().clone(),
                "effort_limit_active": torch.any(
                    torch.abs(finger_applied_torque) >= 0.999 * finger_effort_limit,
                    dim=-1,
                ).detach().clone(),
                "velocity_limit_active": torch.any(
                    torch.abs(finger_velocity) >= 0.999 * finger_velocity_limit,
                    dim=-1,
                ).detach().clone(),
                "joint_target_buffer_updated": torch.all(
                    torch.abs(joint_target_buffer - d_cmd_two) <= 1e-12,
                    dim=-1,
                ).detach().clone(),
                "actuator_target_applied": torch.ones(self.num_envs, dtype=torch.bool, device=self._device),
            }
        elif self.cfg.squeeze_kp != 0.0:
            # 注意这里使用 (预测值 - 测量值)，方便直观调节增益方向
            # squeeze 定义已变为 2*min(|fL_z|,|fR_z|)，误差会整体×2；这里乘 0.5 以保持等效控制幅度
            delta_f_sq = 0.5 * (f_sq_target_eff - f_sq_meas)  # (N,)  = f_pred - f_actual

            # 死区逻辑（位置调整死区）：先算位置修正量 Δd = squeeze_kp * Δf，再基于 |Δd| 判断是否启用修正。
            if self.cfg.squeeze_deadzone > 0.0:
                delta_d = self.cfg.squeeze_kp * delta_f_sq
                dz = abs(float(self.cfg.squeeze_kp)) * float(self.cfg.squeeze_deadzone)
                use_correction = torch.abs(delta_d) >= dz
                d_cmd = d_pred - delta_d
                d_cmd = torch.where(use_correction, d_cmd, d_pred)
            else:
                # 无死区时，直接使用连续增量式
                d_cmd = d_pred - self.cfg.squeeze_kp * delta_f_sq

            # 简单饱和：使用 env.cfg 的 open/close 范围（如果有）
            d_min = torch.zeros_like(d_cmd)
            d_max = torch.full_like(d_cmd, getattr(self._env.cfg, "gripper_open_val", 0.04))
            d_cmd = torch.clamp(d_cmd, d_min, d_max)

            # Both finger DOFs are explicitly actuated by this custom asset.
            # Keep the historical P4-B command contract: one aperture target
            # is written to both finger joints.
            d_cmd_two = torch.stack([d_cmd, d_cmd], dim=-1)  # (N,2)
            self._robot.set_joint_position_target(d_cmd_two, joint_ids=self._gripper_joint_ids)

            # 记录最近一次 d_cmd，供调试可视化使用
            self._last_d_cmd = d_cmd.detach().clone()
        else:
            # squeeze 修正关闭时：直接把预测的 abs gripper 当作目标下发（纯位置式夹爪）
            d_min = torch.zeros_like(d_pred)
            d_max = torch.full_like(d_pred, getattr(self._env.cfg, "gripper_open_val", 0.04))
            d_cmd = torch.clamp(d_pred, d_min, d_max)

            d_cmd_two = torch.stack([d_cmd, d_cmd], dim=-1)  # (N,2)
            self._robot.set_joint_position_target(d_cmd_two, joint_ids=self._gripper_joint_ids)

            # 记录最近一次 d_cmd，供调试可视化使用
            self._last_d_cmd = d_cmd.detach().clone()

        # ------------------------------
        # 5) 构造内部 DiffIK 的动作并调用（绝对位姿）
        # ------------------------------
        # 位置外环混合：P_pos_hybrid = P_pos_target + K_pos * (F_app_target^b - F_app_measured^b)
        if isinstance(self.cfg.pos_kp, tuple):
            k_vec = torch.tensor(self.cfg.pos_kp, device=self._device).view(1, 3)
        else:
            k_vec = torch.full((1, 3), float(self.cfg.pos_kp), device=self._device)
        K_pos = k_vec.expand(self.num_envs, -1)  # (N,3)
        F_err_b = F_app_pred_b - F_app_meas_b    # (N,3) = F_target - F_measured
        pos_hybrid = self._eef_pos_cmd + K_pos * F_err_b

        # 姿态部分保持不变：P_axis_hybrid = P_axis_target
        aa = self._eef_aa_cmd  # (N,3)
        angle = torch.linalg.vector_norm(aa, dim=-1, keepdim=True)  # (N,1)
        eps = 1e-6
        safe_axis = torch.zeros_like(aa)
        safe_axis[:, 0] = 1.0
        axis = torch.where(angle > eps, aa / angle, safe_axis)
        quat = math_utils.quat_from_angle_axis(angle.squeeze(-1), axis)  # (N,4)

        eef_pose_quat = torch.cat([pos_hybrid, quat], dim=-1)  # (N,7)

        ik_action_dim = self._ik_term.action_dim
        ik_actions = torch.zeros(self.num_envs, ik_action_dim, device=self._device)
        ik_actions[:, 0:7] = eef_pose_quat

        # 下发到内部 DiffIK term：它自己会读取当前 EEF pose/vel、Jacobian 等
        self._ik_term.process_actions(ik_actions)
        self._ik_term.apply_actions()

        if authoritative_inner_trace is not None:
            target_after_arm = self._robot.data.joint_pos_target[
                :, self._gripper_joint_ids
            ].detach().clone()
            authoritative_inner_trace.update(
                {
                    "left_finger_joint_target_after_arm": target_after_arm[:, 0],
                    "right_finger_joint_target_after_arm": target_after_arm[:, 1],
                    "arm_apply_overwrote_gripper_target": torch.any(
                        torch.abs(target_after_arm - d_cmd_two) > 1e-12,
                        dim=-1,
                    ).detach().clone(),
                }
            )

        # 「加持力」模长（在 base 系下），用于调试可视化 / 统计
        F_app_norm_pred = torch.linalg.vector_norm(F_app_pred_b, dim=-1)  # (N,)
        F_app_norm_meas = torch.linalg.vector_norm(F_app_meas_b, dim=-1)  # (N,)

        # ------------------------------------------------------------------
        # 6) 更新调试信息缓存（仅 env0 会被可视化使用）
        # ------------------------------------------------------------------
        # 夹爪“实际”开合度：读取当前关节位置（两指取平均，单位：m）
        # NOTE: 这不会影响控制，只用于 debug/可视化。
        try:
            d_meas_two = self._robot.data.joint_pos[:, self._gripper_joint_ids]  # (N,2)
            d_meas = d_meas_two.mean(dim=-1)  # (N,)
        except Exception:
            d_meas = self._last_d_cmd.detach().clone()
            d_meas_two = torch.stack([d_meas, d_meas], dim=-1)

        self._debug = {
            "fL_pred_local": self._fL_target_local.detach().clone(),
            "fR_pred_local": self._fR_target_local.detach().clone(),
            # meas: 默认是控制器实际使用的（可能 EMA 后）；raw 额外保留一份方便对比
            "fL_meas_local": fL_meas_local.detach().clone(),
            "fR_meas_local": fR_meas_local.detach().clone(),
            "fL_meas_local_raw": fL_meas_local_raw.detach().clone(),
            "fR_meas_local_raw": fR_meas_local_raw.detach().clone(),
            # 「加持力」向量及其模长（在 base 系下）；同时兼容旧字段名 F_ext_*
            "F_app_pred_b": F_app_pred_b.detach().clone(),
            "F_app_meas_b": F_app_meas_b.detach().clone(),
            "F_app_norm_pred": F_app_norm_pred.detach().clone(),
            "F_app_norm_meas": F_app_norm_meas.detach().clone(),
            "F_ext_pred_b": F_app_pred_b.detach().clone(),
            "F_ext_meas_b": F_app_meas_b.detach().clone(),
            "f_sq_pred": f_sq_target.detach().clone(),
            "f_sq_meas": f_sq_meas.detach().clone(),
            "f_sq_meas_raw": f_sq_meas_raw.detach().clone(),
            "f_sq_pred_eff": f_sq_target_eff.detach().clone(),
            "target_contact_override_enabled": torch.full_like(
                self._target_contact_detected,
                self.cfg.target_contact_squeeze_enabled,
            ),
            "target_contact_detected": self._target_contact_detected.detach().clone(),
            "target_contact_force_norm": self._target_contact_force_norm.detach().clone(),
            "target_contact_override_latched": self._target_contact_latched.detach().clone(),
            "target_contact_activation_step": self._target_contact_activation_step.detach().clone(),
            "target_contact_configured_single_finger_force": torch.full_like(
                f_sq_target_eff,
                float(
                    self.cfg.target_contact_single_finger_normal_force_n
                ),
            ),
            "target_contact_configured_squeeze_force": torch.full_like(
                f_sq_target_eff,
                2.0
                * float(
                    self.cfg.target_contact_single_finger_normal_force_n
                ),
            ),
            "target_contact_single_finger_target": (
                0.5 * f_sq_target_eff
            ).detach().clone(),
            "target_contact_squeeze_target": f_sq_target_eff.detach().clone(),
            "d_pred": d_pred.detach().clone(),
            # 保持字段名兼容 force_position_debug_viz.py：d_actual 现在是“实测关节位置”
            "d_actual": d_meas.detach().clone(),
            # 保留左右原始关节位置，避免把带符号变换的 policy observation
            # 误当成夹爪行程；两者的均值严格等于 d_actual。
            "d_actual_left": d_meas_two[:, 0].detach().clone(),
            "d_actual_right": d_meas_two[:, 1].detach().clone(),
            # 额外提供控制器下发的目标开合度，便于对比
            "d_cmd": self._last_d_cmd.detach().clone(),
            "authoritative_true_force_inner_loop_enabled": torch.full_like(
                d_pred, self._authoritative_true_force_enabled, dtype=torch.bool
            ),
            "eef_pos_pred": self._eef_pos_cmd.detach().clone(),
            # 记录外环混合后的增量（仅用于可视化）
            "eef_pos_delta": (pos_hybrid - self._eef_pos_cmd).detach().clone(),
        }

        if authoritative_inner_trace is not None:
            sim_step = int(getattr(self._env, "_sim_step_counter", 0))
            decimation = int(getattr(self._env.cfg, "decimation", 1))
            physics_dt = float(getattr(self._env, "physics_dt", self._env.cfg.sim.dt))
            env_step = int(self._env.episode_length_buf[0].item()) + 1
            substep = (sim_step - 1) % max(decimation, 1)
            def scalar(value: Any, index: int = 0) -> Any:
                if isinstance(value, torch.Tensor):
                    value = value[index]
                if isinstance(value, torch.Tensor):
                    return value.detach().cpu().item()
                return value
            trace_row = {
                # apply_actions() runs before the next sim.step(); contact
                # data therefore comes from the scene.update() at the end of
                # the preceding physics step.  Its age is bounded by one dt.
                "sim_time_s": float(sim_step * physics_dt),
                "env_step": env_step,
                "physics_step": sim_step,
                "substep_index": substep,
                "contact_sensor_sample_index": max(sim_step - 1, 0),
                "contact_sensor_sample_time_s": float(max(sim_step - 1, 0) * physics_dt),
                "force_measurement_age_seconds": physics_dt,
                "external_hybrid_updated_this_step": bool(substep == 0),
                "force_filter_updated_this_step": True,
                "gripper_target_updated_this_step": True,
                "actuator_target_applied_this_step": True,
                "hybrid_state": "FORCE_TRACK" if scalar(authoritative_inner_trace["bilateral_contact"]) else "CONTACT_LOSS",
                **{key: scalar(value) for key, value in authoritative_inner_trace.items()},
            }
            self._authoritative_true_force_physics_trace.append(trace_row)


# 把 cfg 的 class_type 指回本 ActionTerm，供 manager 创建
ForcePositionActionCfg.class_type = ForcePositionAction
