"""Torque-safe minimum world-yaw torque adapter for knob turning."""
import types

import numpy as np

from r2h_branch import Controller, nullspace_torques, opspace_matrices, orientation_error


class TorqueSafeMinimumYawTorqueAdapter:
    def __init__(self, env, yaw_sign, enabled, minimum_yaw_torque_nm, torque_margin=0.05):
        self.env = env
        self.controller = env.robots[0].controller
        self.original = self.controller.run_controller
        self.yaw_axis = np.asarray([0.0, 0.0, float(np.sign(yaw_sign) or 1.0)])
        self.enabled = bool(enabled)
        self.minimum_yaw_torque_nm = float(minimum_yaw_torque_nm)
        self.torque_margin = float(torque_margin)
        self.trace = []

        def run(_controller):
            if not self.enabled:
                torque = np.asarray(self.original(), dtype=float)
                self.trace.append({"mode": "disabled_original_bound_method", "raw_returned_torque": torque.tolist(), "finite": bool(np.isfinite(torque).all()), "clip_count": 0})
                return torque
            c = self.controller
            c.update()
            desired_pos = c.interpolator_pos.get_interpolated_goal() if c.interpolator_pos is not None and c.interpolator_pos.order == 1 else np.array(c.goal_pos)
            if c.interpolator_ori is not None:
                c.relative_ori = orientation_error(c.ee_ori_mat, c.ori_ref)
                ori_error = c.interpolator_ori.get_interpolated_goal()
            else:
                ori_error = orientation_error(np.array(c.goal_ori), c.ee_ori_mat)
            position_error = desired_pos - c.ee_pos
            desired_force = position_error * np.asarray(c.kp[:3]) + (-c.ee_pos_vel) * np.asarray(c.kd[:3])
            desired_torque = ori_error * np.asarray(c.kp[3:6]) + (-c.ee_ori_vel) * np.asarray(c.kd[3:6])
            lambda_full, lambda_pos, lambda_ori, nullspace_matrix = opspace_matrices(c.mass_matrix, c.J_full, c.J_pos, c.J_ori)
            if c.uncoupling:
                stock_force = lambda_pos @ desired_force
                stock_orientation = lambda_ori @ desired_torque
            else:
                stock_wrench = lambda_full @ np.concatenate([desired_force, desired_torque])
                stock_force, stock_orientation = stock_wrench[:3], stock_wrench[3:]
            stock_wrench = np.concatenate([stock_force, stock_orientation])
            gravity = np.asarray(c.torque_compensation, dtype=float)
            nullspace = nullspace_torques(c.mass_matrix, nullspace_matrix, c.initial_joint, c.joint_pos, c.joint_vel)
            raw_stock = c.J_full.T @ stock_wrench + gravity + nullspace
            stock_axis = float(self.yaw_axis @ stock_orientation)
            requested_axis = max(stock_axis, self.minimum_yaw_torque_nm)
            delta_orientation = self.yaw_axis * (requested_axis - stock_axis)
            delta_joint = c.J_ori.T @ delta_orientation
            low, high = self.env.robots[0].torque_limits
            safe_low, safe_high = low + self.torque_margin, high - self.torque_margin
            candidates = [1.0]
            for value, delta, lo, hi in zip(raw_stock, delta_joint, safe_low, safe_high):
                if delta > 0:
                    candidates.append((hi - value) / delta)
                elif delta < 0:
                    candidates.append((lo - value) / delta)
            alpha = float(np.clip(min(candidates), 0.0, 1.0))
            raw = raw_stock + alpha * delta_joint
            applied_axis = stock_axis + alpha * (requested_axis - stock_axis)
            c.torques = raw
            Controller.run_controller(c)
            self.trace.append(
                {
                    "mode": "torque_safe_minimum_world_yaw_torque",
                    "yaw_axis": self.yaw_axis.tolist(),
                    "stock_yaw_torque_nm": stock_axis,
                    "minimum_yaw_torque_nm": self.minimum_yaw_torque_nm,
                    "floor_active": bool(self.minimum_yaw_torque_nm > stock_axis),
                    "requested_yaw_torque_nm": requested_axis,
                    "applied_yaw_torque_nm": applied_axis,
                    "safety_alpha": alpha,
                    "raw_stock_torque": raw_stock.tolist(),
                    "raw_returned_torque": raw.tolist(),
                    "finite": bool(np.isfinite(raw).all()),
                    "clip_count": int(np.sum((raw < low) | (raw > high))),
                }
            )
            return raw

        self.controller.run_controller = types.MethodType(run, self.controller)

    def close(self):
        self.controller.run_controller = self.original
