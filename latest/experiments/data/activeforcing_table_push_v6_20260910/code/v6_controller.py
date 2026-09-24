"""Torque-safe minimum-force adapter for the table-push ActiveForcing interface."""
import types

import numpy as np

from r2h_branch import (
    Controller,
    nullspace_torques,
    opspace_matrices,
    orientation_error,
)


class TorqueSafeMinimumForceAdapter:
    """Preserve stock OSC force unless it is below the requested task-axis floor."""

    def __init__(self, env, direction_xy, enabled, minimum_axis_force_n, torque_margin=0.05):
        self.env = env
        self.controller = env.robots[0].controller
        self.original = self.controller.run_controller
        self.direction = np.asarray([direction_xy[0], direction_xy[1], 0.0], dtype=float)
        self.direction /= np.linalg.norm(self.direction)
        self.enabled = bool(enabled)
        self.minimum_axis_force_n = float(minimum_axis_force_n)
        self.torque_margin = float(torque_margin)
        self.trace = []

        def run(_controller):
            if not self.enabled:
                torque = np.asarray(self.original(), dtype=float)
                self.trace.append(
                    {
                        "mode": "disabled_original_bound_method",
                        "raw_returned_torque": torque.tolist(),
                        "finite": bool(np.isfinite(torque).all()),
                        "clip_count": 0,
                    }
                )
                return torque

            c = self.controller
            c.update()
            if c.interpolator_pos is not None:
                desired_pos = c.interpolator_pos.get_interpolated_goal() if c.interpolator_pos.order == 1 else None
            else:
                desired_pos = np.array(c.goal_pos)
            if desired_pos is None:
                raise RuntimeError("unsupported position interpolator")

            if c.interpolator_ori is not None:
                c.relative_ori = orientation_error(c.ee_ori_mat, c.ori_ref)
                ori_error = c.interpolator_ori.get_interpolated_goal()
            else:
                ori_error = orientation_error(np.array(c.goal_ori), c.ee_ori_mat)

            position_error = desired_pos - c.ee_pos
            desired_force = position_error * np.asarray(c.kp[:3]) + (-c.ee_pos_vel) * np.asarray(c.kd[:3])
            desired_torque = ori_error * np.asarray(c.kp[3:6]) + (-c.ee_ori_vel) * np.asarray(c.kd[3:6])
            lambda_full, lambda_pos, lambda_ori, nullspace_matrix = opspace_matrices(
                c.mass_matrix, c.J_full, c.J_pos, c.J_ori
            )
            if c.uncoupling:
                stock_force = lambda_pos @ desired_force
                stock_orientation_torque = lambda_ori @ desired_torque
            else:
                stock_wrench = lambda_full @ np.concatenate([desired_force, desired_torque])
                stock_force = stock_wrench[:3]
                stock_orientation_torque = stock_wrench[3:]

            stock_wrench = np.concatenate([stock_force, stock_orientation_torque])
            task_torque_stock = c.J_full.T @ stock_wrench
            gravity = np.asarray(c.torque_compensation, dtype=float)
            nullspace = nullspace_torques(
                c.mass_matrix, nullspace_matrix, c.initial_joint, c.joint_pos, c.joint_vel
            )
            raw_stock = task_torque_stock + gravity + nullspace

            stock_axis = float(self.direction @ stock_force)
            requested_axis = max(stock_axis, self.minimum_axis_force_n)
            delta_force_requested = self.direction * (requested_axis - stock_axis)
            delta_torque_requested = c.J_pos.T @ delta_force_requested
            low, high = self.env.robots[0].torque_limits
            safe_low = low + self.torque_margin
            safe_high = high - self.torque_margin
            alpha_candidates = [1.0]
            for value, delta, lo, hi in zip(raw_stock, delta_torque_requested, safe_low, safe_high):
                if delta > 0:
                    alpha_candidates.append((hi - value) / delta)
                elif delta < 0:
                    alpha_candidates.append((lo - value) / delta)
            safety_alpha = float(np.clip(min(alpha_candidates), 0.0, 1.0))
            applied_delta_force = safety_alpha * delta_force_requested
            applied_axis = stock_axis + float(self.direction @ applied_delta_force)
            raw = raw_stock + safety_alpha * delta_torque_requested
            c.torques = raw
            Controller.run_controller(c)

            self.trace.append(
                {
                    "mode": "torque_safe_minimum_axis_force",
                    "stock_axis_force_n": stock_axis,
                    "minimum_axis_force_n": self.minimum_axis_force_n,
                    "floor_active": bool(self.minimum_axis_force_n > stock_axis),
                    "requested_axis_force_n": requested_axis,
                    "applied_axis_force_n": applied_axis,
                    "safety_alpha": safety_alpha,
                    "stock_decoupled_force": stock_force.tolist(),
                    "requested_delta_force": delta_force_requested.tolist(),
                    "applied_delta_force": applied_delta_force.tolist(),
                    "raw_stock_torque": raw_stock.tolist(),
                    "raw_returned_torque": raw.tolist(),
                    "finite": bool(np.isfinite(raw).all()),
                    "clip_count": int(np.sum((raw < low) | (raw > high))),
                    "min_margin": float(np.min(np.minimum(raw - low, high - raw))),
                }
            )
            return raw

        self.controller.run_controller = types.MethodType(run, self.controller)

    def close(self):
        self.controller.run_controller = self.original
