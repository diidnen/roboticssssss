import numpy as np


class ShakeTrajectoryEvaluatorMixin:
    """Physics-grounded success and telemetry for the two shake tasks."""

    af_min_position_path_m = 0.0
    af_min_position_span_z_m = 0.0
    af_min_angular_path_rad = 0.0
    af_min_angular_excursion_rad = 0.0
    af_min_direction_reversals = 5
    af_trace_stride = 15
    af_lift_height_m = 0.10
    af_drop_height_m = 0.79
    af_min_contact_ratio = 0.50

    def initialize_activeforcing_dynamic_trace(self):
        self._af_arm_tag = "right" if self.bottle.get_pose().p[0] > 0 else "left"
        self._af_physics_step = 0
        self._af_grasp_started = False
        self._af_table_impact_after_grasp = False
        self._af_trace = []
        self.af_dynamic_metrics = {}

    def configure_activeforcing_physics(self):
        if self.af_object_mass_kg is not None:
            mass_kg = float(self.af_object_mass_kg)
            if not np.isfinite(mass_kg) or mass_kg <= 0:
                raise ValueError("af_object_mass_kg must be finite and positive")
            self.bottle.set_mass(mass_kg)
        if self.af_contact_friction is not None:
            self.set_actor_contact_friction(self.bottle, self.af_contact_friction)
        if self.af_force_limit_n is not None:
            self.robot.set_gripper_force_limit(self.af_force_limit_n, self._af_arm_tag)

    def on_physics_step(self):
        if not self.af_dynamic_evaluator or not hasattr(self, "bottle"):
            return
        self._af_physics_step += 1
        if self._af_physics_step % self.af_trace_stride:
            return

        bottle_position = np.asarray(self.bottle.get_pose().p, dtype=np.float64)
        contact = self.get_actor_gripper_contact_forces(self.bottle, self._af_arm_tag)
        bilateral = bool(contact["bilateral_contact"])
        lifted = bottle_position[2] >= 0.74 + self.table_z_bias + self.af_lift_height_m
        if not self._af_grasp_started and lifted and bilateral:
            self._af_grasp_started = True

        if not self._af_grasp_started:
            return
        if bottle_position[2] <= self.af_drop_height_m + self.table_z_bias:
            self._af_table_impact_after_grasp = True

        ee_pose = np.asarray(self.get_arm_pose(self._af_arm_tag), dtype=np.float64)
        self._af_trace.append(
            {
                "ee_position": ee_pose[:3],
                "ee_quaternion": ee_pose[3:7],
                "bottle_position": bottle_position,
                "bilateral_contact": bilateral,
                "measured_single_finger_force_n": float(contact["single_finger_normal_force_n"]),
            }
        )

    @staticmethod
    def _angular_distance(q1, q2):
        q1 = np.asarray(q1, dtype=np.float64)
        q2 = np.asarray(q2, dtype=np.float64)
        q1 = q1 / np.maximum(np.linalg.norm(q1, axis=-1, keepdims=True), 1e-12)
        q2 = q2 / np.maximum(np.linalg.norm(q2, axis=-1, keepdims=True), 1e-12)
        dots = np.abs(np.sum(q1 * q2, axis=-1))
        return 2.0 * np.arccos(np.clip(dots, -1.0, 1.0))

    @staticmethod
    def _direction_reversals(values, epsilon=0.002):
        values = np.asarray(values, dtype=np.float64)
        if len(values) < 4:
            return 0
        smooth = np.convolve(values, np.ones(3) / 3.0, mode="valid")
        delta = np.diff(smooth)
        signs = np.sign(delta[np.abs(delta) >= epsilon])
        return int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0

    def compute_activeforcing_dynamic_metrics(self):
        if not self._af_trace:
            metrics = {
                "samples": 0,
                "position_path_m": 0.0,
                "position_span_z_m": 0.0,
                "angular_path_rad": 0.0,
                "angular_excursion_rad": 0.0,
                "direction_reversals": 0,
                "contact_ratio": 0.0,
                "final_bilateral_contact": False,
                "table_impact_after_grasp": bool(self._af_table_impact_after_grasp),
            }
            self.af_dynamic_metrics = metrics
            return metrics

        positions = np.stack([sample["ee_position"] for sample in self._af_trace])
        quaternions = np.stack([sample["ee_quaternion"] for sample in self._af_trace])
        contacts = np.asarray([sample["bilateral_contact"] for sample in self._af_trace], dtype=bool)
        measured_forces = np.asarray(
            [sample["measured_single_finger_force_n"] for sample in self._af_trace], dtype=np.float64
        )
        position_path = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
        angular_path = float(self._angular_distance(quaternions[1:], quaternions[:-1]).sum())
        angular_excursion = float(self._angular_distance(quaternions, quaternions[[0]]).max())
        metrics = {
            "samples": int(len(self._af_trace)),
            "position_path_m": position_path,
            "position_span_z_m": float(np.ptp(positions[:, 2])),
            "angular_path_rad": angular_path,
            "angular_excursion_rad": angular_excursion,
            "direction_reversals": self._direction_reversals(positions[:, 2]),
            "contact_ratio": float(np.mean(contacts)),
            "final_bilateral_contact": bool(contacts[-1]),
            "table_impact_after_grasp": bool(self._af_table_impact_after_grasp),
            "measured_force_mean_n": float(np.mean(measured_forces)),
            "measured_force_p95_n": float(np.percentile(measured_forces, 95)),
        }
        self.af_dynamic_metrics = metrics
        return metrics

    def check_activeforcing_dynamic_success(self):
        metrics = self.compute_activeforcing_dynamic_metrics()
        bottle_above_table = self.bottle.get_pose().p[2] > 0.8 + self.table_z_bias
        return bool(
            self._af_grasp_started
            and bottle_above_table
            and not metrics["table_impact_after_grasp"]
            and metrics["final_bilateral_contact"]
            and metrics["contact_ratio"] >= self.af_min_contact_ratio
            and metrics["position_path_m"] >= self.af_min_position_path_m
            and metrics["position_span_z_m"] >= self.af_min_position_span_z_m
            and metrics["angular_path_rad"] >= self.af_min_angular_path_rad
            and metrics["angular_excursion_rad"] >= self.af_min_angular_excursion_rad
            and metrics["direction_reversals"] >= self.af_min_direction_reversals
        )
