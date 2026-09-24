import unittest

import numpy as np

from task_form_contract import (
    CANDIDATE_ORDER, ORIGINAL_TASK_ORDER, expand_original_phasefree,
    expanded_shape, full_task_outcome,
)


class ContractTest(unittest.TestCase):
    def test_seven_task_shape_and_exact_original_mapping(self):
        order = ORIGINAL_TASK_ORDER + CANDIDATE_ORDER
        self.assertEqual(expanded_shape(order), (8, 67, 13))
        x = np.zeros((8, 64), np.float32); x[:, 6] = 1
        x[:, :6] = np.arange(48, dtype=np.float32).reshape(8, 6)
        x[:, 10:] = np.arange(54, dtype=np.float32)
        y = expand_original_phasefree(x, ORIGINAL_TASK_ORDER[0], order)
        np.testing.assert_array_equal(y[:, :6], x[:, :6])
        np.testing.assert_array_equal(y[:, 13:], x[:, 10:])
        np.testing.assert_array_equal(y[:, 6:13], np.tile([1, 0, 0, 0, 0, 0, 0], (8, 1)))

    def test_bowl_needs_clearance_terminal_and_release(self):
        trace = [{"native_goal_reached": True, "drawer_clearance_reached": i == 0,
                  "release_intent": True, "unheld": True} for i in range(20)]
        self.assertEqual(full_task_outcome("libero_spatial/task4", trace)["full_task_success_y"], 1)
        for row in trace: row["drawer_clearance_reached"] = False
        result = full_task_outcome("libero_spatial/task4", trace)
        self.assertEqual(result["full_task_success_y"], 0)
        self.assertEqual(result["failure_class"], "TASK_CONTACT_OR_INTERACTION_FAILURE")

    def test_drop_stays_in_denominator(self):
        trace = [{"native_goal_reached": False, "grasp_retention_failure": i == 3} for i in range(20)]
        result = full_task_outcome("libero_goal/task9", trace)
        self.assertTrue(result["label_valid"])
        self.assertEqual(result["full_task_success_y"], 0)
        self.assertEqual(result["failure_class"], "GRASP_RETENTION_FAILURE")


if __name__ == "__main__":
    unittest.main()
