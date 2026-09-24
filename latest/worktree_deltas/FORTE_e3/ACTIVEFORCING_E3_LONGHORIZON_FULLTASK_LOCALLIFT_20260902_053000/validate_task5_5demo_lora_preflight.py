#!/usr/bin/env python3
"""CPU-only fail-closed preflight for the frozen E3 task5 onboarding config."""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np

from openpi.policies import libero_policy
from openpi.training import config as training_config


BASE_CONFIG = "pi0_lora_tacfield_tabero"
E3_CONFIG = "pi0_lora_tacfield_e3_task5_5demo_7dpf"
EXPECTED_PARAMS = pathlib.Path(
    "/media/volume/newdata/exouser/tabero/models/pi0_lora_tacfield_tabero/checkpoints/"
    "pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999/params"
)
EXPECTED_ASSETS = pathlib.Path(
    "/media/volume/newdata/exouser/activeforcing_e3/"
    "TASK5_ONBOARDING_7DPF_ASSETS_20260902_113000"
)
EXPECTED_CHECKPOINTS = pathlib.Path(
    "/media/volume/newdata/exouser/activeforcing_e3/"
    "TASK5_ONBOARDING_7DPF_CHECKPOINTS_20260902_113000"
)


def assert_tree_equal(left, right, path="root") -> None:
    if isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys(), path
        for key in left:
            assert_tree_equal(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray), path
        assert left.dtype == right.dtype and left.shape == right.shape, path
        assert np.array_equal(left, right), path
        return
    assert left == right, path


def main() -> None:
    base = training_config.get_config(BASE_CONFIG)
    e3 = training_config.get_config(E3_CONFIG)

    base_model = dataclasses.asdict(base.model)
    e3_model = dataclasses.asdict(e3.model)
    model_diff = {key: (base_model[key], e3_model[key]) for key in base_model if base_model[key] != e3_model[key]}
    assert model_diff == {"tactile_loss_weight": (0.1, 0.0), "padding_loss_weight": (1.0, 0.0)}
    assert e3.model.effective_action_dim == 13
    assert e3.model.tactile_dim == 6
    assert e3.model.effective_action_dim - e3.model.tactile_dim == 7
    assert e3.data.repo_id == "activeforcing_e3_task5_5demo_7dpf"
    assert pathlib.Path(e3.weight_loader.params_path) == EXPECTED_PARAMS
    assert e3.weight_loader.missing_regex == ".*"
    assert pathlib.Path(e3.assets_base_dir) == EXPECTED_ASSETS
    assert pathlib.Path(e3.checkpoint_base_dir) == EXPECTED_CHECKPOINTS
    assert (e3.seed, e3.batch_size, e3.num_workers) == (20260902, 8, 2)
    assert (e3.num_train_steps, e3.log_interval, e3.save_interval, e3.keep_period) == (1000, 25, 250, 250)
    assert not e3.wandb_enabled and not e3.overwrite and not e3.resume

    # A real-marker sample must take exactly the authoritative tactile-field path.
    sample = {
        "image": np.zeros((224, 224, 3), dtype=np.uint8),
        "wrist_image": np.ones((224, 224, 3), dtype=np.uint8),
        "state": np.arange(7, dtype=np.float32),
        "tactile_marker_motion": np.arange(9 * 198 * 2, dtype=np.float32).reshape(9, 198, 2),
        "actions": np.arange(13, dtype=np.float32),
        "prompt": "put the black book in the back compartment of the caddy",
    }
    expected = libero_policy.TaberoTacFieldInputs(model_type=e3.model.model_type)(sample)
    observed = libero_policy.E3OptionalTaberoTacFieldInputs(model_type=e3.model.model_type)(sample)
    assert_tree_equal(expected, observed)
    assert observed["tactile_prefix"].shape == (9, 396)
    print("E3_TASK5_5DEMO_LORA_PREFLIGHT_PASS")


if __name__ == "__main__":
    main()
