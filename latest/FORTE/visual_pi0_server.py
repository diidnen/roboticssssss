#!/usr/bin/env python3
"""Frozen π0 server control/instrumentation pair for visual diagnostics.

The control path is the same official policy construction and websocket path.
The instrumented path computes the exact PI0Pytorch image-token representation
from the transformed observation before calling the unchanged action sampler,
then returns it in a diagnostics-only response field.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import logging
import socket
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.shared import normalize as _normalize
from openpi.training import config as _config
from openpi.models import model as _model


def sha256_bytes(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def make_policy(args):
    norm_stats = _normalize.load(Path(args.norm_stats_dir))
    train_config = _config.get_config(args.policy_config)
    policy = _policy_config.create_trained_policy(
        train_config,
        Path(args.policy_dir),
        norm_stats=norm_stats,
        pytorch_device="cuda:0",
    )
    return policy


class InstrumentedPolicy:
    def __init__(self, policy):
        self._policy = policy
        self._model = policy._model
        self._last_diag = None

    @property
    def metadata(self):
        metadata = dict(self._policy.metadata)
        metadata.update({
            "visual_diagnostics": True,
            "visual_diagnostics_source": "PI0Pytorch._preprocess_observation -> PaliGemmaWithExpertModel.embed_image",
            "visual_diagnostics_feature": "per-camera mean pooled image tokens, base then left wrist; right wrist padding excluded",
            "visual_diagnostics_token_width": 2048,
            "visual_diagnostics_camera_order": ["agentview_cam/base_0_rgb", "eye_in_hand_cam/left_wrist_0_rgb"],
        })
        return metadata

    def _capture(self, obs):
        # Match Policy.infer's input transform and PyTorch conversion exactly,
        # but use a copy so diagnostics cannot mutate the action input.
        inputs = jax.tree.map(lambda x: x, copy.deepcopy(obs))
        inputs = self._policy._input_transform(inputs)
        pooled = []
        valid_names = []
        if self._policy._is_pytorch_model:
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to("cuda:0")[None, ...], inputs)
            observation = _model.Observation.from_dict(inputs)
            with torch.inference_mode():
                images, image_masks, _lang, _lang_mask, _state = self._model._preprocess_observation(observation, train=False)
                for name, image, mask in zip(observation.images.keys(), images, image_masks, strict=True):
                    valid = bool(mask.detach().cpu().reshape(-1)[0].item())
                    if not valid or name == "right_wrist_0_rgb":
                        continue
                    tokens = self._model.paligemma_with_expert.embed_image(image)
                    pooled.append(tokens.mean(dim=1)[0].detach().float().cpu().numpy())
                    valid_names.append(name)
        else:
            # This checkpoint uses the authoritative frozen JAX PI0 path.
            # The call sequence mirrors Pi0.sample_actions: preprocess once,
            # then PaliGemma/SigLIP image embedding before action decoding.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
            observation = _model.Observation.from_dict(inputs)
            observation = _model.preprocess_observation(
                None, observation, train=False, tactile_type=self._model.tactile_type
            )
            for name in observation.images:
                valid = bool(np.asarray(observation.image_masks[name]).reshape(-1)[0])
                if not valid or name == "right_wrist_0_rgb":
                    continue
                tokens, _ = self._model.PaliGemma.img(observation.images[name], train=False)
                pooled.append(np.asarray(tokens[0].mean(axis=0), dtype=np.float32))
                valid_names.append(name)
        if valid_names != ["base_0_rgb", "left_wrist_0_rgb"]:
            raise RuntimeError(f"unexpected valid camera order: {valid_names}")
        feature = np.concatenate(pooled, axis=0).astype(np.float32, copy=False)
        return {
            "feature": feature,
            "feature_sha256": sha256_bytes(feature),
            "shape": list(feature.shape),
            "dtype": str(feature.dtype),
        }

    def infer(self, obs):
        diag = self._capture(obs)
        result = self._policy.infer(obs)
        result["diagnostics_visual_feature"] = diag["feature"]
        result["diagnostics_visual_feature_sha256"] = diag["feature_sha256"]
        return result


class ControlPolicy:
    def __init__(self, policy):
        self._policy = policy

    @property
    def metadata(self):
        return self._policy.metadata

    def infer(self, obs):
        return self._policy.infer(obs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--mode", choices=["control", "instrumented"], required=True)
    ap.add_argument("--policy-config", default="pi0_lora_tacfield_tabero")
    ap.add_argument("--policy-dir", required=True)
    ap.add_argument("--norm-stats-dir", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    torch.manual_seed(2026083101)
    torch.cuda.manual_seed_all(2026083101)
    policy = make_policy(args)
    wrapped = InstrumentedPolicy(policy) if args.mode == "instrumented" else ControlPolicy(policy)
    logging.info("Starting frozen π0 %s server on %s:%s", args.mode, socket.gethostname(), args.port)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=wrapped,
        host="0.0.0.0",
        port=args.port,
        metadata=dict(wrapped.metadata),
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
