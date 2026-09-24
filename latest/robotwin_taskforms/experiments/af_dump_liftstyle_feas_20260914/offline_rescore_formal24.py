"""Counterfactual: lift-style feas on frozen formal-24 preaction features.

Does not rerun physics. Not an online AF result.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
FORMAL = Path(
    "/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/"
    "experiments/af_dump_formal_single_root_20260914"
)
FORCE_NORM = 8.0
V4_GRID = np.round(np.arange(0.5, 8.0 + 1e-9, 0.05), 10)
LIFT_GRID = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], np.float32)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.command_gru = nn.GRU(10, 64, batch_first=True)
        self.condition = nn.Sequential(nn.Linear(54, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, step, cond):
        _, h = self.command_gru(step)
        return self.head(torch.cat([h[-1], self.condition(cond)], -1)).squeeze(-1)


def utility(p, f, max_force=8.0):
    return p * (max_force - f) / max_force - (1.0 - p)


def load_pack(device):
    pack = []
    for seed in (0, 1, 2):
        ckpt = torch.load(HERE / "models" / f"member_seed{seed}.pt", map_location=device, weights_only=False)
        model = Network().to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        pack.append((model, np.asarray(ckpt["mean"], np.float32), np.asarray(ckpt["std"], np.float32)))
    return pack


def predict(pack, xs, device):
    ps = []
    for model, mean, std in pack:
        x = (xs - mean[None, None]) / std[None, None]
        with torch.no_grad():
            p = torch.sigmoid(
                model(
                    torch.as_tensor(x[:, :, :10], device=device),
                    torch.as_tensor(x[:, 0, 10:], device=device),
                )
            )
        ps.append(p.cpu().numpy())
    return np.mean(ps, axis=0)


def sweep(pack, base, mu, grid, device):
    xs = np.stack([base] * len(grid)).astype(np.float32)
    for i, force in enumerate(grid):
        xs[i, :, 10] = float(force) / FORCE_NORM
        xs[i, :, 11] = float(mu)
    return predict(pack, xs, device)


def select(grid, p, mode):
    grid = np.asarray(grid, float)
    p = np.asarray(p, float)
    if mode == "argmax_p":
        i = int(np.argmax(p))
    elif mode == "v4_utility_maxF8":
        i = int(np.argmax(utility(p, grid, 8.0)))
    else:
        raise ValueError(mode)
    return float(grid[i]), float(p[i])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pack = load_pack(device)
    payload = json.loads((FORMAL / "FORMAL_PREACTION_INPUTS.json").read_text())
    core = json.loads((FORMAL / "FINAL_RESULTS.json").read_text())
    success = {}
    for rec_path in core["records"]:
        rec = json.loads(Path(rec_path).read_text())
        ctx = rec["context"]["id"]
        success[ctx] = {o["method"]: int(o["success"]) for o in rec["outcomes"]}

    variants = [
        ("pi0_transfer_v4grid_utility", False, V4_GRID, "v4_utility_maxF8"),
        ("pi0_transfer_1to5_utility", False, LIFT_GRID, "v4_utility_maxF8"),
        ("pi0_transfer_1to5_argmaxp", False, LIFT_GRID, "argmax_p"),
        ("hold_v4grid_utility", True, V4_GRID, "v4_utility_maxF8"),
        ("hold_1to5_utility", True, LIFT_GRID, "v4_utility_maxF8"),
        ("hold_1to5_argmaxp", True, LIFT_GRID, "argmax_p"),
    ]

    rows = []
    for ctx in payload["contexts"]:
        cid = ctx["context_id"]
        base = np.asarray(ctx["feature"]["sequence"], np.float32).copy()
        if base.shape != (8, 64):
            raise RuntimeError((cid, base.shape))
        mu_hat = float(ctx["posterior"]["posterior_moments"]["mean"])
        mu_true = float(ctx["friction"])
        v4_f = float(ctx["original_decision"]["selected_force_N"])
        v4_p = float(ctx["original_decision"]["predicted_success"])
        hold = base.copy()
        hold[:, :6] = 0.0
        row = {
            "context_id": cid,
            "friction": mu_true,
            "mu_hat": mu_hat,
            "v4_selected_N": v4_f,
            "v4_predicted_success": v4_p,
            "actual_AF": success[cid]["ActiveForcing"],
            "actual_Fixed8": success[cid]["Fixed-Strong 8N"],
            "actual_Nominal": success[cid]["Nominal Frozen VLA"],
            "choices": {},
        }
        for name, zero_motion, grid, mode in variants:
            feat = hold if zero_motion else base
            p = sweep(pack, feat, mu_hat, grid, device)
            f, psel = select(grid, p, mode)
            row["choices"][name] = {
                "selected_N": f,
                "p_at_selected": psel,
                "p_mean": float(np.mean(p)),
                "p_max": float(np.max(p)),
                "p_at_5N": float(p[int(np.argmin(np.abs(grid - 5.0)))]) if np.min(np.abs(grid - 5.0)) < 1e-6 else None,
            }
        rows.append(row)

    def summarize(name):
        fs = [r["choices"][name]["selected_N"] for r in rows]
        v4 = [r["v4_selected_N"] for r in rows]
        return {
            "mean_selected_N": float(np.mean(fs)),
            "median_selected_N": float(np.median(fs)),
            "min_selected_N": float(np.min(fs)),
            "max_selected_N": float(np.max(fs)),
            "mean_abs_delta_vs_v4": float(np.mean(np.abs(np.asarray(fs) - np.asarray(v4)))),
            "n_within_0.5N_of_v4": int(np.sum(np.abs(np.asarray(fs) - np.asarray(v4)) <= 0.5)),
            "n_exactly_5N": int(np.sum(np.isclose(fs, 5.0))),
            "n_leq_5N": int(np.sum(np.asarray(fs) <= 5.0 + 1e-9)),
        }

    out = {
        "claim_boundary": (
            "offline counterfactual on frozen formal-24 preaction features; "
            "not a new online AF success rate"
        ),
        "n": len(rows),
        "v4_mean_selected_N": float(np.mean([r["v4_selected_N"] for r in rows])),
        "variant_summary": {name: summarize(name) for name, *_ in variants},
        "rows": rows,
    }
    dest = HERE / "OFFLINE_FORMAL24_RESCORE.json"
    dest.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("v4 mean", out["v4_mean_selected_N"])
    for name, stats in out["variant_summary"].items():
        print(name, stats)
    print("wrote", dest)


if __name__ == "__main__":
    main()
