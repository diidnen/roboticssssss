#!/usr/bin/env python3
"""Build and train the E3 causally-clean force-feasibility supervision.

This is an offline read-only analysis of the frozen 72-branch collection.  It
does not modify the collection, evaluator, model architecture, feature
definition, split, seed, selector, or utility.  The only training change is
the binary-loss mask: official successes and audited force-attributable
failures contribute to loss; policy/placement failures are censored.
"""
from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

COLLECTION = Path("/media/volume/newdata/exouser/activeforcing_e3/P1_SIMPLIFIED_COLLECTION_FORMAL_20260903")
DECOMP = Path("/home/exouser/E3_DOWNSTREAM_FAILURE_DECOMPOSITION.csv")
OUT = Path("/home/exouser/E3_FORCE_FEASIBILITY_CLEAN_TRAINING")
CLEAN_CSV = Path("/home/exouser/E3_FORCE_FEASIBILITY_CLEAN_DATASET.csv")
TRAIN_ROOTS = {str(x) for x in range(7700, 7706)}
HELDOUT_ROOTS = {str(x) for x in range(7800, 7803)}
FORCES = [float(x) for x in range(1, 9)]
SEEDS = [0, 1, 2]
FMAX = 8.0


class Direct(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def ffloat(v, default=float("nan")):
    try:
        if v in (None, ""):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")


def load_decomposition() -> dict[tuple[str, float], dict]:
    out = {}
    with DECOMP.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[(str(row["root"]), float(row["requested_force_N"]))] = row
    return out


def step_audit(root: str, force: float, result: dict) -> dict:
    """Fixed audit rule for lift failures.

    A lift failure is force-attributable only if all four predeclared checks
    pass: contact was established, object height stayed below the 0.03 m lift
    threshold, mean measured force was below 1 N (or no force was sustained
    in the final ten samples), and a higher-force sibling lifted successfully.
    Otherwise it is censored.  The rule uses measured force and sibling lift
    outcomes, never requested force alone.
    """
    step_path = next((COLLECTION / f"P1_SIMPLIFIED_ROOT{root}_F{int(force)}N" / "logs").glob("*_steps.csv"))
    with step_path.open(newline="", encoding="utf-8") as f:
        steps = list(csv.DictReader(f))
    contact = [r for r in steps if r.get("contact") == "1"]
    dz = [ffloat(r.get("obj_dz"), 0.0) for r in steps]
    force_vals = [ffloat(r.get("measured_squeeze_N"), 0.0) for r in steps]
    force_vals = [x for x in force_vals if math.isfinite(x)]
    mean_force = ffloat(result["episode_row"].get("mean_measured_force_N"))
    final10 = statistics.fmean(force_vals[-10:]) if force_vals else float("nan")
    higher = []
    for hf in FORCES:
        if hf <= force:
            continue
        p = COLLECTION / f"P1_SIMPLIFIED_ROOT{root}_F{int(hf)}N" / "branch_result.json"
        if p.is_file():
            higher.append(json.loads(p.read_text()).get("y_lift", 0) == 1)
    higher_lift = any(higher)
    contact_established = bool(contact)
    below_lift_threshold = max(dz, default=0.0) < 0.03
    sustained_force_low = (math.isfinite(mean_force) and mean_force < 1.0) or (math.isfinite(final10) and final10 < 0.05)
    include = contact_established and below_lift_threshold and sustained_force_low and higher_lift
    reason = (
        "Contact established; object height remained below 0.03 m; sustained measured force was low; "
        "a higher-force sibling lifted successfully."
        if include else
        "Lift failure did not pass the fixed contact/height/measured-force/higher-force-repair audit."
    )
    return {
        "lift_failure_category": "LIFT_UNDER_FORCE" if include else "LIFT_POLICY_OR_GRASP_ERROR",
        "lift_audit_reason": reason,
        "contact_established": int(contact_established),
        "max_object_dz_m": max(dz, default=float("nan")),
        "lift_threshold_m": 0.03,
        "mean_measured_force_N_audit": mean_force,
        "final10_measured_force_N": final10,
        "higher_force_lift_repair": int(higher_lift),
        "step_log": str(step_path),
    }


def load_rows() -> tuple[list[dict], list[dict]]:
    decomposition = load_decomposition()
    rows = []
    lift_audit_rows = []
    dirs = [p for p in COLLECTION.glob("P1_SIMPLIFIED_ROOT*_F*N") if p.is_dir() and ".incomplete" not in str(p)]
    for branch_dir in sorted(dirs):
        result = json.loads((branch_dir / "branch_result.json").read_text(encoding="utf-8"))
        root, force = str(result["root_id"]), float(result["force_N"])
        if root not in TRAIN_ROOTS | HELDOUT_ROOTS:
            continue
        split = "TRAIN" if root in TRAIN_ROOTS else "HELDOUT"
        e = result["episode_row"]
        key = (root, force)
        lift, full = int(result["y_lift"]), int(result["y_full"])
        audit = {}
        if lift == 0:
            audit = step_audit(root, force, result)
            lift_audit_rows.append({"context_id": f"libero10_task5_root{root}", "root": root, "requested_force_N": force,
                                    "measured_force_N": e.get("mean_measured_force_N"), "lift_success": lift,
                                    "full_task_success": full, **audit})
            category = audit["lift_failure_category"]
            failure_stage, reason = "lift", audit["lift_audit_reason"]
            object_dropped = 0
            slipped = 0
            lost_contact = 0
            transport_completed = 0
            placement_attempted = 0
            final_eval = int(e.get("official_success", full))
        elif full == 1:
            category = "FULL_TASK_SUCCESS"
            failure_stage, reason = "", ""
            object_dropped = slipped = lost_contact = 0
            transport_completed, placement_attempted, final_eval = 1, 1, int(e.get("official_success", full))
        else:
            d = decomposition.get(key)
            if d is None:
                raise RuntimeError(f"missing prior decomposition row for {key}")
            category = d["category"]
            failure_stage, reason = d["failure_stage"], d["failure_reason"]
            object_dropped = int(d["object_dropped_after_lift"])
            slipped = int(d["object_slipped"])
            lost_contact = int(d["lost_contact"])
            transport_completed = int(d["transport_completed"])
            placement_attempted = int(d["placement_attempted"])
            final_eval = int(d["final_evaluator_success"])
        if category == "FULL_TASK_SUCCESS":
            label, include, exclusion = 1, 1, ""
        elif category in {"UNDER_FORCE_STRONG", "LIFT_UNDER_FORCE"}:
            label, include, exclusion = 0, 1, ""
        else:
            label, include = "CENSORED", 0
            exclusion = "Non-force-attributable policy/trajectory/placement failure; excluded from binary force loss."
        x_path = branch_dir / "x_observations.jsonl"
        x0 = json.loads(x_path.open(encoding="utf-8").readline())
        state = np.asarray(x0["x_state_7"], dtype=np.float32)
        if state.shape != (7,) or not np.isfinite(state).all():
            raise RuntimeError(f"invalid x_state_7 in {branch_dir}")
        row = {
            "context_id": f"libero10_task5_root{root}", "root": root, "split": split,
            "branch_id": result["branch_id"], "requested_force": force, "requested_force_N": force,
            "measured_force": e.get("mean_measured_force_N"), "measured_force_N": e.get("mean_measured_force_N"),
            "peak_measured_force_N": e.get("peak_measured_force_N"), "lift_success": lift,
            "full_task_success": full, "original_failure_category": category,
            "force_feasibility_label": label, "include_in_force_training": include,
            "exclusion_reason": exclusion, "failure_stage": failure_stage, "failure_reason": reason,
            "object_dropped_after_lift": object_dropped, "object_slipped": slipped,
            "lost_contact": lost_contact, "transport_completed": transport_completed,
            "placement_attempted": placement_attempted, "final_evaluator_success": final_eval,
            "integrated_measured_force_Ns": e.get("integrated_measured_force_Ns"),
            "object_height_max_m": audit.get("max_object_dz_m", ""),
            "contact_established": audit.get("contact_established", ""),
            "higher_force_lift_repair": audit.get("higher_force_lift_repair", ""),
            "step_log": audit.get("step_log", ""),
        }
        rows.append(row)
    rows.sort(key=lambda r: (int(r["root"]), float(r["requested_force_N"])))
    if len(rows) != 72 or len({(r["root"], r["requested_force_N"]) for r in rows}) != 72:
        raise RuntimeError(f"expected 72 unique branches, got {len(rows)}")
    return rows, lift_audit_rows


def clean_frontiers(rows: list[dict]) -> list[dict]:
    out = []
    for root in sorted({r["root"] for r in rows}, key=int):
        q = sorted([r for r in rows if r["root"] == root], key=lambda r: r["requested_force_N"])
        known = [(float(r["requested_force_N"]), int(r["force_feasibility_label"])) for r in q if r["force_feasibility_label"] != "CENSORED"]
        neg = [f for f, y in known if y == 0]
        pos = [f for f, y in known if y == 1]
        sequence = " ".join(str(r["force_feasibility_label"])[0] if r["force_feasibility_label"] == "CENSORED" else str(r["force_feasibility_label"]) for r in q)
        if not neg or not pos:
            typ, est = "UNIDENTIFIABLE", ""
        elif max(neg) < min(pos):
            has_censored = any(r["force_feasibility_label"] == "CENSORED" for r in q)
            typ, est = ("PARTIAL_FRONTIER" if has_censored else "CLEAR_FRONTIER"), min(pos)
        else:
            typ, est = "UNIDENTIFIABLE", ""
        out.append({"context_id": q[0]["context_id"], "root": root, "split": q[0]["split"],
                    "clean_labels_1_to_8N": sequence, "frontier_type": typ,
                    "estimated_minimum_force_N": est, "known_negative_forces": ",".join(map(str, neg)),
                    "known_positive_forces": ",".join(map(str, pos)),
                    "censored_forces": ",".join(str(int(r["requested_force_N"])) for r in q if r["force_feasibility_label"] == "CENSORED")})
    return out


def auc(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    n1, n0 = int(y.sum()), len(y) - int(y.sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    i = 0
    while i < len(p):
        j = i + 1
        while j < len(p) and p[order[j]] == p[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def spearman(x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])


def fit(x, y, seed, mean, std, mask=None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = Direct()
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    xt, yt = torch.tensor((x - mean) / std, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    usable = np.flatnonzero(mask)
    for epoch in range(80):
        order = np.random.default_rng(seed * 1009 + epoch).permutation(usable)
        for start in range(0, len(order), 64):
            idx = order[start:start + 64]
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(xt[idx]), yt[idx])
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model.eval()


def load_x(rows):
    x = []
    for r in rows:
        p = COLLECTION / f"P1_SIMPLIFIED_ROOT{r['root']}_F{int(float(r['requested_force_N']))}N" / "x_observations.jsonl"
        state = np.asarray(json.loads(p.open(encoding="utf-8").readline())["x_state_7"], dtype=np.float32)
        x.append(np.r_[state, float(r["requested_force_N"]) / FMAX].astype(np.float32))
    return np.stack(x)


def metrics_for(rows, probs, method):
    labeled = [i for i, r in enumerate(rows) if r["force_feasibility_label"] != "CENSORED"]
    y, p = np.asarray([int(rows[i]["force_feasibility_label"]) for i in labeled], float), np.asarray([probs[i] for i in labeled], float)
    out = {"method": method, "heldout_total": len(rows), "heldout_labeled": len(y), "heldout_censored": len(rows) - len(y),
           "positives": int(y.sum()), "negatives": int(len(y) - y.sum()), "MAE": float(np.mean(np.abs(p - y))),
           "Brier": float(np.mean((p - y) ** 2)), "AUROC": auc(y, p),
           "positive_minus_negative_probability": float(np.mean(p[y == 1]) - np.mean(p[y == 0]))}
    pairs = []
    for root in sorted({r["root"] for r in rows}):
        q = [(float(r["requested_force_N"]), probs[i], int(r["force_feasibility_label"])) for i, r in enumerate(rows) if r["root"] == root and r["force_feasibility_label"] != "CENSORED"]
        for fn, pn, yn in q:
            for fp, pp, yp in q:
                if yn == 0 and yp == 1 and fn < fp:
                    pairs.append(float(pp > pn) + 0.5 * float(pp == pn))
    out["within_context_negative_before_positive_pairwise_accuracy"] = float(np.mean(pairs)) if pairs else float("nan")
    out["within_context_order_pairs"] = len(pairs)
    out["low_force_1_to_3_mean_probability"] = float(np.mean([probs[i] for i, r in enumerate(rows) if float(r["requested_force_N"]) <= 3]))
    out["explicit_positive_mean_probability"] = float(np.mean([probs[i] for i in labeled if int(rows[i]["force_feasibility_label"]) == 1]))
    return out


def main():
    rows, lift_audit = load_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    write_csv(CLEAN_CSV, rows, fields)
    write_csv(OUT / "E3_LIFT_FAILURE_AUDIT.csv", lift_audit)
    fronts = clean_frontiers(rows)
    write_csv(OUT / "E3_CLEAN_CONTEXT_FRONTIERS.csv", fronts)
    stats = []
    for split in ("TRAIN", "HELDOUT"):
        q = [r for r in rows if r["split"] == split]
        included = [r for r in q if r["include_in_force_training"] == 1]
        stats.append({"split": split, "original_branch_count": len(q), "clean_included_count": len(included),
                      "positive_count": sum(r["force_feasibility_label"] == 1 for r in included),
                      "negative_count": sum(r["force_feasibility_label"] == 0 for r in included),
                      "censored_count": sum(r["force_feasibility_label"] == "CENSORED" for r in q)})
    write_csv(OUT / "E3_CLEAN_DATASET_SPLIT_STATS.csv", stats)
    context_force = []
    for root in sorted({r["root"] for r in rows}, key=int):
        q = sorted([r for r in rows if r["root"] == root], key=lambda r: r["requested_force_N"])
        context_force.append({"context_id": q[0]["context_id"], "root": root, "split": q[0]["split"],
                              "force_N": int(q[0]["requested_force_N"]), "force_feasibility_label": q[0]["force_feasibility_label"],
                              "include_in_force_training": q[0]["include_in_force_training"], "full_task_success": q[0]["full_task_success"],
                              "original_failure_category": q[0]["original_failure_category"]})
        for r in q[1:]:
            context_force.append({"context_id": r["context_id"], "root": root, "split": r["split"], "force_N": int(r["requested_force_N"]),
                                  "force_feasibility_label": r["force_feasibility_label"], "include_in_force_training": r["include_in_force_training"],
                                  "full_task_success": r["full_task_success"], "original_failure_category": r["original_failure_category"]})
    write_csv(OUT / "E3_CLEAN_CONTEXT_FORCE_LABELS.csv", context_force)

    train, held = [r for r in rows if r["split"] == "TRAIN"], [r for r in rows if r["split"] == "HELDOUT"]
    x_all, x_train = load_x(rows), load_x(train)
    mean, std = x_train.mean(0), x_train.std(0); std[std < 1e-6] = 1.0
    clean_mask = np.asarray([r["include_in_force_training"] == 1 for r in train])
    y_clean = np.asarray([int(r["force_feasibility_label"]) if r["force_feasibility_label"] != "CENSORED" else 0 for r in train], dtype=np.float32)
    x_held = load_x(held)
    models = []
    for seed in SEEDS:
        model = fit(x_train, y_clean, seed, mean, std, clean_mask)
        models.append(model)
        torch.save({"state_dict": model.state_dict(), "target": "force_feasibility_label", "seed": seed,
                    "architecture": "Linear(8,32)-ReLU-Linear(32,16)-ReLU-Linear(16,1)", "epochs": 80, "batch_size": 64,
                    "optimizer": "AdamW", "lr": 8e-4, "weight_decay": 1e-4, "train_roots": sorted(TRAIN_ROOTS),
                    "heldout_roots": sorted(HELDOUT_ROOTS), "normalization_mean": mean.tolist(), "normalization_std": std.tolist(),
                    "clean_train_included": int(clean_mask.sum()), "censored_train_excluded": int((~clean_mask).sum()),
                    "test_used": False, "selector_or_utility_used": False}, OUT / f"FULLTASK_DIRECT_CLEAN_seed{seed}.pt")
    with torch.no_grad():
        clean_seed_p = np.stack([torch.sigmoid(m(torch.tensor((x_held - mean) / std, dtype=torch.float32))).numpy() for m in models])
    clean_p = clean_seed_p.mean(0)
    old_dir = COLLECTION / "P1_SIMPLIFIED_MATCHED_DIRECT"
    old_models = []
    for seed in SEEDS:
        payload = torch.load(old_dir / f"FULLTASK_DIRECT_seed{seed}.pt", map_location="cpu", weights_only=False)
        m = Direct(); m.load_state_dict(payload["state_dict"]); m.eval(); old_models.append(m)
    with torch.no_grad():
        old_seed_p = np.stack([torch.sigmoid(m(torch.tensor((x_held - mean) / std, dtype=torch.float32))).numpy() for m in old_models])
    old_p = old_seed_p.mean(0)
    curve = []
    for i, r in enumerate(held):
        curve.append({"context_id": r["context_id"], "root": r["root"], "force_N": int(r["requested_force_N"]),
                      "original_p": float(old_p[i]), "clean_p": float(clean_p[i]), "force_label": r["force_feasibility_label"],
                      "full_task_success_reference": r["full_task_success"], "category": r["original_failure_category"],
                      "measured_force_N": r["measured_force_N"], "censored": int(r["force_feasibility_label"] == "CENSORED")})
    write_csv(OUT / "E3_HELDOUT_ORIGINAL_VS_CLEAN_FORCE_CURVES.csv", curve)
    compare = [metrics_for(held, old_p, "FULLTASK_DIRECT_original_checkpoint"), metrics_for(held, clean_p, "FULLTASK_DIRECT_CLEAN")]
    write_csv(OUT / "E3_CLEAN_HELDOUT_METRICS.csv", compare)
    # Threshold is a diagnostic convention only; it is not used for selection.
    pred_front = []
    for root in sorted(HELDOUT_ROOTS, key=int):
        q = sorted([(float(r["requested_force_N"]), clean_p[i]) for i, r in enumerate(held) if r["root"] == root], key=lambda z: z[0])
        crossing = next((f for f, p in q if p >= 0.5), "N/A")
        max_force = max(q, key=lambda z: (z[1], -z[0]))[0]
        pred_front.append({"context_id": f"libero10_task5_root{root}", "clean_probability_threshold": 0.5,
                           "predicted_minimum_feasible_force_N_at_threshold": crossing,
                           "clean_probability_argmax_force_N": max_force,
                           "clean_probability_1_to_8N": " ".join(f"{p:.6f}" for _, p in q)})
    write_csv(OUT / "E3_HELDOUT_CLEAN_PREDICTED_FRONTIERS.csv", pred_front)
    type_counts = Counter(x["frontier_type"] for x in fronts)
    summary = {
        "status": "E3_CLEAN_FORCE_SUPERVISION_RETRAIN_COMPLETE", "original_train_samples": 48,
        "clean_train_samples": sum(x["include_in_force_training"] == 1 and x["split"] == "TRAIN" for x in rows),
        "clean_positives": sum(x["force_feasibility_label"] == 1 and x["split"] == "TRAIN" for x in rows),
        "clean_negatives": sum(x["force_feasibility_label"] == 0 and x["split"] == "TRAIN" for x in rows),
        "censored_train": sum(x["force_feasibility_label"] == "CENSORED" and x["split"] == "TRAIN" for x in rows),
        "lift_under_force": f"{sum(x['lift_failure_category'] == 'LIFT_UNDER_FORCE' for x in lift_audit)}/7",
        "lift_censored": f"{sum(x['lift_failure_category'] != 'LIFT_UNDER_FORCE' for x in lift_audit)}/7",
        "frontier_type_counts": dict(type_counts), "contexts_with_clear_force_frontier": type_counts["CLEAR_FRONTIER"],
        "contexts_without_clear_frontier": 9 - type_counts["CLEAR_FRONTIER"],
        "metrics": compare, "artifacts": [str(CLEAN_CSV), str(OUT)],
        "rules": {"positive": "full_task_success=1", "negative": "UNDER_FORCE_STRONG or audited LIFT_UNDER_FORCE", "censored": "POLICY_TRAJECTORY or PLACEMENT or failed lift audit", "threshold_diagnostic_only": 0.5},
    }
    write_json(OUT / "E3_CLEAN_FORCE_SUPERVISION_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
