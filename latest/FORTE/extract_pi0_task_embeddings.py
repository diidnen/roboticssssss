#!/usr/bin/env python3
"""Extract the exact frozen pi0/PaliGemma input-token embeddings for four prompts.

This is deliberately task-only and simulator-free.  It restores the authoritative
JAX checkpoint on CPU, indexes the language input embedding table with the exact
tokens produced by pi0's PaligemmaTokenizer, and mask-mean pools valid prompt
tokens.  No outcome, force, friction, image, or evaluation record is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

from openpi.models import model as model_lib
from openpi.models.tokenizer import PaligemmaTokenizer


PROMPTS = {
    0: "pick up the alphabet soup and place it in the basket",
    1: "pick up the cream cheese and place it in the basket",
    5: "pick up the tomato sauce and place it in the basket",
    6: "pick up the butter and place it in the basket",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    tokenizer = PaligemmaTokenizer(max_len=48)
    params = model_lib.restore_params(
        args.checkpoint / "params", restore_type=np.ndarray, dtype=None
    )
    embedding_leaf = params["PaliGemma"]["llm"]["embedder"]["input_embedding"]
    # OpenPI's restore helper unwraps the nnx ``value`` leaf in current
    # checkpoints; retain compatibility with older structured restores.
    if isinstance(embedding_leaf, dict):
        embedding_leaf = embedding_leaf["value"]
    raw_table = np.asarray(embedding_leaf)
    print(f"restored input embedding: shape={raw_table.shape} dtype={raw_table.dtype}", flush=True)
    if raw_table.shape != (257152, 2048):
        raise RuntimeError(f"unexpected embedding table shape: {raw_table.shape}")

    vectors = []
    rows = []
    token_rows = {}
    for task, prompt in PROMPTS.items():
        ids, mask = tokenizer.tokenize(prompt)
        ids = np.asarray(ids, dtype=np.int64)
        mask = np.asarray(mask, dtype=bool)
        # PI0.embed_prefix multiplies by sqrt(width); this common positive scalar
        # is retained for exactness and has no effect on cosine similarity.
        # Cast only the handful of selected rows.  Converting the full 1 GiB
        # bfloat16 table would create an unnecessary 2 GiB float32 copy.
        token_emb = np.asarray(raw_table[ids[mask]], dtype=np.float32) * np.sqrt(raw_table.shape[1])
        pooled = token_emb.mean(axis=0).astype(np.float32)
        vectors.append(pooled)
        rows.append(
            {
                "task": task,
                "prompt": prompt,
                "valid_tokens": int(mask.sum()),
                "embedding_dim": int(pooled.size),
                "l2_norm": float(np.linalg.norm(pooled)),
            }
        )
        token_rows[str(task)] = ids[mask].tolist()

    out = np.stack(vectors)
    if not np.isfinite(out).all() or np.linalg.matrix_rank(out) < 2:
        raise RuntimeError("invalid or degenerate semantic embeddings")
    np.save(args.output / "PI0_TASK_EMBEDDINGS.npy", out)
    (args.output / "PI0_TASK_EMBEDDING_METADATA.json").write_text(
        json.dumps(
            {
                "status": "EXTRACTED_BEFORE_SEMANTIC_TRANSFER_TRAINING",
                "checkpoint": str(args.checkpoint),
                "checkpoint_metadata_sha256": sha(args.checkpoint / "_CHECKPOINT_METADATA"),
                "tokenizer": "openpi.models.tokenizer.PaligemmaTokenizer(max_len=48)",
                "pooling": "masked mean of valid input token embeddings after pi0 sqrt(2048) scaling",
                "frozen": True,
                "contains_outcome_force_friction": False,
                "rows": rows,
                "token_ids": token_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
