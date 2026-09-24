# Semantic Task Representation Audit

## Finding

The previous Direct did **not** have a semantic task-transfer mechanism. Its 71D input was split into a 17D temporal sequence (6 nominal Cartesian commands + 7 phase indicators + **4 task one-hot coordinates**) and a 54D static condition (candidate force, scalar physics, and physical state/masks). No language, RGB, π0 hidden state, or VLA semantic feature entered Direct.

## What exists in the frozen π0 stack

| Representation | Archived in force branches? | Deterministically recomputable without simulator? | Dimension | Frozen? | Selected? |
|---|---:|---:|---:|---:|---:|
| Raw neutral language instruction | Yes | Yes | string | Yes by protocol | Source text |
| PaliGemma SentencePiece tokens/mask | No | Yes, exact π0 tokenizer | 48 max tokens | Yes | Intermediate |
| PaliGemma/Gemma input-token embeddings | No | Yes, checkpoint 49999 | 2048/token | Yes | **Primary** |
| Masked mean prompt embedding | No | Yes | 2048 | Yes | **Primary task vector** |
| Contextual language hidden states | No | Not as a task-only deterministic vector; π0 contextualizes jointly with current images | 2048/token | Model frozen | Not selected |
| Explicit pooled language output | No | No such π0 interface exists | NA | NA | Not selected |
| Visual-language fused representation | No | Requires an observation and is not a task-only constant | observation-dependent | Model frozen | Not selected |
| Previously cached visual feature | Separate visual diagnostic only | Yes for those scenes | 4096/pre-PCA | Frozen | Not selected |

The primary representation is the exact frozen π0/PaliGemma input embedding used by `Pi0.embed_prefix`: tokenize the **actual neutral instruction**, index checkpoint `PaliGemma/llm/embedder/input_embedding`, multiply by sqrt(2048) exactly as π0 does, and mask-mean valid prompt tokens. It contains no outcome, force, friction, frontier, branch ID, or target-evaluation statistic and requires no task-specific training.

The resulting vector is 2048D. Before any transfer result is viewed it is passed through one fixed, outcome-independent 2048→16 orthogonal Gaussian projection (seed 20260901), followed by deterministic non-affine LayerNorm and GELU. Projection weights are not trained. The frozen 16D vector replaces the role of task identity; the one-hot columns are removed from the GRU input.

## Capacity and leakage audit

- Old one-hot Direct trainable parameters: **27,777**.
- Semantic Direct trainable parameters: **28,033** (+256; only **0.92%** different).
- π0 and the language embedding table are frozen; no encoder fine-tuning occurs.
- All four target instructions are legal at B=0, but no target outcomes or target Direct branches are used.
- Contextual/fused π0 states were rejected because they are not archived and are observation-dependent, which would change more than task representation.
- No alternative embedding, projection dimension, architecture, optimizer, epoch count, threshold, or reward is swept.
- Root-scaling untouched TEST is excluded by path and never read.

Authoritative code: `/media/volume/newdata/exouser/tabero/Tabero-VTLA/src/openpi/models/tokenizer.py` and `/media/volume/newdata/exouser/tabero/Tabero-VTLA/src/openpi/models/pi0.py`. Authoritative checkpoint: `/media/volume/newdata/exouser/tabero/models/pi0_lora_tacfield_tabero/checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999`.
