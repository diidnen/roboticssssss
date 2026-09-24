# Multi-seed original-geom finger-friction sanity

Independent directory. Not 32×20. Not 18/19. Not Fig.B.

**Verdict: MIXED**

high < mid on seed 200014 is **not** a unique seed fluke, and **no seed** shows a stable Coulomb ladder. This dump grasp is not usable as clean friction-to-force evidence.

---

## 1. Seeds / episodes

Selected **only** by PRE validity, first eligible episode in `[1,0,2,…,9]`, never by force-curve appearance.

| seed | episode | deskbin_id | n_hulls | source |
|---|---|---|---|---|
| 200014 | 1 | 10 | 45 | reused previous sanity (not rerun) |
| 200002 | 1 | 9 | 70 | v4 TRAIN, first candidate |
| 200010 | 1 | 9 | 70 | v4 VAL, second candidate |
| 200019 | 1 | 10 | 45 | next stable candidate |
| 200020 | 1 | 9 | 70 | next stable candidate |

Excluded known UnStableError roots: 200003–200009, 200011–200013, 200015–200018. Not used.

First four screened candidates were all PRE_VALID at episode 1; nothing was skipped for looking “better.”

## 2. PRE validity (all three μ)

All five seeds: **VALID / VALID / VALID** at low, mid, high. qCR = 1.0 on every prefix.

Protocol frozen: official collider, deskbin μ = 0.30, finger 0.425 / 0.575 / 0.85 → μ_eff 0.3625 / 0.4375 / 0.575, query 4 N / 12 mm, official 20-point F grid.

## 3–5. Per-seed retention vs F

Bits are 20 points F=0.25 … 5.0. Rate = n/20. First-success F is auxiliary only.

### 200014 (reused)

| μ | n/20 | lowF | midF | highF | first F | longest run | reversals |
|---|---|---|---|---|---|---|---|
| low | **2** | 0.00 | 0.12 | 0.17 | 2.5 | 1 | 3 |
| mid | **17** | 0.67 | 0.88 | 1.00 | 0.75 | 13 | 2 |
| high | **4** | 0.17 | 0.00 | 0.50 | 0.75 | 1 | 6 |

Ordering: **low < mid > high**. high << mid.

low: `00000000010000000010`  
mid: `00111101111111111111`  
high: `00100000000000010101`

### 200002

| μ | n/20 | lowF | midF | highF | first F | longest | rev |
|---|---|---|---|---|---|---|---|
| low | **14** | 0.50 | 0.75 | 0.83 | 1.0 | 5 | 6 |
| mid | **10** | 0.00 | 0.50 | 1.00 | 2.25 | 7 | 4 |
| high | **10** | 0.00 | 0.50 | 1.00 | 2.25 | 9 | 2 |

Ordering: **low > mid ~= high**.

### 200010

| μ | n/20 | lowF | midF | highF | first F | longest | rev |
|---|---|---|---|---|---|---|---|
| low | **12** | 0.00 | 0.75 | 1.00 | 1.75 | 9 | 4 |
| mid | **14** | 0.17 | 0.88 | 1.00 | 1.25 | 10 | 4 |
| high | **12** | 0.33 | 0.62 | 0.83 | 1.25 | 9 | 4 |

Ordering: **low < mid > high** (mild; not the 200014 extreme).

### 200019

| μ | n/20 | lowF | midF | highF | first F | longest | rev |
|---|---|---|---|---|---|---|---|
| low | **19** | 0.83 | 1.00 | 1.00 | 0.5 | 19 | 0 |
| mid | **19** | 0.83 | 1.00 | 1.00 | 0.5 | 19 | 0 |
| high | **20** | 1.00 | 1.00 | 1.00 | 0.25 | 20 | 0 |

Ordering: **low ~= mid ~= high**. Ceiling / 10 g pose. High even retains at 0.25 N.

### 200020

| μ | n/20 | lowF | midF | highF | first F | longest | rev |
|---|---|---|---|---|---|---|---|
| low | **15** | 0.17 | 1.00 | 1.00 | 1.5 | 15 | 0 |
| mid | **13** | 0.17 | 0.75 | 1.00 | 1.0 | 12 | 2 |
| high | **13** | 0.50 | 0.50 | 1.00 | 1.0 | 8 | 4 |

Ordering: **low > mid ~= high**.

## 6–7. Aggregate (5 seeds)

P(retain | F):

| band | low | mid | high |
|---|---|---|---|
| all 20 F | 0.62 | **0.73** | 0.59 |
| low F (0.25–1.5) | 0.30 | 0.37 | 0.40 |
| mid F (1.75–3.5) | 0.73 | **0.80** | 0.53 |
| high F (3.75–5.0) | 0.80 | **1.00** | 0.87 |

If 200014 is removed, rates become low 0.75 / mid 0.70 / high 0.69 — **low is no longer hardest**.

0/5 seeds are `low < mid < high`.

## 8. Does high < mid still appear?

Yes, but the **extreme** 4 vs 17 is 200014-only.

- 200014: high 4 << mid 17
- 200010: high 12 < mid 14
- 200002, 200020: high ~= mid, and **low is higher**
- 200019: all saturated

So 200014 is a special *severity*, not a unique *direction*.

## 9. Anomalous geometry (200014 high << mid)

PRE on high is still an opposing ±x pinch (not split nx+pz). Pair ~58 mm, same as mid.

Classified as:

- post-query squeeze **8.83 N (high) vs 1.08 N (mid)** — high friction keeps a much harder query residual
- high successes are **isolated flicker** (4 singletons)
- on 13 F points where mid retains, high realizes squeeze then **loses contact** (CR < 0.3)
- one finger touches 2 hulls, same axis, not corner wrap

Likely **high-friction sticking after query** (pose/load change) plus **downstream contact loss**, not Coulomb “more friction → more retain.” 200010’s milder mid>high is the same direction without that squeeze gap.

200019’s 0.25 N high retain is the opposite ceiling: this pose/hull set can hold 10 g with almost no friction discrimination.

## 10. MIXED

Not PASS: aggregate is not low < mid < high; high is the *worst* overall rate; 0/5 ladders.

Not FAIL in the “μ does nothing” sense: mid owns the high-F band (1.00), 200014 still shows a huge μ effect, 200019 shows a ceiling.

It **is** MIXED in the intended sense: seed/contact geometry dominate, and this is **not** a clean Coulomb staircase.

## 11. One sentence

**high 不如 mid 不是单纯的 200014 seed effect；这个 dump grasp 本身就没有稳定的 Coulomb ordering。**
