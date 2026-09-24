# Why commanded F = 0.25 N still retains (diagnostic)

Not paper data. Query, controller, task criterion, 32×20, official 18/19, and Fig.B were not changed.

Fixed probe: seed **200014**, episode **1**, PRE_VALID (CR = 1.0, nx+nx), deskbin id **10**, official query 4 N / 12 mm, grasp-surface isolation on, inner/bottom/table/balls/fingers unchanged, maxF = 5, scripted remainder unchanged.

Convention used below:

- `F_cmd = 0.25 N` is the **single-finger** force reference.
- Inner loop reference is `force_ref = 2 F_cmd = 0.50 N` bilateral.
- `N = min(N_L, N_R)`, `s = 2 N`.
- Ideal pinch: `F_friction,max ≈ 2 μ_eff N`.

---

## 1. F_cmd = 0.25 N squeeze time series

Full log: `runs/normal_mu0.425/steps.jsonl` (3286 steps, 13.14 s). Plot: `runs/normal_mu0.425/timeseries.png`.

| Phase | t (s) | s mean | s median | s max | N_min median | bilateral |
|---|---|---|---|---|---|---|
| query end / handoff | 0 | **0.00** | 0.00 | — | 0.00 | no |
| realize (150 steps) | 0.00–0.60 | 0.20 | 0.009 | 7.96 at t = 0.012 (1 frame) | 0.004 | 50% |
| **lift** | 0.61–1.38 | **0.58** | **0.56** | 3.44 | **0.28** | 99% |
| pour1 | 1.39–4.28 | 0.45 | 0.22 | 6.79 (impact) | 0.11 | 80% |
| pour2 | 4.29–5.11 | 0.18 | 0.18 | 0.23 | 0.089 | 100% |
| pour3 | 5.11–5.94 | 0.17 | 0.17 | 0.22 | 0.084 | 100% |
| delay | 5.94–13.14 | **0.17** | **0.17** | 0.26 | **0.083** | 100% |

TEST 1 verdict: **B — actual squeeze is near 0.25 N**, not a stuck high load.

- After F_cmd is set, measured squeeze is **not** held at 1 / 2 / 5 N.
- Lift median `s = 0.56 N` is the inner bilateral reference `0.50 N`.
- Delay lives at `s ≈ 0.17 N` (`N ≈ 0.08 N`).
- A single-frame ~8 N blip happens at realize re-contact (t = 0.012 s); 3% of realize steps have `s > 1 N`. Pour1 has brief impact spikes. These are not a sustained preload.

Retention: CR = 0.957, drop = 0, irrecoverable = false. Official dump still fails (balls not in band).

---

## 2. Query → 0.25 N unload (TEST 5)

Query observables (existing probe, not retuned):

- during query: measured single-finger p95 = **9.52 N**, mean = 3.64 N
- query contact_ratio = **0.75**
- `final_bilateral_contact = false`

Handoff timeline after F_cmd = 0.25 N:

| t after F_cmd | s (N) | N_min (N) | phase |
|---|---|---|---|
| query end | 0.00 | 0.00 | post_query |
| F_cmd set | 0.00 | 0.00 | realize |
| +0.1 s | 0.00 | 0.00 | realize |
| +0.2 s | 0.33 | 0.17 | realize |
| +0.5 s | 0.16 | 0.08 | realize |
| lift start ~0.61 s / +1.0 s | 0.48 | 0.24 | lift, z rising |

**This 0.25 N rollout does not carry query high preload into later motion.** Query already lost bilateral contact on the return. Realize starts from ~0 N and rebuilds a weak pinch. Nominal setpoint 0.25 N and actual contact load are in the same ballpark, not “setpoint 0.25 with retained 5–9 N”.

---

## 3–6. Mass, mg, μ_eff, friction capacity (TEST 2)

Read from PhysX, not guessed:

| Quantity | Value |
|---|---|
| deskbin mass | **0.0100 kg** (`PhysxRigidDynamicComponent.mass`; `af_object_mass_kg` is unset, Actor default `set_mass(0.01)`) |
| 5 balls | 0.0001 kg each |
| g | 9.810 m/s² |
| mg (deskbin only) | **0.0981 N** |
| finger μ | **0.300** (fl_link7/8, measured) |
| object-side grasp μ | 0.425 |
| combine | PhysX 5 **eAVERAGE** (no SAPIEN combine API) |
| **μ_eff** | **0.5 × (0.30 + 0.425) = 0.3625** |

At `F_cmd = 0.25 N`, N is the single-finger reference **0.25 N**, so `s_cmd = 0.50 N`.

| N used | F_friction,max = 2 μ_eff N | / mg |
|---|---|---|
| commanded 0.25 | **0.181 N** | **1.85** (enough) |
| actual lift N_min median 0.279 | 0.203 N | 2.07 (enough) |
| actual delay N_min median 0.083 | **0.060 N** | **0.61** (not enough) |

So: **0.25 N is not physically absurd for this 10 g bin** if that normal force is actually applied. During lift it approximately is. During delay it is not — Coulomb alone cannot explain the hold.

---

## 7. Near-zero-friction counterfactual (TEST 3)

Same PRE_VALID snapshot, F_cmd = 0.25 N, remainder unchanged. Only grasp-surface μ → **0.001**. Finger / inner / bottom / table / balls unchanged.

| | μ_grasp = 0.425 | μ_grasp = 0.001 |
|---|---|---|
| μ_eff (eAVERAGE) | 0.3625 | **0.1505** (finger still 0.30) |
| retention | 1 | **1** |
| contact_ratio | 0.957 | **0.913** |
| drop | 0 | 0 |
| delay s median | 0.165 | 0.112 |
| delay Coulomb / mg | 0.61 | **0.17** |

Near-zero did **not** drop. It is slightly shakier (more bilateral flicker in pour1, CR 0.91 vs 0.96) but still retains.

Caveat: because finger μ stayed 0.30, μ_eff is 0.15, not ~0. Even with that, delay friction capacity is 0.017 N vs mg 0.098 N. Friction is not doing the delay hold.

This fork is diagnostic only. Do not put it in the paper.

---

## 8–9. Contact geometry (TEST 4)

Hull audit (object frame, scale 0.08 applied because raw verts were in cm-like units):

- grasp_nx/px/pz/nz thickness **≈ 5.5 mm**
- AABB of the four grasp slabs **do not overlap**
- inner/bottom not in finger contacts (collision filter still working)

Keyframe contacts (`runs/normal_mu0.425/keyframes/`):

**realize_end (still on table, about to lift)**

- fl_link7: `grasp_nx` + `grasp_pz` (2 hulls)
- fl_link8: `grasp_nx` + `grasp_pz` (2 hulls)
- force-bearing normals nearly antiparallel (dot ≈ −0.97)
- both fingers on the **nx/pz corner**, not a simple left/right wall pinch
- plot: `contacts_realize_end.png`; camera: `frames/realize_end.png`

**terminal / delay (bin in air, z ≈ 1.03 m)**

- both fingers: `grasp_nx` + `grasp_nz` + `grasp_pz` (**3 hulls each**)
- mixed normals (min dot −0.95, max +1.0) — not a single opposing pair
- pair distance ≈ 43 mm vs 5.5 mm wall thickness: contacts wrap around the split convex set

So: **yes, multi-hull contact / corner wedging is present.** Split hulls are not thicker than a visual wall in the AABB sense (5.5 mm), but the corner of two perpendicular 5.5 mm slabs is a geometric pocket. Fingers are simultaneously on two (then three) angled convex hulls.

---

## 10. Classification

**MIXED**

| Class | This evidence |
|---|---|
| A CONTROLLER / PRELOAD | **No.** Actual squeeze tracks ~0.2 N, not leftover query 2–9 N. |
| B LOW MASS / FRICTION ENOUGH | **Yes for lift** (N ≈ 0.28, 2 μ N > mg). **No for delay** (0.61× mg). |
| C GEOMETRIC WEDGING / FORM CLOSURE | **Yes.** nx/pz (then nx/nz/pz) multi-hull; near-zero grasp μ still retains. |
| D MIXED | **Selected.** B explains why 0.25 N is not crazy; C explains why it still holds when Coulomb is short. |
| E UNKNOWN | Not needed; the five tests distinguish A from B/C. |

Do not treat this as a Coulomb F_min result, and do not treat 0.25 N retain as proof of “geometry only” either.

---

## 11. One sentence

**0.25 N 还能拿住，是因为 commanded 力确实卸到了约 0.2 N（不是 query 残载），10 g deskbin 让 lift 阶段 Coulomb 勉强够用，同时两指卡在 split hull 的 nx/pz 转角上，near-zero grasp μ 也仍 retain。**

Artifacts: `DIAG_RESULT.json`, `MASS_MU_HULL.json`, `runs/normal_mu0.425/`, `runs/nearzero_mu0.001/`.
