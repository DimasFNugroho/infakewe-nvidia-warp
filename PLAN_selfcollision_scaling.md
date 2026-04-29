# Plan: Yarn Self-Collision, Denser Winding, and Real-Time Performance

**Status**: Decisions resolved — ready to implement  
**Target hardware**: NVIDIA RTX 6000 Ada  
**Baseline**: 120 particles, 20 substeps, 10 constraint iter, ~60 FPS (estimated)

---

## 1. Goals

| Goal                 | Description                                                                      |
|----------------------|----------------------------------------------------------------------------------|
| **Self-collision**   | Yarn segments collide with each other (detect and resolve yarn-vs-yarn contacts) |
| **Denser winding**   | Significantly more helical wraps on Roll A visible at runtime                    |
| **Particle scaling** | Increase from 120 to ~500–5000 particles without dropping below target FPS       |
| **Real-time budget** | 60 FPS target (16.6 ms/frame); 30 FPS acceptable minimum                         |

---

## 2. Baseline Performance Analysis

### Current bottleneck hypothesis

At 120 particles × 20 substeps × 10 constraint iterations = **2,000 inner passes/frame**.  
Each pass launches approximately 8 small Warp kernels (stretch even/odd, bend, VF project, EE project, etc.)  
→ **~16,000 kernel launches per frame**

Estimated launch overhead: ~5 µs per launch on CUDA  
→ **~80 ms/frame in launch overhead alone**, regardless of arithmetic.

This is the primary bottleneck — not FLOPs, not memory bandwidth.

### Secondary bottlenecks (order of suspicion)

1. Brute-force VF/EE detection every substep: O(N × T) and O(E_yarn × E_obstacle) — tolerable now because the obstacle is small (~hundreds of triangles), but will grow with N.
2. `pos_wp.numpy()` sync once per frame — unavoidable for rendering, but currently the only per-frame CPU–GPU sync point (acceptable).
3. `omega_a_wp.numpy()` and `angle_a_wp.numpy()` reads per frame — also one sync each, manageable.

### Why the OGC paper's 1M-vertex number does not directly transfer

- The paper uses **VBD (Vertex Block Descent)** — embarrassingly parallel per vertex.
- We use **PBD with red-black coloring** — Gauss-Seidel along the yarn chain, which caps parallelism at N/2.
- The paper uses **LBVH + conservative-bound redetection** — contact detection only re-runs when particles move past a threshold.
- We re-run detection every substep.
- Their workload is **2D cloth** (dense grid); ours is **1D rope** (thin chain). Less arithmetic per kernel call, more launch overhead as fraction of total.

**Realistic target for this codebase on RTX 6000**: 2,000–5,000 particles at 60 FPS after optimization. Sufficient for a multi-layer winding demo.

---

## 3. Phase Plan

### Phase 1 — Eliminate Launch Overhead (CUDA Graph Capture)

**Goal**: Make the existing 120-particle scene CPU-overhead-free, then profile to confirm.

**What to do**:
- Wrap the entire inner substep loop (integrate → stretch → bend → VF project → EE project → friction → velocity update → Roll A/B kernel) inside a `wp.capture_begin()` / `wp.capture_end()` block.
- The captured graph is replayed via `wp.capture_launch()` each frame — one GPU submission regardless of how many kernels are inside.
- Per-substep CPU state reads (`omega_a_wp.numpy()`, etc.) must be moved outside the graph (once per frame).

**Constraints**:
- CUDA graphs require that array pointers and launch dimensions are fixed across all replays. Dynamic parameters (pull_speed, bearing_damping, etc.) must be passed via GPU-side scalar arrays that get written before graph replay.
- Contact arrays (VFContacts, EEContacts) may need their count buffers updated outside the graph if detection runs inside. Alternatively, detection can be run outside the graph (it's cheap at current N) and only the projection kernels inside.

**Expected outcome**:
- 5–20× throughput gain at current particle count.
- Unlocks ~500–1000 particles within the same 16.6 ms budget without any other changes.

**Deliverables**:
- Modified `sim_step()` in `yarn_rolls_ogc_gui.py` using CUDA graph replay.
- `wp.ScopedTimer` measurements before and after to confirm the gain.

---

### Phase 2 — Yarn Self-Collision Detection (algorithm5.py + algorithm6.py)

**Goal**: Detect and resolve yarn-vs-yarn edge-edge contacts using the same OGC offset framework.

**Design choice: EE (edge-edge) over VV (vertex-vertex)**
- Yarn is a thin filament; EE contacts capture close-approach of non-adjacent segments accurately.
- VV (particle-particle sphere collision) is cheaper but misses cases where segment midpoints are close while endpoints are far.
- **Decision: EE self-collision**, consistent with OGC algorithms 1–4.

**Detection (new algorithm5.py)**:
- For each pair of yarn edges (i, i+1) and (j, j+1), compute the minimum distance between the two line segments.
- Skip topologically adjacent pairs: skip if |i − j| ≤ 2 (share a vertex or a particle).
- Brute force is O(E²) ≈ O(N²). At N=120 → 14,000 pairs, cheap. At N=2000 → ~4M pairs, expensive.
- **Broad phase via `wp.HashGrid`**: hash edge midpoints into grid cells of size ≈ `REST_LEN + 2r`. For each edge, query only cells within radius. Reduces pairs from O(N²) to O(N × k) where k is the average neighbors (~10–30 for sparse yarn).

**Resolution (new algorithm6.py)**:
- Identical structure to `project_ee` in algorithm4.py, but the two edges are both yarn edges (indices into `pos`), not yarn-vs-obstacle.
- Apply position correction to all four endpoints (two per yarn edge), weighted by `inv_mass`.
- Same OGC contact distance `d = 2r` (both filaments have radius r).

**Contact array capacity**:
- Self-EE contacts expected to be sparse during normal operation (yarn rarely crosses itself in winding).
- Conservative allocation: `max_contacts = N * 4` should be sufficient.

**Skip list for winding**:
- When yarn wraps helically, consecutive loops are always within `2r` of each other by design (they're wound at `orbit_r` on the same roll).
- To avoid self-collision projections fighting the winding geometry, wound particles need a skip list or those contacts need to be suppressed for particle indices within the wound region.
- **Simple approach**: skip self-EE contact pairs where both edges are in the first `n_wound` particles. Free-span particles can collide with each other and with the wound section.

---

### Phase 3 — Conservative-Bound Redetection

**Goal**: Avoid re-running VF and EE detection every substep when particles haven't moved enough.

**How it works (from OGC paper)**:
- After detection, record each particle's position at detection time in a `pos_det` buffer.
- At the start of each substep, check if `|pos - pos_det| > threshold` (e.g., `0.5 * r`).
- Only re-run detection if any particle exceeds the threshold.
- Otherwise, reuse the contact list from the previous detection.

**Expected gain**: 3–10× fewer detection passes in low-motion frames. Especially useful for the yarn wound on Roll A which moves slowly.

**Deliverables**:
- `pos_det_wp` buffer in `OGCSimulation`.
- Per-substep "dirty check" kernel → single scalar flag.
- Conditional detection call.

---

### Phase 4 — Particle Scaling

**Goal**: Increase N to demonstrate multi-layer winding clearly.

**Parameter targets by tier**:

| Tier              | Particles | Substeps | Constraint iter | Hash grid | Est. FPS (RTX 6000) |
|-------------------|-----------|----------|-----------------|-----------|---------------------|
| Current           | 120       | 20       | 10              | No        | ~60 (est.)          |
| T1 (Phase 1 only) | 500       | 20       | 10              | No        | ~60                 |
| T2 (Phase 1+2)    | 2000      | 15       | 8               | Yes       | ~60                 |
| T3 (Phase 1+2+3)  | 5000      | 12       | 6               | Yes       | ~45–60              |

**Substeps vs constraint_iter tradeoff**:
- **Substeps** are more important for contact stability (finer `dt` = less penetration).
- **Constraint_iter** helps convergence within a substep; returns diminish past ~8–10.
- As N grows and REST_LEN shrinks, constraints become stiffer → more substeps needed, not more iter.
- Rule of thumb: keep `substeps × iter ≈ 100–200`, prioritize substeps for contact-heavy scenes.

**Winding increase**:
- More wraps = larger `n_wound` = higher `N` with the same `n_free`.
- At N=500, REST_LEN = 7.0/499 ≈ 0.014 m. With orbit_r_a ≈ 0.17 m, `dtheta = 0.014/0.17 ≈ 0.083 rad` → **~75 particles per full turn** → with 400 wound particles → ~5 full turns.
- At N=2000, wound particles could give 15–20 visible turns.

---

## 4. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| CUDA graph incompatible with dynamic params (pull_speed slider changes mid-run) | High | Medium | Pass mutable params via GPU scalar arrays, written before each replay |
| HashGrid cell size wrong → missed contacts | Medium | High | Unit test: place two segments at exactly `2r` apart, verify detection |
| Self-collision in wound section fights winding geometry | High | Medium | Skip-list for wound-region pairs |
| Conservative-bound threshold too aggressive → missed contacts | Medium | Medium | Default threshold = 0.3r; expose as tunable param |
| Phase 1 CUDA graph doesn't give expected speedup (already compute-bound) | Low | Low | Profile first; if already fast, skip graph capture |

---

## 5. File Change Summary

| File | Changes |
|------|---------|
| `examples/ogc/algorithm5.py` | New: yarn self-collision EE detection with HashGrid broad phase |
| `examples/ogc/algorithm6.py` | New: yarn self-collision EE projection + friction |
| `examples/ogc/algorithm3.py` | Add self-collision detect + project calls; add conservative-bound redetection logic |
| `examples/ogc/algorithm4.py` | Minor: expose any new GPU utility kernels needed |
| `examples/yarn_rolls_ogc_gui.py` | CUDA graph capture; N/substep/iter param tuning; self-collision on/off toggle; wound skip-list wiring |
| `config.py` | Updated default NUM_PARTICLES, SUBSTEPS, CONSTRAINT_ITER for new tier |

---

## 6. Evaluation Criteria

Before moving from one phase to the next:

- **Phase 1 → Phase 2**: `wp.ScopedTimer` confirms ≥ 3× wall-time reduction for the substep loop.
- **Phase 2 → Phase 3**: Yarn-on-yarn penetration visually absent in a "figure-eight cross" stress test. No regression in single-strand winding.
- **Phase 3 → Phase 4**: Stable 60 FPS at T2 (2000 particles) confirmed over 500 frames without blow-up.
- **Phase 4 done**: T3 (5000 particles) achieves ≥ 45 FPS; winding shows ≥ 10 visible turns on Roll A.

---

## 7. Decisions

| # | Question | Decision | Rationale |
|---|----------|----------|-----------|
| 1 | Separate `yarn_yarn_r` vs. shared `r` for self-collision distance? | **Share `r`** (same contact radius for yarn–obstacle and yarn–yarn) | No meaningful performance difference; add separate slider later only if wound turns look too loose or too tight. |
| 2 | Wound-section skip list: hard-code vs. auto-detect? | **Hard-code**: skip self-EE pairs where both edge indices are in `0..n_wound` | Performance identical; `n_wound` is fixed at init/reset in our setup, so auto-detect buys nothing. |
| 3 | CUDA graph rebuild on substeps/iter slider change: stutter or disable? | **Debounce rebuild (300 ms)**: graph rebuilds only after slider is stable | Stutter is one dry-run frame (~16 ms), not ongoing. Debounce prevents choppy rebuilds during slider drag. All other sliders (gravity, pull_speed, damping, etc.) remain live-updating without triggering a rebuild — they write directly into GPU scalar arrays. |
