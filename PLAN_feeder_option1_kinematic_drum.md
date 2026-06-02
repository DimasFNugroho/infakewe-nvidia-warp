# Plan: Feeder Option 1 — Kinematic Drum

Branch: `feeder/o1-kinematic-drum`
Predecessor: `master` after `PLAN_collocated_anchor_feedback.md` lands.

See [`PLAN_feeder_strategy_overview.md`](PLAN_feeder_strategy_overview.md)
for context. This is the minimum-fidelity branch: shrink the wound coil
to a realistic drum-wrap count and make those wrapped particles
**kinematically driven**, treating the EFS drum as a perfect-grip
positive-feed device. The bulk yarn upstream of the drum is *implicit*
— it does not exist in the simulation.

---

## 1. Goal

Replicate a real EFS-32 drum: a few wraps of yarn rigidly carried by the
drum surface. As the drum rotates by $\Delta\theta$, every wound
particle advances by exactly $\Delta\theta$ along the helix. No
elasticity in the wound section; no Capstan dynamics on the drum (the
"perfect-grip" limit of high $\mu$, many wraps).

---

## 2. New parameters

| Key                | Type  | Default | Range  | Description |
|--------------------|-------|---------|--------|-------------|
| `roll_a_wraps`     | float | 3.0     | 0.5–10 | Number of full wraps on the drum (turns). |
| `roll_a_pitch_d`   | float | 1.0     | 0.5–4  | Helix pitch in units of $2r$ (one OGC diameter). 1.0 = current behaviour. |

`yarn_length` keeps its meaning **as total yarn length**. The wound
section consumes
$L_w = n_w \cdot 2\pi R_A^\text{orb}$
of yarn (independent of $r$), leaving
$L_\text{free} = \text{yarn\_length} - L_w$
for the free spans + arc. If $L_\text{free}$ would be non-positive, we
clamp $n_w$ down to fit and emit a warning to the log.

---

## 3. Particle inventory

Let $n_w$ be the user-set wrap count and $\Delta\theta = L_0 / R_A^\text{orb}$
the per-particle angular step. Then:

$$n_\text{wound} = \mathrm{round}\!\left(\frac{2\pi\, n_w}{\Delta\theta}\right)
= \mathrm{round}\!\left(\frac{2\pi\, n_w\, R_A^\text{orb}}{L_0}\right)$$

With default $n_w = 3$, $R_A^\text{orb} \approx 0.155$ m,
$L_0 \approx 0.005$ m: $n_\text{wound} \approx 584$ — still substantial
but a small fraction of typical $N \approx 1500$.

Free-span particles: $n_\text{free} = N - n_\text{wound}$, distributed
across the three free-span segments as today.

**Inverse masses:**

$$w_i = \begin{cases}
  0 & 0 \le i < n_\text{wound} \quad\text{(kinematic: drum-wound)} \\
  1/m_p & n_\text{wound} \le i < N - 1 \\
  0 & i = N - 1 \quad\text{(kinematic: Roll B anchor, unchanged)}
\end{cases}$$

---

## 4. Drum kinematic update

A new kernel `kernel_drum_kinematic_advance` runs every substep,
**before** the existing `roll_a_servo_step` / `roll_a_torque_step`
selection, and writes the position of every drum-wound particle:

```
For i in [0, n_wound):
    θ_i = angle_a[0] + i * dθ
    z_i = az + i * dz                # dz = 2 r · pitch_d · L_0 / (2π R_a^orb)
    pos[i] = (ax + R_a^orb · cos θ_i,
              ay + R_a^orb · sin θ_i,
              z_i)
```

Implementation notes:
- A single Warp kernel with `dim = n_wound` per substep.
- `angle_a` (size-1 wp.array) is updated by the existing
  `kernel_roll_a_servo_update` or `kernel_roll_a_torque_update`. Those
  kernels are kept as-is so the angle integration logic stays in one
  place.
- The kernel only writes positions for $i < n_\text{wound}$; the rest of
  the particles are untouched.

The pre-existing `pos[0] = ...` write at the end of those two angular
kernels becomes redundant for $i = 0$ but does no harm. We leave it for
clarity (the angular kernel still "owns" the kinematic anchor angle, and
the drum kernel "owns" the helical layout).

---

## 5. Anchor segment redefinition (collocated feedback hook)

The anchor segment for the collocated feedback in
`PLAN_collocated_anchor_feedback.md` becomes:

$$\text{segment } [\,n_\text{wound} - 1,\; n_\text{wound}\,]$$

This is the segment where the yarn leaves the drum surface and starts
the free span. Its extension is the direct measure of the yarn's pull
on the drum.

If `roll_a_anchor_k > 1` the average is over segments
$[\,n_\text{wound} - 1,\; n_\text{wound}\,],\;
 [\,n_\text{wound},\; n_\text{wound} + 1\,],\;\ldots$
i.e. $k$ consecutive free-span segments starting at the departure
point.

---

## 6. Files touched

- `examples/ogc/algorithm4.py`
  - Add `kernel_drum_kinematic_advance` and Python wrapper
    `drum_kinematic_step`.
- `examples/yarn_rolls_ogc_gui.py`
  - `DEFAULTS` — add `roll_a_wraps`, `roll_a_pitch_d`.
  - `make_initial_positions()` — replace the
    `n_wound = N - n_free` budgeting with the geometry-driven
    $n_\text{wound}$ formula above. Compute `n_free = N - n_wound` and
    distribute as before.
  - `make_inv_mass()` — kinematic mass for all $i < n_\text{wound}$.
  - `_execute_substeps()` — call `drum_kinematic_step` first inside the
    substep loop.
  - `_snapshot_params()` — include `roll_a_wraps`, `roll_a_pitch_d`
    (changing them rebuilds the graph).
  - `_auto_place_sensors()` — use $i = n_\text{wound}$ as the departure
    point rather than the angle-driven particle 0.
  - GUI section "Roll A — feeding roll" gains two sliders
    (`roll_a_wraps`, `roll_a_pitch_d`).

---

## 7. Out-of-scope

- Upstream package compliance (that is O2).
- Separate package cylinder (that is O3).
- Self-collision in the wound section — already disabled by the
  wound-vs-wound skip; no change required.
- Capstan-residual contribution from the drum — by construction O1 has
  no drum Capstan. The residual measures only the guide.

---

## 8. Risks

- **Kinematic particles in OGC contact arrays.** The existing OGC
  projection kernels skip `inv_mass[i] == 0`, but EE contacts on edges
  where both endpoints are kinematic will still be computed by detection
  and skipped at projection. That is wasteful but correct. If it shows
  up in the profile, add a kinematic-edge skip in
  `detect_edge_edge`.
- **Sub-pixel mismatch between drum kernel and angular kernel.** The
  angular kernel writes `pos[0]` from the same `angle_a[0]` the drum
  kernel reads, so they're consistent. But if the drum kernel reads
  `angle_a[0]` before the angular kernel updates it, particle 0 is one
  substep behind. We must call drum kernel **after** the angular
  kernel. Order:
  1. `roll_a_servo_step` *or* `roll_a_torque_step` → updates
     `angle_a[0]` and `omega_a[0]`.
  2. `drum_kinematic_step` → writes positions for $i < n_\text{wound}$.
  3. (continue with `roll_b_motor_step`, `kernel_integrate`, etc.)
- **Sudden $n_\text{wound}$ jump on slider change.** Changing
  `roll_a_wraps` at runtime triggers a re-budgeting. We re-emit the
  initial positions via the existing `do_reinit` path, which already
  resets velocity to zero and triggers a graph rebuild. Acceptable.

---

## 9. Acceptance tests

- Open-loop step test (strategy §5.1) shows fast rise to steady state
  with no low-frequency oscillation.
- Stability boundary on $k_p$ — should be at least 2× the current value
  before oscillation (we are removing a major source of compliance).
- Visual: drum surface looks like a real EFS — yarn wraps a short
  helical section and leaves on a clean tangent. No "spool" appearance.
