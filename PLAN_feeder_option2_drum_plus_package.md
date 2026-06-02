# Plan: Feeder Option 2 — Drum + Upstream Compliance

Branch: `feeder/o2-drum-plus-package`
Predecessor: `feeder/o1-kinematic-drum`.

See [`PLAN_feeder_strategy_overview.md`](PLAN_feeder_strategy_overview.md)
for context and [`PLAN_feeder_option1_kinematic_drum.md`](PLAN_feeder_option1_kinematic_drum.md)
for the underlying kinematic-drum mechanism. This branch keeps every-
thing in O1 and adds an **upstream compliance** representing the bulk
yarn coming off the package, without modelling the package as a full
cylinder.

---

## 1. Goal

Acknowledge that real bulk yarn upstream of the drum is *not* infinitely
stiff: when the drum demands yarn, it has to pay out from a spool with
some unwind resistance + storage compliance. We model this as a single
soft spring-anchor at the trailing end of the drum-wound helix.

This is the cheapest way to put a finite upstream compliance into the
model. It captures the *low-frequency* effect of bulk-yarn elasticity
without committing to a full package cylinder (which is O3).

---

## 2. Conceptual structure

In O1 the trailing wound particle (particle 0, the "first" particle on
the helix) is kinematically positioned by the drum kernel. In O2 we
instead let particle 0 be a *free* particle that is pulled back toward a
geometric reference position by a 1-D spring representing the package's
unwind resistance.

The reference position is the helical position particle 0 *would* have
in O1, i.e.
$$\vect{r}_0^{\text{ref}}(t) = \vect{C}_A + R_A^\text{orb}\,
(\cos\theta_A(t), \sin\theta_A(t), z_0) .$$

The spring is *only along the surface tangent direction* at particle 0
(i.e. along the helix). Radial and axial restoring is left to OGC
contact with Roll A. This keeps the upstream compliance from leaking
into transverse motion.

---

## 3. New parameters

| Key                       | Type  | Default | Range      | Description |
|---------------------------|-------|---------|------------|-------------|
| `package_stiffness`       | float | 0.5     | 0–10 cN/mm | Spring constant of the upstream "package" (force per mm of payout). |
| `package_damping`         | float | 0.2     | 0–1        | Per-substep damping factor for the package spring (Rayleigh-style). |
| `package_visible`         | int   | 1       | 0/1        | Show a small marker at the reference position vs particle 0. |

`package_stiffness` units: cN/mm gives a number in the same scale the
HUD already uses. Internally it converts to N/m.

---

## 4. Particle inventory

Identical to O1, **except** that particle 0 — and only particle 0 —
becomes free again:

$$w_i = \begin{cases}
  1/m_p & i = 0 \quad\text{(package-anchored, free)} \\
  0 & 1 \le i < n_\text{wound} \quad\text{(kinematic: drum-wound)} \\
  1/m_p & n_\text{wound} \le i < N - 1 \\
  0 & i = N - 1
\end{cases}$$

The single free particle on the drum is what stores the payout
displacement. The remaining $n_\text{wound} - 1$ particles are still
kinematically driven by the drum kernel, exactly as in O1, which means
their positions are *not* the bottleneck for the payout — the bottleneck
is at particle 0 only.

---

## 5. Spring kernel

A new substep kernel `kernel_package_spring_step` applies the spring
force on particle 0 along the helix tangent:

```
ref = drum reference position for particle 0 at current angle_a
disp_tangent = (pos[0] - ref) · t̂_0     # tangent component, signed
force_tangent = -k * disp_tangent - c * (v_tangent)
pos[0] += force_tangent * sub_dt² / m_p    # explicit Euler
vel[0] += force_tangent * sub_dt / m_p
```

In PBD style we apply this as a position correction at the end of the
substep, similar to OGC projection:
```
δ_tangent = -stiffness · disp_tangent   # PBD-style relaxation
pos[0] += δ_tangent · t̂_0
```
where `stiffness ∈ [0, 1]` is derived from `package_stiffness` via the
same impulse-equivalence trick used for the tension reading.

Implementation notes:
- The kernel is `dim = 1`. It belongs after the drum kinematic kernel
  and after the OGC projection passes (so radial/axial corrections from
  OGC don't fight the spring along the tangent).
- The reference position uses `angle_a[0]`, read at substep time —
  i.e. tracks the drum so the spring's rest length shrinks as the drum
  rotates. That is what gives the "package pays out yarn as the drum
  demands it" behaviour.

---

## 6. Anchor segment hook

Same as O1: the anchor segment for the collocated feedback is
$[\,n_\text{wound} - 1,\; n_\text{wound}\,]$. Particle 0's spring sits
*upstream* of all the drum wraps — the controller sees the drum's
downstream side, not the package's.

(In other words, the upstream compliance does not change *where* the
controller measures; it changes *what disturbance dynamics* the
controller measures.)

---

## 7. Files touched

Relative to O1:

- `examples/ogc/algorithm4.py`
  - Add `kernel_package_spring_step` and wrapper `package_spring_step`.
- `examples/yarn_rolls_ogc_gui.py`
  - `DEFAULTS` — add the three keys above.
  - `make_inv_mass()` — set $w_0 = 1/m_p$.
  - `_execute_substeps()` — call `package_spring_step` after the OGC
    projection passes inside the constraint-iter loop, or once per
    substep after constraint iterations.
  - `_snapshot_params()` — include `package_stiffness`,
    `package_damping`.
  - GUI: a new mini-section "Roll A — package compliance" with three
    controls.
  - Visualisation: a small marker at the reference position when
    `package_visible == 1`.

---

## 8. Out-of-scope

- Modelling the package as a separate cylinder with its own wound yarn
  (that is O3).
- Modelling spool start-up inertia (could be added later as a damped
  mass instead of a pure spring).
- Anything axial — the spring is tangent-only.

---

## 9. Risks

- **Spring–OGC interaction at particle 0.** OGC keeps particle 0 on the
  offset surface in the radial direction; the spring acts along the
  helix tangent. Both can act on the same substep without conflict, but
  if `package_stiffness` is large and the drum is rotating fast,
  particle 0 may overshoot the offset surface during the tangent push.
  Acceptance criterion: at default values, particle 0 stays within
  $\pm r$ of the offset surface throughout a 30-second run.
- **Numerical stiffness regime.** Soft spring → low-frequency
  oscillation → controller benefit (filters out high-frequency
  disturbances). Stiff spring → behaviour reverts to O1. Both extremes
  are useful for testing.
- **Reference angle reset on slider change.** Changing `roll_a_wraps`
  re-budgets and re-runs `do_reinit`, which resets particle 0 to its
  reference. No further action needed.

---

## 10. Acceptance tests

- Open-loop step test (strategy §5.1) shows the same fast rise as O1
  *plus* a damped overshoot whose period is set by
  $T \approx 2\pi \sqrt{m_p / k}$ (with $k$ derived from
  `package_stiffness`). Tuning the spring should visibly move this
  pole.
- Closed-loop step test (strategy §5.2) shows reduced overshoot
  compared to O1 at the same $k_p$ when `package_stiffness` is small
  (the upstream compliance damps high-frequency demand).
- With `package_stiffness` cranked to its maximum, the response should
  approach O1 within numerical noise.
