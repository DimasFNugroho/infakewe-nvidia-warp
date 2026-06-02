# Plan: Feeder Option 3 — Two-Cylinder (Package + Drum)

Branch: `feeder/o3-two-cylinder`
Predecessor: `master` after the collocated-feedback work lands. **Not**
predecessor on top of O2 — the geometry diverges enough that diffing
back is painful.

See [`PLAN_feeder_strategy_overview.md`](PLAN_feeder_strategy_overview.md)
for context. This branch builds the most faithful version: the **bulk
yarn package** and the **EFS drum** are two distinct cylindrical
obstacles, each with their own wound section of yarn.

---

## 1. Goal

Model the real layout: a stationary spool/package holds bulk yarn; a
short transit span connects it to the EFS drum; the drum has a few
wraps and is motor-driven; a free span continues to the guide cylinder
and onward to Roll B as before.

Total layout:
```
[Package P] —(transit)→ [Drum D]
                          │
                          (free span)
                          ↓
                       [Guide C]
                          │
                          (free span)
                          ↓
                       [Roll B]
```

The package is passive (no rotation, no motor). It carries a long wound
section of bulk yarn. The transit from package to drum is a short free
span with its own external-tangent geometry.

---

## 2. New parameters

Geometry of the new package cylinder:

| Key                  | Type  | Default | Description |
|----------------------|-------|---------|-------------|
| `package_x`          | float | −1.4    | World X of package centre. |
| `package_y`          | float | −0.1    | World Y. |
| `package_z`          | float | 0.0     | World Z. |
| `package_radius`     | float | 0.30    | Physical radius (m). Larger than drum to look package-like. |
| `package_visible`    | int   | 1       | Show / hide the mesh. |
| `package_wraps`      | float | 5.0     | Wraps of bulk yarn on the package surface. |
| `package_friction`   | float | 0.2     | Local $\mu$ on the package (independent slider). |

Drum parameters from O1 are kept (`roll_a_wraps`, `roll_a_pitch_d`).

Yarn budget: `yarn_length` now distributes across:

$$L_\text{yarn} = L_\text{package wound} + L_\text{transit}
                + L_\text{drum wound} + L_\text{free AC} + L_\text{arc C}
                + L_\text{free CB}$$

If `yarn_length` is too small to satisfy the wound counts, $n_\text{wraps}$
clamps downward with a log warning.

---

## 3. Particle inventory

In free-yarn order from package to Roll B:

1. Particles $0 \ldots n_{wp} - 1$ — wound on the package.
2. Particles $n_{wp} \ldots n_{wp} + n_T - 1$ — transit span.
3. Particles $n_{wp} + n_T \ldots n_{wp} + n_T + n_{wd} - 1$ —
   wound on the drum.
4. Particles after that — free spans + guide arc, exactly as today.

Particle 0 is the kinematic anchor on the package; particle $N-1$ is the
kinematic last on Roll B. Drum wound particles are **kinematic** as in
O1 (the package is the upstream compliance; the drum is the rigid
positive feed).

---

## 4. Mesh + OGC contact additions

- Build a fourth OGC obstacle for the package via
  `build_cylinder(package_x, …)`.
- Add it to the `contacts` list with full friction. The default $\mu$
  is the global $\mu_k$ (we can promote `package_friction` later).
- The package has no kinematic kernel — it is a static obstacle. Yarn
  wound on it stays in place by OGC contact alone.

For O3 the wound-vs-wound self-collision skip from O1 should apply
*per cylinder*: pairs on the same wound section are skipped, but
package-wound vs drum-wound (which never touch in practice) need no
extra rule.

---

## 5. Initial-state geometry extension

`_warp_keypoints()` currently produces tangents A→C and C→B. For O3 we
add P→A (package → drum) and keep A→C and C→B unchanged.

`make_initial_positions()` builds the chain in this order:

1. Helical wound section on the package (`n_{wp}` particles).
2. Straight transit span from package departure tangent to drum
   arrival tangent (`n_T` particles).
3. Helical wound section on the drum (`n_{wd}` particles), starting at
   the drum arrival tangent.
4. Free span drum→guide, arc on guide, free span guide→Roll B —
   identical to today.

Helical pitches: package uses its own `package_pitch_d` if we want
visual fidelity (a real package winds *axially* much faster than a
drum); the drum uses `roll_a_pitch_d` from O1.

Wrap-direction signs around the package depend on layout; we use the
same cross-product rule as for the guide:

$$\chi_\text{P} = (A_x - P_x)(\text{next}_y - P_y) - (A_y - P_y)(\text{next}_x - P_x)$$

with $\text{next}$ being the drum centre.

---

## 6. Anchor segment hook

Same definition as O1/O2: the anchor segment is
$[\,n_\text{drum-wound-last} - 1,\; n_\text{drum-wound-last}\,]$, i.e.
the segment leaving the drum into the guide-side free span. This is
*not* the segment leaving the package — the controller measures
downstream of the drum.

---

## 7. Files touched

- `examples/yarn_rolls_ogc_gui.py`
  - `DEFAULTS` — add the package keys.
  - Obstacle creation block — add `mesh_pkg`, `obs_pkg`.
  - `contacts` list — add an entry for the package.
  - `_warp_keypoints()` — return P→A tangent points alongside the
    existing ones. New keys `T_p_dep`, `T_a_arr`.
  - `make_initial_positions()` — build the four-segment chain. The
    particle-budget formula is extended to seven terms.
  - `make_inv_mass()` — kinematic mask for both wound sections.
  - `_execute_substeps()` — call drum kinematic step (from O1).
    No new kernel needed for the package (it is static).
  - `_snapshot_params()` — include all new geometry / wraps params.
  - GUI: a new section "Yarn package" with the seven sliders.
  - Picking, sensors, rebuild callbacks — extend to include the
    package cylinder.

---

## 8. Out-of-scope

- Package spinning (some real packages do rotate). Could be added as an
  optional motor later. For now the package is purely static.
- Yarn-break / end-of-package events.
- A dancer arm between package and drum.
- The transit span tension *upstream of the drum* as a controller input
  — the controller still measures downstream of the drum (anchor
  segment).

---

## 9. Risks

- **Capstan-amplified payout.** Bulk yarn wound on the package will
  experience real Capstan amplification through the drum wraps when the
  drum pulls. With high $\mu$ and many drum wraps, the package side may
  essentially never move — which is faithful behaviour but may surprise
  users who expected payout. Acceptance criterion: with default $\mu$
  and 3 drum wraps, the package wound section visibly shifts under
  steady-state downstream pull.
- **Performance.** Adding a fourth cylindrical obstacle adds ~25 %
  contact-detection cost. We may want to skip detection on the package
  for particles known to be on the drum or downstream. A simple particle-
  range gate in the detection wrappers suffices.
- **Geometry validity.** If the user places the package on the wrong
  side of the drum (so P→A→C produces a self-crossing yarn path), the
  initial geometry will look obviously broken. We rely on the user to
  place it sensibly; no automatic recovery.
- **Hot-rebuild scope.** Moving the package via sliders triggers a
  rebuild of `mesh_pkg` and the contact list. The existing rebuild
  pattern (`rebuild_roll_a`, `rebuild_guide`, …) is extended to
  `rebuild_package`.

---

## 10. Acceptance tests

- Open-loop step test (strategy §5.1) shows two distinct time scales:
  a fast drum response (O1-like) and a slow package payout.
- Closed-loop test (strategy §5.2) shows the controller never sees the
  slow package-side dynamics — its anchor-segment reading and PI
  response should look essentially the same as O1.
- Capstan residual (strategy §5.3): the drum's wraps now contribute
  their own measurable Capstan amplification, on top of the guide's.
  Validates that the friction model is consistent across multiple
  contact bodies.
- Visual: yarn visibly comes off a "package" cylinder, runs through a
  transit span, wraps a smaller drum a few times, and continues
  downstream. Distinct from the O1 / O2 picture.
