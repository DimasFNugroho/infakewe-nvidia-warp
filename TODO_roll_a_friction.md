# TODO: Fix Roll A Friction — Restore Capstan Strain Gradient

## Problem

After re-enabling Roll A friction (commit `8c2f902`) and adding yarn self-collision,
the strain heatmap no longer shows the expected Capstan tension gradient across the
guide cylinder. Previously:

- **T_before** (Roll A → guide): lower strain
- **T_after** (guide → Roll B): higher strain, visibly redder in heatmap

This gradient is physically correct — the guide's friction creates a tension step-up
when Roll B pulls the yarn. After the changes, the gradient is flat or absent.

---

## Root Cause Analysis

### Primary suspect: Roll A friction on wound particles

Roll A friction is now applied to **all** particles in contact with Roll A — including
the many wound particles that wrap helically on the roll. These particles are always
touching Roll A by design, so they all receive friction corrections every substep.

Effect: large distributed resistance at the feed end → either locks the wound section
or homogenises tension throughout the yarn, wiping out the guide-induced gradient.

The wound particles were never meant to have surface friction with Roll A. They follow
the roll kinematically through the winding geometry. Only the **departure-point region**
(where the free span lifts off Roll A) should experience friction.

### Secondary suspect: Self-collision in the wound section

Wound wraps are separated by exactly one yarn-diameter (2r) by design, so every
adjacent wound-wound pair is an active self-EE contact. The self-EE projection fires
on all of them every substep, pushing them apart and introducing position corrections
that corrupt the strain field.

---

## Diagnostic Steps (to do at session start)

1. Launch sim, load `params_tension_test.json`, start pulling (positive pull speed).
2. Switch heatmap to stretch mode.
3. **Toggle self-collision off** → does the gradient return?
   - Yes → self-collision is the primary cause; proceed to Fix B.
   - No → proceed to step 4.
4. **Set `mu_static` and `mu_kinetic` to ~0** → does the gradient return?
   - Yes → Roll A friction is the primary cause; proceed to Fix A.

---

## Planned Fixes

### Fix A — Apply Roll A friction only to free-span particles (recommended)

In `_execute_substeps`, replace the blanket Roll A friction call with a version
that skips particles with index `< n_wound`:

**Option 1**: Add an optional `min_particle_idx` parameter to `apply_vf_friction`
and `apply_ee_friction` in `algorithm4.py`. The kernel skips thread `i < min_idx`.

**Option 2**: Build a temporary masked `inv_mass` array where wound particles have
`inv_mass = 0` (infinite mass → friction kernel skips them), apply Roll A friction
with it, then restore. Avoids changing algorithm4.py but allocates a temp array.

**Option 3** (simplest): Revert Roll A friction to disabled (`contacts[1:]`) until
a more surgical fix is implemented. The free-span departure point has negligible
contact area, so the physical impact is small.

### Fix B — Suppress self-collision within the wound section

In `algorithm5.py` `kernel_self_ee_detect`, skip pairs where both edge indices
`i` and `j` are `< n_wound - 1`. Pass `n_wound` as a kernel parameter.

This prevents the wound-wrap contacts from corrupting strain while still allowing:
- Free-span self-contacts (yarn crossing itself mid-air)
- Wound-vs-free-span contacts (yarn wrapping over an existing layer)

**Required change**: `detect_self_ee()` signature gains `n_wound: int` parameter;
`_execute_substeps` passes `n_wound` (already computed in `make_initial_positions`
scope — needs to be tracked as a closure variable).

---

## Files to Change

| File | Change |
|------|--------|
| `examples/ogc/algorithm4.py` | Add `min_particle_idx` param to VF/EE friction kernels (Fix A Option 1) |
| `examples/ogc/algorithm5.py` | Add `n_wound` skip condition to `kernel_self_ee_detect` (Fix B) |
| `examples/yarn_rolls_ogc_gui.py` | Track `n_wound` as closure var; pass to friction and detection calls |

---

## Acceptance Criterion

With `pull_speed > 0` and heatmap in stretch mode:
- Strain in the **guide → Roll B** segment is visibly higher than in the
  **Roll A → guide** segment.
- The gradient increases as `pull_speed` increases.
- The gradient increases as guide `mu_static`/`mu_kinetic` increases.
- No simulation blow-up over 500 frames at current particle density.
