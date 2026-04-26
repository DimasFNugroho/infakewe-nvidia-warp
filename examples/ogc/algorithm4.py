"""algorithm4.py — Inner solver iteration with OGC contact (paper Alg. 4).

The paper's Algorithm 4 is a VBD (Vertex Block Descent) pass that, for every
vertex, assembles inertia + elastic + contact force / Hessian and takes a 3×3
Newton step.  Here we adapt the *structure* of that algorithm to the PBD
solver used by the parent yarn simulator:

    one iteration  =  stretch  +  bend  +  OGC contact projection

Stretch and bend are the existing parent kernels, re-used verbatim.  The
contact projection kernels in this module consume the VF / EE contact arrays
produced by Algorithms 1 and 2 and push yarn particles out along the stored
feature normals until the offset-geometry condition ‖x − cp‖ ≥ r is met.

Design notes
------------
* **Coloring** — the parent PBD solver already colors yarn constraints with
  even/odd splitting (see kernels.py).  VF contacts touch one particle at a
  time, so they are trivially race-free.  EE contacts touch two particles of
  the same yarn edge; to stay race-free we split them with the same
  even/odd scheme.

* **Offset geometry** — a particle is projected so that its distance to the
  feature-closest point cp equals exactly r (the rest thickness of the yarn
  + obstacle skin).  This is the position-space analogue of the paper's
  barrier energy, and is what makes the offset surface behave as a smooth
  collider around the polyhedral obstacle.
"""

from __future__ import annotations

import warp as wp


# ── Kernel: vertex-facet contact projection ───────────────────────────────────

@wp.kernel
def kernel_vf_project(
    pos:        wp.array(dtype=wp.vec3),
    inv_mass:   wp.array(dtype=float),
    vf_active:  wp.array(dtype=int),
    vf_cp:      wp.array(dtype=wp.vec3),
    vf_dist:    wp.array(dtype=float),
    vf_normal:  wp.array(dtype=wp.vec3),
    r:          float,
    stiffness:  float,   # 0..1, 1 = fully project (standard PBD contact)
):
    i = wp.tid()
    if inv_mass[i] == 0.0:
        return
    if vf_active[i] == 0:
        return
    if vf_dist[i] >= r:
        return  # already outside offset geometry

    # Push the particle to a point at distance r from cp along the feature normal.
    # Using the stored feature normal (rather than direction p - cp) keeps the
    # projection well-defined when the particle sits exactly on the feature.
    target = vf_cp[i] + vf_normal[i] * r
    pos[i] = pos[i] + (target - pos[i]) * stiffness


# ── Kernel: edge-edge contact projection ──────────────────────────────────────

@wp.kernel
def kernel_ee_project(
    pos:        wp.array(dtype=wp.vec3),
    inv_mass:   wp.array(dtype=float),
    yarn_edges: wp.array(dtype=wp.vec2i),
    parity:     int,                             # 0 = even edges, 1 = odd
    ee_active:  wp.array(dtype=int),
    ee_cp_yarn: wp.array(dtype=wp.vec3),
    ee_cp_obs:  wp.array(dtype=wp.vec3),
    ee_dist:    wp.array(dtype=float),
    ee_normal:  wp.array(dtype=wp.vec3),
    ee_s:       wp.array(dtype=float),
    r:          float,
    stiffness:  float,
):
    i = wp.tid()
    if (i & 1) != parity:
        return
    if ee_active[i] == 0:
        return
    if ee_dist[i] >= r:
        return

    ev = yarn_edges[i]
    wa = inv_mass[ev[0]]
    wb = inv_mass[ev[1]]
    wt = wa + wb
    if wt < 1.0e-12:
        return

    # Desired yarn closest-point position: distance r outward from the obstacle cp.
    target_cp = ee_cp_obs[i] + ee_normal[i] * r
    delta     = (target_cp - ee_cp_yarn[i]) * stiffness

    # Distribute the correction to the two yarn endpoints by barycentric weight
    # of the closest point along the yarn edge.  Barycentric weights at
    # parametric position s are (1 - s) on endpoint a and s on endpoint b.
    s  = ee_s[i]
    ca = 1.0 - s
    cb = s
    # Weighted Jacobian^T * inertia-scaled pseudo-inverse; the standard PBD
    # derivation for a point-on-segment constraint yields:
    w_denom = ca * ca * wa + cb * cb * wb
    if w_denom < 1.0e-12:
        return

    pos[ev[0]] = pos[ev[0]] + delta * (ca * wa / w_denom)
    pos[ev[1]] = pos[ev[1]] + delta * (cb * wb / w_denom)


# ── Kernel: Coulomb friction at vertex-facet contact ─────────────────────────

@wp.kernel
def kernel_vf_friction(
    pos:       wp.array(dtype=wp.vec3),
    prev_pos:  wp.array(dtype=wp.vec3),
    inv_mass:  wp.array(dtype=float),
    vf_active: wp.array(dtype=int),
    vf_dist:   wp.array(dtype=float),
    vf_normal: wp.array(dtype=wp.vec3),
    r:         float,
    mu_s:      float,
    mu_k:      float,
):
    """Position-based Coulomb friction (Müller et al. 2007, Sec. 3.3).

    For each contacting particle, the tangential displacement since substep
    start (pos - prev_pos projected onto the tangent plane of the contact
    normal) is clamped by the Coulomb cone:

        |Δx_T| ≤ μ_s · correction_n   →  static:  cancel Δx_T entirely
        |Δx_T| >  μ_s · correction_n   →  kinetic: scale back to μ_k · correction_n

    correction_n = max(r - vf_dist, r/2): uses the penetration depth as the
    cone reference, floored at half the contact radius so that settled
    particles (penetration ≈ 0 when dist ≈ r) still get a non-degenerate
    cone and μ values remain visually significant across the full slider range.
    """
    i = wp.tid()
    if inv_mass[i] == 0.0:
        return
    if vf_active[i] == 0:
        return
    if vf_dist[i] >= r:
        return

    # Coulomb cone reference = penetration depth, floored at r/2 so that
    # settled particles (penetration ≈ 0 when vf_dist ≈ r) still get a
    # non-degenerate cone and mu values remain visually significant.
    penetration  = r - vf_dist[i]
    correction_n = wp.max(penetration, r * float(0.5))

    n       = vf_normal[i]
    delta   = pos[i] - prev_pos[i]
    delta_t = delta - n * wp.dot(delta, n)
    len_t   = wp.length(delta_t)

    if len_t < 1.0e-6:
        return

    if len_t <= mu_s * correction_n:
        pos[i] = pos[i] - delta_t                                        # static
    else:
        # Clamp scale to 1.0: if mu_k > mu_s (user error via slider) the
        # ratio overshoots 1 and reverses motion, ping-ponging each substep.
        scale = wp.min(mu_k * correction_n / len_t, float(1.0))
        pos[i] = pos[i] - delta_t * scale                                # kinetic


# ── Kernel: Coulomb friction at edge-edge contact ────────────────────────────

@wp.kernel
def kernel_ee_friction(
    pos:        wp.array(dtype=wp.vec3),
    prev_pos:   wp.array(dtype=wp.vec3),
    inv_mass:   wp.array(dtype=float),
    yarn_edges: wp.array(dtype=wp.vec2i),
    parity:     int,
    ee_active:  wp.array(dtype=int),
    ee_dist:    wp.array(dtype=float),
    ee_normal:  wp.array(dtype=wp.vec3),
    ee_s:       wp.array(dtype=float),
    r:          float,
    mu_s:       float,
    mu_k:       float,
):
    """Coulomb friction for edge-edge contacts (same cone rule as VF, per-edge).

    The contact point on the yarn moves tangentially by the barycentric
    interpolation of the two endpoint displacements.  The same Coulomb
    cone correction is distributed back to the endpoints by the same
    barycentric weights used in kernel_ee_project.
    """
    i = wp.tid()
    if (i & 1) != parity:
        return
    if ee_active[i] == 0:
        return
    if ee_dist[i] >= r:
        return

    penetration  = r - ee_dist[i]
    correction_n = wp.max(penetration, r * float(0.5))

    ev = yarn_edges[i]
    wa = inv_mass[ev[0]]
    wb = inv_mass[ev[1]]
    wt = wa + wb
    if wt < 1.0e-12:
        return

    s  = ee_s[i]
    ca = float(1.0) - s
    cb = s

    # Tangential displacement of the yarn contact point since substep start.
    cp_now  = pos[ev[0]]  * ca + pos[ev[1]]  * cb
    cp_prev = prev_pos[ev[0]] * ca + prev_pos[ev[1]] * cb
    delta   = cp_now - cp_prev
    n       = ee_normal[i]
    delta_t = delta - n * wp.dot(delta, n)
    len_t   = wp.length(delta_t)

    if len_t < 1.0e-6:
        return

    if len_t <= mu_s * correction_n:
        scale = float(1.0)                                               # static
    else:
        scale = wp.min(mu_k * correction_n / len_t, float(1.0))         # kinetic

    correction = -delta_t * scale

    # Distribute correction to endpoints by barycentric / inverse-mass weighting.
    w_denom = ca * ca * wa + cb * cb * wb
    if w_denom < 1.0e-12:
        return

    pos[ev[0]] = pos[ev[0]] + correction * (ca * wa / w_denom)
    pos[ev[1]] = pos[ev[1]] + correction * (cb * wb / w_denom)


# ── Kernel: velocity magnitude clamp ─────────────────────────────────────────

@wp.kernel
def kernel_clamp_velocity(
    vel:   wp.array(dtype=wp.vec3),
    v_max: float,
):
    """Hard-cap particle speed to prevent PBD constraint corrections from
    feeding back into runaway velocities.

    In PBD, velocity is derived as  v = (x_after_constraints - x_prev) / dt,
    so large constraint corrections (stretch snap, contact projection) appear
    as large velocities that overshoot even further next substep — a positive
    feedback loop that explodes.  Clamping speed breaks the loop without
    altering direction, so the simulation remains physically plausible.

    A value of 0.0 disables the clamp.
    """
    i = wp.tid()
    if v_max <= 0.0:
        return
    v_mag = wp.length(vel[i])
    if v_mag > v_max:
        vel[i] = vel[i] * (v_max / v_mag)


# ── Kernel: contact normal-velocity damping ───────────────────────────────────

@wp.kernel
def kernel_vf_damp_normal_velocity(
    vel:       wp.array(dtype=wp.vec3),
    inv_mass:  wp.array(dtype=float),
    vf_active: wp.array(dtype=int),
    vf_dist:   wp.array(dtype=float),
    vf_normal: wp.array(dtype=wp.vec3),
    r:         float,
    skin:      float,
):
    """Zero the inward-normal velocity component for particles on the offset surface.

    A particle is "in contact" when vf_active == 1 and its distance to the
    closest obstacle feature is within  r + skin.  For those particles, the
    velocity component pointing back toward the obstacle (negative dot with the
    outward feature normal) is removed — leaving tangential motion intact.
    """
    i = wp.tid()
    if inv_mass[i] == 0.0:
        return
    if vf_active[i] == 0:
        return
    if vf_dist[i] > r + skin:
        return

    n      = vf_normal[i]
    v      = vel[i]
    v_inward = wp.dot(v, n)
    if v_inward < 0.0:
        vel[i] = v - n * v_inward


# ── Python entry point ────────────────────────────────────────────────────────

def project_vf(
    pos:      wp.array,
    inv_mass: wp.array,
    vf,       # VFContacts
    r:        float,
    stiffness: float,
    device:   str,
):
    wp.launch(
        kernel_vf_project, dim=pos.shape[0], device=device,
        inputs=[pos, inv_mass, vf.active, vf.cp, vf.dist, vf.normal, r, stiffness],
    )


def damp_normal_velocity(
    vel:      wp.array,
    inv_mass: wp.array,
    vf,       # VFContacts
    r:        float,
    device:   str,
):
    """Cancel the inward normal velocity for particles resting on the offset surface.

    PBD contact projection repositions a particle each substep, but the
    velocity update  v = (x - x_prev)/dt  then treats that displacement as
    outward velocity.  On the next substep gravity drives the particle back
    in, the contact fires again, and a low-frequency normal oscillation
    (wiggling) builds up.  Zeroing the component of v that points *against*
    the feature normal (i.e. back into the obstacle) kills this loop — the
    same trick used in kernel_cylinder_contact_damping in the analytic example.

    The `skin` band is 0.5 * r beyond the offset surface so that particles
    that have just settled are caught even if numerical drift pushed them
    slightly outside r.
    """
    wp.launch(
        kernel_vf_damp_normal_velocity, dim=vel.shape[0], device=device,
        inputs=[vel, inv_mass, vf.active, vf.dist, vf.normal, r, float(r * 0.5)],
    )


def clamp_velocity(
    vel:    wp.array,
    v_max:  float,
    device: str,
):
    wp.launch(
        kernel_clamp_velocity, dim=vel.shape[0], device=device,
        inputs=[vel, float(v_max)],
    )


def apply_vf_friction(
    pos:      wp.array,
    prev_pos: wp.array,
    inv_mass: wp.array,
    vf,       # VFContacts
    r:        float,
    mu_s:     float,
    mu_k:     float,
    device:   str,
):
    wp.launch(
        kernel_vf_friction, dim=pos.shape[0], device=device,
        inputs=[pos, prev_pos, inv_mass,
                vf.active, vf.dist, vf.normal,
                r, mu_s, mu_k],
    )


def apply_ee_friction(
    pos:        wp.array,
    prev_pos:   wp.array,
    inv_mass:   wp.array,
    yarn_edges: wp.array,
    ee,         # EEContacts
    r:          float,
    mu_s:       float,
    mu_k:       float,
    device:     str,
):
    n = yarn_edges.shape[0]
    for parity in (0, 1):
        wp.launch(
            kernel_ee_friction, dim=n, device=device,
            inputs=[pos, prev_pos, inv_mass, yarn_edges, parity,
                    ee.active, ee.dist, ee.normal, ee.s,
                    r, mu_s, mu_k],
        )


def project_ee(
    pos:      wp.array,
    inv_mass: wp.array,
    yarn_edges: wp.array,
    ee,       # EEContacts
    r:        float,
    stiffness: float,
    device:   str,
):
    n = yarn_edges.shape[0]
    for parity in (0, 1):   # even/odd split = race-free
        wp.launch(
            kernel_ee_project, dim=n, device=device,
            inputs=[pos, inv_mass, yarn_edges, parity,
                    ee.active, ee.cp_yarn, ee.cp_obs, ee.dist, ee.normal, ee.s,
                    r, stiffness],
        )
