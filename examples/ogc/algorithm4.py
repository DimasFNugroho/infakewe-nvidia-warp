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


# ── GPU-params kernel variants (for CUDA graph live-param updates) ───────────
#
# These mirror kernels.py and the contact kernels above, but every mutable
# slider-controlled scalar is accepted as a wp.array(dtype=float) of size 1.
# The CUDA graph records the array *pointer* at capture time; writing a new
# value into the array before each wp.capture_launch() lets params change
# without rebuilding the graph.  Only structural params (substeps,
# constraint_iter) that change the graph node count still require a rebuild.

@wp.kernel
def kernel_integrate_gp(
    pos:       wp.array(dtype=wp.vec3),
    vel:       wp.array(dtype=wp.vec3),
    prev_pos:  wp.array(dtype=wp.vec3),
    inv_mass:  wp.array(dtype=float),
    gravity_y: wp.array(dtype=float),
    dt:        float,
    damping:   wp.array(dtype=float),
):
    i = wp.tid()
    if inv_mass[i] == float(0.0):
        prev_pos[i] = pos[i]
        return
    prev_pos[i] = pos[i]
    g       = wp.vec3(float(0.0), gravity_y[0], float(0.0))
    vel[i]  = vel[i] * damping[0] + g * dt
    pos[i]  = pos[i] + vel[i] * dt


@wp.kernel
def kernel_stretch_even_gp(
    pos:      wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    rest_len: float,
    stiff:    wp.array(dtype=float),
):
    t = wp.tid()
    a = t * 2; b = a + 1
    if b >= pos.shape[0]:
        return
    pa = pos[a]; pb = pos[b]
    d = pb - pa; dist = wp.length(d)
    if dist < float(1.0e-8):
        return
    corr = d * ((dist - rest_len) / dist) * stiff[0]
    wa = inv_mass[a]; wb = inv_mass[b]; wt = wa + wb
    if wt < float(1.0e-8):
        return
    pos[a] = pa + corr * (wa / wt)
    pos[b] = pb - corr * (wb / wt)


@wp.kernel
def kernel_stretch_odd_gp(
    pos:      wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    rest_len: float,
    stiff:    wp.array(dtype=float),
):
    t = wp.tid()
    a = t * 2 + 1; b = a + 1
    if b >= pos.shape[0]:
        return
    pa = pos[a]; pb = pos[b]
    d = pb - pa; dist = wp.length(d)
    if dist < float(1.0e-8):
        return
    corr = d * ((dist - rest_len) / dist) * stiff[0]
    wa = inv_mass[a]; wb = inv_mass[b]; wt = wa + wb
    if wt < float(1.0e-8):
        return
    pos[a] = pa + corr * (wa / wt)
    pos[b] = pb - corr * (wb / wt)


@wp.kernel
def kernel_bend_gp(
    pos:      wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    rest_len: float,
    stiff:    wp.array(dtype=float),
):
    i = wp.tid(); j = i + 2
    if j >= pos.shape[0]:
        return
    pa = pos[i]; pb = pos[j]
    d = pb - pa; dist = wp.length(d)
    if dist < float(1.0e-8):
        return
    corr = d * ((dist - rest_len * float(2.0)) / dist) * stiff[0]
    wa = inv_mass[i]; wb = inv_mass[j]; wt = wa + wb
    if wt < float(1.0e-8):
        return
    pos[i] = pa + corr * (wa / wt)
    pos[j] = pb - corr * (wb / wt)


@wp.kernel
def kernel_vf_project_gp(
    pos:       wp.array(dtype=wp.vec3),
    inv_mass:  wp.array(dtype=float),
    vf_active: wp.array(dtype=int),
    vf_cp:     wp.array(dtype=wp.vec3),
    vf_dist:   wp.array(dtype=float),
    vf_normal: wp.array(dtype=wp.vec3),
    r:         wp.array(dtype=float),
    stiffness: wp.array(dtype=float),
):
    i = wp.tid()
    if inv_mass[i] == float(0.0) or vf_active[i] == 0 or vf_dist[i] >= r[0]:
        return
    target = vf_cp[i] + vf_normal[i] * r[0]
    pos[i]  = pos[i] + (target - pos[i]) * stiffness[0]


@wp.kernel
def kernel_ee_project_gp(
    pos:        wp.array(dtype=wp.vec3),
    inv_mass:   wp.array(dtype=float),
    yarn_edges: wp.array(dtype=wp.vec2i),
    parity:     int,
    ee_active:  wp.array(dtype=int),
    ee_cp_yarn: wp.array(dtype=wp.vec3),
    ee_cp_obs:  wp.array(dtype=wp.vec3),
    ee_dist:    wp.array(dtype=float),
    ee_normal:  wp.array(dtype=wp.vec3),
    ee_s:       wp.array(dtype=float),
    r:          wp.array(dtype=float),
    stiffness:  wp.array(dtype=float),
):
    i = wp.tid()
    if (i & 1) != parity or ee_active[i] == 0 or ee_dist[i] >= r[0]:
        return
    ev = yarn_edges[i]
    wa = inv_mass[ev[0]]; wb = inv_mass[ev[1]]; wt = wa + wb
    if wt < float(1.0e-12):
        return
    target_cp = ee_cp_obs[i] + ee_normal[i] * r[0]
    delta     = (target_cp - ee_cp_yarn[i]) * stiffness[0]
    s = ee_s[i]; ca = float(1.0) - s; cb = s
    w_denom = ca * ca * wa + cb * cb * wb
    if w_denom < float(1.0e-12):
        return
    pos[ev[0]] = pos[ev[0]] + delta * (ca * wa / w_denom)
    pos[ev[1]] = pos[ev[1]] + delta * (cb * wb / w_denom)


@wp.kernel
def kernel_vf_friction_gp(
    pos:       wp.array(dtype=wp.vec3),
    prev_pos:  wp.array(dtype=wp.vec3),
    inv_mass:  wp.array(dtype=float),
    vf_active: wp.array(dtype=int),
    vf_dist:   wp.array(dtype=float),
    vf_normal: wp.array(dtype=wp.vec3),
    r:         wp.array(dtype=float),
    mu_s:      wp.array(dtype=float),
    mu_k:      wp.array(dtype=float),
):
    i = wp.tid()
    if inv_mass[i] == float(0.0) or vf_active[i] == 0 or vf_dist[i] >= r[0]:
        return
    penetration  = r[0] - vf_dist[i]
    correction_n = wp.max(penetration, r[0] * float(0.5))
    n       = vf_normal[i]
    delta   = pos[i] - prev_pos[i]
    delta_t = delta - n * wp.dot(delta, n)
    len_t   = wp.length(delta_t)
    if len_t < float(1.0e-6):
        return
    if len_t <= mu_s[0] * correction_n:
        pos[i] = pos[i] - delta_t
    else:
        scale  = wp.min(mu_k[0] * correction_n / len_t, float(1.0))
        pos[i] = pos[i] - delta_t * scale


@wp.kernel
def kernel_ee_friction_gp(
    pos:        wp.array(dtype=wp.vec3),
    prev_pos:   wp.array(dtype=wp.vec3),
    inv_mass:   wp.array(dtype=float),
    yarn_edges: wp.array(dtype=wp.vec2i),
    parity:     int,
    ee_active:  wp.array(dtype=int),
    ee_dist:    wp.array(dtype=float),
    ee_normal:  wp.array(dtype=wp.vec3),
    ee_s:       wp.array(dtype=float),
    r:          wp.array(dtype=float),
    mu_s:       wp.array(dtype=float),
    mu_k:       wp.array(dtype=float),
):
    i = wp.tid()
    if (i & 1) != parity or ee_active[i] == 0 or ee_dist[i] >= r[0]:
        return
    penetration  = r[0] - ee_dist[i]
    correction_n = wp.max(penetration, r[0] * float(0.5))
    ev = yarn_edges[i]
    wa = inv_mass[ev[0]]; wb = inv_mass[ev[1]]; wt = wa + wb
    if wt < float(1.0e-12):
        return
    s = ee_s[i]; ca = float(1.0) - s; cb = s
    cp_now  = pos[ev[0]] * ca + pos[ev[1]] * cb
    cp_prev = prev_pos[ev[0]] * ca + prev_pos[ev[1]] * cb
    delta   = cp_now - cp_prev
    n       = ee_normal[i]
    delta_t = delta - n * wp.dot(delta, n)
    len_t   = wp.length(delta_t)
    if len_t < float(1.0e-6):
        return
    if len_t <= mu_s[0] * correction_n:
        scale = float(1.0)
    else:
        scale = wp.min(mu_k[0] * correction_n / len_t, float(1.0))
    correction = -delta_t * scale
    w_denom = ca * ca * wa + cb * cb * wb
    if w_denom < float(1.0e-12):
        return
    pos[ev[0]] = pos[ev[0]] + correction * (ca * wa / w_denom)
    pos[ev[1]] = pos[ev[1]] + correction * (cb * wb / w_denom)


@wp.kernel
def kernel_vf_damp_normal_velocity_gp(
    vel:       wp.array(dtype=wp.vec3),
    inv_mass:  wp.array(dtype=float),
    vf_active: wp.array(dtype=int),
    vf_dist:   wp.array(dtype=float),
    vf_normal: wp.array(dtype=wp.vec3),
    r:         wp.array(dtype=float),
):
    i = wp.tid()
    if inv_mass[i] == float(0.0) or vf_active[i] == 0:
        return
    skin = r[0] * float(0.5)
    if vf_dist[i] > r[0] + skin:
        return
    n        = vf_normal[i]
    v_inward = wp.dot(vel[i], n)
    if v_inward < float(0.0):
        vel[i] = vel[i] - n * v_inward


@wp.kernel
def kernel_clamp_velocity_gp(
    vel:   wp.array(dtype=wp.vec3),
    v_max: wp.array(dtype=float),
):
    i = wp.tid()
    if v_max[0] <= float(0.0):
        return
    v_mag = wp.length(vel[i])
    if v_mag > v_max[0]:
        vel[i] = vel[i] * (v_max[0] / v_mag)


# ── Python wrappers for GP contact kernels ────────────────────────────────────

def project_vf_gp(pos, inv_mass, vf, gp_r, gp_stiff, device):
    wp.launch(kernel_vf_project_gp, dim=pos.shape[0], device=device,
              inputs=[pos, inv_mass, vf.active, vf.cp, vf.dist, vf.normal,
                      gp_r, gp_stiff])


def project_ee_gp(pos, inv_mass, yarn_edges, ee, gp_r, gp_stiff, device):
    n = yarn_edges.shape[0]
    for parity in (0, 1):
        wp.launch(kernel_ee_project_gp, dim=n, device=device,
                  inputs=[pos, inv_mass, yarn_edges, parity,
                          ee.active, ee.cp_yarn, ee.cp_obs,
                          ee.dist, ee.normal, ee.s, gp_r, gp_stiff])


def apply_vf_friction_gp(pos, prev_pos, inv_mass, vf, gp_r, gp_mu_s, gp_mu_k, device):
    wp.launch(kernel_vf_friction_gp, dim=pos.shape[0], device=device,
              inputs=[pos, prev_pos, inv_mass,
                      vf.active, vf.dist, vf.normal, gp_r, gp_mu_s, gp_mu_k])


def apply_ee_friction_gp(pos, prev_pos, inv_mass, yarn_edges, ee,
                          gp_r, gp_mu_s, gp_mu_k, device):
    n = yarn_edges.shape[0]
    for parity in (0, 1):
        wp.launch(kernel_ee_friction_gp, dim=n, device=device,
                  inputs=[pos, prev_pos, inv_mass, yarn_edges, parity,
                          ee.active, ee.dist, ee.normal, ee.s,
                          gp_r, gp_mu_s, gp_mu_k])


def damp_normal_velocity_gp(vel, inv_mass, vf, gp_r, device):
    wp.launch(kernel_vf_damp_normal_velocity_gp, dim=vel.shape[0], device=device,
              inputs=[vel, inv_mass, vf.active, vf.dist, vf.normal, gp_r])


def clamp_velocity_gp(vel, gp_v_max, device):
    wp.launch(kernel_clamp_velocity_gp, dim=vel.shape[0], device=device,
              inputs=[vel, gp_v_max])


# ── Kernel: passive roll rotation driven by yarn tension ─────────────────────

@wp.kernel
def kernel_roll_a_torque_update(
    pos:             wp.array(dtype=wp.vec3),
    center:          wp.vec3,
    ra:              float,   # physical roll radius (torque arm)
    orbit_r:         float,   # ra + ogc_r — particle sits on the OGC offset surface
    rest_len:        float,
    stretch_stiff:   float,
    particle_mass:   float,
    roll_mass:       float,
    sub_dt:          float,
    bearing_damping: float,   # per-substep viscous drag on omega  [0..1]
    torque_scale:    float,   # dimensionless gain on Δω to compensate PBD overestimate
    omega_max:       float,
    angle:           wp.array(dtype=float),   # size-1: current departure angle
    omega:           wp.array(dtype=float),   # size-1: current angular velocity
):
    """Single-thread kernel.

    Reads pos[1] (yarn particle adjacent to departure) to estimate the
    tangential tension the yarn exerts on Roll A, integrates Roll A's
    angular velocity (I = 0.5 * M * r²), then writes the new pos[0].

    Force model (PBD-compatible):
        F ≈ particle_mass * stretch_stiff * stretch / sub_dt²
        tau = F_tangential * ra
        Δω  = torque_scale * tau * sub_dt / I
            = torque_scale * particle_mass * stretch_stiff * stretch
              * tan_comp * 2 / (roll_mass * ra * sub_dt)

    bearing_damping multiplies omega each substep, acting as axle friction.
    """
    ang     = angle[0]
    tangent = wp.vec3(-wp.sin(ang), wp.cos(ang), float(0.0))

    seg     = pos[1] - pos[0]
    seg_len = wp.length(seg)

    new_omega = omega[0]
    if seg_len > float(1.0e-6):
        seg_dir   = seg / seg_len
        stretch   = wp.max(seg_len - rest_len, float(0.0))
        tan_comp  = wp.dot(seg_dir, tangent)
        # Δω from tangential yarn impulse on roll surface (ra/I = 2/(M*ra))
        delta_omega = (torque_scale * particle_mass * stretch_stiff * stretch
                       * tan_comp * float(2.0) / (roll_mass * ra * sub_dt))
        new_omega = new_omega + delta_omega

    new_omega = new_omega * bearing_damping
    new_omega = wp.clamp(new_omega, -omega_max, omega_max)
    new_angle = ang + new_omega * sub_dt

    omega[0] = new_omega
    angle[0] = new_angle
    pos[0]   = center + wp.vec3(orbit_r * wp.cos(new_angle),
                                orbit_r * wp.sin(new_angle),
                                float(0.0))


def roll_a_torque_step(
    pos:             wp.array,
    center:          wp.vec3,
    ra:              float,
    orbit_r:         float,
    rest_len:        float,
    stretch_stiff:   float,
    particle_mass:   float,
    roll_mass:       float,
    sub_dt:          float,
    bearing_damping: float,
    torque_scale:    float,
    omega_max:       float,
    angle_wp:        wp.array,
    omega_wp:        wp.array,
    device:          str,
):
    wp.launch(
        kernel_roll_a_torque_update, dim=1, device=device,
        inputs=[pos, center, ra, orbit_r, rest_len, stretch_stiff,
                particle_mass, roll_mass, sub_dt,
                bearing_damping, torque_scale, omega_max,
                angle_wp, omega_wp],
    )


# ── Kernel: motor-driven roll B rotation ─────────────────────────────────────

@wp.kernel
def kernel_roll_b_motor_step(
    pos:        wp.array(dtype=wp.vec3),
    center:     wp.vec3,
    rb:         float,   # physical roll radius (used for angular speed)
    orbit_r:    float,   # rb + ogc_r — particle sits on the OGC offset surface
    pull_speed: float,
    sub_dt:     float,
    n_last:     int,
    angle:      wp.array(dtype=float),   # size-1: current winding angle
):
    """Advance Roll B's winding angle by pull_speed * sub_dt / rb and write pos[n_last].

    Roll B is motor-driven: its angular velocity is prescribed as pull_speed / rb.
    The yarn end (particle n_last) is placed at orbit_r = rb + ogc_r from the
    center so it sits exactly on the OGC offset surface and OGC never fires there.
    """
    new_angle   = angle[0] + pull_speed * sub_dt / rb
    angle[0]    = new_angle
    pos[n_last] = center + wp.vec3(orbit_r * wp.cos(new_angle),
                                   orbit_r * wp.sin(new_angle),
                                   float(0.0))


def roll_b_motor_step(
    pos:        wp.array,
    center:     wp.vec3,
    rb:         float,
    orbit_r:    float,
    pull_speed: float,
    sub_dt:     float,
    n_last:     int,
    angle_wp:   wp.array,
    device:     str,
):
    wp.launch(
        kernel_roll_b_motor_step, dim=1, device=device,
        inputs=[pos, center, rb, orbit_r, pull_speed, sub_dt, n_last, angle_wp],
    )


# ── Kernel: set a single kinematic particle ───────────────────────────────────

@wp.kernel
def kernel_set_particle(
    pos: wp.array(dtype=wp.vec3),
    idx: int,
    p:   wp.vec3,
):
    pos[idx] = p


def set_particle(
    pos:    wp.array,
    idx:    int,
    p:      wp.vec3,
    device: str,
):
    wp.launch(kernel_set_particle, dim=1, device=device,
              inputs=[pos, idx, p])


# ── Kernel: set two kinematic endpoint particles ──────────────────────────────

@wp.kernel
def kernel_set_endpoints(
    pos:    wp.array(dtype=wp.vec3),
    p0:     wp.vec3,
    pN:     wp.vec3,
    n_last: int,
):
    pos[0]      = p0
    pos[n_last] = pN


def set_endpoints(
    pos:    wp.array,
    p0:     wp.vec3,
    pN:     wp.vec3,
    n_last: int,
    device: str,
):
    wp.launch(kernel_set_endpoints, dim=1, device=device,
              inputs=[pos, p0, pN, n_last])
