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
