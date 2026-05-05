"""algorithm6.py — Yarn self-collision projection and friction.

Mirrors algorithm4.py's EE kernels but both edges are yarn edges.

Each yarn edge i corrects only its own two endpoints (ev[0], ev[1]).
The complementary edge j corrects its own endpoints when thread j runs —
so all four endpoints are corrected without data races.

Even/odd parity splitting makes the two-pass execution race-free:
  pass 0 → even edges  (0, 2, 4, …) write to endpoints (0,1), (2,3), …
  pass 1 → odd  edges  (1, 3, 5, …) write to endpoints (1,2), (3,4), …
Endpoints are owned by exactly one pass, so no simultaneous writes.
"""

from __future__ import annotations

import warp as wp


@wp.kernel
def kernel_self_ee_project(
    pos:        wp.array(dtype=wp.vec3),
    inv_mass:   wp.array(dtype=float),
    yarn_edges: wp.array(dtype=wp.vec2i),
    parity:     int,
    active:     wp.array(dtype=int),
    cp_self:    wp.array(dtype=wp.vec3),
    cp_other:   wp.array(dtype=wp.vec3),
    dist:       wp.array(dtype=float),
    normal:     wp.array(dtype=wp.vec3),
    s_arr:      wp.array(dtype=float),
    r:          float,
    stiffness:  float,
):
    i = wp.tid()
    if (i & 1) != parity or active[i] == 0 or dist[i] >= r:
        return
    ev = yarn_edges[i]
    wa = inv_mass[ev[0]]; wb = inv_mass[ev[1]]
    wt = wa + wb
    if wt < float(1.0e-12):
        return
    # Push cp_self to distance r from cp_other along the outward normal.
    target_cp = cp_other[i] + normal[i] * r
    delta     = (target_cp - cp_self[i]) * stiffness
    s  = s_arr[i]
    ca = float(1.0) - s
    cb = s
    w_denom = ca * ca * wa + cb * cb * wb
    if w_denom < float(1.0e-12):
        return
    pos[ev[0]] = pos[ev[0]] + delta * (ca * wa / w_denom)
    pos[ev[1]] = pos[ev[1]] + delta * (cb * wb / w_denom)


@wp.kernel
def kernel_self_ee_friction(
    pos:        wp.array(dtype=wp.vec3),
    prev_pos:   wp.array(dtype=wp.vec3),
    inv_mass:   wp.array(dtype=float),
    yarn_edges: wp.array(dtype=wp.vec2i),
    parity:     int,
    active:     wp.array(dtype=int),
    dist:       wp.array(dtype=float),
    normal:     wp.array(dtype=wp.vec3),
    s_arr:      wp.array(dtype=float),
    r:          float,
    mu_s:       float,
    mu_k:       float,
):
    i = wp.tid()
    if (i & 1) != parity or active[i] == 0 or dist[i] >= r:
        return
    penetration  = r - dist[i]
    correction_n = wp.max(penetration, r * float(0.5))
    ev = yarn_edges[i]
    wa = inv_mass[ev[0]]; wb = inv_mass[ev[1]]
    wt = wa + wb
    if wt < float(1.0e-12):
        return
    s  = s_arr[i]
    ca = float(1.0) - s
    cb = s
    cp_now  = pos[ev[0]] * ca      + pos[ev[1]] * cb
    cp_prev = prev_pos[ev[0]] * ca + prev_pos[ev[1]] * cb
    delta   = cp_now - cp_prev
    n       = normal[i]
    delta_t = delta - n * wp.dot(delta, n)
    len_t   = wp.length(delta_t)
    if len_t < float(1.0e-6):
        return
    if len_t <= mu_s * correction_n:
        scale = float(1.0)
    else:
        scale = wp.min(mu_k * correction_n / len_t, float(1.0))
    correction = -delta_t * scale
    w_denom = ca * ca * wa + cb * cb * wb
    if w_denom < float(1.0e-12):
        return
    pos[ev[0]] = pos[ev[0]] + correction * (ca * wa / w_denom)
    pos[ev[1]] = pos[ev[1]] + correction * (cb * wb / w_denom)


def project_self_ee(
    pos:        wp.array,
    inv_mass:   wp.array,
    yarn_edges: wp.array,
    self_ee,    # SelfEEContacts
    r:          float,
    stiffness:  float,
    device:     str,
):
    n = yarn_edges.shape[0]
    for parity in (0, 1):
        wp.launch(
            kernel_self_ee_project, dim=n, device=device,
            inputs=[pos, inv_mass, yarn_edges, parity,
                    self_ee.active, self_ee.cp_self, self_ee.cp_other,
                    self_ee.dist, self_ee.normal, self_ee.s,
                    float(r), float(stiffness)],
        )


def apply_self_ee_friction(
    pos:        wp.array,
    prev_pos:   wp.array,
    inv_mass:   wp.array,
    yarn_edges: wp.array,
    self_ee,    # SelfEEContacts
    r:          float,
    mu_s:       float,
    mu_k:       float,
    device:     str,
):
    n = yarn_edges.shape[0]
    for parity in (0, 1):
        wp.launch(
            kernel_self_ee_friction, dim=n, device=device,
            inputs=[pos, prev_pos, inv_mass, yarn_edges, parity,
                    self_ee.active, self_ee.dist, self_ee.normal, self_ee.s,
                    float(r), float(mu_s), float(mu_k)],
        )
